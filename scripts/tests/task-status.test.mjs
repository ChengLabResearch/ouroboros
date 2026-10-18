import './helpers/renderer-loader.mjs'
import assert from 'node:assert/strict'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

const { openTaskStatusStream, emptyStreamResult } =
	await import('../../src/renderer/src/interfaces/task-status-stream.ts')
const { parseTaskStatus } = await import('../../src/renderer/src/schemas/task-status-schema.ts')
const { default: ProgressPanel } =
	await import('../../src/renderer/src/components/ProgressPanel/Progress.tsx')

class FakeEventSource extends EventTarget {
	static instances = []
	closeCount = 0
	constructor(url) {
		super()
		this.url = url
		FakeEventSource.instances.push(this)
	}
	close() {
		this.closeCount++
	}
	emit(type, payload, taskId = 'current-task') {
		this.dispatchEvent(
			new MessageEvent(type, { data: JSON.stringify(payload), lastEventId: taskId })
		)
	}
}

function fixture(t, operation) {
	t.mock.method(globalThis, 'EventSource', FakeEventSource) // Installed below for Node without EventSource.
	let state = emptyStreamResult()
	let deliveries = 0
	const stream = openTaskStatusStream(
		`http://test/${operation}_status_stream/?task_id=current-task`,
		'current-task',
		(update) => {
			state = { ...state, ...update }
			deliveries++
		}
	)
	t.after(stream.close)
	return {
		stream,
		source: FakeEventSource.instances.at(-1),
		get state() {
			return state
		},
		get deliveries() {
			return deliveries
		},
		render() {
			return renderToStaticMarkup(
				createElement(ProgressPanel, {
					operation: operation === 'slice' ? 'Slicing' : 'Backprojection',
					stream: state,
					connected: true
				})
			)
		}
	}
}
// EventSource is provided by the browser in production; all network I/O here is simulated.
if (!globalThis.EventSource) globalThis.EventSource = FakeEventSource

function payload(status = 'done', publication = 'published') {
	const scienceError =
		status === 'error'
			? 'Scientific processing failed'
			: status === 'interrupted'
				? 'Scientific processing interrupted'
				: null
	const reportError = ['failed', 'interrupted', 'unavailable'].includes(publication)
		? 'Report transfer failed'
		: null
	return {
		status,
		progress: [['Data operation', status === 'done' ? 1 : 0.4, 1.25]],
		error: scienceError || reportError,
		progress_error: null,
		statistics: {
			status: publication,
			path:
				publication === 'disabled' || publication === 'unavailable'
					? null
					: '/host/output/report.json',
			recovery_path: reportError ? '/volume/main-statistics/report.json' : null,
			error: reportError,
			presentation_error: null,
			presentation_interrupted: false
		}
	}
}

for (const operation of ['slice', 'backproject']) {
	const label = operation === 'slice' ? 'Slicing' : 'Backprojection'
	for (const science of ['done', 'error', 'interrupted']) {
		for (const publication of [
			'published',
			'failed',
			'interrupted',
			'unavailable',
			'disabled'
		]) {
			test(`${operation}: ${science} science with ${publication} report`, async (t) => {
				const run = fixture(t, operation)
				const input = payload(science, publication)
				// Feed the HTTP status body through the same parser used at SSE ingress.
				assert.equal(parseTaskStatus(input).error, null)
				const event = input.error ? 'error_event' : 'done_event'
				run.source.emit(event, input)
				await run.stream.finished
				assert.equal(run.state.results.status, science)
				assert.deepEqual(run.state.results.progress, input.progress)
				assert.equal(run.state.error.status, false)
				assert.equal(run.state.done, true)
				const html = run.render()
				assert.match(
					html,
					new RegExp(
						`${label} ${science === 'done' ? 'completed' : science === 'error' ? 'failed' : 'interrupted'}\\.`
					)
				)
				assert.match(html, new RegExp(`width:${science === 'done' ? 100 : 40}%`))
				if (science !== 'done') assert.match(html, /Scientific processing/)
				if (['failed', 'interrupted', 'unavailable'].includes(publication)) {
					assert.match(html, /alert-warning/)
					assert.match(html, /Report transfer failed/)
					assert.match(
						html,
						/Statistics recovery location: \/volume\/main-statistics\/report.json/
					)
					assert.doesNotMatch(html, /Statistics report: \/host/)
					if (publication !== 'unavailable')
						assert.match(
							html,
							/Intended statistics destination \(not published\): \/host/
						)
					assert.ok(html.indexOf(label) < html.indexOf('Report transfer failed'))
				} else if (publication === 'published') {
					assert.match(html, /Statistics report: \/host\/output\/report.json/)
					assert.doesNotMatch(html, /alert-warning|recovery location|not published/)
				} else {
					assert.doesNotMatch(html, /Statistics|alert-warning/)
				}
				if (science === 'done') assert.doesNotMatch(html, /alert-error/)
				// Duplicate terminal events and a subsequent network error cannot re-alert.
				const count = run.deliveries
				run.source.emit(event, input)
				run.source.dispatchEvent(new Event('error'))
				assert.equal(run.deliveries, count)
				assert.equal(run.source.closeCount, 1)
				assert.equal(run.render(), html)
				assert.equal(
					(
						html.match(new RegExp(`${label} (completed|failed|interrupted)\\.`, 'g')) ??
						[]
					).length,
					1
				)
			})
		}
	}

	test(`${operation}: pending publication remains active and preserves final step progress`, (t) => {
		const run = fixture(t, operation)
		const input = payload('started', 'pending')
		input.progress[0][1] = 1
		run.source.emit('update_event', input)
		assert.equal(run.state.done, false)
		assert.equal(run.source.closeCount, 0)
		assert.match(run.render(), /width:100%/)
		assert.doesNotMatch(run.render(), /completed|alert-error|alert-warning|Statistics report:/)
	})

	test(`${operation}: transport failure preserves last progress without inventing an outcome`, async (t) => {
		const run = fixture(t, operation)
		run.source.emit('update_event', payload('started', 'pending'))
		const sources = FakeEventSource.instances.length
		run.source.dispatchEvent(new Event('error'))
		await run.stream.finished
		assert.equal(run.state.results.status, 'started')
		assert.equal(run.state.done, true)
		assert.match(run.render(), /width:40%/)
		assert.match(run.render(), /Connection to the status stream was lost/)
		assert.match(run.render(), /task outcome is unknown/)
		assert.doesNotMatch(run.render(), new RegExp(`${label} (failed|completed|interrupted)`))
		assert.equal(FakeEventSource.instances.length, sources)
	})

	test(`${operation}: stale task IDs and replaced streams cannot update presentation`, async (t) => {
		const old = fixture(t, operation)
		old.source.emit('error_event', payload('done', 'failed'), 'stale-task')
		assert.equal(old.state.results, null)
		assert.equal(old.source.closeCount, 0)
		old.stream.close()
		await old.stream.finished
		const current = fixture(t, operation)
		old.source.emit('done_event', payload())
		old.source.dispatchEvent(new Event('error'))
		assert.equal(old.state.results, null)
		assert.equal(current.state.results, null)
		current.source.emit('done_event', payload())
		assert.equal(current.state.results.status, 'done')
	})

	test(`${operation}: delivered report with log presentation failure stays delivered`, (t) => {
		const run = fixture(t, operation)
		const input = payload()
		input.error = input.statistics.presentation_error = 'Logging unavailable'
		input.statistics.presentation_interrupted = true
		run.source.emit('error_event', input)
		const html = run.render()
		assert.match(html, /completed/)
		assert.match(html, /Statistics log presentation interrupted: Logging unavailable/)
		assert.match(html, /Statistics report: \/host/)
		assert.doesNotMatch(html, /not published|Statistics delivery failed|alert-error/)
	})
}

for (const malformed of [
	'not JSON',
	JSON.stringify({ status: 'done', progress: [], error: null, statistics: { status: 'invalid' } })
]) {
	test(`malformed status is a communication problem: ${malformed}`, (t) => {
		const run = fixture(t, 'slice')
		run.source.emit('update_event', payload('started', 'pending'))
		run.source.dispatchEvent(
			new MessageEvent('done_event', { data: malformed, lastEventId: 'current-task' })
		)
		assert.equal(run.state.results.status, 'started')
		assert.equal(run.state.done, true)
		assert.match(run.render(), /task outcome could not be determined/)
		assert.doesNotMatch(run.render(), /Slicing completed|Slicing failed/)
	})
}

test('legacy status without a publication descriptor does not invent a missing report', (t) => {
	const run = fixture(t, 'slice')
	const input = payload()
	delete input.statistics
	run.source.emit('done_event', input)
	assert.match(run.render(), /Slicing completed/)
	assert.doesNotMatch(run.render(), /Statistics|alert-warning/)
})

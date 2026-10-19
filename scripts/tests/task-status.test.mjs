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
		render(props = {}) {
			return renderToStaticMarkup(
				createElement(ProgressPanel, {
					operation: operation === 'slice' ? 'Slicing' : 'Backprojection',
					stream: state,
					connected: true,
					...props
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

function structuredPayload(operation, overrides = {}) {
	return {
		...payload('started', 'disabled'),
		phase: 'processing',
		failure_phase: null,
		// Deliberately different from the structured snapshot: the renderer must
		// derive the displayed status, fraction and timing from a single source.
		progress: [['Legacy progress representation', 0.9, 99]],
		steps: [
			{
				name:
					operation === 'slice' ? 'SliceParallelPipelineStep' : 'BackprojectPipelineStep',
				status: 'running',
				progress: 0.4,
				duration_seconds: 1.25,
				stage:
					operation === 'slice' ? 'downloading_and_slicing' : 'backprojecting_and_writing'
			}
		],
		...overrides
	}
}

for (const operation of ['slice', 'backproject']) {
	const label = operation === 'slice' ? 'Slicing' : 'Backprojection'

	test(`${operation}: awaiting the initial snapshot does not claim there are no active tasks`, (t) => {
		const run = fixture(t, operation)
		assert.equal(run.state.results, null)
		assert.match(run.render(), /No task status yet\./)
		assert.doesNotMatch(run.render(), /No active tasks|Task active|role="progressbar"/)

		const input = structuredPayload(operation)
		input.steps[0].name = 'ParseJSONPipelineStep'
		input.steps[0].stage = null
		run.source.emit('update_event', input)
		assert.match(run.render(), /Processing/)
		assert.match(run.render(), /Read annotations and source/)
		assert.doesNotMatch(
			run.render(),
			/No task status yet|No active tasks|ParseJSONPipelineStep/
		)
	})

	test(`${operation}: a stream failure before the first snapshot leaves the task outcome unknown`, async (t) => {
		const run = fixture(t, operation)
		run.source.dispatchEvent(new Event('error'))
		await run.stream.finished
		assert.equal(run.state.results, null)
		assert.equal(run.state.done, true)
		const html = run.render()
		assert.match(html, /Connection to the status stream was lost/)
		assert.match(html, /task outcome is unknown/)
		assert.doesNotMatch(
			html,
			/No task status yet|No active tasks|Task active|role="progressbar"/
		)
		assert.doesNotMatch(html, new RegExp(`${label} (failed|completed|interrupted)`))
	})

	test(`${operation}: structured status is preserved through SSE and owns the displayed progress`, (t) => {
		const run = fixture(t, operation)
		const input = structuredPayload(operation)
		const parsed = parseTaskStatus(input)
		assert.equal(parsed.error, null)
		assert.deepEqual(parsed.result.steps, input.steps)
		assert.equal(parsed.result.phase, input.phase)
		run.source.emit('update_event', input)
		assert.deepEqual(run.state.results.steps, input.steps)
		assert.equal(run.state.done, false)
		const html = run.render()
		assert.match(html, new RegExp(label))
		assert.match(html, /Running/)
		assert.match(html, /Processing/)
		assert.match(
			html,
			operation === 'slice'
				? /Downloading data and slicing/
				: /Backprojecting and writing output/
		)
		assert.match(html, /role="progressbar"/)
		assert.match(html, /aria-valuenow="40"/)
		assert.match(html, /width:40%/)
		assert.match(html, /1\.3s/)
		assert.doesNotMatch(html, /PipelineStep|Legacy progress representation|90%|99s/)
		assert.doesNotMatch(html, /Statistics|alert-warning|alert-error/)
	})

	for (const [stepStatus, taskStatus, badge] of [
		['succeeded', 'done', 'Complete'],
		['failed', 'error', 'Failed'],
		['interrupted', 'interrupted', 'Interrupted']
	]) {
		test(`${operation}: ${stepStatus} lifecycle remains informative with reporting disabled`, async (t) => {
			const run = fixture(t, operation)
			const input = structuredPayload(operation, { status: 'enqueued', phase: 'queued' })
			input.steps[0].status = 'pending'
			input.steps[0].progress = 0
			input.steps[0].stage = null
			run.source.emit('update_event', input)
			assert.match(run.render(), /Waiting in queue/)
			assert.match(run.render(), /Waiting/)
			assert.doesNotMatch(run.render(), /Running|completed|failed|interrupted/)

			input.status = 'started'
			input.phase = 'processing'
			input.steps[0].status = 'running'
			input.steps[0].progress = 0.4
			run.source.emit('update_event', input)
			assert.match(run.render(), /Running/)
			assert.equal(run.state.done, false)

			input.status = taskStatus
			input.phase = 'complete'
			input.steps[0].status = stepStatus
			input.steps[0].progress = stepStatus === 'succeeded' ? 1 : 0.4
			input.error = stepStatus === 'succeeded' ? null : 'Scientific processing stopped'
			input.failure_phase = stepStatus === 'succeeded' ? null : 'processing'
			run.source.emit(taskStatus === 'done' ? 'done_event' : 'error_event', input)
			await run.stream.finished
			assert.equal(run.state.done, true)
			assert.match(run.render(), new RegExp(badge))
			assert.match(
				run.render(),
				new RegExp(
					`${label} ${taskStatus === 'done' ? 'completed' : taskStatus === 'error' ? 'failed' : 'interrupted'}\\.`
				)
			)
			assert.doesNotMatch(run.render(), /Running|Statistics/)
			if (stepStatus !== 'succeeded') {
				assert.match(run.render(), /Scientific processing stopped/)
				assert.match(run.render(), /Failure stage: Processing/)
			}
		})
	}

	test(`${operation}: execution phases stay visible outside scientific processing`, (t) => {
		const run = fixture(t, operation)
		for (const [phase, text] of [
			['loading_input', 'Loading input'],
			['returning_output', 'Returning output'],
			['cleanup', 'Cleaning up'],
			['publishing_statistics', 'Publishing statistics report']
		]) {
			const input = structuredPayload(operation, { phase, steps: [], progress: [] })
			if (phase === 'publishing_statistics') input.statistics.status = 'pending'
			run.source.emit('update_event', input)
			assert.equal(run.state.done, false)
			assert.equal(run.source.closeCount, 0)
			assert.match(run.render(), new RegExp(text))
			assert.doesNotMatch(run.render(), new RegExp(`${label} completed`))
		}
	})

	for (const reporting of ['disabled', 'pending']) {
		test(`${operation}: 100% scientific progress does not finish pending output (${reporting} reporting)`, (t) => {
			const run = fixture(t, operation)
			const input = structuredPayload(operation, { phase: 'returning_output' })
			input.statistics.status = reporting
			input.steps[0].status = 'succeeded'
			input.steps[0].progress = 1
			run.source.emit('update_event', input)
			assert.equal(run.state.done, false)
			assert.equal(run.source.closeCount, 0)
			assert.match(run.render(), /width:100%/)
			assert.match(run.render(), /Complete/)
			assert.match(run.render(), /Returning output/)
			assert.doesNotMatch(run.render(), new RegExp(`${label} completed`))
		})
	}

	const indeterminateStages =
		operation === 'slice'
			? [
					['preparing_input', 'Preparing input'],
					['normalizing_and_writing', 'Finalizing TIFF output']
				]
			: [
					['preparing_input', 'Preparing input'],
					['exporting_tiff', 'Exporting TIFF'],
					['cleaning_intermediates', 'Cleaning intermediate files']
				]
	for (const [stage, stageLabel] of indeterminateStages) {
		test(`${operation}: ${stage} shows unknown progress without fabricating a percentage`, (t) => {
			const run = fixture(t, operation)
			const input = structuredPayload(operation)
			input.steps[0].progress = null
			input.steps[0].stage = stage
			run.source.emit('update_event', input)
			assert.equal(run.state.done, false)
			const html = run.render()
			assert.match(html, new RegExp(stageLabel))
			assert.match(html, /Running/)
			assert.match(html, /role="progressbar"/)
			assert.match(html, /aria-busy="true"/)
			assert.doesNotMatch(html, /aria-valuenow|\d+%|Legacy progress representation/)
		})
	}

	test(`${operation}: output failure retains successful scientific steps and identifies the failure phase`, async (t) => {
		const run = fixture(t, operation)
		const input = structuredPayload(operation, {
			status: 'error',
			phase: 'complete',
			failure_phase: 'returning_output',
			error: 'Output destination is unavailable'
		})
		input.steps[0].status = 'succeeded'
		input.steps[0].progress = 1
		run.source.emit('error_event', input)
		await run.stream.finished
		assert.equal(run.state.results.failure_phase, 'returning_output')
		const html = run.render()
		assert.match(html, new RegExp(`${label} failed`))
		assert.match(html, /Complete/)
		assert.match(html, /width:100%/)
		assert.match(html, /Failure stage: Returning output/)
		assert.match(html, /Output destination is unavailable/)
	})

	test(`${operation}: lost stream preserves the last structured snapshot without claiming live progress`, async (t) => {
		const run = fixture(t, operation)
		const input = structuredPayload(operation)
		input.steps[0].progress = null
		input.steps[0].stage = operation === 'slice' ? 'normalizing_and_writing' : 'exporting_tiff'
		run.source.emit('update_event', input)
		const lastSnapshot = run.state.results
		run.source.dispatchEvent(new Event('error'))
		await run.stream.finished
		assert.equal(run.state.results, lastSnapshot)
		const html = run.render()
		assert.match(html, /Last reported status/)
		assert.match(html, /Connection to the status stream was lost/)
		assert.match(html, /task outcome is unknown/)
		assert.match(html, /aria-busy="false"/)
		assert.doesNotMatch(html, /aria-busy="true"|indeterminate|Task active/)
		assert.doesNotMatch(html, /aria-valuenow|\d+%/)
		assert.doesNotMatch(html, new RegExp(`${label} (failed|completed|interrupted)`))
	})

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
				assert.match(html, /1\.3s/)
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

	test(`${operation}: reporting disabled preserves progress updates, timings and completion`, async (t) => {
		const run = fixture(t, operation)
		run.source.emit('update_event', payload('started', 'disabled'))
		assert.equal(run.state.done, false)
		assert.match(run.render(), /width:40%/)
		assert.match(run.render(), /1\.3s/)
		run.source.emit('done_event', payload('done', 'disabled'))
		await run.stream.finished
		assert.match(run.render(), /width:100%/)
		assert.match(run.render(), /1\.3s/)
		assert.match(run.render(), new RegExp(`${label} completed`))
		assert.doesNotMatch(run.render(), /Statistics|alert-warning|alert-error/)
	})

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

	for (const hasProgress of [false, true]) {
		test(`${operation}: unknown task ${hasProgress ? 'after progress' : 'before any progress'} has unknown outcome`, async (t) => {
			const run = fixture(t, operation)
			if (hasProgress) run.source.emit('update_event', payload('started', 'pending'))
			const lastStatus = run.state.results
			const missing = { status: 'error', progress: [], error: 'Item ID Not Found' }
			const sources = FakeEventSource.instances.length
			// A missing-task event from another task must not close the current stream.
			run.source.emit('error_event', missing, 'stale-task')
			assert.equal(run.source.closeCount, 0)
			run.source.emit('error_event', missing)
			await run.stream.finished
			assert.equal(run.state.results, lastStatus)
			assert.equal(run.state.done, true)
			const html = run.render()
			assert.match(html, /server cannot find this task/)
			assert.match(html, /outcome is unknown/)
			assert.match(html, /alert-warning/)
			assert.doesNotMatch(html, /alert-error/)
			assert.doesNotMatch(html, new RegExp(`${label} (failed|completed|interrupted)`))
			if (hasProgress) assert.match(html, /width:40%/)
			else assert.doesNotMatch(html, /width:/)
			const count = run.deliveries
			run.source.emit('error_event', missing)
			run.source.dispatchEvent(new Event('error'))
			assert.equal(run.deliveries, count)
			assert.equal(run.source.closeCount, 1)
			assert.equal(FakeEventSource.instances.length, sources)
		})
	}

	test(`${operation}: legacy scientific errors and known tasks are not lookup failures`, (t) => {
		const legacy = fixture(t, operation)
		const input = payload('error', 'disabled')
		delete input.statistics
		legacy.source.emit('error_event', input)
		assert.match(legacy.render(), new RegExp(`${label} failed`))
		assert.match(legacy.render(), /Scientific processing failed/)
		assert.doesNotMatch(legacy.render(), /outcome is unknown/)

		const known = fixture(t, operation)
		const knownError = payload('error', 'published')
		knownError.error = 'Item ID Not Found'
		known.source.emit('error_event', knownError)
		assert.match(known.render(), new RegExp(`${label} failed`))
		assert.match(known.render(), /Statistics report:/)
		assert.doesNotMatch(known.render(), /outcome is unknown/)
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

test('structured status with no steps does not fall back to unrelated legacy progress', (t) => {
	const run = fixture(t, 'slice')
	run.source.emit(
		'update_event',
		structuredPayload('slice', { phase: 'loading_input', steps: [] })
	)
	assert.match(run.render(), /Loading input/)
	assert.doesNotMatch(run.render(), /Legacy progress representation|90%|99s/)
})

test('an unreachable backend freezes the last reported activity while retaining its status', (t) => {
	const run = fixture(t, 'slice')
	const input = structuredPayload('slice')
	input.steps[0].progress = null
	input.steps[0].stage = 'normalizing_and_writing'
	run.source.emit('update_event', input)
	assert.match(run.render(), /aria-busy="true"/)
	const html = run.render({ connected: false })
	assert.match(html, /Last reported status/)
	assert.match(html, /Finalizing TIFF output/)
	assert.match(html, /aria-busy="false"/)
	assert.doesNotMatch(html, /aria-busy="true"|indeterminate|Task active/)
	assert.doesNotMatch(html, /Slicing (completed|failed|interrupted)/)
})

for (const [state, connected, expected] of [
	['idle', false, 'Waiting for backend'],
	['starting', false, 'Starting backend'],
	['started', false, 'Waiting for backend'],
	['started', true, 'Ready']
]) {
	test(`backend launch ${state}, API ${connected ? 'healthy' : 'unreachable'}: ${expected}`, (t) => {
		const run = fixture(t, 'slice')
		const html = run.render({ connected, launchStatus: { state, error: null } })
		assert.equal(run.state.results, null)
		assert.match(html, new RegExp(expected))
		assert.doesNotMatch(html, /Backend launch failed|Slicing (completed|failed)/)
		if (!connected) assert.doesNotMatch(html, /Ready/)
	})
}

test('backend launch failure is visible before a task or status stream response exists', (t) => {
	const run = fixture(t, 'backproject')
	const html = run.render({
		connected: false,
		launchStatus: { state: 'failed', error: 'Docker Desktop is not running' }
	})
	assert.equal(run.state.results, null)
	assert.match(html, /Backend launch failed/)
	assert.match(html, /Docker Desktop is not running/)
	assert.doesNotMatch(html, /Ready|Backprojection (completed|failed|interrupted)/)
})

import './helpers/renderer-loader.mjs'
import assert from 'node:assert/strict'
import { registerHooks } from 'node:module'
import { posix } from 'node:path'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

// Capture the real page's form callbacks while rendering the real form controls.
// Only SVG artwork and the unused WebGL visualization are replaced in Node.
const panelURL = new URL(
	'../../src/renderer/src/components/OptionsPanel/OptionsPanel.tsx',
	import.meta.url
)
const alertURL = new URL(
	'../../src/renderer/src/components/AlertArea/components/Alert/Alert.tsx',
	import.meta.url
)
const tooltipURL = new URL('../../src/renderer/src/components/Tooltip/Tooltip.tsx', import.meta.url)
registerHooks({
	resolve(specifier, context, nextResolve) {
		let source
		if (specifier === '@renderer/components/OptionsPanel/OptionsPanel') {
			source = `import Panel from ${JSON.stringify(panelURL.href)};
export default function Capture(props) {
  globalThis.performanceLoggingPanel = props;
  return Panel(props);
}`
		} else if (specifier.endsWith('/Alert/Alert')) {
			source = `import Alert from ${JSON.stringify(alertURL.href)};
export default function CaptureNotice(props) {
  globalThis.pageNotices.push(props);
  return Alert(props);
}`
		} else if (specifier === '@renderer/components/Tooltip/Tooltip') {
			source = `import { useTooltip as realTooltip } from ${JSON.stringify(tooltipURL.href)};
export function useTooltip(...args) {
  globalThis.performanceLoggingTooltips.push(args[1]);
  return realTooltip(...args);
}`
		} else if (specifier.endsWith('.svg?react') || specifier.endsWith('/VisualizeSlicing')) {
			source = 'export default function Artwork() { return null }'
		}
		if (source) {
			return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true }
		}
		return nextResolve(specifier, context)
	}
})

const { ServerContext } = await import('../../src/renderer/src/contexts/ServerContext.tsx')
const { DirectoryContext } = await import('../../src/renderer/src/contexts/DirectoryContext.tsx')
const { AlertContext } = await import('../../src/renderer/src/contexts/AlertContext.tsx')
const { DragContext } = await import('../../src/renderer/src/contexts/DragContext.tsx')
const { emptyStreamResult } =
	await import('../../src/renderer/src/interfaces/task-status-stream.ts')
const { default: OptionsPanel } = await import(panelURL.href)
const { default: SlicesPage } =
	await import('../../src/renderer/src/routes/SlicesPage/SlicesPage.tsx')
const { default: BackprojectPage } =
	await import('../../src/renderer/src/routes/BackprojectPage/BackprojectPage.tsx')
const { safeParse } = await import('valibot')

function fixture(t, operation, fetchStates = new Map()) {
	const requests = []
	const files = new Map()
	const alerts = []
	const fetchResult = { results: null, pending: false, error: { status: false, message: '' } }
	const stream = emptyStreamResult()
	const server = {
		connected: true,
		useFetchListener: (url) => fetchStates.get(url) ?? fetchResult,
		useStreamListener: () => stream,
		clearFetchError: (url) => {
			const previous = fetchStates.get(url)
			if (previous) fetchStates.set(url, { ...previous, error: fetchResult.error })
		},
		clearFetch: (url) => fetchStates.delete(url),
		clearStream: () => {},
		performFetch: async (url, query, options) => {
			requests.push({ url, query, options })
			fetchStates.set(url, fetchResult)
			return fetchResult
		}
	}
	const ipc = {
		invoke: async (channel, args) => {
			if (channel === 'join-path') return posix.join(...args)
			const path = posix.join(args.folder, args.name)
			if (channel === 'save-file') {
				files.set(path, args.data)
				return true
			}
			if (channel === 'read-file') {
				assert.ok(files.has(path), `Missing fixture file: ${path}`)
				return files.get(path)
			}
			assert.fail(`Unexpected IPC call: ${channel}`)
		}
	}
	globalThis.window = { electron: { ipcRenderer: ipc } }
	globalThis.performanceLoggingTooltips = []
	t.after(() => {
		delete globalThis.window
		delete globalThis.performanceLoggingPanel
		delete globalThis.pageNotices
		delete globalThis.performanceLoggingTooltips
	})
	const render = (component) => {
		for (const [context, value] of [
			[ServerContext, server],
			[DirectoryContext, { directoryPath: '/workspace' }],
			[AlertContext, { addAlert: (...args) => alerts.push(args) }],
			[DragContext, { parentChildData: null }]
		]) {
			component = createElement(context.Provider, { value }, component)
		}
		return renderToStaticMarkup(component)
	}
	const renderPage = (page = operation) => {
		globalThis.pageNotices = []
		return render(createElement(page === 'slice' ? SlicesPage : BackprojectPage))
	}
	const html = renderPage()
	const panel = globalThis.performanceLoggingPanel
	const entries = panel.entries[0]
	const logging = entries.findEntry('logging_mode')
	return {
		requests,
		files,
		alerts,
		server,
		fetchStates,
		stream,
		renderPage,
		get notices() {
			return globalThis.pageNotices
		},
		get tooltips() {
			return globalThis.performanceLoggingTooltips
		},
		get panel() {
			return globalThis.performanceLoggingPanel
		},
		entries,
		logging,
		html,
		renderForm: () => render(createElement(OptionsPanel, panel)),
		async submit() {
			const currentPanel = globalThis.performanceLoggingPanel
			// Fill required inputs without triggering an unrelated visualization request.
			for (const name of [
				'neuroglancer_json',
				'straightened_volume_path',
				'slice_options_path'
			]) {
				currentPanel.entries[0].findEntry(name)?.setValue(`./${name}.json`)
			}
			await currentPanel.onSubmit()
			const request = requests.at(-1)
			assert.equal(request.url, `/${operation}/`)
			assert.equal(request.options.method, 'POST')
			const saved = JSON.parse(files.get(request.query.options))
			assert.equal(request.query.logging_mode, saved.logging_mode)
			assert.equal('reporting_enabled' in request.query, false)
			assert.equal('reporting_enabled' in saved, false)
			assert.equal(alerts.length, 0)
			return request.query
		}
	}
}

function selectedMode(html) {
	const input = html.match(/<select[^>]*name="logging_mode"[^>]*>(.*?)<\/select>/)?.[1]
	assert.ok(input, 'Performance Logging must have a select')
	assert.deepEqual(
		[...input.matchAll(/<option value="([^"]+)"[^>]*>(.*?)<\/option>/g)].map(
			([, value, label]) => [value, label]
		),
		[
			['off', 'Off'],
			['standard', 'Standard'],
			['detailed', 'Detailed (with RAM)']
		]
	)
	assert.match(html, /Performance Logging/)
	return input.match(/<option value="([^"]+)" selected=""/)[1]
}

for (const operation of ['slice', 'backproject']) {
	test(`${operation}: submission errors are inline, dismiss at their owner, and clear on another submission`, async (t) => {
		const url = `/${operation}/`
		const failed = {
			results: null,
			pending: false,
			error: { status: true, message: 'Input could not be opened' }
		}
		const run = fixture(t, operation, new Map([[url, failed]]))
		assert.match(run.html, /Task submission failed: Input could not be opened/)
		assert.match(run.html, /type="button"[^>]*>Dismiss/)
		assert.equal(run.notices.filter((notice) => notice.onDismiss).length, 1)
		run.notices.find((notice) => notice.onDismiss).onDismiss()
		assert.equal(run.fetchStates.get(url).error.status, false)
		assert.doesNotMatch(run.renderPage(), /Input could not be opened|Dismiss/)

		run.fetchStates.set(url, failed)
		const performFetch = run.server.performFetch
		run.server.performFetch = async (...args) => {
			assert.equal(run.fetchStates.get(url).error.status, false)
			return performFetch(...args)
		}
		run.renderPage()
		await run.submit()
		assert.doesNotMatch(run.renderPage(), /Input could not be opened|Dismiss/)
		assert.deepEqual(run.alerts, [])
	})

	test(`${operation}: deletion errors belong to the previous task and dismissal preserves its outcome`, async (t) => {
		const task = { task_id: `previous-${operation}` }
		const url = `/${operation}/`
		const deleteURL = `/delete/?task_id=${task.task_id}`
		const success = { results: task, pending: false, error: { status: false, message: '' } }
		const failed = {
			...success,
			results: null,
			error: { status: true, message: 'Delete denied' }
		}
		const run = fixture(
			t,
			operation,
			new Map([
				[url, success],
				[deleteURL, failed]
			])
		)
		run.stream.results = { status: 'done', progress: [] }
		run.stream.done = true
		assert.match(run.renderPage(), /Previous task could not be deleted: Delete denied/)
		run.notices.find((notice) => notice.onDismiss).onDismiss()
		assert.equal(run.fetchStates.get(url).results, task)
		assert.equal(run.stream.results.status, 'done')
		assert.match(run.renderPage(), /completed\./)
		assert.doesNotMatch(run.renderPage(), /Delete denied|Dismiss/)

		run.fetchStates.set(deleteURL, failed)
		assert.doesNotMatch(
			run.renderPage(operation === 'slice' ? 'backproject' : 'slice'),
			/Delete denied|Dismiss/
		)
		run.server.performFetch = async (requestURL, query, options) => {
			assert.equal(requestURL, deleteURL)
			assert.deepEqual(query, {})
			assert.equal(options.method, 'POST')
			assert.equal(run.fetchStates.get(deleteURL).error.status, false)
			run.fetchStates.set(deleteURL, failed)
			return failed
		}
		run.renderPage()
		await run.panel.onSubmit()
		assert.match(run.renderPage(), /Previous task could not be deleted: Delete denied/)
		assert.equal(run.fetchStates.get(url).results, task)
		assert.deepEqual(run.alerts, [])
	})

	test(`${operation}: default off, saved value and successive run requests follow the select`, async (t) => {
		const run = fixture(t, operation)
		assert.equal(selectedMode(run.html), 'off')
		assert.equal((await run.submit()).logging_mode, 'off')
		for (const mode of ['standard', 'detailed', 'off', 'detailed']) {
			run.logging.setValue(mode)
			assert.equal(selectedMode(run.renderForm()), mode)
			assert.equal((await run.submit()).logging_mode, mode)
			const saved = run.files.get(run.requests.at(-1).query.options)
			run.logging.setValue(mode === 'off' ? 'detailed' : 'off')
			// Avoid visualization I/O when the slice options are dropped onto the header.
			const input = JSON.parse(saved)
			if (operation === 'slice') input.neuroglancer_json = ''
			run.files.set('/saved.json', JSON.stringify(input))
			await run.panel.onHeaderDrop('/saved.json')
			assert.equal(run.logging.value, mode)
			assert.equal((await run.submit()).logging_mode, mode)
		}
	})

	test(`${operation}: legacy settings migrate only at load, explicit modes win, and missing settings reset off`, async (t) => {
		const run = fixture(t, operation)
		const legacy = run.entries.toObject()
		delete legacy.logging_mode
		for (const [input, expected] of [
			[{ ...legacy, reporting_enabled: true }, 'detailed'],
			[{ ...legacy, reporting_enabled: false }, 'off'],
			[legacy, 'off'],
			[{ ...legacy, reporting_enabled: true, logging_mode: 'standard' }, 'standard'],
			[{ ...legacy, reporting_enabled: false, logging_mode: 'detailed' }, 'detailed'],
			[{ ...legacy, reporting_enabled: true, logging_mode: 'off' }, 'off'],
			[{ ...legacy, reporting_enabled: 'invalid', logging_mode: 'standard' }, 'standard']
		]) {
			run.logging.setValue('detailed')
			const parsed = safeParse(run.entries.toSchema(), input)
			assert.equal(parsed.success, true)
			assert.equal('reporting_enabled' in parsed.output, false)
			run.files.set('/legacy.json', JSON.stringify(input))
			await run.panel.onHeaderDrop('/legacy.json')
			assert.equal(run.logging.value, expected)
			assert.equal(selectedMode(run.renderForm()), expected)
			assert.equal((await run.submit()).logging_mode, expected)
		}
	})

	test(`${operation}: invalid saved logging settings are rejected without changing the selected mode`, async (t) => {
		const run = fixture(t, operation)
		const input = run.entries.toObject()
		delete input.logging_mode
		run.logging.setValue('standard')
		for (const invalid of [
			...['false', 'off', 'standard', 'detailed', null, 1].map((reporting_enabled) => ({
				reporting_enabled
			})),
			...['verbose', '', null, 1, true, false].map((logging_mode) => ({ logging_mode })),
			{ logging_mode: 'invalid', reporting_enabled: true }
		]) {
			run.files.set('/invalid.json', JSON.stringify({ ...input, ...invalid }))
			await run.panel.onHeaderDrop('/invalid.json')
			assert.match(run.alerts.at(-1)[0], /Wrong JSON file format/)
			assert.equal(run.logging.value, 'standard')
		}
	})

	test(`${operation}: the existing tooltip explains the three modes and RAM overhead`, (t) => {
		const run = fixture(t, operation)
		assert.ok(run.tooltips.includes(run.logging.description))
		assert.match(run.logging.description, /Off: Do not save performance reports/)
		assert.match(
			run.logging.description,
			/Standard: Save performance reports without RAM monitoring/
		)
		assert.match(
			run.logging.description,
			/Detailed \(with RAM\): Save full performance reports/
		)
		assert.match(run.logging.description, /RAM monitoring adds overhead/)
		assert.match(
			run.logging.description,
			/Progress bars and step timings remain available in all modes/
		)
	})

	for (const mode of ['off', 'standard', 'detailed']) {
		test(`${operation}: ${mode} submissions retain progress, basic timings, and completion`, async (t) => {
			const run = fixture(t, operation)
			run.logging.setValue(mode)
			assert.equal((await run.submit()).logging_mode, mode)
			const stepName =
				operation === 'slice' ? 'SliceParallelPipelineStep' : 'BackprojectPipelineStep'
			run.stream.results = {
				status: 'started',
				phase: 'processing',
				progress: [[stepName, 0.5, 1.25]],
				error: null,
				statistics: {
					status: mode === 'off' ? 'disabled' : 'pending',
					path: null,
					recovery_path: null,
					error: null
				}
			}
			const running = run.renderPage()
			assert.match(running, /role="progressbar"[^>]*aria-valuenow="50"/)
			assert.match(running, />1\.3s</)
			assert.match(running, />Processing</)
			run.stream.results.status = 'done'
			run.stream.results.phase = 'complete'
			run.stream.results.progress = [[stepName, 1, 2.5]]
			run.stream.results.statistics.status = mode === 'off' ? 'disabled' : 'published'
			run.stream.done = true
			const completed = run.renderPage()
			assert.match(completed, /role="progressbar"[^>]*aria-valuenow="100"/)
			assert.match(completed, />2\.5s</)
			assert.match(completed, /completed\./)
		})
	}
}

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
	t.after(() => {
		delete globalThis.window
		delete globalThis.performanceLoggingPanel
		delete globalThis.pageNotices
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
	const logging = entries.findEntry('reporting_enabled')
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
			assert.equal(request.query.reporting_enabled, saved.reporting_enabled)
			assert.equal(alerts.length, 0)
			return request.query
		}
	}
}

function checkbox(html) {
	const input = html.match(/<input[^>]*name="reporting_enabled"[^>]*>/)?.[0]
	assert.ok(input, 'Performance Logging must have an input')
	assert.match(input, /type="checkbox"/)
	assert.match(html, /Performance Logging/)
	return input
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

	test(`${operation}: default off, saved value and successive run requests follow the checkbox`, async (t) => {
		const run = fixture(t, operation)
		assert.doesNotMatch(checkbox(run.html), /checked/)
		assert.equal((await run.submit()).reporting_enabled, false)
		for (const enabled of [true, false, true]) {
			run.logging.setValue(enabled)
			assert.equal(/checked/.test(checkbox(run.renderForm())), enabled)
			assert.equal((await run.submit()).reporting_enabled, enabled)
			const saved = run.files.get(run.requests.at(-1).query.options)
			run.logging.setValue(!enabled)
			// Avoid visualization I/O when the slice options are dropped onto the header.
			const input = JSON.parse(saved)
			if (operation === 'slice') input.neuroglancer_json = ''
			run.files.set('/saved.json', JSON.stringify(input))
			await run.panel.onHeaderDrop('/saved.json')
			assert.equal(run.logging.value, enabled)
			assert.equal((await run.submit()).reporting_enabled, enabled)
		}
	})

	test(`${operation}: loading legacy options resets an enabled checkbox and rejects invalid values`, async (t) => {
		const run = fixture(t, operation)
		const legacy = run.entries.toObject()
		delete legacy.reporting_enabled
		run.logging.setValue(true)
		run.files.set('/legacy.json', JSON.stringify(legacy))
		await run.panel.onHeaderDrop('/legacy.json')
		assert.equal(run.logging.value, false)
		assert.doesNotMatch(checkbox(run.renderForm()), /checked/)
		assert.equal((await run.submit()).reporting_enabled, false)
		for (const invalid of ['false', null, 1]) {
			run.files.set(
				'/invalid.json',
				JSON.stringify({ ...legacy, reporting_enabled: invalid })
			)
			await run.panel.onHeaderDrop('/invalid.json')
			assert.match(run.alerts.at(-1)[0], /Wrong JSON file format/)
			assert.equal(run.logging.value, false)
		}
	})
}

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
registerHooks({
	resolve(specifier, context, nextResolve) {
		let source
		if (specifier === '@renderer/components/OptionsPanel/OptionsPanel') {
			source = `import Panel from ${JSON.stringify(panelURL.href)};
export default function Capture(props) {
  globalThis.performanceLoggingPanel = props;
  return Panel(props);
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

function fixture(t, operation) {
	const requests = []
	const files = new Map()
	const alerts = []
	const fetchResult = { results: null, pending: false, error: { status: false, message: '' } }
	const server = {
		connected: true,
		useFetchListener: () => fetchResult,
		useStreamListener: emptyStreamResult,
		performFetch: async (url, query, options) => {
			requests.push({ url, query, options })
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
	const html = render(createElement(operation === 'slice' ? SlicesPage : BackprojectPage))
	const panel = globalThis.performanceLoggingPanel
	const entries = panel.entries[0]
	const logging = entries.findEntry('reporting_enabled')
	return {
		requests,
		files,
		alerts,
		panel,
		entries,
		logging,
		html,
		renderForm: () => render(createElement(OptionsPanel, panel)),
		async submit() {
			// Fill required inputs without triggering an unrelated visualization request.
			for (const name of [
				'neuroglancer_json',
				'straightened_volume_path',
				'slice_options_path'
			]) {
				entries.findEntry(name)?.setValue(`./${name}.json`)
			}
			await panel.onSubmit()
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

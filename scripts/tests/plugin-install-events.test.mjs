import assert from 'node:assert/strict'
import { readFile, stat } from 'node:fs/promises'
import { join } from 'node:path'
import test from 'node:test'
import { loadCommonJsModule } from './helpers/load-commonjs-module.mjs'
import {
	pluginInstallFixture,
	pluginManifest,
	writePlugin
} from './helpers/plugin-install-fixture.mjs'

async function eventFixture(t, options = {}) {
	const run = await pluginInstallFixture(t, options)
	const handlers = new Map()
	const watchers = []
	const windowHandlers = new Map()
	const windowOptions = []
	const sent = []
	const dialogs = []
	const lifecycle = []
	const completion = Promise.withResolvers()
	let nextSend
	const window = {
		on: (name, listener) => windowHandlers.set(name, listener),
		webContents: {
			send: (...args) => {
				sent.push(args)
				nextSend?.resolve()
			}
		}
	}
	const details = [
		{ id: 'running-plugin', name: 'Running Plugin', indexPath: 'plugin/index.html' }
	]
	const events = loadCommonJsModule('src/main/event-handlers/plugins.ts', {
		electron: { dialog: { showErrorBox: (...args) => dialogs.push(args) } },
		'../plugins': {
			...run.plugins,
			// The real install/delete operations run against the fixture directory.
			// Record lifecycle dispatch instead of starting the app's plugin services.
			restartPlugins: (...args) => {
				lifecycle.push(['restart', ...args])
				completion.resolve()
			},
			initializePlugins: (...args) => {
				lifecycle.push(['initialize', ...args])
				completion.resolve()
			},
			stopAllPlugins: async () => {
				lifecycle.push(['stop'])
				await options.stop?.()
			},
			...(options.getPluginFolder ? { getPluginFolder: options.getPluginFolder } : {})
		},
		// Emit explicit filesystem notifications; OS watcher timing is irrelevant
		// to subscription replacement/closure and the actual directory scan.
		watcher: {
			default: class {
				constructor(folder) {
					this.folder = folder
					this.listeners = new Map()
					this.closeCount = 0
					watchers.push(this)
				}
				on(name, listener) {
					this.listeners.set(name, listener)
				}
				close() {
					this.closeCount += 1
				}
			}
		},
		'../helpers': {
			makeExtraWindow: (configuration) => {
				windowOptions.push(configuration)
				return window
			}
		}
	})
	events.addPluginEventHandlers(
		{ on: (channel, handler) => handlers.set(channel, handler) },
		() => details,
		() => window
	)
	return {
		...run,
		window,
		details,
		watchers,
		windowHandlers,
		windowOptions,
		sent,
		dialogs,
		lifecycle,
		completion: completion.promise,
		emit: (channel, ...args) => {
			assert.ok(handlers.has(channel), `Register the ${channel} IPC handler`)
			return handlers.get(channel)({}, ...args)
		},
		expectSend: () => {
			nextSend = Promise.withResolvers()
			return nextSend.promise
		}
	}
}

test(
	'add-local-plugin waits for the copied plugin and Docker build before restarting',
	{ timeout: 5000 },
	async (t) => {
		const building = Promise.withResolvers()
		const finishBuild = Promise.withResolvers()
		t.after(() => finishBuild.resolve())
		const run = await eventFixture(t, {
			build: () => {
				building.resolve()
				return finishBuild.promise
			}
		})
		const source = await writePlugin(
			join(run.root, 'source'),
			{ ...pluginManifest, dockerCompose: 'compose.yml' },
			{ 'compose.yml': 'services: {}\n' }
		)
		const installation = run.emit('add-local-plugin', source)
		await building.promise
		assert.deepEqual(run.lifecycle, [], 'Do not restart while the install build is pending')
		finishBuild.resolve()
		await installation
		await run.completion
		assert.equal(
			await readFile(join(run.pluginFolder, pluginManifest.name, 'index.html'), 'utf8'),
			'<main>Local plugin</main>\n',
			'Install the selected local plugin'
		)
		assert.deepEqual(
			run.lifecycle,
			[['restart', run.details, run.window]],
			'Restart with the current plugin list and main window only after installation'
		)
	}
)

test(
	'download-plugin passes the URL through the real downloader before restarting',
	{ timeout: 5000 },
	async (t) => {
		const requested = Promise.withResolvers()
		const release = Promise.withResolvers()
		t.after(() => release.resolve(Response.json([])))
		const run = await eventFixture(t, {
			fetchImplementation: async (url) => {
				requested.resolve(url)
				return release.promise
			}
		})
		const download = run.emit(
			'download-plugin',
			'https://github.com/fixture-owner/fixture-repo'
		)
		assert.equal(
			await requested.promise,
			'https://api.github.com/repos/fixture-owner/fixture-repo/releases',
			'Pass the requested GitHub owner and repository through to the release lookup'
		)
		assert.deepEqual(run.lifecycle, [], 'Do not restart while the release lookup is pending')
		// The downloader reports lookup failures and resolves; the IPC handler's
		// existing contract still restarts after that completed attempt.
		release.resolve(Response.json([], { status: 503 }))
		await download
		await run.completion
		assert.equal(
			run.logs[0]?.args[0]?.message,
			'GitHub release lookup failed for fixture-owner/fixture-repo: 503',
			'Retain the download failure reason'
		)
		assert.deepEqual(
			run.lifecycle,
			[['restart', run.details, run.window]],
			'Restart after the download attempt settles'
		)
	}
)

test(
	'delete-plugin waits for shutdown, removes the selected directory, then initializes',
	{ timeout: 5000 },
	async (t) => {
		const stopping = Promise.withResolvers()
		const finishStop = Promise.withResolvers()
		t.after(() => finishStop.resolve())
		const run = await eventFixture(t, {
			stop: () => {
				stopping.resolve()
				return finishStop.promise
			}
		})
		const target = await writePlugin(join(run.pluginFolder, pluginManifest.name))
		const other = await writePlugin(join(run.pluginFolder, 'other-plugin'), {
			...pluginManifest,
			name: 'other-plugin'
		})
		const deletion = run.emit('delete-plugin', target)
		await stopping.promise
		assert.ok(await stat(target), 'Keep plugin files while shutdown is pending')
		assert.deepEqual(run.lifecycle, [['stop']], 'Do not initialize before deletion completes')
		finishStop.resolve()
		await deletion
		await run.completion
		await assert.rejects(
			stat(target),
			{ code: 'ENOENT' },
			'Delete the selected plugin directory'
		)
		assert.ok(await stat(other), 'Retain other installed plugins')
		assert.deepEqual(
			run.lifecycle,
			[['stop'], ['initialize', run.details, run.window]],
			'Initialize after shutdown and deletion without a second stop'
		)
	}
)

test(
	'plugin manager publishes real contents and closes replaced and window-owned watchers',
	{ timeout: 5000 },
	async (t) => {
		const run = await eventFixture(t)
		await run.emit('manage-plugins')
		assert.deepEqual(run.windowOptions, [
			{ name: 'Manage Plugins', width: 550, height: 400, path: '#/extras/plugins' }
		])
		const firstUpdate = run.expectSend()
		await run.emit('get-plugin-folder-contents')
		await firstUpdate
		assert.deepEqual(
			run.sent,
			[['plugin-folder-contents', []]],
			'Publish an initial empty folder snapshot'
		)
		assert.equal(
			run.watchers[0].folder,
			run.pluginFolder,
			'Watch the same application plugin folder that is scanned'
		)
		const installed = await writePlugin(join(run.pluginFolder, pluginManifest.name))
		const changedUpdate = run.expectSend()
		await run.watchers[0].listeners.get('all')()
		await changedUpdate
		assert.deepEqual(
			run.sent.at(-1),
			[
				'plugin-folder-contents',
				[
					{
						name: pluginManifest.pluginName,
						id: pluginManifest.name,
						folder: installed,
						folderName: pluginManifest.name
					}
				]
			],
			'Rescan actual contents after a watcher event'
		)
		const replacementUpdate = run.expectSend()
		await run.emit('get-plugin-folder-contents')
		await replacementUpdate
		assert.equal(
			run.watchers[0].closeCount,
			1,
			'Close the previous subscription before replacing it'
		)
		assert.equal(run.watchers.length, 2, 'Create one replacement subscription')
		run.windowHandlers.get('close')()
		run.windowHandlers.get('close')()
		assert.equal(
			run.watchers[1].closeCount,
			1,
			'Close and clear the active subscription with the manager window'
		)
		assert.equal(run.watchers[0].closeCount, 1, 'Do not close the retired subscription again')
	}
)

test('missing plugin folder shows a reason without creating a watcher', async (t) => {
	// Exercise the IPC adapter's explicit sentinel contract independently of
	// getPluginFolder(), which normally creates and returns a real directory.
	const run = await eventFixture(t, { getPluginFolder: async () => '' })
	await run.emit('get-plugin-folder-contents')
	assert.deepEqual(run.dialogs, [['Plugin Folder Not Found', 'The plugin folder was not found.']])
	assert.deepEqual(run.watchers, [], 'Do not subscribe to an unavailable directory')
	assert.deepEqual(run.sent, [], 'Do not publish a fabricated directory listing')
})

test('get-plugin-paths publishes the current plugin details to the main window', async (t) => {
	const run = await eventFixture(t)
	run.emit('get-plugin-paths')
	assert.deepEqual(run.sent, [['plugin-paths', run.details]])
	assert.strictEqual(
		run.sent[0][1],
		run.details,
		'Pass through the current canonical plugin details array'
	)
})

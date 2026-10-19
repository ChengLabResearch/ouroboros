import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runInThisContext } from 'node:vm'
import test from 'node:test'
import ts from 'typescript'

const require = createRequire(import.meta.url)

// Execute the real launcher, shared compose helper and IPC adapter with only
// external operations stubbed. No Docker daemon or Electron window is needed.
function loadModule(path, imports, logger) {
	const url = new URL(`../../${path}`, import.meta.url)
	const filename = fileURLToPath(url)
	const { outputText } = ts.transpileModule(readFileSync(url, 'utf8'), {
		fileName: filename,
		compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
	})
	const module = { exports: {} }
	const evaluate = runInThisContext(
		`(function(require, module, exports, __dirname, console) { ${outputText}\n})`,
		{ filename }
	)
	evaluate(
		(name) => {
			if (name in imports) return imports[name]
			if (name === 'path') return require(name)
			assert.fail(`Unexpected dependency: ${name}`)
		},
		module,
		module.exports,
		dirname(filename),
		logger
	)
	return module.exports
}

function fixture() {
	const starts = []
	const stops = []
	const logs = []
	const handlers = new Map()
	const sent = []
	const logger = {
		info: (...args) => logs.push(['info', ...args]),
		error: (...args) => logs.push(['error', ...args])
	}
	const docker = loadModule(
		'src/main/servers/docker.ts',
		{
			'docker-compose': {
				upAll: (options) => {
					const completion = Promise.withResolvers()
					starts.push({ options, ...completion })
					return completion.promise
				},
				downAll: async (options) => stops.push(options)
			},
			'../helpers': {}
		},
		logger
	)
	const server = loadModule('src/main/servers/main-server.ts', { './docker': docker }, logger)
	const shared = loadModule('src/shared/main-server-status.ts', {}, logger)
	const window = {
		isDestroyed: () => false,
		webContents: {
			isDestroyed: () => false,
			send: (channel, status) => sent.push({ channel, status })
		}
	}
	const events = loadModule(
		'src/main/events.ts',
		{
			electron: {
				ipcMain: {
					on: () => undefined,
					handle: (channel, handler) => handlers.set(channel, handler)
				}
			},
			'@electron-toolkit/utils': { is: { dev: false } },
			'./event-handlers/filesystem': { addFSEventHandlers: () => undefined },
			'./event-handlers/plugins': { addPluginEventHandlers: () => undefined },
			'./servers/main-server': server,
			'../shared/main-server-status': shared
		},
		logger
	)
	events.handleEvents({ getMainWindow: () => window, getPluginDetails: () => [] })
	return {
		server,
		docker,
		starts,
		stops,
		logs,
		sent,
		window,
		shared,
		snapshot: () => handlers.get(shared.MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL)()
	}
}

for (const mode of ['Development', 'Production']) {
	test(`${mode}: compose completion owns startup status and late IPC snapshots`, async () => {
		const run = fixture()
		assert.deepEqual(run.snapshot(), { state: 'idle', error: null })
		const launch = run.server[`startMainServer${mode}`]()
		assert.deepEqual(run.snapshot(), { state: 'starting', error: null })
		assert.equal(run.logs.length, 0, 'No success log while compose is pending')
		assert.equal(run.starts.length, 1)
		assert.match(
			run.starts[0].options.config,
			mode === 'Development' ? /compose\.dev\.yml$/ : /compose\.yml$/
		)
		assert.strictEqual(run.sent.at(-1).status, run.server.getMainServerLaunchStatus())
		assert.equal(run.sent.at(-1).channel, run.shared.MAIN_SERVER_LAUNCH_STATUS_CHANNEL)
		run.starts[0].resolve()
		await launch
		// A renderer that missed both push events obtains the final canonical snapshot.
		assert.deepEqual(run.snapshot(), { state: 'started', error: null })
		assert.strictEqual(run.snapshot(), run.server.getMainServerLaunchStatus())
		assert.deepEqual(
			run.sent.map(({ status }) => status.state),
			['starting', 'started']
		)
		assert.deepEqual(run.logs, [['info', 'Started Main Server']])
	})
}

for (const error of [
	{ exitCode: 1, out: '', err: '  Cannot connect to the Docker daemon.\n' },
	new Error('spawn docker ENOENT')
]) {
	test(`startup failure remains visible before HTTP is available: ${(error.err ?? error.message).trim()}`, async () => {
		const run = fixture()
		const launch = run.server.startMainServerProduction()
		run.starts[0].reject(error)
		await launch
		assert.deepEqual(run.snapshot(), {
			state: 'failed',
			error: (error.err ?? error.message).trim()
		})
		assert.deepEqual(
			run.sent.map(({ status }) => status.state),
			['starting', 'failed']
		)
		assert.equal(run.logs.length, 1)
		assert.equal(run.logs[0][0], 'error')
		assert.strictEqual(run.logs[0][2], error)
		// A retry clears the prior failure as soon as a new attempt begins.
		const retry = run.server.startMainServerProduction()
		assert.deepEqual(run.snapshot(), { state: 'starting', error: null })
		run.starts[1].resolve()
		await retry
		assert.equal(run.snapshot().state, 'started')
	})
}

test('window recreation shares the active launch and reuses a completed launch', async () => {
	const run = fixture()
	const first = run.server.startMainServerProduction()
	let secondFinished = false
	const second = run.server.startMainServerProduction().then(() => (secondFinished = true))
	await Promise.resolve()
	assert.equal(run.starts.length, 1)
	assert.equal(secondFinished, false)
	run.starts[0].resolve()
	await Promise.all([first, second])
	await run.server.startMainServerProduction()
	assert.equal(run.starts.length, 1, 'A completed launch is reused until shutdown')
})

test('shutdown still calls compose-down while launch is pending', async () => {
	const run = fixture()
	const launch = run.server.startMainServerProduction()
	await run.server.stopMainServerProduction()
	assert.equal(run.stops.length, 1)
	assert.deepEqual(run.snapshot(), { state: 'idle', error: null })
	// Settling startup after shutdown retains the existing compose race; shutdown
	// must not gain an unbounded dependency on an image pull completing.
	run.starts[0].resolve()
	await launch
})

test('normal shutdown resets the launcher so a later start can launch again', async () => {
	const run = fixture()
	const first = run.server.startMainServerProduction()
	run.starts[0].resolve()
	await first
	await run.server.stopMainServerProduction()
	assert.equal(run.stops.length, 1)
	assert.deepEqual(run.snapshot(), { state: 'idle', error: null })
	const restart = run.server.startMainServerProduction()
	assert.equal(run.starts.length, 2)
	run.starts[1].resolve()
	await restart
	assert.deepEqual(run.snapshot(), { state: 'started', error: null })
})

test('destroyed renderer does not turn a successful compose launch into a failure', async () => {
	const run = fixture()
	run.window.webContents.isDestroyed = () => true
	const launch = run.server.startMainServerProduction()
	run.starts[0].resolve()
	await launch
	assert.deepEqual(run.sent, [])
	assert.deepEqual(run.snapshot(), { state: 'started', error: null })
})

test('shared compose helper still leaves plugin error policy to the existing callback', async () => {
	const run = fixture()
	const errors = []
	const pluginStart = run.docker.startDockerCompose({
		cwd: '/plugin',
		config: '/plugin/compose.yml',
		onError: (error) => errors.push(error)
	})
	const error = { exitCode: 1, out: '', err: 'Plugin fixture failure' }
	run.starts[0].reject(error)
	await pluginStart
	assert.deepEqual(errors, [error])
	assert.deepEqual(run.snapshot(), { state: 'idle', error: null })
})

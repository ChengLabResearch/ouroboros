import assert from 'node:assert/strict'
import test from 'node:test'
import * as React from 'react'
import * as jsxRuntime from 'react/jsx-runtime'
import * as valibot from 'valibot'
import { loadCommonJsModule } from './helpers/load-commonjs-module.mjs'

const schemas = loadCommonJsModule('src/renderer/src/schemas/iframe-message-schema.ts', {
	valibot
})

function fixture(t, { invoke = async () => 'file contents' } = {}) {
	const effects = []
	const listeners = new Map()
	const calls = []
	const responses = []
	const errors = []
	const window = {
		addEventListener: (type, listener) => listeners.set(type, listener),
		removeEventListener: (type, listener) => {
			assert.strictEqual(listeners.get(type), listener, 'Cleanup removes its own listener')
			listeners.delete(type)
		},
		electron: {
			ipcRenderer: {
				invoke: async (...args) => {
					calls.push(args)
					return invoke(...args)
				}
			}
		}
	}
	const source = { postMessage: (...args) => responses.push(args) }
	const directory = { directoryPath: '/project', directoryName: 'project' }
	const scope = { window, console: { error: (...args) => errors.push(args) } }
	const files = loadCommonJsModule('src/renderer/src/interfaces/file.tsx', {}, scope)
	const { IFrameProvider } = loadCommonJsModule(
		'src/renderer/src/contexts/IFrameContext.tsx',
		{
			'@renderer/schemas/iframe-message-schema': schemas,
			valibot,
			'../interfaces/file': files,
			'@renderer/contexts/DirectoryContext': { DirectoryContext: {} },
			'react/jsx-runtime': jsxRuntime,
			// Capture a single mount's effects so the real browser-message listener can
			// run in Node. These stubs do not model React rendering or state updates.
			react: {
				...React,
				useState: (value) => [
					value,
					() =>
						assert.fail('This single-mount fixture does not model React state updates')
				],
				useContext: () => directory,
				useRef: (value) => ({ current: value }),
				useCallback: (callback) => callback,
				useEffect: (effect) => effects.push(effect)
			}
		},
		scope
	)
	IFrameProvider({ children: null })
	const cleanups = effects.map((effect) => effect()).filter(Boolean)
	let mounted = true
	function unmount() {
		if (!mounted) return
		mounted = false
		for (const cleanup of cleanups) cleanup()
	}
	t.after(unmount)
	assert.equal(listeners.size, 1, 'Mount installs exactly one browser-message listener')
	return {
		calls,
		responses,
		errors,
		listeners,
		unmount,
		async send(data, { origin = 'http://localhost:5173', eventSource = source } = {}) {
			const listener = listeners.get('message')
			assert.equal(typeof listener, 'function', 'A mounted provider receives messages')
			listener({ origin, data, source: eventSource })
			// File handlers await the real IPC wrapper and return through an async
			// listener. Drain that promise chain without sleeps or timing assumptions.
			await new Promise((resolve) => setImmediate(resolve))
		}
	}
}

const readRequest = () => ({
	type: 'read-file',
	data: { folder: '/project', fileName: 'options.json' }
})

const saveRequest = () => ({
	type: 'save-file',
	data: { folder: '/project', fileName: 'options.json', contents: '{"slice_width":512}' }
})

for (const origin of [
	'http://localhost:5173',
	'http://127.0.0.1:9988',
	'http://0.0.0.0:8080',
	'http://localhost:43210'
]) {
	test(`iframe read: loopback origin ${origin} reaches IPC and replies to the sender`, async (t) => {
		const contents = '{"slice_width":512}\n'
		const run = fixture(t, { invoke: async () => contents })
		await run.send(readRequest(), { origin })
		assert.deepEqual(
			run.calls,
			[['read-file', { folder: '/project', name: 'options.json' }]],
			'The file adapter must preserve the requested folder and file name'
		)
		assert.deepEqual(
			run.responses,
			[
				[
					{
						type: 'read-file-response',
						data: { fileName: 'options.json', contents }
					},
					{ targetOrigin: '*' }
				]
			],
			'The requesting iframe receives the unchanged file contents'
		)
		assert.deepEqual(run.errors, [], 'A successful read must not log an error')
	})
}

for (const [reason, origin] of [
	['hostname prefix attack', 'http://localhost.evil.example:5173'],
	['IPv4 prefix attack', 'http://127.0.0.1.evil.example'],
	['userinfo disguising the remote host', 'http://localhost@evil.example'],
	['non-loopback host', 'https://plugins.example'],
	['opaque origin', 'null'],
	['empty origin', ''],
	['malformed URL', 'http://[broken']
]) {
	test(`iframe messages: deny ${reason} before reading or saving a file`, async (t) => {
		const run = fixture(t)
		await run.send(readRequest(), { origin })
		await run.send(saveRequest(), { origin })
		assert.deepEqual(
			run.calls,
			[],
			`Denied origin ${JSON.stringify(origin)} must not reach IPC`
		)
		assert.deepEqual(run.responses, [], 'A denied origin must not receive file contents')
		assert.deepEqual(run.errors, [], 'Rejected origins are ignored without a handler error')
	})
}

for (const [reason, request] of [
	['null envelope', null],
	['non-string message type', { type: 7, data: {} }],
	['unknown message type', { type: 'unknown-file-operation', data: readRequest().data }],
	[
		'read type substring without a registered handler',
		{ ...readRequest(), type: 'prefix-read-file' }
	],
	['read without file name', { type: 'read-file', data: { folder: '/project' } }],
	[
		'read with non-string folder',
		{ type: 'read-file', data: { folder: 7, fileName: 'options.json' } }
	],
	['save without contents', { type: 'save-file', data: readRequest().data }],
	[
		'save with non-string contents',
		{ type: 'save-file', data: { ...saveRequest().data, contents: {} } }
	]
]) {
	test(`iframe messages: ignore ${reason} without accessing files`, async (t) => {
		const run = fixture(t)
		await run.send(request)
		assert.deepEqual(run.calls, [], `${reason} must not invoke a file operation`)
		assert.deepEqual(run.responses, [], `${reason} must not send a successful response`)
		assert.deepEqual(run.errors, [], 'Schema rejection is not an asynchronous handler failure')
	})
}

test('iframe messages: a missing source cannot read or save a file', async (t) => {
	const run = fixture(t)
	await run.send(readRequest(), { eventSource: null })
	await run.send(saveRequest(), { eventSource: null })
	assert.deepEqual(run.calls, [], 'No file operation is dispatched without a requesting window')
	assert.deepEqual(run.responses, [], 'A missing source cannot receive a response')
})

test('iframe save: forwards contents through the real file adapter without a read reply', async (t) => {
	const run = fixture(t, { invoke: async () => true })
	await run.send(saveRequest())
	assert.deepEqual(
		run.calls,
		[
			[
				'save-file',
				{
					folder: '/project',
					name: 'options.json',
					data: '{"slice_width":512}'
				}
			]
		],
		'The save IPC payload must preserve the requested file and contents'
	)
	assert.deepEqual(run.responses, [], 'The existing save protocol does not acknowledge writes')
	assert.deepEqual(run.errors, [], 'A successful write must not log an error')
})

for (const [operation, request] of [
	['read', readRequest],
	['save', saveRequest]
]) {
	test(`iframe ${operation}: reports the original IPC failure without a success response`, async (t) => {
		const failure = new Error(`${operation} denied by filesystem permissions`)
		const run = fixture(t, {
			invoke: async () => {
				throw failure
			}
		})
		await run.send(request())
		assert.equal(run.calls.length, 1, 'A failed file operation must not be retried silently')
		assert.deepEqual(
			run.errors,
			[[failure]],
			'Log the original error once, retaining its reason'
		)
		assert.deepEqual(
			run.responses,
			[],
			'Failed file access must not look like a successful reply'
		)
	})
}

test('iframe provider cleanup removes the message listener installed by its effect', (t) => {
	const run = fixture(t)
	run.unmount()
	assert.equal(run.listeners.size, 0, 'Unmounted providers must stop accepting plugin messages')
})

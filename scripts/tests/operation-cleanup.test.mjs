import './helpers/renderer-loader.mjs'
import assert from 'node:assert/strict'
import { registerHooks } from 'node:module'
import test from 'node:test'

const providerURL = new URL('../../src/renderer/src/contexts/ServerContext.tsx', import.meta.url)
const reactURL = import.meta.resolve('react')
// Exercise the actual provider without a browser. Only its React scheduling is
// replaced: state/ref identity and effect dependency/cleanup behavior are retained.
registerHooks({
	resolve(specifier, context, nextResolve) {
		if (specifier === 'react' && context.parentURL === providerURL.href) {
			const source = `export { createContext } from ${JSON.stringify(reactURL)};
export const useState = (...args) => globalThis.operationCleanupHooks.state(...args);
export const useRef = (...args) => globalThis.operationCleanupHooks.ref(...args);
export const useCallback = (...args) => globalThis.operationCleanupHooks.callback(...args);
export const useEffect = (...args) => globalThis.operationCleanupHooks.effect(...args);`
			return { url: `data:text/javascript,${encodeURIComponent(source)}`, shortCircuit: true }
		}
		return nextResolve(specifier, context)
	}
})
const { default: ServerProvider } = await import(providerURL.href)
const { deleteOperation } = await import('../../src/renderer/src/interfaces/operation-cleanup.ts')

function deferred() {
	let resolve
	const promise = new Promise((done) => {
		resolve = done
	})
	return { promise, resolve }
}

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
	emit(payload) {
		this.dispatchEvent(
			new MessageEvent('update_event', {
				data: JSON.stringify(payload),
				lastEventId: new URL(this.url).searchParams.get('task_id')
			})
		)
	}
}
if (!globalThis.EventSource) globalThis.EventSource = FakeEventSource

const successful = (results) => ({ results, pending: false, error: { status: false, message: '' } })
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data })
function status(outcome = 'done', publication = 'published') {
	return {
		status: outcome,
		progress: [['Native step', 1, 1.25]],
		error: null,
		statistics: {
			status: publication,
			path: '/output/statistics.json',
			recovery_path: null,
			error: null
		}
	}
}

function provider(t, operation) {
	const slots = []
	let cursor = 0
	let effects = []
	const same = (left, right) =>
		left &&
		right &&
		left.length === right.length &&
		left.every((item, index) => Object.is(item, right[index]))
	const hooks = {
		state(initial) {
			const index = cursor++
			if (!slots[index]) {
				const slot = { value: typeof initial === 'function' ? initial() : initial }
				slot.set = (next) => {
					slot.value = typeof next === 'function' ? next(slot.value) : next
				}
				slots[index] = slot
			}
			return [slots[index].value, slots[index].set]
		},
		ref(initial) {
			return (slots[cursor++] ??= { current: initial })
		},
		callback(value, dependencies) {
			const index = cursor++
			if (!same(slots[index]?.dependencies, dependencies))
				slots[index] = { value, dependencies }
			return slots[index].value
		},
		effect(effect, dependencies) {
			const index = cursor++
			if (!same(slots[index]?.dependencies, dependencies)) {
				const previous = slots[index]
				const slot = (slots[index] = { dependencies })
				effects.push(() => {
					previous?.cleanup?.()
					slot.cleanup = effect()
				})
			}
		}
	}
	const calls = []
	const requestURL = `/${operation}/`
	const streamURL = `/${operation}_status_stream/`
	const task = { task_id: `${operation}-task` }
	const deleteURL = `/delete/?task_id=${task.task_id}`
	let deletion = response({ success: true })
	let submission = response(task)
	let eviction = Promise.resolve()
	const pollingTimers = []
	t.mock.method(globalThis, 'setTimeout', (callback, delay) => {
		assert.equal(delay, 5000)
		pollingTimers.push(callback)
	})
	t.mock.method(globalThis, 'EventSource', FakeEventSource)
	t.mock.method(globalThis, 'fetch', async (input, options) => {
		const url = new URL(input)
		if (url.pathname === '/') return response({})
		calls.push({ kind: 'fetch', url: url.pathname + url.search, method: options.method })
		return url.pathname === '/delete/' ? deletion : submission
	})
	globalThis.window = {
		electron: {
			ipcRenderer: {
				invoke: async (channel, path) => {
					calls.push({ kind: 'ipc', channel, path })
					await eviction
				}
			}
		}
	}
	globalThis.operationCleanupHooks = hooks
	let context
	function render() {
		cursor = 0
		effects = []
		context = ServerProvider({ baseURL: 'http://test', children: null }).props.value
		for (const effect of effects) effect()
		return context
	}
	render()
	t.after(() => {
		for (const slot of slots) slot?.cleanup?.()
		for (const timer of pollingTimers) timer()
		delete globalThis.operationCleanupHooks
		delete globalThis.window
	})
	return {
		calls,
		task,
		requestURL,
		streamURL,
		deleteURL,
		render,
		get context() {
			return render()
		},
		get state() {
			return render().useStreamListener(streamURL)
		},
		get submission() {
			return render().useFetchListener(requestURL)
		},
		get source() {
			return FakeEventSource.instances.at(-1)
		},
		set deleteResponse(value) {
			deletion = value
		},
		set submitResponse(value) {
			submission = value
		},
		set eviction(value) {
			eviction = value
		},
		async settle() {
			await new Promise(setImmediate)
			render()
		},
		async submit(query = { options: '/output/original-options.json' }) {
			await new Promise(setImmediate)
			return render().performFetch(requestURL, query, { method: 'POST' })
		}
	}
}

for (const operation of ['slice', 'backproject']) {
	test(`${operation}: provider waits for terminal publication, deletion and eviction without a mounted page`, async (t) => {
		const run = provider(t, operation)
		const releaseDelete = deferred()
		const releaseEviction = deferred()
		run.deleteResponse = releaseDelete.promise
		run.eviction = releaseEviction.promise
		const query = { options: '/output/original-options.json' }
		await run.submit(query)
		query.options = '/elsewhere/edited-options.json'
		assert.equal(run.submission.optionsPath, '/output/original-options.json')
		assert.equal(run.source.closeCount, 0)
		run.source.emit(status('done', 'pending'))
		await run.settle()
		assert.equal(run.calls.length, 1)
		assert.equal(run.state.cleanupPending, false)

		run.source.emit(status())
		assert.equal(run.state.done, true)
		assert.equal(run.state.cleanupPending, true)
		const terminal = run.state.results
		await run.settle()
		assert.equal(run.calls.at(-1).url, run.deleteURL)
		assert.equal(run.calls.at(-1).method, 'POST')
		assert.equal(
			run.calls.some((call) => call.kind === 'ipc'),
			false
		)
		releaseDelete.resolve(response({ success: true }))
		await run.settle()
		assert.deepEqual(run.calls.at(-1), {
			kind: 'ipc',
			channel: 'evict-output-cache',
			path: '/output/original-options.json'
		})
		assert.equal(run.state.cleanupPending, true)
		assert.deepEqual(run.submission.results, run.task)
		releaseEviction.resolve()
		await run.settle()
		assert.equal(run.state.cleanupPending, false)
		assert.equal(run.submission.results, null)
		assert.equal(run.submission.optionsPath, undefined)
		assert.equal(run.state.results, terminal)

		// Provider rerenders and duplicate terminal events must not repeat deletion.
		// This fixture does not mount/unmount route pages or simulate the router.
		const source = run.source
		run.render()
		run.source.emit(status())
		await run.settle()
		assert.equal(run.source, source)
		assert.equal(source.closeCount, 1)
		assert.equal(run.calls.filter((call) => call.url === run.deleteURL).length, 1)
	})
}

test('submission keeps the options path sent on the wire if its caller edits the query while awaiting a response', async (t) => {
	const run = provider(t, 'slice')
	const submitted = deferred()
	run.submitResponse = submitted.promise
	const query = { options: '/output/original-options.json' }
	const pending = run.submit(query)
	await run.settle()
	query.options = '/elsewhere/edited-options.json'
	submitted.resolve(response(run.task))
	await pending
	assert.equal(run.submission.optionsPath, '/output/original-options.json')
	run.source.emit(status())
	await run.settle()
	assert.equal(run.calls.at(-1).path, '/output/original-options.json')
})

test('failed deletion retains the completed task and request notice for the shared manual retry', async (t) => {
	const run = provider(t, 'backproject')
	run.deleteResponse = response({ detail: 'Delete denied' }, 409)
	await run.submit()
	run.source.emit(status('error', 'failed'))
	await run.settle()
	const terminal = run.state.results
	assert.deepEqual(run.submission.results, run.task)
	assert.equal(run.state.cleanupPending, false)
	assert.equal(run.context.useFetchListener(run.deleteURL).error.message, 'Delete denied')
	assert.equal(
		run.calls.some((call) => call.kind === 'ipc'),
		false
	)
	run.deleteResponse = response({ success: true })
	const releaseEviction = deferred()
	run.eviction = releaseEviction.promise
	const retry = run.context.deleteTask(run.streamURL, run.deleteURL, run.submission.optionsPath)
	assert.equal(run.state.cleanupPending, true)
	await run.settle()
	assert.equal(run.context.useFetchListener(run.deleteURL).error.status, false)
	assert.deepEqual(run.calls.at(-1), {
		kind: 'ipc',
		channel: 'evict-output-cache',
		path: '/output/original-options.json'
	})
	// The pending state belongs to the provider, so a newly rendered consumer
	// observes it even though the original page's local submit lock is gone.
	run.render()
	assert.equal(run.state.cleanupPending, true)
	assert.deepEqual(run.submission.results, run.task)
	assert.equal(run.submission.optionsPath, '/output/original-options.json')
	assert.equal(run.state.results, terminal)
	releaseEviction.resolve()
	assert.equal(await retry, true)
	assert.equal(run.state.cleanupPending, false)
	assert.equal(run.state.results, terminal)
})

test('late cleanup cannot clear a replacement submission or its active stream', async (t) => {
	const run = provider(t, 'slice')
	const releaseEviction = deferred()
	run.eviction = releaseEviction.promise
	await run.submit()
	run.source.emit(status())
	await run.settle()
	assert.equal(run.state.cleanupPending, true)
	const replacement = { task_id: 'replacement-task' }
	run.submitResponse = response(replacement)
	await run.submit({ options: '/replacement/options.json' })
	const source = run.source
	releaseEviction.resolve()
	await run.settle()
	assert.deepEqual(run.submission.results, replacement)
	assert.equal(run.submission.optionsPath, '/replacement/options.json')
	assert.equal(run.state.done, false)
	assert.equal(source.closeCount, 0)
	assert.equal(run.calls.filter((call) => call.url === run.deleteURL).length, 1)
})

for (const failure of ['transport', 'unknown-task']) {
	test(`${failure} closure never deletes or evicts a task with an unknown outcome`, async (t) => {
		const run = provider(t, 'slice')
		await run.submit()
		run.source.emit(status('started', 'pending'))
		if (failure === 'transport') run.source.dispatchEvent(new Event('error'))
		else run.source.emit({ status: 'error', progress: [], error: 'Item ID Not Found' })
		await run.settle()
		assert.equal(run.state.done, true)
		assert.equal(run.state.cleanupPending, false)
		assert.equal(run.state.error.status, true)
		assert.deepEqual(run.submission.results, run.task)
		assert.equal(run.calls.length, 1)
	})
}

test('cleanup requires confirmed successful deletion, including when a request was merely queued', async (t) => {
	const ipc = t.mock.fn()
	globalThis.window = { electron: { ipcRenderer: { invoke: ipc } } }
	t.after(() => {
		delete globalThis.window
	})
	for (const result of [
		successful(null),
		successful({ success: false }),
		{
			...successful(null),
			error: { status: true, message: 'Delete failed' }
		}
	]) {
		assert.equal(
			await deleteOperation(
				'/delete/?task_id=test',
				'/saved/options.json',
				async () => result
			),
			false
		)
	}
	assert.equal(ipc.mock.callCount(), 0)
})

test('eviction failure is reported without changing successful task retirement', async (t) => {
	const error = new Error('File remains open')
	globalThis.window = {
		electron: {
			ipcRenderer: {
				invoke: async () => {
					throw error
				}
			}
		}
	}
	t.after(() => {
		delete globalThis.window
	})
	const warning = t.mock.method(console, 'warn', () => {})
	assert.equal(
		await deleteOperation('/delete/?task_id=test', '/saved/options.json', async () =>
			successful({ success: true })
		),
		true
	)
	assert.equal(warning.mock.callCount(), 1)
	assert.match(warning.mock.calls[0].arguments[0], /Could not evict operation outputs/)
	assert.equal(warning.mock.calls[0].arguments[1], error)
})

import './helpers/renderer-loader.mjs'
import assert from 'node:assert/strict'
import test from 'node:test'

const { subscribeToMainServerLaunchStatus } =
	await import('../../src/renderer/src/interfaces/main-server-status.ts')
const { MAIN_SERVER_LAUNCH_STATUS_CHANNEL, MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL } =
	await import('../../src/shared/main-server-status.ts')

function client() {
	let receive
	let resolve
	let reject
	let removed = false
	const updates = []
	const initial = new Promise((done, fail) => {
		resolve = done
		reject = fail
	})
	const close = subscribeToMainServerLaunchStatus(
		{
			on(channel, listener) {
				assert.equal(channel, MAIN_SERVER_LAUNCH_STATUS_CHANNEL)
				receive = listener
				return () => {
					removed = true
				}
			},
			invoke(channel) {
				assert.equal(channel, MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL)
				assert.ok(receive, 'subscription must precede the initial snapshot request')
				return initial
			}
		},
		(status) => updates.push(status)
	)
	return {
		close,
		resolve,
		reject,
		updates,
		emit: (status) => receive({}, status),
		removed: () => removed
	}
}

test('late renderer receives the current launch snapshot', async () => {
	const run = client()
	const snapshot = { state: 'failed', error: 'Docker daemon is not running' }
	run.resolve(snapshot)
	await new Promise(setImmediate)
	assert.deepEqual(run.updates, [snapshot])
	run.close()
})

test('new launch event cannot be overwritten by the initial snapshot', async () => {
	const run = client()
	const latest = { state: 'started', error: null }
	run.emit(latest)
	run.resolve({ state: 'starting', error: null })
	await new Promise(setImmediate)
	assert.deepEqual(run.updates, [latest])
	run.close()
})

test('unsubscribed renderers ignore queued events and late snapshots', async () => {
	const run = client()
	run.close()
	assert.equal(run.removed(), true)
	run.emit({ state: 'failed', error: 'late failure' })
	run.resolve({ state: 'started', error: null })
	await new Promise(setImmediate)
	assert.deepEqual(run.updates, [])
})

test('unavailable launch IPC does not invent a launch outcome', async () => {
	const run = client()
	run.reject(new Error('No handler registered'))
	await new Promise(setImmediate)
	assert.deepEqual(run.updates, [])
	run.close()
})

import assert from 'node:assert/strict'
import { constants } from 'node:fs'
import * as fs from 'node:fs/promises'
import { tmpdir } from 'node:os'
import * as path from 'node:path'
import test from 'node:test'
import { loadCommonJsModule } from './helpers/load-commonjs-module.mjs'

const loadModule = (source, imports, platform, warnings) =>
	loadCommonJsModule(
		source,
		{ fs: { constants }, ...imports },
		{ process: { platform }, console: { warn: (...args) => warnings.push(args) } }
	)

function fixture(filesystem, platform = 'win32') {
	const warnings = []
	const helper = loadModule(
		'src/main/cache-eviction.ts',
		{ 'fs/promises': filesystem },
		platform,
		warnings
	)
	return { ...helper, warnings }
}

async function outputTree(t) {
	const root = await fs.mkdtemp(path.join(tmpdir(), 'ouroboros-cache-eviction-'))
	t.after(() => fs.rm(root, { recursive: true, force: true }))
	const output = path.join(root, 'output')
	const outside = path.join(root, 'outside')
	await fs.mkdir(path.join(output, 'nested'), { recursive: true })
	await fs.mkdir(outside)
	const files = [
		path.join(output, 'image.tif'),
		path.join(output, '.hidden-output'),
		path.join(output, 'nested', 'chunk.bin')
	]
	for (const file of files) await fs.writeFile(file, `contents of ${file}`)
	await fs.writeFile(path.join(outside, 'keep.tif'), 'outside the output tree')
	const link = path.join(output, 'outside-link')
	await fs.symlink(outside, link, process.platform === 'win32' ? 'junction' : 'dir')
	return { output, outside, files, link }
}

test('Windows uses the libuv direct-open flag with read-only access and closes the handle', async () => {
	const calls = []
	const run = fixture({
		open: async (...args) => {
			calls.push(['open', ...args])
			return { close: async () => calls.push(['close']) }
		}
	})
	await run.evictFromCache('C:\\output\\image.tif')
	assert.deepEqual(calls, [
		['open', 'C:\\output\\image.tif', constants.O_RDONLY | 0x02000000],
		['close']
	])
	assert.deepEqual(run.warnings, [])
})

for (const platform of ['linux', 'darwin']) {
	test(`${platform} performs no filesystem access through any eviction entry point`, async () => {
		const run = fixture(
			new Proxy(
				{},
				{ get: () => assert.fail('Non-Windows eviction touched the filesystem') }
			),
			platform
		)
		await run.evictFromCache('/output/image.tif')
		await run.evictTree('/output')
		await run.evictOperationOutput('/output/options.json')
		assert.deepEqual(run.warnings, [])
	})
}

test('the submitted options location owns the complete output walk with one open file at a time', async (t) => {
	const { output, outside, files, link } = await outputTree(t)
	const options = path.join(output, 'options.json')
	// A later edit must not redirect the completed task's cache pass elsewhere.
	await fs.writeFile(options, JSON.stringify({ output_folder: outside }))
	files.push(options)
	const before = await Promise.all(files.map((file) => fs.readFile(file)))
	const opened = []
	let active = 0
	const run = fixture({
		...fs,
		open: async (file, flags) => {
			assert.equal(flags, constants.O_RDONLY | 0x02000000)
			assert.equal(++active, 1, 'File handles must remain bounded during a large output walk')
			opened.push(file)
			// Exercise real handles and contents on every host. Windows additionally
			// exercises its native direct-open flag; Unix cannot interpret that flag.
			const handle = await fs.open(
				file,
				process.platform === 'win32' ? flags : constants.O_RDONLY
			)
			return {
				close: async () => {
					await new Promise(setImmediate)
					await handle.close()
					active--
				}
			}
		}
	})
	await run.evictOperationOutput(options)
	assert.deepEqual(opened.sort(), [...files].sort())
	assert.equal(active, 0)
	assert.deepEqual(run.warnings, [])
	// The same symlink/junction exclusion also applies to the requested root.
	await run.evictTree(link)
	assert.equal(opened.length, files.length)
	const after = await Promise.all(files.map((file) => fs.readFile(file)))
	assert.deepEqual(after, before)
	assert.equal(
		await fs.readFile(path.join(outside, 'keep.tif'), 'utf8'),
		'outside the output tree'
	)
})

test('file open and close failures warn with their paths and continue through sibling files', async (t) => {
	const { output, files } = await outputTree(t)
	const openError = Object.assign(new Error('File is locked'), { code: 'EACCES' })
	const closeError = new Error('Close failed')
	const attempted = []
	const closed = []
	const run = fixture({
		...fs,
		open: async (file) => {
			attempted.push(file)
			if (file === files[0]) throw openError
			return {
				close: async () => {
					closed.push(file)
					if (file === files[1]) throw closeError
				}
			}
		}
	})
	await run.evictTree(output)
	assert.deepEqual(attempted.sort(), [...files].sort())
	assert.deepEqual(closed.sort(), files.slice(1).sort())
	assert.equal(run.warnings.length, 2)
	for (const [file, error] of [
		[files[0], openError],
		[files[1], closeError]
	]) {
		assert.ok(
			run.warnings.some(([message, cause]) => message.includes(file) && cause === error)
		)
	}
})

test('an unreadable child directory warns and does not stop sibling output eviction', async (t) => {
	const { output, files } = await outputTree(t)
	const inaccessible = path.join(output, 'nested')
	const error = Object.assign(new Error('Access denied'), { code: 'EACCES' })
	const opened = []
	const run = fixture({
		...fs,
		readdir: async (dir, options) => {
			if (dir === inaccessible) throw error
			return fs.readdir(dir, options)
		},
		open: async (file) => {
			opened.push(file)
			return { close: async () => undefined }
		}
	})
	await run.evictTree(output)
	assert.deepEqual(opened.sort(), files.slice(0, 2).sort())
	assert.equal(run.warnings.length, 1)
	assert.ok(run.warnings[0][0].includes(inaccessible))
	assert.strictEqual(run.warnings[0][1], error)
	await run.evictTree(path.join(output, 'missing'))
	assert.equal(run.warnings.length, 2)
	assert.equal(run.warnings[1][1].code, 'ENOENT')
})

test('filesystem IPC awaits the cache helper using the submitted options path unchanged', async () => {
	const handlers = new Map()
	const calls = []
	const completion = Promise.withResolvers()
	const { addFSEventHandlers } = loadModule(
		'src/main/event-handlers/filesystem.ts',
		{
			'fs/promises': {},
			watcher: {},
			'../helpers': {},
			'../../shared/constants': {},
			'../cache-eviction': {
				evictOperationOutput: (options) => {
					calls.push(options)
					return completion.promise
				}
			}
		},
		'win32',
		[]
	)
	addFSEventHandlers(
		{ handle: (channel, callback) => handlers.set(channel, callback) },
		() => null
	)
	let finished = false
	const pending = handlers
		.get('evict-output-cache')({}, 'C:\\output\\options.json')
		.then(() => {
			finished = true
		})
	await Promise.resolve()
	assert.deepEqual(calls, ['C:\\output\\options.json'])
	assert.equal(finished, false)
	completion.resolve()
	await pending
	assert.equal(finished, true)
})

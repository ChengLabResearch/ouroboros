import assert from 'node:assert/strict'
import { registerHooks } from 'node:module'
import test from 'node:test'

const serverURL = new URL('../../resources/processes/volume-server-script.mjs', import.meta.url)
let fixtureId = 0

async function serverFixture(t, dialect = 'posix') {
	const routes = new Map()
	const calls = []
	const fixture = {
		failure: null,
		app: {
			use() {
				// Middleware is unnecessary when invoking captured handlers directly.
			},
			post: (route, handler) => routes.set(route, handler),
			listen() {
				// Keep this route fixture independent of sockets and server lifetime.
			}
		},
		execFile(file, args, callback) {
			calls.push({ file, args })
			callback(
				fixture.failure ? new Error(fixture.failure.message) : null,
				'',
				fixture.failure?.stderr ?? ''
			)
		}
	}
	globalThis.volumeServerFixture = fixture
	// Exercise the real routes and runner without starting an HTTP server or Docker.
	// Only the server's platform/dependency boundaries are replaced; copy and cleanup
	// helpers remain the production modules.
	const hooks = registerHooks({
		resolve(specifier, context, nextResolve) {
			if (!context.parentURL?.startsWith(serverURL.href)) {
				return nextResolve(specifier, context)
			}
			let source
			if (specifier === 'express') {
				source = 'export default () => globalThis.volumeServerFixture.app'
			} else if (specifier === 'cors') {
				source = 'export default () => () => {}'
			} else if (specifier === 'body-parser') {
				source = 'export default { json: () => () => {} }'
			} else if (specifier === 'child_process' || specifier === 'node:child_process') {
				source = `export const execFile = (...args) => globalThis.volumeServerFixture.execFile(...args);
export const exec = () => { throw new Error('Volume operations must not invoke a host shell') }`
			} else if (specifier === 'path' || specifier === 'node:path') {
				source = `import path from 'node:path';
export default path.${dialect};
export const { basename, dirname } = path.${dialect}`
			}
			if (source) {
				return {
					url: `data:text/javascript,${encodeURIComponent(source)}`,
					shortCircuit: true
				}
			}
			return nextResolve(specifier, context)
		}
	})
	t.after(() => {
		hooks.deregister()
		delete globalThis.volumeServerFixture
	})
	await import(`${serverURL.href}?fixture=${fixtureId++}`)
	return {
		calls,
		fail: (message, stderr = message) => (fixture.failure = { message, stderr }),
		async request(route, body) {
			const response = {}
			const res = {
				status(code) {
					response.status = code
					return res
				},
				send(message) {
					response.body = message
					return res
				}
			}
			assert.ok(routes.has(route), `Missing route: ${route}`)
			await routes.get(route)({ body }, res)
			return response
		}
	}
}

const pathCases = [
	{
		name: 'POSIX',
		dialect: 'posix',
		folder: '/tmp/O\'Brien  "$HOME" `echo nope`',
		fileName: 'scan  \'$VALUE\' "`date`".tif',
		separator: '/'
	},
	{
		name: 'Windows drive',
		dialect: 'win32',
		folder: "C:\\O'Brien  $USER `whoami` & %TEMP%",
		fileName: "scan  '$VALUE' `date`.tif",
		separator: '\\'
	},
	{
		name: 'Windows UNC',
		dialect: 'win32',
		folder: "\\\\server\\share\\O'Brien  $USER `whoami`",
		fileName: "scan  '$VALUE' `date`.tif",
		separator: '\\'
	}
]

for (const { name, dialect, folder, fileName, separator } of pathCases) {
	for (const toVolume of [true, false]) {
		const route = toVolume ? '/copy-to-volume' : '/copy-to-host'
		test(`${route} preserves ${name} paths as Docker arguments`, async (t) => {
			const server = await serverFixture(t, dialect)
			// Docker owns volume-name validity; the route must pass even shell syntax literally.
			const volumeName = 'volume\' "$NAME" `date`'
			const pluginFolderName = 'plugin  O\'Brien "$HOME" `date`'
			const targetPath = 'nested  \'$VALUE\' "`date`"'
			assert.deepEqual(
				await server.request(route, {
					volumeName,
					pluginFolderName,
					files: [{ sourcePath: `${folder}${separator}${fileName}`, targetPath }]
				}),
				{ status: 200, body: 'Files copied successfully.' }
			)
			assert.equal(server.calls.length, 1)
			const [{ file, args }] = server.calls
			assert.equal(file, 'docker')
			assert.deepEqual(args.slice(0, 11), [
				'run',
				'--rm',
				'-v',
				`${folder}:/host`,
				'-v',
				`${volumeName}:/volume`,
				'-w',
				'/host',
				'alpine',
				'sh',
				'-c'
			])
			assert.equal(args[12], 'sh')
			const volumeDir = `/volume/${pluginFolderName}/${targetPath}`
			const hostFile = `/host/${fileName}`
			const volumeFile = `${volumeDir}/${fileName}`
			assert.deepEqual(
				args.slice(13),
				toVolume ? [volumeDir, hostFile, volumeFile] : [hostFile, volumeFile]
			)
			for (const value of [folder, fileName, volumeName, pluginFolderName, targetPath]) {
				assert.ok(!args[11].includes(value), 'Request data must not enter the shell script')
			}
		})
	}
}

test('copy routes retain their HTTP failure responses', async (t) => {
	const server = await serverFixture(t)
	for (const [message, stderr] of [
		['fixture copy failure', 'fixture copy failure'],
		['spawn docker ENOENT', '']
	]) {
		server.fail(message, stderr)
		for (const route of ['/copy-to-volume', '/copy-to-host']) {
			assert.deepEqual(
				await server.request(route, {
					volumeName: 'fixture-volume',
					pluginFolderName: 'main',
					files: [{ sourcePath: '/tmp/scan.tif', targetPath: 'results' }]
				}),
				{ status: 500, body: `Error copying files: ${message}` }
			)
		}
	}
})

test('cleanup routes use the same argument-based Docker runner', async (t) => {
	const server = await serverFixture(t)
	const volumeName = 'fixture-volume'
	const pluginFolderName = "plugin  O'Brien $HOME `date`"
	assert.deepEqual(
		await server.request('/clear-plugin-folder', { volumeName, pluginFolderName }),
		{ status: 200, body: 'Plugin folder cleared successfully.' }
	)
	assert.deepEqual(await server.request('/clear-volume', { volumeName }), {
		status: 200,
		body: 'Volume cleared successfully.'
	})
	assert.equal(server.calls.length, 2)
	for (const { file, args } of server.calls) {
		assert.equal(file, 'docker')
		assert.deepEqual(args.slice(0, 7), [
			'run',
			'--rm',
			'-v',
			`${volumeName}:/volume`,
			'alpine',
			'sh',
			'-c'
		])
		assert.ok(!args[7].includes(pluginFolderName))
	}
	assert.deepEqual(server.calls[0].args.slice(8), ['sh', `/volume/${pluginFolderName}`])
})

import assert from 'node:assert/strict'
import * as childProcess from 'node:child_process'
import * as fs from 'node:fs'
import * as fileSystem from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import * as valibot from 'valibot'
import { downloadRelease } from '../../../src/main/plugin-release.mts'
import { loadCommonJsModule } from './load-commonjs-module.mjs'

export const pluginManifest = {
	name: 'fixture-plugin',
	pluginName: 'Fixture Plugin',
	index: 'index.html',
	version: '2.0.0'
}

// A minimal packaged plugin: real JSON and files go through the application's
// filesystem helpers and Valibot schema, rather than a replacement parser.
export async function writePlugin(folder, manifest = pluginManifest, files = {}) {
	await fileSystem.mkdir(folder, { recursive: true })
	const contents = {
		'package.json':
			manifest === null || typeof manifest === 'string' ? manifest : JSON.stringify(manifest),
		'index.html': '<main>Local plugin</main>\n',
		...files
	}
	for (const [name, content] of Object.entries(contents)) {
		if (content === null) continue
		const path = join(folder, name)
		await fileSystem.mkdir(dirname(path), { recursive: true })
		await fileSystem.writeFile(path, content)
	}
	return folder
}

export async function pluginInstallFixture(t, { fetchImplementation, build } = {}) {
	const root = await fileSystem.mkdtemp(join(tmpdir(), 'ouroboros-plugin-install-'))
	t.after(() => fileSystem.rm(root, { recursive: true, force: true }))
	const userData = join(root, 'user data')
	const temp = join(root, 'downloads')
	const pluginFolder = join(userData, 'plugins')
	const builds = []
	const logs = []
	const logger = Object.fromEntries(
		['info', 'error', 'log'].map((level) => [level, (...args) => logs.push({ level, args })])
	)
	const helpers = loadCommonJsModule('src/main/helpers.ts', {
		'@electron-toolkit/utils': { is: { dev: false } },
		child_process: childProcess,
		electron: {},
		'fs/promises': { default: fileSystem },
		// Reading plugin files must not rewrite this test process's PATH.
		'fix-path': { default: () => undefined }
	})
	const schemas = loadCommonJsModule('src/main/schemas.ts', { valibot })
	const plugins = loadCommonJsModule(
		'src/main/plugins.ts',
		{
			electron: {
				app: {
					getPath(name) {
						assert.ok(
							['userData', 'temp'].includes(name),
							`Unexpected Electron path: ${name}`
						)
						return name === 'userData' ? userData : temp
					}
				}
			},
			fs,
			'fs/promises': { default: fileSystem },
			'./schemas': schemas,
			'./helpers': helpers,
			'./plugin-release.mjs': {
				// Keep HTTP responses deterministic while executing the real download,
				// stream-to-file and ZIP extraction path. Never contact GitHub in tests.
				downloadRelease: (user, repo, output) =>
					downloadRelease(user, repo, output, {
						fetchImplementation:
							fetchImplementation ??
							(() => assert.fail('This scenario must not download a plugin')),
						token: ''
					})
			},
			'./servers/docker': {
				// Installation must construct the actual compose paths, but no Docker
				// daemon or plugin process is needed to verify those arguments.
				buildDockerCompose: async (options) => {
					builds.push(options)
					await build?.(options)
				}
			},
			'./servers/file-server': {}
		},
		{ console: logger }
	)
	return { root, userData, temp, pluginFolder, plugins, builds, logs }
}

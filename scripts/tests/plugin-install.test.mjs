import assert from 'node:assert/strict'
import { mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import test from 'node:test'
import {
	pluginInstallFixture,
	pluginManifest,
	writePlugin
} from './helpers/plugin-install-fixture.mjs'

// Stored ZIP containing package.json (pluginManifest + dockerCompose), index.html
// and compose.yml. Keep the real extractor in the path: a mocked unpack operation
// would miss failures between the release download and the installed plugin.
const pluginArchive = Buffer.from(
	'UEsDBBQAAAAAAAAAIVyrsTE2hQAAAIUAAAAMAAAAcGFja2FnZS5qc29ueyJuYW1lIjogImZpeHR1cmUtcGx1Z2luIiwgInBsdWdpbk5hbWUiOiAiRml4dHVyZSBQbHVnaW4iLCAiaW5kZXgiOiAiaW5kZXguaHRtbCIsICJ2ZXJzaW9uIjogIjIuMC4wIiwgImRvY2tlckNvbXBvc2UiOiAiY29tcG9zZS55bWwifVBLAwQUAAAAAAAAACFcA81TEB8AAAAfAAAACgAAAGluZGV4Lmh0bWw8bWFpbj5Eb3dubG9hZGVkIHBsdWdpbjwvbWFpbj4KUEsDBBQAAAAAAAAAIVy0XyhVNAAAADQAAAALAAAAY29tcG9zZS55bWxzZXJ2aWNlczoKICBwbHVnaW46CiAgICBpbWFnZTogZml4dHVyZS9wbHVnaW46Mi4wLjAKUEsBAhQDFAAAAAAAAAAhXKuxMTaFAAAAhQAAAAwAAAAAAAAAAAAAAIABAAAAAHBhY2thZ2UuanNvblBLAQIUAxQAAAAAAAAAIVwDzVMQHwAAAB8AAAAKAAAAAAAAAAAAAACAAa8AAABpbmRleC5odG1sUEsBAhQDFAAAAAAAAAAhXLRfKFU0AAAANAAAAAsAAAAAAAAAAAAAAIAB9gAAAGNvbXBvc2UueW1sUEsFBgAAAAADAAMAqwAAAFMBAAAAAA==',
	'base64'
)

test('local installation uses the manifest ID and copies nested files before building compose', async (t) => {
	let run
	run = await pluginInstallFixture(t, {
		build: async ({ cwd, config }) => {
			assert.equal(
				cwd,
				join(run.pluginFolder, pluginManifest.name),
				'Build the installed plugin'
			)
			assert.equal(
				config,
				join(cwd, 'backend', 'compose.yml'),
				'Resolve compose from the install'
			)
			assert.equal(
				await readFile(config, 'utf8'),
				'services: {}\n',
				'Copy compose before building'
			)
			assert.equal(
				await readFile(join(cwd, 'assets', 'plugin.txt'), 'utf8'),
				'nested payload'
			)
		}
	})
	const source = await writePlugin(
		join(run.root, 'folder name differs from plugin ID'),
		{ ...pluginManifest, dockerCompose: 'backend/compose.yml' },
		{ 'backend/compose.yml': 'services: {}\n', 'assets/plugin.txt': 'nested payload' }
	)
	await run.plugins.addLocalPlugin(source)
	assert.equal(run.builds.length, 1, 'A compose plugin is built once after copying')
	assert.deepEqual(await readdir(run.pluginFolder), [pluginManifest.name])
	assert.equal(
		await readFile(join(source, 'assets', 'plugin.txt'), 'utf8'),
		'nested payload',
		'Keep the selected source folder'
	)
})

test('frontend-only plugins install without a Docker build', async (t) => {
	const run = await pluginInstallFixture(t)
	const source = await writePlugin(join(run.root, 'source'))
	await run.plugins.addLocalPlugin(source)
	assert.equal(
		await readFile(join(run.pluginFolder, pluginManifest.name, 'index.html'), 'utf8'),
		'<main>Local plugin</main>\n'
	)
	assert.deepEqual(run.builds, [], 'No compose file means no Docker build')
})

for (const [label, manifest, expectedError] of [
	['missing package.json', null, null],
	['malformed JSON', '{', 'Could not parse plugin package.json file'],
	[
		'missing required fields',
		{ name: 'invalid-plugin' },
		/Unable to parse plugin package.json file/
	]
]) {
	test(`local installation ignores ${label} without copying or building`, async (t) => {
		const run = await pluginInstallFixture(t)
		const source = await writePlugin(join(run.root, 'invalid-source'), manifest)
		await run.plugins.addLocalPlugin(source)
		assert.deepEqual(
			await readdir(run.pluginFolder),
			[],
			`${label} must not create an installed plugin`
		)
		assert.deepEqual(run.builds, [], `${label} must not reach Docker`)
		if (expectedError instanceof RegExp) {
			assert.match(
				run.logs[0]?.args[0] ?? '',
				expectedError,
				'Explain the manifest validation failure'
			)
		} else {
			assert.deepEqual(
				run.logs,
				expectedError ? [{ level: 'error', args: [expectedError] }] : []
			)
		}
	})
}

test('Docker build errors identify the plugin and retain the underlying failure', async (t) => {
	const failure = new Error('Fixture Docker build failed')
	const run = await pluginInstallFixture(t, { build: ({ onError }) => onError(failure) })
	const source = await writePlugin(
		join(run.root, 'source'),
		{ ...pluginManifest, dockerCompose: 'compose.yml' },
		{ 'compose.yml': 'services: {}\n' }
	)
	await run.plugins.addLocalPlugin(source)
	assert.deepEqual(run.logs, [
		{
			level: 'error',
			args: ["An error occurred while building plugin Fixture Plugin's Dockerfile:", failure]
		}
	])
	assert.ok(
		await stat(join(run.pluginFolder, pluginManifest.name, 'package.json')),
		'A reported build failure does not remove the copied plugin'
	)
})

test('plugin discovery excludes unusable entries and returns validated metadata only on request', async (t) => {
	const run = await pluginInstallFixture(t)
	const folder = await run.plugins.getPluginFolder()
	const valid = await writePlugin(
		join(folder, 'installed-directory'),
		{ ...pluginManifest, dockerCompose: 'compose.yml' },
		{ 'compose.yml': 'services: {}\n' }
	)
	await writePlugin(join(folder, '.hidden-plugin'))
	await writeFile(join(folder, 'loose-file'), 'not a plugin directory')
	await writePlugin(join(folder, 'missing-package'), null)
	await writePlugin(join(folder, 'bad-json'), '{')
	await writePlugin(join(folder, 'missing-index'), pluginManifest, { 'index.html': null })
	await writePlugin(join(folder, 'missing-compose'), {
		...pluginManifest,
		dockerCompose: 'compose.yml'
	})
	const entry = {
		name: pluginManifest.pluginName,
		id: pluginManifest.name,
		folder: valid,
		folderName: 'installed-directory'
	}
	assert.deepEqual(
		await run.plugins.getPluginList(folder),
		[entry],
		'Only runnable, visible plugin directories appear'
	)
	assert.deepEqual(
		await run.plugins.getPluginList(folder, true),
		[{ ...entry, json: { ...pluginManifest, dockerCompose: 'compose.yml' } }],
		'Include the real parsed manifest for lifecycle consumers'
	)
	assert.ok(
		run.logs.some(({ args }) => args[0] === 'Could not parse plugin package.json file'),
		'Explain malformed manifests'
	)
	assert.ok(
		run.logs.some(
			({ args }) =>
				args[0] ===
				`Index html file not found: ${join(folder, 'missing-index', 'index.html')}`
		),
		'Name the missing frontend file'
	)
	assert.ok(
		run.logs.some(
			({ args }) =>
				args[0] ===
				`Dockerfile not found: ${join(folder, 'missing-compose', 'compose.yml')}`
		),
		'Name the missing compose file'
	)
	const sent = []
	await run.plugins.sendPluginFolderContents(
		{ webContents: { send: (...args) => sent.push(args) } },
		folder
	)
	assert.deepEqual(sent, [['plugin-folder-contents', [entry]]])
	await run.plugins.sendPluginFolderContents(null, folder)
})

for (const [label, installedVersion, expectedAction] of [
	['missing install', null, 'Installed'],
	['older installed version', '1.9.9', 'Upgraded'],
	['same installed version', '2.0.0', 'current or newer'],
	['newer installed version', '10.0.0', 'current or newer'],
	['unversioned install', undefined, 'no comparable version']
]) {
	test(`bundled installation handles ${label} without overwriting a retained install`, async (t) => {
		const run = await pluginInstallFixture(t)
		const bundled = join(run.root, 'bundled')
		await writePlugin(join(bundled, 'bundle-directory'), pluginManifest, {
			'index.html': 'bundled frontend'
		})
		const target = join(run.pluginFolder, pluginManifest.name)
		const replaces = installedVersion === null || expectedAction === 'Upgraded'
		if (installedVersion !== null) {
			await writePlugin(
				target,
				{ ...pluginManifest, version: installedVersion },
				{ 'index.html': 'installed frontend', 'user-file.txt': 'retained payload' }
			)
		}
		await run.plugins.installPreinstalledPlugins(run.pluginFolder, bundled)
		assert.equal(
			await readFile(join(target, 'index.html'), 'utf8'),
			replaces ? 'bundled frontend' : 'installed frontend',
			`${label} must follow the existing version comparison policy`
		)
		assert.match(
			run.logs[0]?.args[0] ?? '',
			new RegExp(expectedAction),
			'Explain the install/skip decision'
		)
		if (expectedAction === 'Upgraded') {
			await assert.rejects(
				stat(join(target, 'user-file.txt')),
				{ code: 'ENOENT' },
				'An upgrade replaces the old directory instead of merging obsolete files'
			)
		} else if (!replaces) {
			assert.equal(await readFile(join(target, 'user-file.txt'), 'utf8'), 'retained payload')
		}
	})
}

test('bundled installation reports absent and invalid bundles while retaining an unreadable install', async (t) => {
	const run = await pluginInstallFixture(t)
	const bundled = join(run.root, 'bundled')
	await run.plugins.installPreinstalledPlugins(run.pluginFolder, bundled)
	assert.deepEqual(run.logs, [
		{ level: 'info', args: [`No preinstalled plugin folder found at ${bundled}`] }
	])
	await mkdir(bundled)
	await writeFile(join(bundled, 'README.txt'), 'Not a plugin directory')
	await writePlugin(join(bundled, 'missing-package'), null)
	await writePlugin(join(bundled, 'bad-package'), '{')
	await writePlugin(join(bundled, 'valid-plugin'))
	const target = await writePlugin(join(run.pluginFolder, pluginManifest.name), '{', {
		'index.html': 'keep unreadable install'
	})
	await run.plugins.installPreinstalledPlugins(run.pluginFolder, bundled)
	assert.equal(
		await readFile(join(target, 'index.html'), 'utf8'),
		'keep unreadable install',
		'Do not replace a local install whose version cannot be compared'
	)
	assert.ok(
		run.logs.some(({ args }) =>
			args[0].includes(
				'Skipping bundled plugin missing-package: Plugin package.json not found:'
			)
		)
	)
	assert.ok(
		run.logs.some(
			({ args }) =>
				args[0] ===
				'Skipping bundled plugin bad-package: Could not parse plugin package.json file'
		)
	)
	assert.ok(
		run.logs.some(
			({ args }) =>
				args[0] ===
				'Skipping bundled plugin fixture-plugin; existing install has no comparable version'
		)
	)
})

test('GitHub download extracts and installs a real ZIP, then removes the staging directory', async (t) => {
	const requests = []
	const releasesURL = 'https://api.github.com/repos/fixture-owner/fixture-repo/releases'
	const assetURL = 'https://api.github.com/repos/fixture-owner/fixture-repo/releases/assets/1'
	const run = await pluginInstallFixture(t, {
		fetchImplementation: async (url) => {
			requests.push(url)
			if (url === assetURL) return new Response(pluginArchive)
			return Response.json([{ assets: [{ name: 'plugin.zip', url: assetURL }] }])
		}
	})
	const staging = join(run.temp, 'fixture-owner-fixture-repo')
	await mkdir(staging, { recursive: true })
	await writeFile(join(staging, 'stale.txt'), 'A previous incomplete download')
	await run.plugins.downloadPlugin('https://github.com/fixture-owner/fixture-repo')
	assert.deepEqual(
		requests,
		[releasesURL, assetURL],
		'Fetch the requested repository releases once, then its selected archive'
	)
	assert.deepEqual(run.logs, [], 'A successful download must not hide an extraction failure')
	const target = join(run.pluginFolder, pluginManifest.name)
	assert.deepEqual(JSON.parse(await readFile(join(target, 'package.json'), 'utf8')), {
		...pluginManifest,
		dockerCompose: 'compose.yml'
	})
	assert.equal(
		await readFile(join(target, 'index.html'), 'utf8'),
		'<main>Downloaded plugin</main>\n'
	)
	assert.equal(
		run.builds.length,
		1,
		'A downloaded compose plugin uses the same build path as a local install'
	)
	assert.equal(run.builds[0].cwd, target)
	assert.equal(run.builds[0].config, join(target, 'compose.yml'))
	await assert.rejects(
		stat(join(target, 'stale.txt')),
		{ code: 'ENOENT' },
		'Clear the previous staging directory before downloading'
	)
	await assert.rejects(stat(staging), { code: 'ENOENT' }, 'Remove staging after installation')
})

test('non-GitHub URLs are reported without downloading or installing', async (t) => {
	const run = await pluginInstallFixture(t)
	await run.plugins.downloadPlugin('https://example.invalid/owner/plugin')
	assert.deepEqual(run.logs, [{ level: 'error', args: ['URL is not a github repository URL'] }])
	await assert.rejects(
		stat(run.pluginFolder),
		{ code: 'ENOENT' },
		'Invalid URLs must not create an install directory'
	)
})

for (const [label, status, body, reason] of [
	[
		'release lookup failure',
		503,
		null,
		'GitHub release lookup failed for fixture-owner/fixture-repo: 503'
	],
	['no downloadable release', 200, [], 'Could not find a release for fixture-owner/fixture-repo']
]) {
	test(`download reports ${label} without installing`, async (t) => {
		const run = await pluginInstallFixture(t, {
			fetchImplementation: async () => Response.json(body, { status })
		})
		await run.plugins.downloadPlugin('https://github.com/fixture-owner/fixture-repo')
		assert.equal(run.logs[0]?.level, 'error')
		assert.equal(
			run.logs[0]?.args[0]?.message,
			reason,
			'Preserve the downloader failure reason'
		)
		await assert.rejects(
			stat(run.pluginFolder),
			{ code: 'ENOENT' },
			'A failed release lookup must not install a plugin'
		)
		assert.deepEqual(run.builds, [], 'A failed download must not reach Docker')
	})
}

for (const [label, status, body, reason] of [
	[
		'asset download failure',
		502,
		null,
		'GitHub release asset download failed for plugin.zip: 502'
	],
	['invalid ZIP', 200, 'This is not a ZIP archive', 'Bad archive']
]) {
	test(`download reports ${label} without copying partial contents into plugins`, async (t) => {
		const assetURL = 'https://api.github.com/repos/fixture-owner/fixture-repo/releases/assets/1'
		const run = await pluginInstallFixture(t, {
			fetchImplementation: async (url) =>
				url === assetURL
					? new Response(body, { status })
					: Response.json([{ assets: [{ name: 'plugin.zip', url: assetURL }] }])
		})
		await run.plugins.downloadPlugin('https://github.com/fixture-owner/fixture-repo')
		assert.equal(run.logs[0]?.level, 'error')
		assert.equal(
			run.logs[0]?.args[0]?.message,
			reason,
			'Preserve the asset or extractor failure reason'
		)
		await assert.rejects(
			stat(run.pluginFolder),
			{ code: 'ENOENT' },
			'A failed archive download or extraction must not install a plugin'
		)
		assert.deepEqual(run.builds, [], 'A failed archive must not reach Docker')
	})
}

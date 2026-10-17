import assert from 'node:assert/strict'
import { copyFile, mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, relative, resolve } from 'node:path'
import test from 'node:test'
import { load as loadYaml } from 'js-yaml'
import { copyFiles, getFileMatchers, getMainFileMatchers } from 'app-builder-lib/out/fileMatcher.js'
import { expandMacro } from 'app-builder-lib/out/util/macroExpander.js'
import { Platform } from 'app-builder-lib'
import { getRepositoryInfo } from 'app-builder-lib/out/util/repositoryInfo.js'
import { getAppUpdatePublishConfiguration } from 'app-builder-lib/out/publish/PublishManager.js'
import {
	RELEASE_PLATFORMS,
	loadReleaseLock,
	pluginLocksForFlavor,
	releaseFingerprint,
	sha256File,
	validateReleaseLock
} from '../lib/release-lock.mjs'
import { gitOutput } from '../lib/git.mjs'
import { npmInvocation } from '../lib/npm.mjs'
import { isReleaseAssetName, parsePlatform } from '../build-release-artifacts.mjs'
import { verifyReleaseArtifacts } from '../verify-release-artifacts.mjs'

const root = process.cwd()

test('the release lock independently pins CPU and CUDA autoseg inputs', async () => {
	const { lock } = await loadReleaseLock(root)
	const cpuPlugins = pluginLocksForFlavor(lock, 'with-plugins-cpu')
	const cudaPlugins = pluginLocksForFlavor(lock, 'with-plugins-cuda')

	assert.strictEqual(cpuPlugins[0], lock.plugins.neuroglancer)
	assert.strictEqual(cpuPlugins[1], lock.plugins.autosegCpu)
	assert.strictEqual(cudaPlugins[0], lock.plugins.neuroglancer)
	assert.strictEqual(cudaPlugins[1], lock.plugins.autosegCuda)
	assert.notEqual(lock.plugins.autosegCpu.asset, lock.plugins.autosegCuda.asset)
	assert.notEqual(
		lock.plugins.autosegCpu.backendImageDigest,
		lock.plugins.autosegCuda.backendImageDigest
	)
	assert.equal(lock.plugins.autosegCpu.variant, 'cpu')
	assert.equal(lock.plugins.autosegCuda.variant, 'cuda')
	assert.equal(lock.plugins.autosegCpu.verifyEmbeddedBackendImage, true)
	assert.equal(lock.plugins.autosegCuda.verifyEmbeddedBackendImage, true)
})

test('release lock validation rejects mutable or malformed identities', async () => {
	const { lock } = await loadReleaseLock(root)
	const invalidServer = structuredClone(lock)
	invalidServer.serverImage.digest = 'latest'
	assert.throws(() => validateReleaseLock(invalidServer), /serverImage\.digest/)

	const invalidAsset = structuredClone(lock)
	invalidAsset.plugins.autosegCuda.assetSha256 = 'not-a-sha'
	assert.throws(() => validateReleaseLock(invalidAsset), /assetSha256/)

	const duplicateFlavor = structuredClone(lock)
	duplicateFlavor.packageFlavors.push(duplicateFlavor.packageFlavors[0])
	assert.throws(() => validateReleaseLock(duplicateFlavor), /duplicates/)
})

test('release lock identity is independent of checkout line endings', async () => {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-release-lock-test-'))
	try {
		const releaseDirectory = join(directory, 'release')
		const lockPath = join(releaseDirectory, 'release-lock.json')
		const source = await readFile(join(root, 'release', 'release-lock.json'), 'utf8')
		await mkdir(releaseDirectory)

		await writeFile(lockPath, source.replace(/\r?\n/g, '\n'))
		const lf = await loadReleaseLock(directory)
		await writeFile(lockPath, source.replace(/\r?\n/g, '\r\n'))
		const crlf = await loadReleaseLock(directory)

		assert.deepEqual(crlf.lock, lf.lock)
		assert.equal(crlf.sha256, lf.sha256)
	} finally {
		await rm(directory, { recursive: true, force: true })
	}
})

test('release fingerprints change with the source tree or input lock', () => {
	const first = releaseFingerprint({
		lockSha256: 'a'.repeat(64),
		sourceTree: 'b'.repeat(40),
		releaseVersion: '1.5.1'
	})
	const second = releaseFingerprint({
		lockSha256: 'c'.repeat(64),
		sourceTree: 'b'.repeat(40),
		releaseVersion: '1.5.1'
	})
	assert.notEqual(first, second)
})

test('release artifact selection excludes electron-builder diagnostics', () => {
	for (const name of [
		'ouroboros-1.5.1-with-plugins-cuda.AppImage',
		'ouroboros-1.5.1-with-plugins-cuda-setup.exe',
		'latest-linux-with-plugins-cuda.yml',
		'Ouroboros-1.5.1-with-plugins-cuda-mac.zip.blockmap'
	]) {
		assert.equal(isReleaseAssetName(name), true, name)
	}
	for (const name of ['builder-debug.yml', 'builder-effective-config.yaml', 'linux-unpacked']) {
		assert.equal(isReleaseAssetName(name), false, name)
	}
	assert.equal(parsePlatform(['--platform', 'linux']), 'linux')
	assert.throws(() => parsePlatform(['--platform', 'solaris']), /must be one of/)
})

test('staged package metadata preserves GitHub update configuration without a git checkout', async (t) => {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-package-repository-test-'))
	t.after(() => rm(directory, { recursive: true, force: true }))
	await copyFile(join(root, 'package.json'), join(directory, 'package.json'))
	await assert.rejects(stat(join(directory, '.git')), { code: 'ENOENT' })
	const metadata = JSON.parse(await readFile(join(directory, 'package.json'), 'utf8'))
	const repositoryInfo = await getRepositoryInfo(directory, metadata)
	assert.equal(repositoryInfo?.type, 'github')
	assert.equal(repositoryInfo.user, 'ChengLabResearch')
	assert.equal(repositoryInfo.project, 'ouroboros')

	const config = loadYaml(await readFile(join(root, 'electron-builder.yml'), 'utf8'))
	const appInfo = { channel: null, updaterCacheDirName: 'ouroboros-updater' }
	const updateConfiguration = await getAppUpdatePublishConfiguration(
		{
			info: { repositoryInfo: Promise.resolve(repositoryInfo), config, appInfo },
			config,
			appInfo,
			platformSpecificBuildOptions: {},
			platform: Platform.LINUX,
			expandMacro: (value) => value
		},
		null,
		1,
		false
	)
	assert.equal(updateConfiguration?.provider, 'github')
	assert.equal(updateConfiguration.owner, 'ChengLabResearch')
	assert.equal(updateConfiguration.repo, 'ouroboros')
})

test('electron-builder selects runtime files and copies only external flavor resources to existing package paths', async (t) => {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-package-allowlist-test-'))
	t.after(() => rm(directory, { recursive: true, force: true }))
	const config = loadYaml(await readFile(join(root, 'electron-builder.yml'), 'utf8'))
	const projectRoot = join(directory, 'compiled')
	const resourcesRoot = join(directory, 'flavors', 'with-plugins-cpu')
	const output = join(directory, 'dist')
	const destination = join(output, 'resources', 'app')
	const originalResources = process.env.OUROBOROS_PACKAGE_RESOURCES_DIR
	t.after(() => {
		if (originalResources === undefined) delete process.env.OUROBOROS_PACKAGE_RESOURCES_DIR
		else process.env.OUROBOROS_PACKAGE_RESOURCES_DIR = originalResources
	})
	// Builder resolves FileSet.from against the project before expanding the macro.
	process.env.OUROBOROS_PACKAGE_RESOURCES_DIR = relative(projectRoot, resourcesRoot)
	const macroExpander = (value) => expandMacro(value, 'x64', {})
	const info = {
		config,
		projectDir: projectRoot,
		buildResourcesDir: 'build',
		debugLogger: { isEnabled: false }
	}
	const mainMatchers = getMainFileMatchers(
		projectRoot,
		destination,
		macroExpander,
		{},
		{ info },
		output,
		false
	)
	const extraMatchers = getFileMatchers(config, 'extraResources', join(output, 'resources'), {
		macroExpander,
		customBuildOptions: {},
		globalOutDir: output,
		defaultSrc: projectRoot
	})
	assert.equal(extraMatchers.length, 1)
	assert.equal(resolve(extraMatchers[0].from), resourcesRoot)
	assert.equal(extraMatchers[0].to, join(output, 'resources', 'extra-resources'))
	const expectedRuntime = [
		'package.json',
		'out/main/index.js',
		'out/preload/index.js',
		'out/renderer/index.html',
		'resources/icon.png',
		'resources/processes/volume-server-script.mjs'
	]
	const expectedExtra = [
		'server/compose.yml',
		'server/server-image.json',
		'package-flavor.json',
		'preinstalled-plugins/example/package.json',
		'preinstalled-plugins/example/assets/index.js'
	]
	const unwanted = [
		'src/main/index.ts',
		'python/payload.bin',
		'documentation/README.md',
		'release-assets/previous.zip',
		'.package-plugin-artifacts/plugin.zip',
		'release/release-lock.json',
		'scripts/build-release-artifacts.mjs',
		...expectedExtra.map((name) => `extra-resources/${name}`),
		'notes.json',
		'package-lock.json',
		'.env',
		'.npmrc',
		'build/icon.png'
	]
	const selectedRuntime = []
	const runtimeFilters = mainMatchers.map((matcher) => matcher.createFilter())
	for (const name of [...expectedRuntime, ...unwanted]) {
		const path = join(projectRoot, name)
		await mkdir(dirname(path), { recursive: true })
		await writeFile(path, name)
		const details = await stat(path)
		if (runtimeFilters.some((filter) => filter(path, details))) selectedRuntime.push(name)
	}
	assert.deepEqual(selectedRuntime.sort(), expectedRuntime.sort())
	for (const name of [
		...expectedExtra,
		'unapproved.json',
		'server/unapproved.json',
		'node_modules/unapproved/index.js',
		'package-lock.json',
		'.env'
	]) {
		const path = join(resourcesRoot, name)
		await mkdir(dirname(path), { recursive: true })
		await writeFile(path, `external resource: ${name}`)
	}
	await copyFiles(extraMatchers, null, false)
	const resourcesDestination = join(output, 'resources')
	const copiedResources = []
	for (const name of await readdir(resourcesDestination, { recursive: true })) {
		const path = join(resourcesDestination, name)
		if ((await stat(path)).isFile()) copiedResources.push(name)
	}
	assert.deepEqual(
		copiedResources.sort(),
		expectedExtra.map((name) => join('extra-resources', name)).sort()
	)
	for (const name of expectedExtra) {
		assert.equal(
			await readFile(join(resourcesDestination, 'extra-resources', name), 'utf8'),
			`external resource: ${name}`
		)
	}
})

test('npm lifecycle commands stay shell-free on Windows', () => {
	const currentInvocation = npmInvocation(['--version'])
	if (process.env.npm_execpath) {
		assert.equal(currentInvocation.command, process.execPath)
		assert.equal(currentInvocation.args[0], process.env.npm_execpath)
		assert.equal(currentInvocation.args[1], '--version')
	} else {
		assert.deepEqual(currentInvocation, { command: 'npm', args: ['--version'] })
	}

	assert.deepEqual(
		npmInvocation(['run', 'build'], {
			nodeExecutable: 'C:\\Program Files\\nodejs\\node.exe',
			npmExecutable: 'C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js',
			platform: 'win32'
		}),
		{
			command: 'C:\\Program Files\\nodejs\\node.exe',
			args: ['C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npm-cli.js', 'run', 'build']
		}
	)
	assert.throws(
		() => npmInvocation(['run', 'build'], { npmExecutable: '', platform: 'win32' }),
		/npm_execpath is required/
	)
})

test('tag publication verification fails closed when a prebuilt file changes', async () => {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-release-test-'))
	try {
		const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
		const { lock, sha256: releaseLockSha256 } = await loadReleaseLock(root)
		const sourceCommit = await gitOutput(root, ['rev-parse', 'HEAD'])
		const sourceTree = await gitOutput(root, ['rev-parse', 'HEAD^{tree}'])
		const inputFingerprint = releaseFingerprint({
			lockSha256: releaseLockSha256,
			sourceTree,
			releaseVersion: packageJson.version
		})

		for (const platform of RELEASE_PLATFORMS) {
			const artifacts = []
			for (const flavor of lock.packageFlavors) {
				for (const [name, type] of [
					[`ouroboros-${platform}-${flavor}.zip`, 'installer'],
					[`package-metadata-${platform}-${flavor}.json`, 'package-metadata']
				]) {
					const path = join(directory, name)
					await writeFile(path, `${platform}:${flavor}:${type}\n`)
					const bytes = (await readFile(path)).length
					artifacts.push({
						name,
						flavor,
						type,
						bytes,
						sha256: await sha256File(path)
					})
				}
			}

			await writeFile(
				join(directory, `release-build-${platform}.json`),
				`${JSON.stringify(
					{
						schemaVersion: 1,
						releaseVersion: packageJson.version,
						platform,
						flavors: lock.packageFlavors,
						sourceCommit,
						certifiedCommit: sourceCommit,
						sourceTree,
						certification: {
							mode: 'built',
							workflowRun: 'test-run'
						},
						releaseLockSha256,
						inputFingerprint,
						releaseLock: lock,
						artifacts,
						metrics: {
							operations: {
								electronCompilations: 1,
								packageOperations: lock.packageFlavors.length,
								serverWheelBuilds: 0,
								serverImageBuilds: 0
							}
						}
					},
					null,
					2
				)}\n`
			)
		}

		await verifyReleaseArtifacts({
			root,
			directory,
			commit: sourceCommit,
			ref: `v${packageJson.version}`
		})

		const tampered = join(
			directory,
			`ouroboros-${RELEASE_PLATFORMS[0]}-${lock.packageFlavors[0]}.zip`
		)
		await writeFile(tampered, 'tampered\n')
		await assert.rejects(
			verifyReleaseArtifacts({
				root,
				directory,
				commit: sourceCommit,
				ref: `v${packageJson.version}`
			}),
			/(size|SHA-256) mismatch/
		)
	} finally {
		await rm(directory, { recursive: true, force: true })
	}
})

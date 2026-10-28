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
	sha256File,
	validateReleaseLock
} from '../lib/release-lock.mjs'
import { gitOutput } from '../lib/git.mjs'
import { getReleaseInputs } from '../lib/release-inputs.mjs'
import { npmInvocation } from '../lib/npm.mjs'
import { isReleaseAssetName, parsePlatform } from '../build-release-artifacts.mjs'
import { selectReleaseBuild } from '../select-release-build.mjs'
import { verifyReleaseArtifacts } from '../verify-release-artifacts.mjs'
import { releaseRepository } from './helpers/release-repository.mjs'

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

test('release reuse follows real Git ancestry and committed artifact inputs', async (t) => {
	const repository = await releaseRepository(t, root)
	const { built, pythonOnly, artifactChanged, artifactReverted, side } = repository
	const inputs = await getReleaseInputs(repository.root, built)
	const artifacts = () =>
		RELEASE_PLATFORMS.map((platform) => ({
			name: `release-assets-${platform}`,
			expired: false
		}))
	const run = (id, head_sha, overrides = {}) => ({
		id,
		head_sha,
		head_branch: 'main',
		event: 'push',
		status: 'completed',
		conclusion: 'success',
		...overrides
	})
	const select = (commit, runs, listArtifacts = async () => artifacts()) =>
		selectReleaseBuild({ root: repository.root, commit, runs, listArtifacts })

	await t.test(
		'Python-only history preserves inputs while retaining the original source tree',
		async () => {
			const later = await getReleaseInputs(repository.root, pythonOnly)
			assert.match(
				await gitOutput(repository.root, ['show', `${pythonOnly}:python/pyproject.toml`]),
				/^version = "0\.0\.0\+release\.fixture"/m
			)
			assert.equal(
				await gitOutput(repository.root, ['rev-parse', `${pythonOnly}:package-lock.json`]),
				await gitOutput(repository.root, ['rev-parse', `${built}:package-lock.json`])
			)
			assert.equal(later.inputFingerprint, inputs.inputFingerprint)
			assert.equal(later.releaseLockSha256, inputs.releaseLockSha256)
			assert.notEqual(later.sourceTree, inputs.sourceTree)
			assert.notEqual(later.sourceCommit, inputs.sourceCommit)
			assert.equal((await select(pythonOnly, [run(1, built)])).id, 1)
		}
	)

	await t.test(
		'an artifact change requires a build but reverting it restores input identity',
		async () => {
			const changed = await getReleaseInputs(repository.root, artifactChanged)
			const reverted = await getReleaseInputs(repository.root, artifactReverted)
			assert.notEqual(changed.inputFingerprint, inputs.inputFingerprint)
			assert.equal(reverted.inputFingerprint, inputs.inputFingerprint)
			await assert.rejects(select(artifactChanged, [run(1, built)]))
			assert.equal(
				(await select(artifactReverted, [run(1, built), run(2, artifactChanged)])).id,
				1
			)
		}
	)

	await t.test(
		'selection prefers the nearest matching ancestor over workflow run recency',
		async () => {
			const selected = await select(artifactReverted, [
				run(100, built),
				run(200, side),
				run(300, artifactChanged),
				run(10, pythonOnly)
			])
			assert.equal(selected.id, 10)
			assert.equal(selected.head_sha, pythonOnly)
		}
	)

	await t.test('successful main dispatches recertify changed artifact inputs', async () => {
		const selected = await select(artifactChanged, [
			run(1, built),
			run(2, artifactChanged, { event: 'workflow_dispatch' })
		])
		assert.equal(selected.id, 2)
	})

	await t.test(
		'pull requests, failed builds, other branches, and nonancestor builds cannot certify',
		async () => {
			const invalidRuns = [
				run(2, built, { event: 'pull_request' }),
				run(3, built, { conclusion: 'failure' }),
				run(4, built, { head_branch: 'feature' }),
				run(5, side)
			]
			await assert.rejects(select(pythonOnly, invalidRuns))
			assert.equal((await select(pythonOnly, [...invalidRuns, run(1, built)])).id, 1)
			await assert.rejects(select(side, [run(1, side)]))
		}
	)

	await t.test('expired or incomplete native artifact sets are skipped', async () => {
		const lookedUp = []
		const selected = await select(
			pythonOnly,
			[run(3, pythonOnly), run(2, pythonOnly), run(1, built)],
			async (id) => {
				lookedUp.push(id)
				if (id === 3) return artifacts().map((artifact) => ({ ...artifact, expired: true }))
				if (id === 2) return artifacts().filter(({ name }) => name !== 'release-assets-mac')
				return artifacts()
			}
		)
		assert.equal(selected.id, 1)
		assert.deepEqual(new Set(lookedUp), new Set([1, 2, 3]))
		await assert.rejects(select(pythonOnly, [run(3, pythonOnly)], async () => []))
	})

	const directory = join(repository.directory, 'artifacts')
	await mkdir(directory)
	const manifests = new Map()
	for (const platform of RELEASE_PLATFORMS) {
		const listedArtifacts = []
		for (const flavor of inputs.lock.packageFlavors) {
			for (const [name, type] of [
				[`ouroboros-${platform}-${flavor}.zip`, 'installer'],
				[`package-metadata-${platform}-${flavor}.json`, 'package-metadata']
			]) {
				const path = join(directory, name)
				await writeFile(path, `${platform}:${flavor}:${type}\n`)
				listedArtifacts.push({
					name,
					flavor,
					type,
					bytes: (await stat(path)).size,
					sha256: await sha256File(path)
				})
			}
		}
		manifests.set(platform, {
			schemaVersion: 2,
			releaseVersion: inputs.releaseVersion,
			platform,
			flavors: inputs.lock.packageFlavors,
			sourceCommit: built,
			certifiedCommit: built,
			sourceTree: inputs.sourceTree,
			certification: { mode: 'built', workflowRun: '123' },
			releaseLockSha256: inputs.releaseLockSha256,
			inputFingerprint: inputs.inputFingerprint,
			releaseLock: inputs.lock,
			artifacts: listedArtifacts,
			metrics: {
				operations: {
					electronCompilations: 1,
					packageOperations: inputs.lock.packageFlavors.length,
					serverWheelBuilds: 0,
					serverImageBuilds: 0
				}
			}
		})
	}
	const writeManifest = (platform, manifest) =>
		writeFile(join(directory, `release-build-${platform}.json`), JSON.stringify(manifest))
	for (const [platform, manifest] of manifests) await writeManifest(platform, manifest)
	const verify = (overrides = {}) =>
		verifyReleaseArtifacts({
			root: repository.root,
			directory,
			commit: pythonOnly,
			ref: `v${inputs.releaseVersion}`,
			buildCommit: built,
			workflowRun: '123',
			...overrides
		})

	await t.test(
		'verification accepts exact and equivalent descendant tags without rewriting provenance',
		async () => {
			for (const commit of [built, pythonOnly, artifactReverted]) await verify({ commit })
			for (const [platform, manifest] of manifests) {
				assert.deepEqual(
					JSON.parse(
						await readFile(join(directory, `release-build-${platform}.json`), 'utf8')
					),
					manifest
				)
			}
		}
	)

	await t.test(
		'verification rejects changed inputs, unrelated tags, wrong versions, and wrong runs',
		async () => {
			await assert.rejects(verify({ commit: artifactChanged }))
			await assert.rejects(verify({ commit: side }))
			await assert.rejects(verify({ ref: 'v0.0.0' }))
			await assert.rejects(verify({ workflowRun: '456' }))
			await assert.rejects(verify({ buildCommit: side }))
		}
	)

	const invalidManifestChanges = [
		[
			'legacy whole-tree schema',
			(manifest) => {
				manifest.schemaVersion = 1
			}
		],
		[
			'incorrect source tree',
			(manifest) => {
				manifest.sourceTree = 'a'.repeat(40)
			}
		],
		[
			'incorrect input fingerprint',
			(manifest) => {
				manifest.inputFingerprint = 'b'.repeat(64)
			}
		],
		[
			'incorrect lock hash',
			(manifest) => {
				manifest.releaseLockSha256 = 'c'.repeat(64)
			}
		],
		[
			'rewritten certification commit',
			(manifest) => {
				manifest.certifiedCommit = pythonOnly
			}
		],
		[
			'mixed platform source commits',
			(manifest) => {
				manifest.sourceCommit = pythonOnly
				manifest.certifiedCommit = pythonOnly
			}
		],
		[
			'mixed platform workflow runs',
			(manifest) => {
				manifest.certification.workflowRun = '456'
			}
		],
		[
			'PR artifact recertification',
			(manifest) => {
				manifest.certification = {
					mode: 'reused-pr-build',
					sourceRun: '123',
					operationsInCertificationRun: {
						electronCompilations: 0,
						packageOperations: 0,
						serverWheelBuilds: 0,
						serverImageBuilds: 0
					}
				}
			}
		]
	]
	for (const [label, mutate] of invalidManifestChanges) {
		await t.test(`verification rejects ${label}`, async () => {
			// The first platform establishes certification identity; later platforms
			// must derive from that same build and workflow run.
			const platform = RELEASE_PLATFORMS[1]
			const manifest = structuredClone(manifests.get(platform))
			mutate(manifest)
			await writeManifest(platform, manifest)
			try {
				await assert.rejects(verify())
			} finally {
				await writeManifest(platform, manifests.get(platform))
			}
		})
	}

	await t.test(
		'verification checks installer bytes even when source certification is valid',
		async () => {
			const name = manifests.get('linux').artifacts[0].name
			const path = join(directory, name)
			const original = await readFile(path)
			const tampered = Buffer.from(original)
			tampered[0] ^= 1
			await writeFile(path, tampered)
			try {
				await assert.rejects(verify(), /SHA-256 mismatch/)
			} finally {
				await writeFile(path, original)
			}
		}
	)
})

import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import {
	lstat,
	mkdir,
	mkdtemp,
	readFile,
	readdir,
	readlink,
	realpath,
	rm,
	symlink,
	writeFile
} from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { promisify } from 'node:util'
import test from 'node:test'
import { createPackageWorkspace } from '../lib/package-workspace.mjs'
import { buildReleaseArtifacts } from '../build-release-artifacts.mjs'
import { validatePackageConfiguration } from '../preflight-release.mjs'
import { prepareProductionServer } from '../prepare-production-server.mjs'
import { preparePackageFlavor } from '../prepare-package-flavor.mjs'
import { loadReleaseLock } from '../lib/release-lock.mjs'
import { getReleaseInputs } from '../lib/release-inputs.mjs'

const run = promisify(execFile)
const repository = process.cwd()

async function put(root, name, value) {
	const path = join(root, name)
	await mkdir(dirname(path), { recursive: true })
	await writeFile(path, typeof value === 'string' ? value : JSON.stringify(value))
}

function inside(parent, child) {
	const path = relative(parent, child)
	return (
		path === '' ||
		(!isAbsolute(path) &&
			path !== '..' &&
			!path.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
	)
}

async function snapshot(root) {
	const result = {}
	async function walk(directory) {
		for (const entry of await readdir(directory, { withFileTypes: true })) {
			const path = join(directory, entry.name)
			const name = relative(root, path)
			if (entry.isSymbolicLink()) result[name] = { link: await readlink(path) }
			else if (entry.isDirectory()) {
				result[name] = { directory: true }
				await walk(path)
			} else result[name] = { bytes: (await readFile(path)).toString('hex') }
		}
	}
	await walk(root)
	return result
}

async function fixture(t, { git = false } = {}) {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-package-workspace-test-'))
	t.after(() => rm(directory, { recursive: true, force: true }))
	const root = join(directory, 'checkout')
	const tempRoot = join(directory, 'runner-temp')
	await mkdir(root)
	await mkdir(tempRoot)
	const lock = JSON.parse(await readFile(join(repository, 'release/release-lock.json'), 'utf8'))
	const packageJson = {
		name: 'ouroboros',
		productName: 'Ouroboros',
		version: lock.releaseVersion,
		main: './out/main/index.js',
		dependencies: { runtime: '1.0.0' },
		devDependencies: { compiler: '1.0.0' }
	}
	for (const [name, value] of Object.entries({
		'package.json': packageJson,
		'package-lock.json': {
			name: packageJson.name,
			version: packageJson.version,
			lockfileVersion: 3,
			packages: { '': { version: packageJson.version } }
		},
		'tsconfig.json': {},
		'tsconfig.node.json': {},
		'tsconfig.web.json': {},
		'electron.vite.config.ts': 'export default {}\n',
		'.npmrc': 'audit=false\n',
		'src/main/index.ts': 'export const source = true\n',
		'resources/processes/worker.mjs': 'export const worker = true\n',
		'build/icon.png': 'fixture icon',
		'node_modules/runtime/package.json': {
			name: 'runtime',
			version: '1.0.0',
			main: 'index.js'
		},
		'node_modules/runtime/index.js': 'original dependency\n',
		'node_modules/compiler/package.json': { name: 'compiler', version: '1.0.0' },
		'release/release-lock.json': lock,
		'release/artifact-inputs.json': await readFile(
			join(repository, 'release/artifact-inputs.json'),
			'utf8'
		),
		'out/main/index.js': 'stale checkout compilation\n',
		'dist/old-installer.AppImage': 'keep checkout installer\n',
		'extra-resources/package-flavor.json': { flavor: 'keep checkout metadata' },
		'extra-resources/preinstalled-plugins/stale/index.html': 'keep checkout plugin\n',
		'release-assets/previous.zip': 'keep checkout release\n',
		'.package-plugin-artifacts/plugin.zip': 'keep checkout cache\n',
		'documentation/private-note.md': 'not a build input\n',
		'python/large-fixture.bin': 'not a build input\n',
		'.env': 'FIXTURE_SECRET=not-a-build-input\n'
	}))
		await put(root, name, value)
	await put(
		root,
		'electron-builder.yml',
		await readFile(join(repository, 'electron-builder.yml'), 'utf8')
	)
	if (git) {
		await run('git', ['init', '--quiet', root])
		await run('git', ['-C', root, 'add', '.'])
		await run('git', [
			'-C',
			root,
			'-c',
			'user.name=Fixture',
			'-c',
			'user.email=fixture@example.invalid',
			'-c',
			'commit.gpgsign=false',
			'commit',
			'--quiet',
			'-m',
			'fixture'
		])
	}
	return { directory, root, tempRoot, lock, packageJson }
}

test('source staging and core preparation leave the original checkout unchanged', async (t) => {
	const { root, tempRoot, packageJson } = await fixture(t)
	const before = await snapshot(root)
	const workspace = await createPackageWorkspace({ root, tempRoot })
	t.after(() => workspace.cleanup())
	await workspace.stageSource()
	assert.equal(inside(root, workspace.directory), false)
	assert.equal(inside(tempRoot, workspace.directory), true)
	assert.equal(
		await readFile(join(workspace.compiledDir, 'src/main/index.ts'), 'utf8'),
		'export const source = true\n'
	)
	for (const name of [
		'out',
		'dist',
		'extra-resources',
		'release-assets',
		'.package-plugin-artifacts',
		'documentation',
		'python',
		'.env'
	]) {
		await assert.rejects(lstat(join(workspace.compiledDir, name)), { code: 'ENOENT' })
	}
	for (const name of [
		'src',
		'node_modules',
		'src/main/index.ts',
		'node_modules/runtime/index.js'
	]) {
		const staged = await lstat(join(workspace.compiledDir, name))
		const original = await lstat(join(root, name))
		assert.equal(staged.isSymbolicLink(), false)
		assert.notDeepEqual([staged.dev, staged.ino], [original.dev, original.ino])
	}
	const project = await workspace.stageFlavor('core')
	const preparation = {
		project,
		packageJson,
		releaseLock: await loadReleaseLock(root),
		environment: { OUROBOROS_PACKAGE_FLAVOR: 'core' }
	}
	await prepareProductionServer(preparation)
	await preparePackageFlavor(preparation)
	const metadata = JSON.parse(
		await readFile(join(project.resourcesDir, 'package-flavor.json'), 'utf8')
	)
	assert.equal(metadata.flavor, 'core')
	assert.deepEqual(metadata.plugins, [])
	assert.equal(metadata.serverImage.digest, preparation.releaseLock.lock.serverImage.digest)
	await assert.rejects(lstat(join(project.resourcesDir, 'preinstalled-plugins')), {
		code: 'ENOENT'
	})
	await assert.rejects(lstat(join(workspace.compiledDir, 'extra-resources')), { code: 'ENOENT' })
	assert.deepEqual(await snapshot(root), before)
})

test('preflight staging needs no installed dependency tree', async (t) => {
	const { root, tempRoot } = await fixture(t)
	await rm(join(root, 'node_modules'), { recursive: true })
	const workspace = await createPackageWorkspace({ root, tempRoot })
	t.after(() => workspace.cleanup())
	await workspace.stageSource({ includeDependencies: false })
	const project = await workspace.stageFlavor('core')
	await assert.rejects(lstat(join(project.root, 'node_modules')), { code: 'ENOENT' })
	assert.equal(
		await readFile(join(project.root, 'package.json'), 'utf8'),
		await readFile(join(root, 'package.json'), 'utf8')
	)
})

test('relative dependency symlinks stay inside the single copied dependency tree', async (t) => {
	const { root, tempRoot } = await fixture(t)
	await mkdir(join(root, 'node_modules/.bin'))
	await symlink('../runtime/index.js', join(root, 'node_modules/.bin/runtime'))
	const workspace = await createPackageWorkspace({ root, tempRoot })
	t.after(() => workspace.cleanup())
	await workspace.stageSource()
	const project = await workspace.stageFlavor('core')
	assert.equal(
		await readlink(join(project.root, 'node_modules/.bin/runtime')),
		'../runtime/index.js'
	)
	assert.equal(project.root, workspace.compiledDir)
	assert.equal(
		await realpath(join(project.root, 'node_modules/.bin/runtime')),
		join(workspace.compiledDir, 'node_modules/runtime/index.js')
	)
	assert.equal(
		await readFile(join(root, 'node_modules/runtime/index.js'), 'utf8'),
		'original dependency\n'
	)
	assert.equal(
		await readFile(join(workspace.compiledDir, 'node_modules/runtime/index.js'), 'utf8'),
		'original dependency\n'
	)
})

test('source snapshots reject symlinks to inputs outside the copied tree', async (t) => {
	const { directory, root, tempRoot } = await fixture(t)
	const outside = join(directory, 'private-input.txt')
	await writeFile(outside, 'outside input must remain untouched\n')
	for (const target of [outside, relative(join(root, 'src'), outside)]) {
		const link = join(root, 'src/escaped-input.txt')
		await symlink(target, link)
		const workspace = await createPackageWorkspace({ root, tempRoot })
		try {
			await assert.rejects(workspace.stageSource(), /symlink/)
			assert.equal(await readFile(outside, 'utf8'), 'outside input must remain untouched\n')
		} finally {
			await workspace.cleanup()
			await rm(link)
		}
	}
})

test('all flavors reuse one compiled dependency tree with separate fresh resources', async (t) => {
	const { root, tempRoot } = await fixture(t)
	const workspace = await createPackageWorkspace({ root, tempRoot })
	t.after(() => workspace.cleanup())
	await workspace.stageSource()
	await put(workspace.compiledDir, 'out/main/index.js', 'fresh compilation\n')
	const identities = new Map()
	for (const name of [
		'src/main/index.ts',
		'out/main/index.js',
		'node_modules/runtime/index.js'
	]) {
		const file = await lstat(join(workspace.compiledDir, name))
		identities.set(name, [file.dev, file.ino])
	}
	const compiledBasis = await snapshot(workspace.compiledDir)
	const resources = new Set()
	const projects = []
	for (const flavor of ['with-plugins-cpu', 'core', 'with-plugins-cuda']) {
		const project = await workspace.stageFlavor(flavor)
		projects.push(project)
		assert.equal(project.root, workspace.compiledDir)
		assert.equal(resources.has(project.resourcesDir), false)
		resources.add(project.resourcesDir)
		assert.equal(inside(workspace.directory, project.root), true)
		assert.equal(inside(workspace.directory, project.resourcesDir), true)
		assert.equal(inside(project.root, project.resourcesDir), false)
		assert.equal(inside(project.root, project.distDir), false)
		assert.deepEqual(await readdir(project.resourcesDir), [])
		assert.equal((await lstat(project.resourcesDir)).isSymbolicLink(), false)
		for (const [name, identity] of identities) {
			const file = await lstat(join(project.root, name))
			assert.equal(file.isSymbolicLink(), false)
			assert.deepEqual([file.dev, file.ino], identity)
		}
		await put(project.resourcesDir, 'preinstalled-plugins/previous-flavor/index.html', flavor)
		await put(project.distDir, 'unpacked.txt', flavor)
	}
	const dependencyTrees = Object.entries(await snapshot(workspace.directory))
		.filter(([name, entry]) => entry.directory && name.split(/[\\/]/).at(-1) === 'node_modules')
		.map(([name]) => name)
	assert.deepEqual(dependencyTrees, [
		relative(workspace.directory, join(workspace.compiledDir, 'node_modules'))
	])
	assert.deepEqual(await snapshot(workspace.compiledDir), compiledBasis)
	for (const [index, project] of projects.entries()) {
		const keepOutput = index === 0
		await project.cleanup({ keepOutput })
		await assert.rejects(lstat(project.resourcesDir), { code: 'ENOENT' })
		if (keepOutput)
			assert.equal(
				await readFile(join(project.distDir, 'unpacked.txt'), 'utf8'),
				'with-plugins-cpu'
			)
		else await assert.rejects(lstat(project.distDir), { code: 'ENOENT' })
		assert.deepEqual(await snapshot(workspace.compiledDir), compiledBasis)
	}
})

for (const setting of ['npmRebuild', 'nodeGypRebuild']) {
	test(`configuration rejects ${setting} on the shared compiled basis`, async (t) => {
		const { root } = await fixture(t)
		const configurationPath = join(root, 'electron-builder.yml')
		const configuration = await readFile(configurationPath, 'utf8')
		const key = new RegExp(`^${setting}:.*$`, 'm')
		const changed = key.test(configuration)
			? configuration.replace(key, `${setting}: true`)
			: `${configuration}\n${setting}: true\n`
		await writeFile(configurationPath, changed)
		await assert.rejects(validatePackageConfiguration(root), /rebuild|dependency snapshot/i)
	})
}

test('plugin input cache is copied to an owned cache without modifying the seed', async (t) => {
	const { directory, root, tempRoot } = await fixture(t)
	const seed = join(directory, 'external-plugin-inputs')
	await put(seed, 'plugin.zip', 'read-only input archive\n')
	const before = await snapshot(seed)
	const workspace = await createPackageWorkspace({ root, tempRoot, pluginCacheDir: seed })
	t.after(() => workspace.cleanup())
	await workspace.stageSource()
	assert.notEqual(workspace.cacheDir, seed)
	assert.equal(inside(workspace.directory, workspace.cacheDir), true)
	assert.equal(
		await readFile(join(workspace.cacheDir, 'plugin.zip'), 'utf8'),
		'read-only input archive\n'
	)
	await rm(join(workspace.cacheDir, 'plugin.zip'))
	await put(workspace.cacheDir, 'plugin.zip', 'replacement downloaded into owned cache\n')
	assert.deepEqual(await snapshot(seed), before)
})

test('workspace creation rejects destructive output overlap before changing existing files', async (t) => {
	const { directory, root, tempRoot } = await fixture(t)
	const seed = join(tempRoot, 'cache-parent', 'seed')
	await put(seed, 'plugin.zip', 'external input\n')
	await put(tempRoot, 'unrelated.txt', 'unrelated runner data\n')
	const cases = [
		{ artifactDir: root },
		{ artifactDir: join(root, 'out') },
		{ artifactDir: directory },
		{ artifactDir: tempRoot },
		{ tempRoot: join(root, 'temporary') },
		{ pluginCacheDir: join(root, '.package-plugin-artifacts') },
		{ artifactDir: seed, pluginCacheDir: seed },
		{ artifactDir: join(seed, 'new-output'), pluginCacheDir: seed },
		{ artifactDir: dirname(seed), pluginCacheDir: seed }
	]
	for (const override of cases) {
		const before = await snapshot(directory)
		await assert.rejects(createPackageWorkspace({ root, tempRoot, ...override }))
		assert.deepEqual(await snapshot(directory), before, JSON.stringify(override))
	}
})

test('path guards resolve symlink ancestors before accepting workspace destinations', async (t) => {
	const { directory, root, tempRoot } = await fixture(t)
	const alias = join(tempRoot, 'checkout-alias')
	await symlink(root, alias, process.platform === 'win32' ? 'junction' : 'dir')
	for (const override of [
		{ artifactDir: alias },
		{ artifactDir: join(alias, 'not-created', 'output') },
		{ tempRoot: join(alias, 'not-created') },
		{ root: alias, artifactDir: join(root, 'out') }
	]) {
		const before = await snapshot(directory)
		await assert.rejects(createPackageWorkspace({ root, tempRoot, ...override }))
		assert.deepEqual(await snapshot(directory), before)
	}
})

for (const failPackaging of [false, true]) {
	test(`release orchestration leaves the checkout unchanged on ${failPackaging ? 'failure' : 'success'}`, async (t) => {
		const { root, tempRoot, lock, packageJson } = await fixture(t, { git: true })
		const before = await snapshot(root)
		const artifactDir = join(tempRoot, 'release-results')
		let compilations = 0
		let compiledRoot
		let compiledBasis
		let preparedLock
		const packaged = []
		const packageRoots = []
		const resourceRoots = []
		const runCommand = async (_command, args, { cwd, env }) => {
			assert.equal(inside(root, cwd), false)
			assert.equal(inside(tempRoot, cwd), true)
			if (args.includes('run') && args.includes('build')) {
				compilations++
				await assert.rejects(lstat(join(cwd, 'out')), { code: 'ENOENT' })
				await put(cwd, 'out/main/index.js', 'fresh compilation\n')
				compiledRoot = cwd
				compiledBasis = await snapshot(cwd)
				return
			}
			assert.equal(args.includes('electron-builder'), true, args.join(' '))
			assert.equal(cwd, compiledRoot)
			packageRoots.push(cwd)
			assert.deepEqual(await snapshot(cwd), compiledBasis)
			assert.equal(
				await readFile(join(cwd, 'out/main/index.js'), 'utf8'),
				'fresh compilation\n'
			)
			assert.equal(
				await readFile(join(cwd, 'node_modules/runtime/index.js'), 'utf8'),
				'original dependency\n'
			)
			const outputArg = args.find((arg) => arg.startsWith('-c.directories.output='))
			assert.ok(outputArg)
			const output = outputArg.slice('-c.directories.output='.length)
			assert.equal(inside(cwd, output), false)
			assert.equal(isAbsolute(env.OUROBOROS_PACKAGE_RESOURCES_DIR), false)
			const resourcesDir = resolve(cwd, env.OUROBOROS_PACKAGE_RESOURCES_DIR)
			assert.equal(inside(tempRoot, resourcesDir), true)
			assert.equal(inside(cwd, resourcesDir), false)
			assert.equal(resourceRoots.includes(resourcesDir), false)
			for (const previous of resourceRoots) {
				await assert.rejects(lstat(previous), { code: 'ENOENT' })
			}
			resourceRoots.push(resourcesDir)
			const metadata = JSON.parse(
				await readFile(join(resourcesDir, 'package-flavor.json'), 'utf8')
			)
			assert.equal(metadata.flavor, env.OUROBOROS_PACKAGE_FLAVOR)
			packaged.push(metadata.flavor)
			await put(resourcesDir, 'packager-note.txt', `processed ${metadata.flavor}\n`)
			if (failPackaging) throw new Error('fixture packaging failure')
			await put(
				output,
				`${packageJson.name}-${packageJson.version}.AppImage`,
				metadata.flavor
			)
		}
		const prepareFlavor = async ({ project, releaseLock, environment }) => {
			assert.deepEqual(releaseLock.lock, lock)
			if (preparedLock) assert.strictEqual(releaseLock.lock, preparedLock)
			else preparedLock = releaseLock.lock
			assert.equal(project.root, compiledRoot)
			assert.equal(
				resolve(project.root, environment.OUROBOROS_PACKAGE_RESOURCES_DIR),
				project.resourcesDir
			)
			const server = JSON.parse(
				await readFile(join(project.resourcesDir, 'server/server-image.json'), 'utf8')
			)
			assert.equal(server.digest, lock.serverImage.digest)
			await assert.rejects(lstat(join(project.resourcesDir, 'preinstalled-plugins/stale')), {
				code: 'ENOENT'
			})
			await put(project.resourcesDir, 'package-flavor.json', {
				flavor: environment.OUROBOROS_PACKAGE_FLAVOR
			})
		}
		const promise = buildReleaseArtifacts({
			root,
			platform: 'linux',
			tempRoot,
			artifactDir,
			environment: {
				...process.env,
				GITHUB_SHA: undefined,
				GITHUB_OUTPUT: undefined,
				OUROBOROS_PACKAGE_FLAVOR: undefined
			},
			runCommand,
			prepareFlavor
		})
		if (failPackaging) await assert.rejects(promise, /fixture packaging failure/)
		else {
			const manifest = await promise
			const inputs = await getReleaseInputs(root)
			assert.equal(manifest.schemaVersion, 2)
			assert.equal(manifest.inputFingerprint, inputs.inputFingerprint)
			assert.equal(manifest.sourceCommit, inputs.sourceCommit)
			assert.strictEqual(manifest.releaseLock, preparedLock)
			assert.deepEqual(packaged, lock.packageFlavors)
			assert.equal(manifest.artifacts.length, lock.packageFlavors.length * 2)
			assert.equal(manifest.metrics.operations.electronCompilations, 1)
			for (const flavor of lock.packageFlavors) {
				assert.equal(
					await readFile(
						join(
							artifactDir,
							`${packageJson.name}-${packageJson.version}-${flavor}.AppImage`
						),
						'utf8'
					),
					flavor
				)
			}
			assert.equal(
				JSON.parse(await readFile(join(artifactDir, 'release-build-linux.json'), 'utf8'))
					.sourceTree,
				manifest.sourceTree
			)
		}
		assert.equal(compilations, 1)
		assert.deepEqual([...new Set(packageRoots)], [compiledRoot])
		for (const path of [compiledRoot, ...resourceRoots]) {
			await assert.rejects(lstat(path), { code: 'ENOENT' })
		}
		assert.deepEqual(await snapshot(root), before)
	})
}

test('release orchestration rejects an uncommitted release-lock override', async () => {
	await assert.rejects(
		buildReleaseArtifacts({
			platform: 'linux',
			environment: { OUROBOROS_RELEASE_LOCK: 'release/alternate-lock.json' }
		}),
		/require the committed release lock/
	)
})

test('release orchestration rejects provenance from a different checked out commit', async (t) => {
	const { root } = await fixture(t, { git: true })
	await assert.rejects(
		buildReleaseArtifacts({
			root,
			platform: 'linux',
			environment: { GITHUB_SHA: '0'.repeat(40) }
		}),
		/does not match checked out release commit/
	)
})

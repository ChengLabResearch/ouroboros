import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import {
	lstat,
	mkdir,
	mkdtemp,
	readFile,
	readdir,
	readlink,
	rm,
	symlink,
	writeFile
} from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, relative } from 'node:path'
import { promisify } from 'node:util'
import test from 'node:test'
import { createPackageWorkspace } from '../lib/package-workspace.mjs'
import { buildReleaseArtifacts } from '../build-release-artifacts.mjs'

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
			packages: {}
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

test('source staging copies only build inputs and never shares mutable dependencies', async (t) => {
	const { root, tempRoot } = await fixture(t)
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
	await put(workspace.compiledDir, 'node_modules/runtime/index.js', 'changed staged dependency\n')
	await put(workspace.compiledDir, 'src/main/index.ts', 'changed staged source\n')
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

test('relative dependency symlinks remain inside independent snapshots', async (t) => {
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
	await writeFile(
		join(project.root, 'node_modules/.bin/runtime'),
		'modified through staged symlink\n'
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

test('each package flavor starts from the same clean compiled basis', async (t) => {
	const { root, tempRoot } = await fixture(t)
	const workspace = await createPackageWorkspace({ root, tempRoot })
	t.after(() => workspace.cleanup())
	await workspace.stageSource()
	await put(workspace.compiledDir, 'out/main/index.js', 'fresh compilation\n')
	const roots = new Set()
	for (const flavor of ['with-plugins-cpu', 'core', 'with-plugins-cuda']) {
		const project = await workspace.stageFlavor(flavor)
		assert.equal(roots.has(project.root), false)
		roots.add(project.root)
		assert.equal(inside(workspace.directory, project.root), true)
		assert.equal(inside(project.root, project.distDir), false)
		assert.equal(
			await readFile(join(project.root, 'out/main/index.js'), 'utf8'),
			'fresh compilation\n'
		)
		assert.equal(
			await readFile(join(project.root, 'node_modules/runtime/index.js'), 'utf8'),
			'original dependency\n'
		)
		await assert.rejects(
			lstat(join(project.root, 'extra-resources/preinstalled-plugins/previous-flavor')),
			{ code: 'ENOENT' }
		)
		await put(project.root, 'out/main/index.js', `modified by ${flavor}\n`)
		await put(project.root, 'node_modules/runtime/index.js', `rebuilt by ${flavor}\n`)
		await put(
			project.root,
			'extra-resources/preinstalled-plugins/previous-flavor/index.html',
			flavor
		)
	}
	assert.equal(
		await readFile(join(workspace.compiledDir, 'out/main/index.js'), 'utf8'),
		'fresh compilation\n'
	)
})

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
		const packaged = []
		const runCommand = async (_command, args, { cwd, env }) => {
			assert.equal(inside(root, cwd), false)
			assert.equal(inside(tempRoot, cwd), true)
			if (args.includes('run') && args.includes('build')) {
				compilations++
				await assert.rejects(lstat(join(cwd, 'out')), { code: 'ENOENT' })
				await put(cwd, 'out/main/index.js', 'fresh compilation\n')
				return
			}
			assert.equal(args.includes('electron-builder'), true, args.join(' '))
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
			const metadata = JSON.parse(
				await readFile(join(cwd, 'extra-resources/package-flavor.json'), 'utf8')
			)
			assert.equal(metadata.flavor, env.OUROBOROS_PACKAGE_FLAVOR)
			packaged.push(metadata.flavor)
			await put(cwd, 'out/main/index.js', 'changed by packager\n')
			await put(cwd, 'node_modules/runtime/index.js', 'changed by packager\n')
			if (failPackaging) throw new Error('fixture packaging failure')
			await put(
				output,
				`${packageJson.name}-${packageJson.version}.AppImage`,
				metadata.flavor
			)
		}
		const prepareFlavor = async ({ project, releaseLock, environment }) => {
			assert.deepEqual(releaseLock.lock, lock)
			const server = JSON.parse(
				await readFile(
					join(project.root, 'extra-resources/server/server-image.json'),
					'utf8'
				)
			)
			assert.equal(server.digest, lock.serverImage.digest)
			await assert.rejects(
				lstat(join(project.root, 'extra-resources/preinstalled-plugins/stale')),
				{ code: 'ENOENT' }
			)
			await put(project.root, 'extra-resources/package-flavor.json', {
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
		assert.deepEqual(await snapshot(root), before)
	})
}

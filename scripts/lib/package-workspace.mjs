import { constants } from 'node:fs'
import { cp, lstat, mkdir, mkdtemp, readdir, readlink, realpath, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'
import { SUPPORTED_FLAVORS } from './release-lock.mjs'

// Compilation inputs, distinct from Electron Builder's runtime-file allowlist.
export const BUILD_INPUTS = [
	'package.json',
	'package-lock.json',
	'electron-builder.yml',
	'electron.vite.config.ts',
	'tsconfig.json',
	'tsconfig.node.json',
	'tsconfig.web.json',
	'src',
	'resources',
	'build'
]

export async function canonicalPath(path) {
	try {
		return await realpath(path)
	} catch (error) {
		if (error.code !== 'ENOENT') throw error
		const parent = dirname(path)
		if (parent === path) throw error
		return join(await canonicalPath(parent), basename(path))
	}
}

function contains(parent, child) {
	const path = relative(parent, child)
	return path === '' || (!isAbsolute(path) && path !== '..' && !path.startsWith(`..${sep}`))
}

function assertSeparate(left, right) {
	if (contains(left, right) || contains(right, left)) {
		throw new Error(`Packaging paths must not overlap: ${left} and ${right}`)
	}
}

async function copyInput(source, destination) {
	await cp(source, destination, {
		recursive: true,
		mode: constants.COPYFILE_FICLONE,
		verbatimSymlinks: true,
		filter: async (path) => {
			if ((await lstat(path)).isSymbolicLink()) {
				const target = await readlink(path)
				if (isAbsolute(target) || !contains(source, await realpath(path))) {
					throw new Error(`Packaging input symlink escapes its snapshot: ${path}`)
				}
			}
			return true
		}
	})
}

export async function createPackageWorkspace({
	root = process.cwd(),
	tempRoot = process.env.RUNNER_TEMP ?? tmpdir(),
	artifactDir = process.env.OUROBOROS_RELEASE_ARTIFACT_DIR,
	pluginCacheDir = process.env.OUROBOROS_PLUGIN_ARTIFACT_DIR
} = {}) {
	root = await canonicalPath(resolve(root))
	tempRoot = await canonicalPath(resolve(tempRoot))
	if (contains(root, tempRoot))
		throw new Error('Packaging temporary root must be outside the checkout')
	const workspaceRoot = process.env.GITHUB_WORKSPACE
	const checkoutRoots = [root]
	if (workspaceRoot) checkoutRoots.push(await canonicalPath(resolve(workspaceRoot)))
	for (const checkout of checkoutRoots) {
		if (contains(checkout, tempRoot))
			throw new Error('RUNNER_TEMP must be outside GITHUB_WORKSPACE')
	}
	if (artifactDir) {
		if (!isAbsolute(artifactDir))
			throw new Error('OUROBOROS_RELEASE_ARTIFACT_DIR must be absolute')
		artifactDir = await canonicalPath(artifactDir)
		if (artifactDir === tempRoot || !contains(tempRoot, artifactDir)) {
			throw new Error(
				'Release artifacts must be beneath RUNNER_TEMP (or the local OS temporary directory)'
			)
		}
		for (const checkout of checkoutRoots) assertSeparate(checkout, artifactDir)
		try {
			if ((await readdir(artifactDir)).length)
				throw new Error(`Release artifact directory must be empty: ${artifactDir}`)
		} catch (error) {
			if (error.code !== 'ENOENT') throw error
		}
	}
	if (pluginCacheDir) {
		pluginCacheDir = await canonicalPath(resolve(pluginCacheDir))
		for (const checkout of checkoutRoots) assertSeparate(checkout, pluginCacheDir)
		if (artifactDir) assertSeparate(artifactDir, pluginCacheDir)
		// An external cache is a read-only seed, never a directory we clear or repair.
		if (contains(pluginCacheDir, tempRoot))
			throw new Error('Plugin cache must not contain the packaging temporary root')
	}

	await mkdir(tempRoot, { recursive: true })
	const directory = await mkdtemp(join(tempRoot, 'ouroboros-package-'))
	const compiledDir = join(directory, 'compiled')
	const cacheDir = join(directory, 'plugin-cache')
	const finalDir = artifactDir ?? join(directory, 'artifacts')
	if (artifactDir) assertSeparate(directory, artifactDir)
	if (pluginCacheDir) assertSeparate(directory, pluginCacheDir)
	return Object.freeze({
		root,
		directory,
		compiledDir,
		artifactDir: finalDir,
		cacheDir,
		builderCacheDir: join(directory, 'builder-cache'),
		electronCacheDir: join(directory, 'electron-cache'),
		async stageSource({ includeDependencies = true } = {}) {
			await mkdir(compiledDir)
			for (const input of BUILD_INPUTS)
				await copyInput(join(root, input), join(compiledDir, input))
			try {
				await copyInput(join(root, '.npmrc'), join(compiledDir, '.npmrc'))
			} catch (error) {
				if (error.code !== 'ENOENT') throw error
			}
			if (includeDependencies)
				await copyInput(join(root, 'node_modules'), join(compiledDir, 'node_modules'))
			if (pluginCacheDir) await copyInput(pluginCacheDir, cacheDir)
			else await mkdir(cacheDir)
		},
		async stageFlavor(flavor) {
			if (!SUPPORTED_FLAVORS.includes(flavor))
				throw new Error(`Unknown package flavor: ${flavor}`)
			const projectRoot = join(directory, 'inputs', flavor)
			const distDir = join(directory, 'outputs', flavor)
			await mkdir(dirname(projectRoot), { recursive: true })
			await mkdir(projectRoot)
			await copyInput(compiledDir, projectRoot)
			await mkdir(distDir, { recursive: true })
			return Object.freeze({
				root: projectRoot,
				distDir,
				cacheDir,
				async cleanup({ keepOutput = false } = {}) {
					await rm(projectRoot, { recursive: true, force: true })
					if (!keepOutput) await rm(distDir, { recursive: true, force: true })
				}
			})
		},
		async cleanup({ keepArtifacts = false, keepOutputs = false } = {}) {
			if (!keepArtifacts && !keepOutputs) {
				await rm(directory, { recursive: true, force: true })
				return
			}
			for (const entry of await readdir(directory)) {
				if (entry === 'artifacts' && keepArtifacts) continue
				if (entry === 'outputs' && keepOutputs) continue
				await rm(join(directory, entry), { recursive: true, force: true })
			}
		}
	})
}

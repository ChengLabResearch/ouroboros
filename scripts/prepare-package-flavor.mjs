import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { isAbsolute, join } from 'node:path'
import { spawn } from 'node:child_process'
import { loadReleaseLock, pluginLocksForFlavor, sha256File } from './lib/release-lock.mjs'

const root = process.cwd()
const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const { lock, sha256: releaseLockSha256 } = await loadReleaseLock(root)
const flavor = process.env.OUROBOROS_PACKAGE_FLAVOR

if (packageJson.version !== lock.releaseVersion) {
	throw new Error(
		`package.json version ${packageJson.version} does not match release lock version ${lock.releaseVersion}`
	)
}
if (!flavor) {
	throw new Error('OUROBOROS_PACKAGE_FLAVOR must be set explicitly')
}
if (!lock.packageFlavors.includes(flavor)) {
	throw new Error(
		`Package flavor "${flavor}" is not selected by the release lock (${lock.packageFlavors.join(', ')})`
	)
}

const extraResourcesDir = join(root, 'extra-resources')
const preinstalledPluginDir = join(extraResourcesDir, 'preinstalled-plugins')
const artifactDir = resolvePathFromRoot(
	process.env.OUROBOROS_PLUGIN_ARTIFACT_DIR ?? '.package-plugin-artifacts'
)
const plugins = []

await mkdir(extraResourcesDir, { recursive: true })
await rm(preinstalledPluginDir, { recursive: true, force: true })

if (flavor !== 'core') {
	await mkdir(preinstalledPluginDir, { recursive: true })

	for (const pluginLock of pluginLocksForFlavor(lock, flavor)) {
		const label =
			pluginLock.id === 'neuroglancer-plugin'
				? 'Neuroglancer'
				: `Automatic Segmentation (${pluginLock.variant.toUpperCase()})`
		await installPlugin({ pluginLock, label })
	}
}

await writeFile(
	join(extraResourcesDir, 'package-flavor.json'),
	`${JSON.stringify(
		{
			flavor,
			appVersion: packageJson.version,
			releaseLockSha256,
			serverImage: await readServerImageMetadata(),
			plugins,
			commit: process.env.GITHUB_SHA ?? null,
			ref: process.env.GITHUB_REF_NAME ?? null
		},
		null,
		2
	)}\n`
)

async function installPlugin({ pluginLock, label }) {
	const artifactPath = await resolveArtifact(pluginLock)
	const target = join(preinstalledPluginDir, pluginLock.id)

	await rm(target, { recursive: true, force: true })
	await mkdir(target, { recursive: true })
	await extractZip(artifactPath, target)
	await normalizePluginRoot(target)

	const pluginPackage = await validatePluginPackage(target, pluginLock.id)
	const releaseManifest = await readPluginReleaseManifest(target, pluginLock.id)
	validateReleaseManifest(releaseManifest, pluginLock)
	plugins.push({
		id: pluginLock.id,
		name: pluginPackage.pluginName,
		version: pluginPackage.version ?? null,
		packageVersion: pluginPackage.version ?? null,
		releaseVersion: releaseManifest?.version ?? null,
		label,
		repo: pluginLock.repository,
		tag: pluginLock.tag,
		artifact: pluginLock.asset,
		assetSha256: pluginLock.assetSha256,
		sourceCommit: pluginLock.sourceCommit,
		releaseTag: pluginLock.tag,
		releaseArtifact: pluginLock.asset,
		releaseManifest: summarizeReleaseManifest(releaseManifest),
		backendImage: pluginLock.backendImage ?? null,
		backendImageDigest: pluginLock.backendImageDigest ?? null,
		embeddedBackendImageVerified: pluginLock.verifyEmbeddedBackendImage ?? false,
		variant: pluginLock.variant ?? null
	})
}

async function resolveArtifact({ repository, tag, asset, assetSha256 }) {
	await mkdir(artifactDir, { recursive: true })

	const artifactPath = join(artifactDir, asset)
	if (existsSync(artifactPath)) {
		const actualSha256 = await sha256File(artifactPath)
		if (actualSha256 === assetSha256) return artifactPath

		console.warn(
			`Removing stale plugin asset ${asset}: expected ${assetSha256}, found ${actualSha256}`
		)
		await rm(artifactPath, { force: true })
	}

	await run('gh', [
		'release',
		'download',
		tag,
		'--repo',
		repository,
		'--pattern',
		asset,
		'--dir',
		artifactDir,
		'--clobber'
	])

	if (!existsSync(artifactPath)) {
		throw new Error(`Expected release asset was not downloaded: ${repository} ${tag} ${asset}`)
	}

	const actualSha256 = await sha256File(artifactPath)
	if (actualSha256 !== assetSha256) {
		throw new Error(
			`Plugin asset SHA-256 mismatch for ${repository} ${tag} ${asset}: expected ${assetSha256}, found ${actualSha256}`
		)
	}

	return artifactPath
}

async function normalizePluginRoot(target) {
	if (existsSync(join(target, 'package.json'))) return

	const packageRoot = await findPackageRoot(target)
	if (!packageRoot || packageRoot === target) return

	const normalizedTarget = `${target}.normalized`
	await rm(normalizedTarget, { recursive: true, force: true })
	await rename(packageRoot, normalizedTarget)
	await rm(target, { recursive: true, force: true })
	await rename(normalizedTarget, target)
}

async function findPackageRoot(directory, depth = 0) {
	if (existsSync(join(directory, 'package.json'))) return directory
	if (depth >= 2) return null

	const entries = await readdir(directory, { withFileTypes: true })
	for (const entry of entries) {
		if (!entry.isDirectory()) continue

		const found = await findPackageRoot(join(directory, entry.name), depth + 1)
		if (found) return found
	}

	return null
}

async function validatePluginPackage(pluginRoot, expectedId) {
	const packagePath = join(pluginRoot, 'package.json')
	if (!existsSync(packagePath)) {
		throw new Error(`Plugin artifact for ${expectedId} does not contain package.json`)
	}

	const pluginPackage = JSON.parse(await readFile(packagePath, 'utf8'))
	if (pluginPackage.name !== expectedId) {
		throw new Error(
			`Plugin artifact name mismatch: expected ${expectedId}, found ${pluginPackage.name}`
		)
	}

	if (!pluginPackage.index || !existsSync(join(pluginRoot, pluginPackage.index))) {
		throw new Error(`Plugin artifact for ${expectedId} does not contain ${pluginPackage.index}`)
	}

	if (pluginPackage.dockerCompose && !existsSync(join(pluginRoot, pluginPackage.dockerCompose))) {
		throw new Error(
			`Plugin artifact for ${expectedId} does not contain ${pluginPackage.dockerCompose}`
		)
	}

	return pluginPackage
}

async function readPluginReleaseManifest(pluginRoot, expectedId) {
	const manifestPath = join(pluginRoot, 'plugin-release.json')
	if (!existsSync(manifestPath)) {
		throw new Error(`Plugin artifact for ${expectedId} does not contain plugin-release.json`)
	}

	const releaseManifest = JSON.parse(await readFile(manifestPath, 'utf8'))
	if (releaseManifest.name && releaseManifest.name !== expectedId) {
		throw new Error(
			`Plugin release manifest name mismatch: expected ${expectedId}, found ${releaseManifest.name}`
		)
	}

	return releaseManifest
}

function validateReleaseManifest(releaseManifest, pluginLock) {
	const expectedFields = {
		releaseTag: pluginLock.tag,
		artifactName: pluginLock.asset,
		commit: pluginLock.sourceCommit
	}
	if (pluginLock.variant) expectedFields.variant = pluginLock.variant

	for (const [field, expected] of Object.entries(expectedFields)) {
		if (releaseManifest[field] !== expected) {
			throw new Error(
				`Plugin release manifest ${field} mismatch for ${pluginLock.id}: expected ${expected}, found ${releaseManifest[field]}`
			)
		}
	}

	if (
		pluginLock.verifyEmbeddedBackendImage &&
		releaseManifest.backendImage !== pluginLock.backendImage
	) {
		throw new Error(
			`Plugin release manifest backendImage mismatch for ${pluginLock.id} (${pluginLock.variant}): expected ${pluginLock.backendImage}, found ${releaseManifest.backendImage}`
		)
	}
	if (
		pluginLock.verifyEmbeddedBackendImage &&
		!releaseManifest.backendImage.endsWith(`@${pluginLock.backendImageDigest}`)
	) {
		throw new Error(
			`Plugin release manifest backendImage does not contain locked digest ${pluginLock.backendImageDigest}`
		)
	}
}

function summarizeReleaseManifest(releaseManifest) {
	return {
		version: releaseManifest.version ?? null,
		packageVersion: releaseManifest.packageVersion ?? null,
		releaseTag: releaseManifest.releaseTag ?? null,
		artifactName: releaseManifest.artifactName ?? null,
		variant: releaseManifest.variant ?? null,
		backendImage: releaseManifest.backendImage ?? null,
		backendImageRepository: releaseManifest.backendImageRepository ?? null,
		backendImageTag: releaseManifest.backendImageTag ?? null,
		cuda: releaseManifest.cuda ?? null,
		commit: releaseManifest.commit ?? null,
		ref: releaseManifest.ref ?? null
	}
}

async function readServerImageMetadata() {
	const serverImagePath = join(extraResourcesDir, 'server', 'server-image.json')
	if (!existsSync(serverImagePath)) return null
	return JSON.parse(await readFile(serverImagePath, 'utf8'))
}

function resolvePathFromRoot(path) {
	return isAbsolute(path) ? path : join(root, path)
}

async function extractZip(artifactPath, target) {
	if (process.platform === 'win32') {
		await run('tar', ['-xf', artifactPath, '-C', target])
		return
	}

	await run('unzip', ['-q', artifactPath, '-d', target])
}

async function run(command, args) {
	await new Promise((resolve, reject) => {
		const child = spawn(command, args, { stdio: 'inherit' })
		child.on('error', reject)
		child.on('close', (code) => {
			if (code === 0) {
				resolve()
			} else {
				reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`))
			}
		})
	})
}

import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { isAbsolute, join } from 'node:path'
import { spawn } from 'node:child_process'

const supportedFlavors = new Set(['core', 'with-plugins-cpu', 'with-plugins-cuda'])
const productionPluginPins = {
	neuroglancer: {
		tag: 'v1.0.0',
		artifact: 'release.zip'
	},
	autoseg: {
		tag: 'v0.4.0-beta',
		cpuArtifact: 'auto-segmentation-v0.4.0-beta-cpu.zip',
		cudaArtifact: 'auto-segmentation-v0.4.0-beta-cuda.zip'
	}
}

const root = process.cwd()
const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const flavor = process.env.OUROBOROS_PACKAGE_FLAVOR ?? 'core'

if (!supportedFlavors.has(flavor)) {
	throw new Error(
		`Unsupported OUROBOROS_PACKAGE_FLAVOR "${flavor}". Expected one of: ${[
			...supportedFlavors
		].join(', ')}`
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

	const neuroglancerTag =
		process.env.OUROBOROS_NEUROGLANCER_PLUGIN_TAG || productionPluginPins.neuroglancer.tag
	await installPlugin({
		id: 'neuroglancer-plugin',
		label: 'Neuroglancer',
		repo: 'ChengLabResearch/neuroglancer-plugin',
		tag: neuroglancerTag,
		artifact:
			process.env.OUROBOROS_NEUROGLANCER_PLUGIN_ARTIFACT ||
			neuroglancerArtifactForTag(neuroglancerTag)
	})

	const autosegTag = process.env.OUROBOROS_AUTOSEG_PLUGIN_TAG || productionPluginPins.autoseg.tag
	const autosegVariant = flavor === 'with-plugins-cuda' ? 'cuda' : 'cpu'
	await installPlugin({
		id: 'auto-segmentation',
		label: `Automatic Segmentation (${autosegVariant.toUpperCase()})`,
		repo: 'ChengLabResearch/ouroboros_autoseg_plugin',
		tag: autosegTag,
		artifact:
			process.env[`OUROBOROS_AUTOSEG_${autosegVariant.toUpperCase()}_PLUGIN_ARTIFACT`] ||
			autosegArtifactForTag(autosegTag, autosegVariant),
		variant: autosegVariant
	})
}

await writeFile(
	join(extraResourcesDir, 'package-flavor.json'),
	`${JSON.stringify(
		{
			flavor,
			appVersion: packageJson.version,
			serverImage: await readServerImageMetadata(),
			plugins,
			commit: process.env.GITHUB_SHA ?? null,
			ref: process.env.GITHUB_REF_NAME ?? null
		},
		null,
		2
	)}\n`
)

async function installPlugin({ id, label, repo, tag, artifact, variant = null }) {
	const artifactPath = await resolveArtifact({ repo, tag, artifact })
	const target = join(preinstalledPluginDir, id)

	await rm(target, { recursive: true, force: true })
	await mkdir(target, { recursive: true })
	await extractZip(artifactPath, target)
	await normalizePluginRoot(target)

	const pluginPackage = await validatePluginPackage(target, id)
	const releaseManifest = await readPluginReleaseManifest(target, id)
	plugins.push({
		id,
		name: pluginPackage.pluginName,
		version: pluginPackage.version ?? null,
		packageVersion: pluginPackage.version ?? null,
		releaseVersion: releaseManifest?.version ?? null,
		label,
		repo,
		tag,
		artifact,
		releaseTag: tag,
		releaseArtifact: artifact,
		releaseManifest: summarizeReleaseManifest(releaseManifest),
		variant
	})
}

async function resolveArtifact({ repo, tag, artifact }) {
	await mkdir(artifactDir, { recursive: true })

	const artifactPath = join(artifactDir, artifact)
	if (existsSync(artifactPath)) return artifactPath

	await run('gh', [
		'release',
		'download',
		tag,
		'--repo',
		repo,
		'--pattern',
		artifact,
		'--dir',
		artifactDir,
		'--clobber'
	])

	if (!existsSync(artifactPath)) {
		throw new Error(`Expected release asset was not downloaded: ${repo} ${tag} ${artifact}`)
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

	if (
		pluginPackage.dockerCompose &&
		!existsSync(join(pluginRoot, pluginPackage.dockerCompose))
	) {
		throw new Error(
			`Plugin artifact for ${expectedId} does not contain ${pluginPackage.dockerCompose}`
		)
	}

	return pluginPackage
}

async function readPluginReleaseManifest(pluginRoot, expectedId) {
	const manifestPath = join(pluginRoot, 'plugin-release.json')
	if (!existsSync(manifestPath)) return null

	const releaseManifest = JSON.parse(await readFile(manifestPath, 'utf8'))
	if (releaseManifest.name && releaseManifest.name !== expectedId) {
		throw new Error(
			`Plugin release manifest name mismatch: expected ${expectedId}, found ${releaseManifest.name}`
		)
	}

	return releaseManifest
}

function summarizeReleaseManifest(releaseManifest) {
	if (!releaseManifest) return null

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

function neuroglancerArtifactForTag(tag) {
	if (tag === productionPluginPins.neuroglancer.tag) {
		return productionPluginPins.neuroglancer.artifact
	}

	return `neuroglancer-plugin-${tag}.zip`
}

function autosegArtifactForTag(tag, variant) {
	if (tag === productionPluginPins.autoseg.tag) {
		return variant === 'cuda'
			? productionPluginPins.autoseg.cudaArtifact
			: productionPluginPins.autoseg.cpuArtifact
	}

	return `auto-segmentation-${tag}-${variant}.zip`
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

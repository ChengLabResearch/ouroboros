import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'
import yaml from 'js-yaml'
import { gitOutput } from './lib/git.mjs'
import { createPackageWorkspace } from './lib/package-workspace.mjs'
import { loadReleaseLock, validateReleaseVersions } from './lib/release-lock.mjs'

export async function validatePackageConfiguration(root) {
	const configuration = yaml.load(await readFile(join(root, 'electron-builder.yml'), 'utf8'))
	validateAllowlist(configuration.files, 'files')
	assert.equal(
		configuration.extraResources?.length,
		1,
		'Flavor resources must use one external source'
	)
	const resources = configuration.extraResources[0]
	assert.equal(
		resources.from,
		'${env.OUROBOROS_PACKAGE_RESOURCES_DIR}',
		'Flavor resources must come from the workspace resource directory'
	)
	assert.equal(resources.to, 'extra-resources', 'Packaged resource paths must stay unchanged')
	validateAllowlist(resources.filter, 'extraResources.filter')
	assert.equal(
		configuration.npmRebuild,
		false,
		'Packaging uses the installed dependency snapshot without rebuilding it'
	)
	assert.notEqual(
		configuration.nodeGypRebuild,
		true,
		'nodeGypRebuild must not mutate the shared dependency snapshot'
	)
	return configuration
}

function validateAllowlist(patterns, name) {
	assert.ok(Array.isArray(patterns) && patterns.length, `${name} must be an explicit allowlist`)
	for (const pattern of patterns) {
		assert.ok(
			typeof pattern === 'string' &&
				/^[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*(?:\/|$)/.test(pattern) &&
				!pattern.split('/').includes('..'),
			`${name} must name explicit relative input files or directories: ${pattern}`
		)
	}
}

export async function validateServerReleaseManifest(root, lock, manifestPath) {
	assert.ok(
		typeof manifestPath === 'string' && manifestPath.length,
		'--server-manifest requires a manifest file path'
	)
	const manifest = JSON.parse(await readFile(resolve(root, manifestPath), 'utf8'))
	assert.equal(manifest.schemaVersion, 1, 'Unsupported server image manifest schemaVersion')
	for (const field of ['repository', 'digest', 'sourceCommit']) {
		assert.equal(
			manifest[field],
			lock.serverImage[field],
			`Server manifest ${field} does not match release lock; repin serverImage from the certified manifest`
		)
	}
	assert.equal(
		manifest.image,
		`${lock.serverImage.repository}@${lock.serverImage.digest}`,
		'Server manifest image must match its certified repository and digest'
	)
	assert.equal(
		manifest.tag,
		`sha-${lock.serverImage.sourceCommit}`,
		'Server manifest tag must identify its certified source commit'
	)
	let pyproject
	try {
		pyproject = await gitOutput(root, [
			'show',
			`${lock.serverImage.sourceCommit}:python/pyproject.toml`
		])
	} catch (cause) {
		throw new Error(
			`Cannot inspect pinned server wheel version; fetch server source commit ${lock.serverImage.sourceCommit} and rerun preflight`,
			{ cause }
		)
	}
	const poetry = pyproject.split(/^\[tool\.poetry\][ \t]*(?:#.*)?\r?$/m)[1]?.split(/^\[/m)[0]
	const wheelVersion = poetry?.match(
		/^version[ \t]*=[ \t]*(["'])([^"'\r\n]+)\1[ \t]*(?:#.*)?$/m
	)?.[2]
	assert.ok(wheelVersion, 'Pinned server source must declare its wheel version in [tool.poetry]')
	return wheelVersion
}

export async function preflightRelease({
	root = process.cwd(),
	serverManifest,
	...workspaceOptions
} = {}) {
	const { lock } = await loadReleaseLock(root)
	const [packageJson, packageLock] = await Promise.all(
		['package.json', 'package-lock.json'].map(async (name) =>
			JSON.parse(await readFile(join(root, name), 'utf8'))
		)
	)
	validateReleaseVersions(packageJson, packageLock, lock)
	if (serverManifest !== undefined) {
		const wheelVersion = await validateServerReleaseManifest(root, lock, serverManifest)
		console.log(
			`Server manifest matches release lock: ${lock.serverImage.repository}@${lock.serverImage.digest}; source ${lock.serverImage.sourceCommit}; wheel ${wheelVersion} (independent of application ${lock.releaseVersion}).`
		)
	}
	await validatePackageConfiguration(root)
	const workspace = await createPackageWorkspace({ root, ...workspaceOptions })
	try {
		await workspace.stageSource({ includeDependencies: false })
		for (const flavor of lock.packageFlavors) await workspace.stageFlavor(flavor)
	} finally {
		await workspace.cleanup()
	}
	console.log(
		'Release preflight passed: locked inputs, explicit file allowlists and isolated flavor staging; no installers built.'
	)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	const { values } = parseArgs({ options: { 'server-manifest': { type: 'string' } } })
	await preflightRelease({ serverManifest: values['server-manifest'] })
}

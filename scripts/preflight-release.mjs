import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import yaml from 'js-yaml'
import { createPackageWorkspace } from './lib/package-workspace.mjs'
import { loadReleaseLock } from './lib/release-lock.mjs'

export async function validatePackageConfiguration(root) {
	const configuration = yaml.load(await readFile(join(root, 'electron-builder.yml'), 'utf8'))
	for (const key of ['files', 'extraResources']) {
		const patterns = configuration[key]
		assert.ok(
			Array.isArray(patterns) && patterns.length,
			`${key} must be an explicit allowlist`
		)
		for (const pattern of patterns) {
			assert.ok(
				typeof pattern === 'string' &&
					/^[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)*(?:\/|$)/.test(pattern) &&
					!pattern.split('/').includes('..'),
				`${key} must name explicit relative input files or directories: ${pattern}`
			)
		}
	}
	assert.equal(
		configuration.npmRebuild,
		false,
		'Packaging uses the installed dependency snapshot without rebuilding it'
	)
	return configuration
}

export async function preflightRelease({ root = process.cwd(), ...workspaceOptions } = {}) {
	const { lock } = await loadReleaseLock(root)
	const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
	assert.equal(
		packageJson.version,
		lock.releaseVersion,
		'Package and release lock versions must match'
	)
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
	await preflightRelease()
}

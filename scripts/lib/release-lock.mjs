import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { isAbsolute, join } from 'node:path'

export const RELEASE_LOCK_PATH = 'release/release-lock.json'
export const SUPPORTED_FLAVORS = Object.freeze(['core', 'with-plugins-cpu', 'with-plugins-cuda'])
export const RELEASE_PLATFORMS = Object.freeze(['linux', 'mac', 'win'])

const sha256Pattern = /^[a-f0-9]{64}$/
const digestPattern = /^sha256:[a-f0-9]{64}$/
const commitPattern = /^[a-f0-9]{40}$/

export async function loadReleaseLock(root = process.cwd()) {
	const configuredPath = process.env.OUROBOROS_RELEASE_LOCK ?? RELEASE_LOCK_PATH
	const path = isAbsolute(configuredPath) ? configuredPath : join(root, configuredPath)
	const bytes = await readFile(path)
	const lock = JSON.parse(bytes.toString('utf8'))
	validateReleaseLock(lock)

	return {
		lock,
		path,
		sha256: sha256(bytes)
	}
}

export function validateReleaseLock(lock) {
	if (!lock || typeof lock !== 'object' || Array.isArray(lock)) {
		throw new Error('Release lock must be a JSON object')
	}
	if (lock.schemaVersion !== 1) {
		throw new Error(`Unsupported release lock schemaVersion: ${lock.schemaVersion}`)
	}
	requireString(lock.releaseVersion, 'releaseVersion')

	if (!Array.isArray(lock.packageFlavors) || lock.packageFlavors.length === 0) {
		throw new Error('release lock packageFlavors must be a non-empty array')
	}
	const uniqueFlavors = new Set(lock.packageFlavors)
	if (uniqueFlavors.size !== lock.packageFlavors.length) {
		throw new Error('release lock packageFlavors contains duplicates')
	}
	for (const flavor of lock.packageFlavors) {
		if (!SUPPORTED_FLAVORS.includes(flavor)) {
			throw new Error(
				`Unsupported package flavor "${flavor}". Expected one of: ${SUPPORTED_FLAVORS.join(', ')}`
			)
		}
	}

	validateServerImage(lock.serverImage)
	validatePlugin(lock.plugins?.neuroglancer, 'plugins.neuroglancer')
	validatePlugin(lock.plugins?.autosegCpu, 'plugins.autosegCpu', 'cpu')
	validatePlugin(lock.plugins?.autosegCuda, 'plugins.autosegCuda', 'cuda')
}

export function pluginLocksForFlavor(lock, flavor) {
	if (flavor === 'core') return []
	if (flavor === 'with-plugins-cpu') {
		return [lock.plugins.neuroglancer, lock.plugins.autosegCpu]
	}
	if (flavor === 'with-plugins-cuda') {
		return [lock.plugins.neuroglancer, lock.plugins.autosegCuda]
	}
	throw new Error(`Unsupported package flavor "${flavor}"`)
}

export function releaseFingerprint({ lockSha256, sourceTree, releaseVersion }) {
	return sha256(
		Buffer.from(
			JSON.stringify({
				schemaVersion: 1,
				lockSha256,
				sourceTree,
				releaseVersion
			})
		)
	)
}

export async function sha256File(path) {
	return sha256(await readFile(path))
}

export function sha256(value) {
	return createHash('sha256').update(value).digest('hex')
}

function validateServerImage(serverImage) {
	if (!serverImage || typeof serverImage !== 'object') {
		throw new Error('release lock serverImage must be an object')
	}
	requireString(serverImage.repository, 'serverImage.repository')
	requirePattern(serverImage.digest, digestPattern, 'serverImage.digest')
	requirePattern(serverImage.sourceCommit, commitPattern, 'serverImage.sourceCommit')
}

function validatePlugin(plugin, field, expectedVariant = null) {
	if (!plugin || typeof plugin !== 'object') {
		throw new Error(`release lock ${field} must be an object`)
	}
	for (const key of ['id', 'repository', 'tag', 'asset']) {
		requireString(plugin[key], `${field}.${key}`)
	}
	requirePattern(plugin.assetSha256, sha256Pattern, `${field}.assetSha256`)
	requirePattern(plugin.sourceCommit, commitPattern, `${field}.sourceCommit`)

	if (expectedVariant && plugin.variant !== expectedVariant) {
		throw new Error(`${field}.variant must be "${expectedVariant}"`)
	}
	if (expectedVariant) {
		requireString(plugin.backendImage, `${field}.backendImage`)
		requirePattern(plugin.backendImageDigest, digestPattern, `${field}.backendImageDigest`)
		if (typeof plugin.verifyEmbeddedBackendImage !== 'boolean') {
			throw new Error(`${field}.verifyEmbeddedBackendImage must be a boolean`)
		}
	}
}

function requireString(value, field) {
	if (typeof value !== 'string' || value.length === 0) {
		throw new Error(`release lock ${field} must be a non-empty string`)
	}
}

function requirePattern(value, pattern, field) {
	requireString(value, field)
	if (!pattern.test(value)) {
		throw new Error(`release lock ${field} has an invalid value`)
	}
}

import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { gitOutput } from './git.mjs'
import {
	RELEASE_LOCK_PATH,
	parseReleaseLock,
	sha256,
	validateReleaseVersions
} from './release-lock.mjs'

export const ARTIFACT_INPUTS_PATH = 'release/artifact-inputs.json'

export async function loadArtifactInputDefinition(root = process.cwd(), commit) {
	const content = commit
		? await gitOutput(root, ['show', `${commit}:${ARTIFACT_INPUTS_PATH}`])
		: await readFile(join(root, ARTIFACT_INPUTS_PATH), 'utf8')
	const definition = JSON.parse(content)
	if (definition.schemaVersion !== 1 || !Array.isArray(definition.paths)) {
		throw new Error('Unsupported artifact input definition')
	}
	if (!definition.paths.length || new Set(definition.paths).size !== definition.paths.length) {
		throw new Error('Artifact input paths must be non-empty and unique')
	}
	for (const path of definition.paths) artifactInputPattern(path)
	if (!definition.paths.some((path) => artifactInputPattern(path).test(ARTIFACT_INPUTS_PATH))) {
		throw new Error('Artifact input paths must include their own definition')
	}
	return definition
}

// Use the same limited glob dialect in Git identity and the generated workflow paths:
// literal paths, a single-segment * wildcard, and a trailing /** directory wildcard.
function artifactInputPattern(path) {
	if (
		typeof path !== 'string' ||
		!path.length ||
		path.startsWith('/') ||
		path.split('/').some((part) => !part || part === '.' || part === '..') ||
		/[^a-zA-Z0-9_./*-]/.test(path) ||
		path.replace(/\/\*\*$/, '').includes('**')
	) {
		throw new Error(`Unsupported artifact input path: ${path}`)
	}
	const recursive = path.endsWith('/**')
	const prefix = recursive ? path.slice(0, -3) : path
	const expression = prefix
		.split('*')
		.map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
		.join('[^/]*')
	return new RegExp(`^${expression}${recursive ? '/.*' : ''}$`)
}

export async function getReleaseInputs(root = process.cwd(), commit = 'HEAD') {
	const sourceCommit = await gitOutput(root, ['rev-parse', '--verify', `${commit}^{commit}`])
	const [definition, packageContent, packageLockContent, lockContent, sourceTree, treeContent] =
		await Promise.all([
			loadArtifactInputDefinition(root, sourceCommit),
			gitOutput(root, ['show', `${sourceCommit}:package.json`]),
			gitOutput(root, ['show', `${sourceCommit}:package-lock.json`]),
			gitOutput(root, ['show', `${sourceCommit}:${RELEASE_LOCK_PATH}`]),
			gitOutput(root, ['rev-parse', `${sourceCommit}^{tree}`]),
			gitOutput(root, ['ls-tree', '-r', '-z', sourceCommit])
		])
	const packageJson = JSON.parse(packageContent)
	const { version: releaseVersion } = packageJson
	const { lock, sha256: releaseLockSha256 } = parseReleaseLock(lockContent)
	validateReleaseVersions(packageJson, JSON.parse(packageLockContent), lock)
	const patterns = definition.paths.map(artifactInputPattern)
	const entries = treeContent
		.split('\0')
		.filter(Boolean)
		.map((entry) => {
			const separator = entry.indexOf('\t')
			const [mode, type, object] = entry.slice(0, separator).split(' ')
			return { path: entry.slice(separator + 1), mode, type, object }
		})
		.filter((entry) => patterns.some((pattern) => pattern.test(entry.path)))
		.sort((left, right) => (left.path < right.path ? -1 : left.path > right.path ? 1 : 0))
	const inputFingerprint = sha256(
		JSON.stringify({ schemaVersion: 2, releaseVersion, releaseLockSha256, entries })
	)
	return { inputFingerprint, releaseVersion, releaseLockSha256, lock, sourceCommit, sourceTree }
}

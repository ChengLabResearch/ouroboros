import { existsSync } from 'node:fs'
import { readFile, readdir, stat } from 'node:fs/promises'
import { basename, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { gitOutput } from './lib/git.mjs'
import {
	RELEASE_PLATFORMS,
	loadReleaseLock,
	releaseFingerprint,
	sha256File
} from './lib/release-lock.mjs'

export async function verifyReleaseArtifacts({
	root = process.cwd(),
	directory,
	commit,
	ref,
	platforms = RELEASE_PLATFORMS,
	requireCertifiedCommit = true
}) {
	if (!directory) throw new Error('Release artifact directory is required')
	if (!commit) throw new Error('Release commit is required')

	const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
	const { lock, sha256: releaseLockSha256 } = await loadReleaseLock(root)
	if (packageJson.version !== lock.releaseVersion) {
		throw new Error(
			`package.json version ${packageJson.version} does not match release lock version ${lock.releaseVersion}`
		)
	}
	if (ref && ref !== `v${packageJson.version}`) {
		throw new Error(`Release tag ${ref} does not match package version v${packageJson.version}`)
	}

	const sourceTree = await gitOutput(root, ['rev-parse', `${commit}^{tree}`])
	const inputFingerprint = releaseFingerprint({
		lockSha256: releaseLockSha256,
		sourceTree,
		releaseVersion: packageJson.version
	})
	const artifactDir = resolve(root, directory)
	const listedNames = new Set()
	let producedBytes = 0
	let artifactCount = 0

	for (const platform of platforms) {
		if (!RELEASE_PLATFORMS.includes(platform)) {
			throw new Error(`Unsupported release platform: ${platform}`)
		}
		const manifestName = `release-build-${platform}.json`
		const manifestPath = join(artifactDir, manifestName)
		if (!existsSync(manifestPath)) {
			throw new Error(`Missing release build manifest: ${manifestName}`)
		}
		const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
		validateManifest({
			manifest,
			platform,
			commit,
			sourceTree,
			inputFingerprint,
			releaseLockSha256,
			lock,
			releaseVersion: packageJson.version,
			requireCertifiedCommit
		})

		for (const artifact of manifest.artifacts) {
			if (artifact.name !== basename(artifact.name)) {
				throw new Error(
					`Release manifest contains a non-flat artifact path: ${artifact.name}`
				)
			}
			if (listedNames.has(artifact.name)) {
				throw new Error(`Release artifact is listed more than once: ${artifact.name}`)
			}
			listedNames.add(artifact.name)

			const artifactPath = join(artifactDir, artifact.name)
			if (!existsSync(artifactPath)) {
				throw new Error(`Release artifact is missing: ${artifact.name}`)
			}
			const details = await stat(artifactPath)
			if (details.size !== artifact.bytes) {
				throw new Error(
					`Release artifact size mismatch for ${artifact.name}: expected ${artifact.bytes}, found ${details.size}`
				)
			}
			const actualSha256 = await sha256File(artifactPath)
			if (actualSha256 !== artifact.sha256) {
				throw new Error(
					`Release artifact SHA-256 mismatch for ${artifact.name}: expected ${artifact.sha256}, found ${actualSha256}`
				)
			}
			producedBytes += details.size
			artifactCount += 1
		}
	}

	const directoryEntries = await readdir(artifactDir, { withFileTypes: true })
	const allowedManifests = new Set(platforms.map((platform) => `release-build-${platform}.json`))
	for (const entry of directoryEntries) {
		if (!entry.isFile()) {
			throw new Error(`Unexpected directory in release artifacts: ${entry.name}`)
		}
		if (!listedNames.has(entry.name) && !allowedManifests.has(entry.name)) {
			throw new Error(`Unlisted file in release artifacts: ${entry.name}`)
		}
	}

	const summary = {
		releaseVersion: packageJson.version,
		commit,
		sourceTree,
		releaseLockSha256,
		inputFingerprint,
		platforms,
		flavors: lock.packageFlavors,
		artifactCount,
		producedBytes
	}
	console.log(JSON.stringify(summary, null, 2))
	return summary
}

function validateManifest({
	manifest,
	platform,
	commit,
	sourceTree,
	inputFingerprint,
	releaseLockSha256,
	lock,
	releaseVersion,
	requireCertifiedCommit
}) {
	const exactFields = {
		schemaVersion: 1,
		releaseVersion,
		platform,
		sourceTree,
		releaseLockSha256,
		inputFingerprint
	}
	if (requireCertifiedCommit) exactFields.certifiedCommit = commit
	for (const [field, expected] of Object.entries(exactFields)) {
		if (manifest[field] !== expected) {
			throw new Error(
				`Release build manifest ${platform} ${field} mismatch: expected ${expected}, found ${manifest[field]}`
			)
		}
	}
	if (typeof manifest.sourceCommit !== 'string' || manifest.sourceCommit.length === 0) {
		throw new Error(`Release build manifest ${platform} has no sourceCommit`)
	}
	validateCertification(manifest, platform)
	if (JSON.stringify(manifest.flavors) !== JSON.stringify(lock.packageFlavors)) {
		throw new Error(`Release build manifest ${platform} package flavors do not match the lock`)
	}
	if (JSON.stringify(manifest.releaseLock) !== JSON.stringify(lock)) {
		throw new Error(
			`Release build manifest ${platform} does not contain the exact release lock`
		)
	}
	if (!Array.isArray(manifest.artifacts) || manifest.artifacts.length === 0) {
		throw new Error(`Release build manifest ${platform} has no artifacts`)
	}

	const operations = manifest.metrics?.operations
	if (operations?.electronCompilations !== 1) {
		throw new Error(
			`Release build manifest ${platform} did not record one Electron compilation`
		)
	}
	if (operations?.packageOperations !== lock.packageFlavors.length) {
		throw new Error(`Release build manifest ${platform} package operation count is incorrect`)
	}
	if (operations?.serverWheelBuilds !== 0 || operations?.serverImageBuilds !== 0) {
		throw new Error(`Release build manifest ${platform} records an unexpected server build`)
	}

	for (const flavor of lock.packageFlavors) {
		const metadataName = `package-metadata-${platform}-${flavor}.json`
		if (!manifest.artifacts.some((artifact) => artifact.name === metadataName)) {
			throw new Error(`Release build manifest ${platform} is missing ${metadataName}`)
		}
		if (
			!manifest.artifacts.some(
				(artifact) => artifact.flavor === flavor && artifact.type === 'installer'
			)
		) {
			throw new Error(`Release build manifest ${platform} has no installer for ${flavor}`)
		}
	}
}

function validateCertification(manifest, platform) {
	if (manifest.certification?.mode === 'built') {
		if (manifest.certifiedCommit !== manifest.sourceCommit) {
			throw new Error(
				`Release build manifest ${platform} built certification does not match its source commit`
			)
		}
		return
	}
	if (manifest.certification?.mode === 'reused-pr-build') {
		if (!manifest.certification.sourceRun) {
			throw new Error(
				`Release build manifest ${platform} reused certification has no source run`
			)
		}
		const operations = manifest.certification.operationsInCertificationRun
		if (
			operations?.electronCompilations !== 0 ||
			operations?.packageOperations !== 0 ||
			operations?.serverWheelBuilds !== 0 ||
			operations?.serverImageBuilds !== 0
		) {
			throw new Error(
				`Release build manifest ${platform} reused certification records build operations`
			)
		}
		return
	}
	throw new Error(`Release build manifest ${platform} has an invalid certification mode`)
}

function parseArguments(args) {
	const values = new Map()
	for (let index = 0; index < args.length; index += 2) {
		values.set(args[index], args[index + 1])
	}
	return {
		directory: values.get('--dir'),
		commit: values.get('--commit'),
		ref: values.get('--ref')
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	await verifyReleaseArtifacts(parseArguments(process.argv.slice(2)))
}

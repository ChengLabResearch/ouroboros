import { existsSync } from 'node:fs'
import { readFile, readdir, stat } from 'node:fs/promises'
import { basename, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { isAncestor } from './lib/git.mjs'
import { getReleaseInputs } from './lib/release-inputs.mjs'
import { RELEASE_PLATFORMS, sha256File } from './lib/release-lock.mjs'

export async function verifyReleaseArtifacts({
	root = process.cwd(),
	directory,
	commit,
	ref,
	platforms = RELEASE_PLATFORMS,
	mainRef = 'origin/main',
	buildCommit,
	workflowRun
}) {
	if (!directory) throw new Error('Release artifact directory is required')
	if (!commit) throw new Error('Release commit is required')

	const target = await getReleaseInputs(root, commit)
	const { lock, releaseVersion, releaseLockSha256, inputFingerprint } = target
	if (ref && ref !== `v${releaseVersion}`) {
		throw new Error(`Release tag ${ref} does not match package version v${releaseVersion}`)
	}
	if (!(await isAncestor(root, target.sourceCommit, mainRef))) {
		throw new Error(`Release commit ${target.sourceCommit} is not on ${mainRef}`)
	}
	let certifiedInputs
	let certifiedRun = workflowRun == null ? null : String(workflowRun)
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
		if (!certifiedInputs) {
			if (!/^[a-f0-9]{40}$/.test(manifest.sourceCommit ?? '')) {
				throw new Error(`Release build manifest ${platform} has no valid sourceCommit`)
			}
			if (buildCommit && manifest.sourceCommit !== buildCommit) {
				throw new Error(
					`Release build manifest ${platform} sourceCommit does not match selected build`
				)
			}
			certifiedInputs = await getReleaseInputs(root, manifest.sourceCommit)
			if (!(await isAncestor(root, certifiedInputs.sourceCommit, target.sourceCommit))) {
				throw new Error(
					`Certified build ${certifiedInputs.sourceCommit} is not an ancestor of release commit ${target.sourceCommit}`
				)
			}
			if (certifiedInputs.inputFingerprint !== inputFingerprint) {
				throw new Error(
					'Certified build inputFingerprint does not match release artifact inputs'
				)
			}
			certifiedRun ??= manifest.certification?.workflowRun
		}
		validateManifest({
			manifest,
			platform,
			sourceCommit: certifiedInputs.sourceCommit,
			sourceTree: certifiedInputs.sourceTree,
			workflowRun: certifiedRun,
			inputFingerprint,
			releaseLockSha256,
			lock,
			releaseVersion
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
		releaseVersion,
		commit: target.sourceCommit,
		buildCommit: certifiedInputs.sourceCommit,
		workflowRun: certifiedRun,
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
	sourceCommit,
	sourceTree,
	workflowRun,
	inputFingerprint,
	releaseLockSha256,
	lock,
	releaseVersion
}) {
	const exactFields = {
		schemaVersion: 2,
		releaseVersion,
		platform,
		sourceCommit,
		certifiedCommit: sourceCommit,
		sourceTree,
		releaseLockSha256,
		inputFingerprint
	}
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
	validateCertification(manifest, platform, workflowRun)
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

function validateCertification(manifest, platform, workflowRun) {
	if (manifest.certification?.mode !== 'built') {
		throw new Error(`Release build manifest ${platform} has an invalid certification mode`)
	}
	if (
		typeof workflowRun !== 'string' ||
		workflowRun.length === 0 ||
		manifest.certification.workflowRun !== workflowRun
	) {
		throw new Error(
			`Release build manifest ${platform} workflowRun does not match selected build`
		)
	}
}

function parseArguments(args) {
	const values = new Map()
	for (let index = 0; index < args.length; index += 2) {
		values.set(args[index], args[index + 1])
	}
	return {
		directory: values.get('--dir'),
		commit: values.get('--commit'),
		ref: values.get('--ref'),
		mainRef: values.get('--main-ref'),
		buildCommit: values.get('--build-commit'),
		workflowRun: values.get('--run')
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	await verifyReleaseArtifacts(parseArguments(process.argv.slice(2)))
}

import { readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { RELEASE_PLATFORMS } from './lib/release-lock.mjs'
import { verifyReleaseArtifacts } from './verify-release-artifacts.mjs'

export async function certifyReleaseArtifacts({
	root = process.cwd(),
	directory,
	platform,
	commit,
	sourceRun
}) {
	if (!RELEASE_PLATFORMS.includes(platform)) {
		throw new Error(`Unsupported release platform: ${platform}`)
	}
	if (!sourceRun) throw new Error('Source workflow run is required for reused artifacts')
	await verifyReleaseArtifacts({
		root,
		directory,
		commit,
		platforms: [platform],
		requireCertifiedCommit: false
	})

	const manifestPath = join(resolve(root, directory), `release-build-${platform}.json`)
	const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
	manifest.certifiedCommit = commit
	manifest.certification = {
		mode: 'reused-pr-build',
		sourceRun,
		certifiedAt: new Date().toISOString(),
		operationsInCertificationRun: {
			electronCompilations: 0,
			packageOperations: 0,
			serverWheelBuilds: 0,
			serverImageBuilds: 0
		}
	}
	await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

	await verifyReleaseArtifacts({
		root,
		directory,
		commit,
		platforms: [platform]
	})
	console.log(`Certified ${platform} artifacts from run ${sourceRun} for ${commit}`)
}

function parseArguments(args) {
	const values = new Map()
	for (let index = 0; index < args.length; index += 2) {
		values.set(args[index], args[index + 1])
	}
	return {
		directory: values.get('--dir'),
		platform: values.get('--platform'),
		commit: values.get('--commit'),
		sourceRun: values.get('--source-run')
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	await certifyReleaseArtifacts(parseArguments(process.argv.slice(2)))
}

import { execFile } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { promisify } from 'node:util'
import { pathToFileURL } from 'node:url'
import { getReleaseInputs } from './lib/release-inputs.mjs'
import { RELEASE_PLATFORMS } from './lib/release-lock.mjs'
import { findReleaseBuild, releaseArguments } from './select-release-build.mjs'
import { verifyReleaseArtifacts } from './verify-release-artifacts.mjs'

export async function checkRelease({
	root = process.cwd(),
	commit = 'HEAD',
	mainRef = 'origin/main',
	ref
} = {}) {
	const inputs = await getReleaseInputs(root, commit)
	const tag = ref ?? `v${inputs.releaseVersion}`
	if (tag !== `v${inputs.releaseVersion}`)
		throw new Error(`Release tag ${tag} does not match v${inputs.releaseVersion}`)
	const run = await findReleaseBuild({ root, commit: inputs.sourceCommit, mainRef })
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-release-check-'))
	try {
		console.log(`Checking ${tag} at ${inputs.sourceCommit} using ${run.html_url ?? run.id}`)
		for (const platform of RELEASE_PLATFORMS) {
			await promisify(execFile)(
				'gh',
				[
					'run',
					'download',
					String(run.id),
					'--name',
					`release-assets-${platform}`,
					'--dir',
					directory
				],
				{ cwd: root }
			)
		}
		await verifyReleaseArtifacts({
			root,
			directory,
			commit: inputs.sourceCommit,
			ref: tag,
			mainRef,
			buildCommit: run.head_sha,
			workflowRun: String(run.id)
		})
		console.log(`Pre-tag check passed for ${tag} at ${inputs.sourceCommit}.`)
	} finally {
		await rm(directory, { recursive: true, force: true })
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	await checkRelease(releaseArguments(process.argv.slice(2)))
}

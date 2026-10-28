import { execFile } from 'node:child_process'
import { appendFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { promisify } from 'node:util'
import { pathToFileURL } from 'node:url'
import { gitOutput, isAncestor } from './lib/git.mjs'
import { ARTIFACT_INPUTS_PATH, getReleaseInputs } from './lib/release-inputs.mjs'
import { RELEASE_PLATFORMS } from './lib/release-lock.mjs'

const execute = promisify(execFile)

export async function selectReleaseBuild({
	root = process.cwd(),
	commit,
	mainRef = 'origin/main',
	runs,
	listArtifacts
}) {
	const target = await getReleaseInputs(root, commit)
	if (!(await isAncestor(root, target.sourceCommit, mainRef))) {
		throw new Error(`Release commit ${target.sourceCommit} is not on ${mainRef}`)
	}
	// Topological order visits a descendant before any of its ancestors, independent
	// of run time: a later rebuild of an older commit must not win over a nearer one.
	const history = (
		await gitOutput(root, ['rev-list', '--topo-order', target.sourceCommit])
	).split('\n')
	const rank = new Map(history.map((sha, index) => [sha, index]))
	const candidates = runs
		.filter(
			(run) =>
				run.head_branch === 'main' &&
				['push', 'workflow_dispatch'].includes(run.event) &&
				run.conclusion === 'success' &&
				rank.has(run.head_sha)
		)
		.sort(
			(left, right) =>
				rank.get(left.head_sha) - rank.get(right.head_sha) ||
				Number(right.id) - Number(left.id)
		)
	const fingerprints = new Map()
	for (const run of candidates) {
		if (!fingerprints.has(run.head_sha)) {
			// Older runs without the new input definition cannot certify this format.
			const definition = await gitOutput(root, [
				'ls-tree',
				run.head_sha,
				'--',
				ARTIFACT_INPUTS_PATH
			])
			fingerprints.set(
				run.head_sha,
				definition ? (await getReleaseInputs(root, run.head_sha)).inputFingerprint : null
			)
		}
		if (fingerprints.get(run.head_sha) !== target.inputFingerprint) continue
		const artifacts = await listArtifacts(run.id)
		if (
			!RELEASE_PLATFORMS.every((platform) =>
				artifacts.some(
					(artifact) =>
						artifact.name === `release-assets-${platform}` &&
						!artifact.expired &&
						(!artifact.expires_at || Date.parse(artifact.expires_at) > Date.now())
				)
			)
		)
			continue
		return run
	}
	throw new Error(
		`No retained successful main release-artifact build matches ${target.sourceCommit}. Dispatch Build Release Artifacts on main and retry; artifacts expire after 14 days.`
	)
}

export async function findReleaseBuild({
	root = process.cwd(),
	commit = 'HEAD',
	mainRef = 'origin/main'
} = {}) {
	const pages = await githubPages(
		root,
		'actions/workflows/build-and-release.yml/runs?branch=main&status=success&per_page=100'
	)
	return selectReleaseBuild({
		root,
		commit,
		mainRef,
		runs: pages.flatMap((page) => page.workflow_runs),
		listArtifacts: async (runId) =>
			(await githubPages(root, `actions/runs/${runId}/artifacts?per_page=100`)).flatMap(
				(page) => page.artifacts
			)
	})
}

async function githubPages(root, endpoint) {
	const { stdout } = await execute(
		'gh',
		['api', '--paginate', '--slurp', `repos/{owner}/{repo}/${endpoint}`],
		{ cwd: root, maxBuffer: 32 * 1024 * 1024 }
	)
	return JSON.parse(stdout)
}

export function releaseArguments(args) {
	const options = {}
	const names = { '--commit': 'commit', '--main-ref': 'mainRef', '--ref': 'ref' }
	for (let index = 0; index < args.length; index += 2) {
		const name = names[args[index]]
		if (!name || !args[index + 1] || args[index + 1].startsWith('--'))
			throw new Error(`Invalid release argument: ${args[index]}`)
		options[name] = args[index + 1]
	}
	return options
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	const run = await findReleaseBuild(releaseArguments(process.argv.slice(2)))
	console.log(`Selected certified build ${run.html_url ?? run.id} at ${run.head_sha}`)
	if (process.env.GITHUB_OUTPUT) {
		await appendFile(
			process.env.GITHUB_OUTPUT,
			`run-id=${run.id}\nsource-commit=${run.head_sha}\n`
		)
	}
}

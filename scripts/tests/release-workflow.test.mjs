import assert from 'node:assert/strict'
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import yaml from 'js-yaml'
import { loadArtifactInputDefinition } from '../lib/release-inputs.mjs'
import { syncReleaseInputPaths } from '../sync-release-input-paths.mjs'

async function workflow(name) {
	return yaml.load(
		await readFile(new URL(`../../.github/workflows/${name}`, import.meta.url), 'utf8')
	)
}

test('native installers run only for main while pull requests and branch dispatches use preflight', async () => {
	const build = await workflow('build-and-release.yml')
	const native = build.jobs['build-packages']
	const allowsNative = new Function('github', `return (${native.if})`)
	for (const [event_name, ref, expected] of [
		['push', 'refs/heads/main', true],
		['workflow_dispatch', 'refs/heads/main', true],
		['pull_request', 'refs/pull/135/merge', false],
		['pull_request', 'refs/heads/main', false],
		['workflow_dispatch', 'refs/heads/feature', false],
		['workflow_dispatch', 'refs/tags/v1.5.3', false]
	]) {
		assert.equal(allowsNative({ event_name, ref }), expected, `${event_name}:${ref}`)
	}
	assert.equal(native.needs, 'release-preflight')
	assert.deepEqual(
		native.strategy.matrix.include.map(({ platform }) => platform),
		['linux', 'mac', 'win']
	)

	const preflight = build.jobs['release-preflight']
	assert.equal(preflight['runs-on'], 'ubuntu-24.04')
	assert.equal(preflight.if, undefined)
	for (const command of ['npm ci', 'npm run test:release', 'npm run preflight:release']) {
		assert(
			preflight.steps.some(({ run }) => run === command),
			`Missing preflight command: ${command}`
		)
	}
	assert.equal(
		preflight.steps.find(({ uses }) => uses?.startsWith('actions/setup-node')).with[
			'node-version'
		],
		24
	)
	assert.doesNotMatch(
		JSON.stringify(preflight),
		/build:release-artifacts|prepare-package-flavor|upload-artifact|download-artifact/
	)
	assert.equal(build.jobs['resolve-pr-build'], undefined)
	assert.doesNotMatch(JSON.stringify(native), /certify-release-artifacts|download-artifact/)
})

test('native build and publication artifacts stay outside the checkout', async () => {
	const native = (await workflow('build-and-release.yml')).jobs['build-packages']
	const nativeSetup = native.steps.find(({ id }) => id === 'artifact-directory')
	assert.equal(
		nativeSetup.env.OUROBOROS_RELEASE_ARTIFACT_DIR,
		'${{ runner.temp }}/ouroboros-release-assets/${{ matrix.platform }}'
	)
	const upload = native.steps.find(({ uses }) => uses?.startsWith('actions/upload-artifact'))
	assert.equal(upload.with.path, '${{ env.OUROBOROS_RELEASE_ARTIFACT_DIR }}/*')
	assert.equal(upload.with.name, 'release-assets-${{ matrix.platform }}')

	const publish = (await workflow('publish-release.yml')).jobs.publish
	const publishSetup = publish.steps.find(({ id }) => id === 'artifact-directory')
	assert.equal(
		publishSetup.env.OUROBOROS_RELEASE_ARTIFACT_DIR,
		'${{ runner.temp }}/ouroboros-release-assets'
	)
	const download = publish.steps.find(({ uses }) => uses?.startsWith('actions/download-artifact'))
	assert.equal(download.with.pattern, 'release-assets-*')
	assert.equal(download.with.path, '${{ env.OUROBOROS_RELEASE_ARTIFACT_DIR }}')
	const verification = publish.steps.find(({ run }) =>
		run?.includes('verify-release-artifacts.mjs')
	)
	assert(verification.run.includes('--dir "${{ env.OUROBOROS_RELEASE_ARTIFACT_DIR }}"'))
	const release = publish.steps.find(({ uses }) =>
		uses?.startsWith('softprops/action-gh-release')
	)
	assert.equal(release.with.files, '${{ env.OUROBOROS_RELEASE_ARTIFACT_DIR }}/*')
	for (const job of [native, publish]) {
		// runner is available in step env, but GitHub rejects it in job env.
		assert.doesNotMatch(JSON.stringify(job.env ?? {}), /\$\{\{[^}]*\brunner\./)
		assert.equal(job.env?.OUROBOROS_RELEASE_ARTIFACT_DIR, undefined)
		const setupIndex = job.steps.findIndex(({ id }) => id === 'artifact-directory')
		const nodeIndex = job.steps.findIndex(({ uses }) => uses?.startsWith('actions/setup-node'))
		assert(setupIndex > nodeIndex)
		assert.match(job.steps[setupIndex].run, /appendFileSync\(process\.env\.GITHUB_ENV,/)
		assert.match(job.steps[setupIndex].run, /process\.env\.OUROBOROS_RELEASE_ARTIFACT_DIR/)
		const firstConsumer = job.steps.findIndex(
			({ uses, run }) =>
				uses?.startsWith('actions/download-artifact') ||
				run?.includes('build:release-artifacts')
		)
		assert(setupIndex < firstConsumer)
	}
})

test('release trigger paths derive exactly from the canonical artifact input definition', async () => {
	const build = await workflow('build-and-release.yml')
	const definition = await loadArtifactInputDefinition()
	for (const event of ['push', 'pull_request']) {
		assert.deepEqual(build.on[event].paths, definition.paths, event)
	}
	assert(
		build.jobs['release-preflight'].steps.some(
			({ run }) => run === 'node scripts/sync-release-input-paths.mjs --check'
		)
	)
	await syncReleaseInputPaths()
})

test('generated workflow filter drift fails preflight and is repaired from the definition', async (t) => {
	const root = await mkdtemp(join(tmpdir(), 'ouroboros-release-paths-'))
	t.after(() => rm(root, { recursive: true, force: true }))
	await mkdir(join(root, 'release'))
	await mkdir(join(root, '.github', 'workflows'), { recursive: true })
	await copyFile(
		new URL('../../release/artifact-inputs.json', import.meta.url),
		join(root, 'release', 'artifact-inputs.json')
	)
	const original = await readFile(
		new URL('../../.github/workflows/build-and-release.yml', import.meta.url),
		'utf8'
	)
	const path = join(root, '.github', 'workflows', 'build-and-release.yml')
	await writeFile(path, original.replace('            - resources/**\n', ''))
	await assert.rejects(syncReleaseInputPaths({ root }), /Artifact workflow paths differ/)
	await syncReleaseInputPaths({ root, check: false })
	assert.equal(await readFile(path, 'utf8'), original)
	await syncReleaseInputPaths({ root })
})

test('manual tags publish verified certified artifacts as a draft without rebuilding', async () => {
	await assert.rejects(workflow('pr-tag.yml'), { code: 'ENOENT' })
	const publish = await workflow('publish-release.yml')
	assert.deepEqual(publish.on.push.tags, ['v*.*.*'])
	const steps = publish.jobs.publish.steps
	const selectionIndex = steps.findIndex(({ id }) => id === 'certified-build')
	const downloadIndex = steps.findIndex(({ uses }) =>
		uses?.startsWith('actions/download-artifact')
	)
	const verificationIndex = steps.findIndex(({ run }) =>
		run?.includes('verify-release-artifacts.mjs')
	)
	const releaseIndex = steps.findIndex(({ uses }) =>
		uses?.startsWith('softprops/action-gh-release')
	)
	assert(selectionIndex >= 0 && selectionIndex < downloadIndex)
	assert(downloadIndex < verificationIndex && verificationIndex < releaseIndex)
	assert.equal(steps[releaseIndex].with.draft, true)
	assert.doesNotMatch(JSON.stringify(steps), /build:release-artifacts|electron-builder/)
	assert.match(steps[selectionIndex].run, /node scripts\/select-release-build\.mjs/)
	assert.match(steps[selectionIndex].run, /--commit "\$\{\{ github\.sha \}\}"/)
	assert.equal(steps[selectionIndex].env.GH_TOKEN, '${{ secrets.GITHUB_TOKEN }}')
	assert.equal(steps[downloadIndex].with['run-id'], '${{ steps.certified-build.outputs.run-id }}')
	assert.match(
		steps[verificationIndex].run,
		/--build-commit \$\{\{ steps\.certified-build\.outputs\.source-commit \}\}/
	)
	assert.match(
		steps[verificationIndex].run,
		/--run \$\{\{ steps\.certified-build\.outputs\.run-id \}\}/
	)
	assert.equal(
		steps.find(({ uses }) => uses?.startsWith('actions/checkout')).with['fetch-depth'],
		0
	)
})

import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import yaml from 'js-yaml'

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
	assert.equal(preflight['runs-on'], 'ubuntu-latest')
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
	assert.equal(
		native.env.OUROBOROS_RELEASE_ARTIFACT_DIR,
		'${{ runner.temp }}/ouroboros-release-assets/${{ matrix.platform }}'
	)
	const upload = native.steps.find(({ uses }) => uses?.startsWith('actions/upload-artifact'))
	assert.equal(upload.with.path, '${{ env.OUROBOROS_RELEASE_ARTIFACT_DIR }}/*')
	assert.equal(upload.with.name, 'release-assets-${{ matrix.platform }}')

	const publish = (await workflow('publish-release.yml')).jobs.publish
	assert.equal(
		publish.env.OUROBOROS_RELEASE_ARTIFACT_DIR,
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
})

test('release preflight watches packaging resources and build configuration', async () => {
	const build = await workflow('build-and-release.yml')
	for (const event of ['push', 'pull_request']) {
		for (const input of [
			'resources/**',
			'build/**',
			'tsconfig*.json',
			'.npmrc',
			'release/**',
			'scripts/**',
			'electron-builder.yml',
			'electron.vite.config.ts',
			'package.json',
			'package-lock.json'
		]) {
			assert(build.on[event].paths.includes(input), `${event} does not watch ${input}`)
		}
	}
})

test('successful main certification still precedes tagging and verified draft publication', async () => {
	const build = await workflow('build-and-release.yml')
	const tag = await workflow('pr-tag.yml')
	assert(tag.on.workflow_run.workflows.includes(build.name))
	assert.deepEqual(tag.on.workflow_run.branches, ['main'])
	assert.deepEqual(tag.on.workflow_run.types, ['completed'])
	const allowsTag = new Function('github', `return (${tag.jobs.tag.if})`)
	assert.equal(
		allowsTag({ event: { workflow_run: { conclusion: 'success', event: 'push' } } }),
		true
	)
	assert.equal(
		allowsTag({ event: { workflow_run: { conclusion: 'failure', event: 'push' } } }),
		false
	)
	assert.equal(
		allowsTag({
			event: { workflow_run: { conclusion: 'success', event: 'workflow_dispatch' } }
		}),
		false
	)

	const publish = await workflow('publish-release.yml')
	assert.deepEqual(publish.on.push.tags, ['v*.*.*'])
	const steps = publish.jobs.publish.steps
	const verificationIndex = steps.findIndex(({ run }) =>
		run?.includes('verify-release-artifacts.mjs')
	)
	const releaseIndex = steps.findIndex(({ uses }) =>
		uses?.startsWith('softprops/action-gh-release')
	)
	assert(verificationIndex >= 0 && verificationIndex < releaseIndex)
	assert.equal(steps[releaseIndex].with.draft, true)
	assert.doesNotMatch(JSON.stringify(steps), /build:release-artifacts|electron-builder/)
	const certifiedBuild = steps.find(({ id }) => id === 'certified-build').with.script
	assert(certifiedBuild.includes("workflow_id: 'build-and-release.yml'"))
	assert(certifiedBuild.includes("candidate.head_branch === 'main'"))
	assert(certifiedBuild.includes("candidate.event === 'push'"))
	assert(certifiedBuild.includes("candidate.conclusion === 'success'"))
})

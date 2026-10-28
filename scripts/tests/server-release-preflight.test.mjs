import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import test from 'node:test'
import { gitOutput } from '../lib/git.mjs'
import { loadReleaseLock, validateReleaseVersions } from '../lib/release-lock.mjs'
import { preflightRelease, validateServerReleaseManifest } from '../preflight-release.mjs'

const run = promisify(execFile)
const repository = process.cwd()

async function temporaryDirectory(t) {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-server-preflight-'))
	t.after(() => rm(directory, { recursive: true, force: true }))
	return directory
}

async function writeManifest(directory, lock) {
	await run(process.execPath, [join(repository, 'scripts/write-server-image-manifest.mjs')], {
		cwd: directory,
		env: {
			...process.env,
			OUROBOROS_SERVER_IMAGE_REPOSITORY: lock.serverImage.repository,
			OUROBOROS_SERVER_IMAGE_DIGEST: lock.serverImage.digest,
			OUROBOROS_SERVER_SOURCE_COMMIT: lock.serverImage.sourceCommit
		}
	})
	return join(directory, 'server-image-manifest.json')
}

async function applicationVersions() {
	return await Promise.all(
		['package.json', 'package-lock.json'].map(async (name) =>
			JSON.parse(await readFile(join(repository, name), 'utf8'))
		)
	)
}

test('real application and npm root package versions agree with the release lock', async () => {
	const { lock } = await loadReleaseLock(repository)
	const [packageJson, packageLock] = await applicationVersions()
	assert.doesNotThrow(() => validateReleaseVersions(packageJson, packageLock, lock))
})

for (const field of ['application', 'npm lock', 'npm root package']) {
	test(`release preflight rejects a forgotten ${field} version bump`, async () => {
		const { lock } = await loadReleaseLock(repository)
		const [packageJson, packageLock] = await applicationVersions()
		if (field === 'application') packageJson.version = '0.0.0-omitted-bump'
		if (field === 'npm lock') packageLock.version = '0.0.0-omitted-bump'
		if (field === 'npm root package') packageLock.packages[''].version = '0.0.0-omitted-bump'
		assert.throws(
			() => validateReleaseVersions(packageJson, packageLock, lock),
			/does not match release lock version/
		)
	})
}

test('ordinary preflight stages the real checkout without requiring server history', async (t) => {
	const tempRoot = await temporaryDirectory(t)
	await preflightRelease({ root: repository, tempRoot })
	assert.deepEqual(await readdir(tempRoot), [])
})

test('the real server manifest writer and pinned source provide the bundled wheel version', async (t) => {
	const directory = await temporaryDirectory(t)
	const { lock } = await loadReleaseLock(repository)
	try {
		await gitOutput(repository, ['cat-file', '-e', `${lock.serverImage.sourceCommit}^{commit}`])
	} catch {
		// Pull-request preflight uses a shallow checkout. The independent history
		// fixture below still exercises source inspection there.
		t.skip('Pinned server history is not present in this checkout')
		return
	}
	const manifestPath = await writeManifest(directory, lock)
	const wheelVersion = await validateServerReleaseManifest(repository, lock, manifestPath)
	const pyproject = await gitOutput(repository, [
		'show',
		`${lock.serverImage.sourceCommit}:python/pyproject.toml`
	])
	assert.ok(pyproject.includes(`version = "${wheelVersion}"`))
})

for (const [field, replacement] of [
	['repository', 'ghcr.io/other/server'],
	['digest', `sha256:${'0'.repeat(64)}`],
	['sourceCommit', '0'.repeat(40)],
	['image', 'ghcr.io/other/server@sha256:incorrect'],
	['tag', 'latest'],
	['schemaVersion', 2]
]) {
	test(`server preflight rejects a mismatched manifest ${field}`, async (t) => {
		const directory = await temporaryDirectory(t)
		const { lock } = await loadReleaseLock(repository)
		const manifestPath = await writeManifest(directory, lock)
		const manifest = JSON.parse(await readFile(manifestPath, 'utf8'))
		manifest[field] = replacement
		await writeFile(manifestPath, JSON.stringify(manifest))
		await assert.rejects(
			validateServerReleaseManifest(repository, lock, manifestPath),
			new RegExp(field)
		)
	})
}

test('server preflight inspects commit A after a later Python change, without app version coupling', async (t) => {
	const directory = await temporaryDirectory(t)
	await mkdir(join(directory, 'python'))
	const pyprojectPath = join(directory, 'python/pyproject.toml')
	const pyproject = await readFile(join(repository, 'python/pyproject.toml'), 'utf8')
	await writeFile(pyprojectPath, pyproject.replace(/^version = .*$/m, 'version = "0.1.0"'))
	await gitOutput(directory, ['init', '--quiet'])
	await gitOutput(directory, ['add', 'python/pyproject.toml'])
	const commit = async (message) =>
		await gitOutput(directory, [
			'-c',
			'user.name=Release fixture',
			'-c',
			'user.email=release@example.invalid',
			'-c',
			'commit.gpgsign=false',
			'commit',
			'--quiet',
			'-m',
			message
		])
	await commit('A: publish server')
	const { lock } = await loadReleaseLock(repository)
	lock.serverImage.sourceCommit = await gitOutput(directory, ['rev-parse', 'HEAD'])
	const manifestPath = await writeManifest(directory, lock)
	await writeFile(pyprojectPath, pyproject.replace(/^version = .*$/m, 'version = "0.2.0"'))
	await gitOutput(directory, ['add', 'python/pyproject.toml'])
	await commit('Later server work does not change the already pinned image')
	assert.notEqual(lock.releaseVersion, '0.1.0')
	assert.equal(await validateServerReleaseManifest(directory, lock, manifestPath), '0.1.0')
})

test('explicit server preflight explains how to fetch unavailable pinned source', async (t) => {
	const directory = await temporaryDirectory(t)
	await gitOutput(directory, ['init', '--quiet'])
	const { lock } = await loadReleaseLock(repository)
	const manifestPath = await writeManifest(directory, lock)
	await assert.rejects(
		validateServerReleaseManifest(directory, lock, manifestPath),
		/fetch server source commit .* and rerun preflight/
	)
})

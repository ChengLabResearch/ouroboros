import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { chmod, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import test from 'node:test'
import { copyFileToHostCommand } from '../../resources/processes/volume-copy.mjs'

const run = promisify(execFile)

async function copyFixture(plugin, runTest, failOwnership = false) {
	const root = await mkdtemp(join(tmpdir(), 'ouroboros-volume-copy-'))
	const host = join(root, 'host')
	const volume = join(root, 'volume')
	const bin = join(root, 'bin')
	const ownershipLog = join(root, 'ownership')
	try {
		await mkdir(host)
		await mkdir(join(volume, plugin), { recursive: true })
		await mkdir(bin)
		await writeFile(join(volume, plugin, 'report.json'), '{}\n', { mode: 0o600 })
		const realChown = (await run('/bin/sh', ['-c', 'command -v chown'])).stdout.trim()
		// Execute the generated command against isolated directories, never real Docker mounts.
		await writeFile(
			join(bin, 'docker'),
			`#!${process.execPath}
const assert = require('node:assert/strict')
const { execFile } = require('node:child_process')
assert.deepEqual(process.argv.slice(2, -1), ['run', '--rm', '-v', process.env.FIXTURE_HOST + ':/host', '-v', 'fixture-volume:/volume', '-w', '/host', 'alpine', 'sh', '-c'])
const script = process.argv.at(-1).replaceAll('/volume/', process.env.FIXTURE_VOLUME + '/').replaceAll('/host/', process.env.FIXTURE_HOST + '/')
execFile('/bin/sh', ['-c', script], { env: process.env }, (error, stdout, stderr) => {
  process.stdout.write(stdout)
  process.stderr.write(stderr)
  process.exit(error ? 1 : 0)
})
`
		)
		await writeFile(
			join(bin, 'chown'),
			`#!${process.execPath}
require('node:fs').writeFileSync(process.env.FIXTURE_OWNERSHIP_LOG, process.argv.slice(2).join(' '))
if (process.env.FIXTURE_FAIL_OWNERSHIP === '1') {
  process.stderr.write('fixture ownership failure')
  process.exit(1)
}
const result = require('node:child_process').spawnSync(process.env.FIXTURE_REAL_CHOWN, process.argv.slice(2), { encoding: 'utf8' })
process.stderr.write(result.stderr || '')
process.exit(result.status ?? 1)
`
		)
		await chmod(join(bin, 'docker'), 0o755)
		await chmod(join(bin, 'chown'), 0o755)
		const copy = () =>
			run(
				'/bin/sh',
				[
					'-c',
					copyFileToHostCommand(host, 'report.json', 'fixture-volume', plugin, plugin)
				],
				{
					env: {
						...process.env,
						PATH: `${bin}:${process.env.PATH}`,
						FIXTURE_HOST: host,
						FIXTURE_VOLUME: volume,
						FIXTURE_OWNERSHIP_LOG: ownershipLog,
						FIXTURE_REAL_CHOWN: realChown,
						FIXTURE_FAIL_OWNERSHIP: failOwnership ? '1' : '0'
					}
				}
			)
		await runTest({
			copy,
			delivered: join(host, 'report.json'),
			source: join(volume, plugin, 'report.json'),
			ownershipLog
		})
	} finally {
		await rm(root, { recursive: true, force: true })
	}
}

test(
	'report export assigns the host identity while retaining private permissions',
	{ skip: process.platform === 'win32' },
	async () => {
		await copyFixture('main-statistics', async ({ copy, delivered, source, ownershipLog }) => {
			await copy()
			assert.equal(await readFile(delivered, 'utf8'), '{}\n')
			const metadata = await stat(delivered)
			assert.equal(metadata.mode & 0o777, 0o600)
			assert.equal(metadata.uid, process.getuid())
			assert.equal(metadata.gid, process.getgid())
			assert.equal(
				await readFile(ownershipLog, 'utf8'),
				`${process.getuid()}:${process.getgid()} ${delivered}`
			)
			assert.equal(await readFile(source, 'utf8'), '{}\n')
		})
	}
)

test(
	'scientific exports retain their existing copy behavior without ownership changes',
	{ skip: process.platform === 'win32' },
	async () => {
		await copyFixture('main', async ({ copy, delivered, ownershipLog }) => {
			await copy()
			assert.equal(await readFile(delivered, 'utf8'), '{}\n')
			assert.equal((await stat(delivered)).mode & 0o777, 0o600)
			await assert.rejects(readFile(ownershipLog), { code: 'ENOENT' })
		})
	}
)

test(
	'ownership failure fails the export and leaves the private source for recovery',
	{ skip: process.platform === 'win32' },
	async () => {
		await copyFixture(
			'main-statistics',
			async ({ copy, source }) => {
				await assert.rejects(copy(), /fixture ownership failure/)
				assert.equal(await readFile(source, 'utf8'), '{}\n')
				assert.equal((await stat(source)).mode & 0o777, 0o600)
			},
			true
		)
	}
)

test('hosts without POSIX identity APIs retain the existing Docker mapping', async () => {
	const moduleUrl = new URL('../../resources/processes/volume-copy.mjs', import.meta.url).href
	const { stdout } = await run(process.execPath, [
		'--input-type=module',
		'-e',
		`
process.getuid = undefined
process.getgid = undefined
const { copyFileToHostCommand } = await import(${JSON.stringify(moduleUrl)})
console.log(copyFileToHostCommand('output', 'report.json', 'fixture-volume', 'main-statistics', 'main-statistics'))
`
	])
	assert.match(stdout, /cp -r/)
	assert.doesNotMatch(stdout, /chown/)
})

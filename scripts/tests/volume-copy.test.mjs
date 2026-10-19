import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { chmod, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import test from 'node:test'
import {
	copyFileToHostCommand,
	copyFileToVolumeCommand
} from '../../resources/processes/volume-copy.mjs'

const run = promisify(execFile)

async function copyFixture(plugin, runTest, failOwnership = false) {
	const root = await mkdtemp(join(tmpdir(), 'ouroboros-volume-copy-'))
	const host = join(root, 'host \' " $HOME `printf EXPANDED` $(printf EXPANDED)  spaces')
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
const args = process.argv.slice(2)
assert.deepEqual(args.slice(0, 11), ['run', '--rm', '-v', process.env.FIXTURE_HOST + ':/host', '-v', 'fixture-volume:/volume', '-w', '/host', 'alpine', 'sh', '-c'])
// Substitute mount roots only in positional data, never in the shell script.
const parameters = args.slice(13).map(value => value.startsWith('/volume/')
  ? process.env.FIXTURE_VOLUME + value.slice('/volume'.length)
  : value.startsWith('/host/') ? process.env.FIXTURE_HOST + value.slice('/host'.length) : value)
execFile('/bin/sh', ['-c', args[11], args[12], ...parameters], { env: process.env }, (error, stdout, stderr) => {
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
		const execute = (args) =>
			run('docker', args, {
				env: {
					...process.env,
					PATH: `${bin}:${process.env.PATH}`,
					FIXTURE_HOST: host,
					FIXTURE_VOLUME: volume,
					FIXTURE_OWNERSHIP_LOG: ownershipLog,
					FIXTURE_REAL_CHOWN: realChown,
					FIXTURE_FAIL_OWNERSHIP: failOwnership ? '1' : '0'
				}
			})
		const copy = (fileName = 'report.json') =>
			execute(copyFileToHostCommand(host, fileName, 'fixture-volume', plugin, plugin))
		const copyToVolume = (fileName) =>
			execute(copyFileToVolumeCommand(host, fileName, 'fixture-volume', plugin))

		await runTest({
			copy,
			copyToVolume,
			host,
			volume,
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

for (const directory of [false, true]) {
	test(
		`both copy directions preserve literal ${directory ? 'directory' : 'file'} paths`,
		{ skip: process.platform === 'win32' },
		async () => {
			const plugin = 'plugin \' " $HOME `printf EXPANDED` $(printf EXPANDED)  folder'
			const name = 'Bob\'s "stack" $HOME `printf EXPANDED` $(printf EXPANDED)  01.tif'
			await copyFixture(plugin, async ({ copy, copyToVolume, host, volume }) => {
				const hostPath = join(host, name)
				const payload = directory ? join(hostPath, 'slice  01.tif') : hostPath
				if (directory) await mkdir(hostPath)
				await writeFile(payload, 'literal path payload')
				await copyToVolume(name)
				assert.equal(
					await readFile(
						join(volume, plugin, name, ...(directory ? ['slice  01.tif'] : [])),
						'utf8'
					),
					'literal path payload'
				)
				await writeFile(payload, 'stale output')
				if (directory)
					await writeFile(join(hostPath, 'obsolete'), 'replace the old directory')
				await copy(name)
				assert.equal(await readFile(payload, 'utf8'), 'literal path payload')
				if (directory)
					await assert.rejects(stat(join(hostPath, 'obsolete')), { code: 'ENOENT' })
			})
		}
	)
}

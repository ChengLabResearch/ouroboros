import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { access, chmod, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import test from 'node:test'
import { deleteFilesFromVolumeFolder } from '../../resources/processes/volume-cleanup.mjs'

const run = promisify(execFile)

async function cleanupFixture(runTest) {
	const root = await mkdtemp(join(tmpdir(), 'ouroboros-volume-cleanup-'))
	const volume = join(root, 'mounted-files')
	const bin = join(root, 'bin')
	await mkdir(volume)
	await mkdir(bin)

	// This executable verifies the real generated Docker command, then confines
	// the inner shell to the test's temporary volume. It never invokes Docker.
	const docker = join(bin, 'docker')
	await writeFile(
		docker,
		`#!${process.execPath}
const assert = require('node:assert/strict')
const { execFile } = require('node:child_process')
const args = process.argv.slice(2)
assert.deepEqual(args.slice(0, 7), ['run', '--rm', '-v', 'fixture-volume:/volume', 'alpine', 'sh', '-c'])
// The whole-volume find script has one fixed mount root. Plugin paths travel
// exclusively in positional data, so their shell metacharacters stay literal.
const script = args[7].replaceAll('/volume/', process.env.FIXTURE_VOLUME + '/')
const parameters = args.slice(9).map(value => {
  assert.ok(value.startsWith('/volume/'))
  return process.env.FIXTURE_VOLUME + value.slice('/volume'.length)
})
assert.ok(!script.includes('/volume/'), 'Never execute against a real volume')
execFile('/bin/sh', ['-c', script, args[8], ...parameters], { env: process.env, encoding: 'utf8' }, (error, stdout, stderr) => {
  process.stdout.write(stdout)
  process.stderr.write(stderr)
  if (error) process.stderr.write(String(error))
  process.exit(error ? 1 : 0)
})
`
	)
	await chmod(docker, 0o755)

	try {
		// Exercise Alpine-compatible BusyBox find/rm whenever available.
		try {
			await access('/usr/bin/busybox')
			await symlink('/usr/bin/busybox', join(bin, 'find'))
			await symlink('/usr/bin/busybox', join(bin, 'rm'))
		} catch {
			// Other Unix test hosts use their standard shell utilities.
		}
		const clear = (targetFolder) =>
			run('docker', deleteFilesFromVolumeFolder('fixture-volume', targetFolder), {
				env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, FIXTURE_VOLUME: volume }
			})
		await runTest(volume, clear)
	} finally {
		await rm(root, { recursive: true, force: true })
	}
}

test(
	'whole-volume shutdown cleanup retains failed reports and existing hidden entries',
	{
		skip: process.platform === 'win32'
	},
	async () => {
		await cleanupFixture(async (volume, clear) => {
			await mkdir(join(volume, 'main-statistics'))
			await writeFile(
				join(volume, 'main-statistics', 'failed-run.json'),
				'{"outcome":"failed"}'
			)
			await mkdir(join(volume, 'main'))
			await writeFile(join(volume, 'main', '.intermediate'), 'scratch')
			await writeFile(join(volume, 'another plugin output'), 'scratch')
			await mkdir(join(volume, '.hidden'))
			await writeFile(join(volume, '.hidden', 'keep'), 'hidden state')
			await writeFile(join(volume, '.hidden-file'), 'hidden file')
			await clear('/')
			assert.equal(
				await readFile(join(volume, 'main-statistics', 'failed-run.json'), 'utf8'),
				'{"outcome":"failed"}'
			)
			assert.equal(await readFile(join(volume, '.hidden', 'keep'), 'utf8'), 'hidden state')
			assert.equal(await readFile(join(volume, '.hidden-file'), 'utf8'), 'hidden file')
			await assert.rejects(access(join(volume, 'main')))
			await assert.rejects(access(join(volume, 'another plugin output')))
			// Repeated shutdown is safe when only retained reports/hidden entries remain.
			await clear('/')
		})
	}
)

test(
	'plugin-folder cleanup preserves its previous wildcard scope',
	{
		skip: process.platform === 'win32'
	},
	async () => {
		await cleanupFixture(async (volume, clear) => {
			const plugin = 'plugin \' " $HOME `printf EXPANDED` $(printf EXPANDED)  folder'
			await mkdir(join(volume, plugin))
			await writeFile(join(volume, plugin, 'output.tif'), 'scratch')
			await writeFile(join(volume, plugin, '.hidden'), 'preserved')
			await mkdir(join(volume, 'main-statistics'))
			await writeFile(join(volume, 'main-statistics', 'report.json'), '{}')
			await clear(plugin)
			await assert.rejects(access(join(volume, plugin, 'output.tif')))
			assert.equal(await readFile(join(volume, plugin, '.hidden'), 'utf8'), 'preserved')
			assert.equal(
				await readFile(join(volume, 'main-statistics', 'report.json'), 'utf8'),
				'{}'
			)
		})
	}
)

test(
	'whole-volume cleanup succeeds for empty scratch volumes',
	{
		skip: process.platform === 'win32'
	},
	async () => {
		await cleanupFixture(async (_volume, clear) => {
			await clear('/')
		})
	}
)

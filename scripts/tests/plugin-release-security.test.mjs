import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
	parseGitHubRepositoryUrl,
	withDownloadedPluginRelease
} from '../../src/main/plugin-release.mts'

function crc32(data) {
	let crc = 0xffffffff
	for (const byte of data) {
		crc ^= byte
		for (let bit = 0; bit < 8; bit += 1) {
			crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0)
		}
	}
	return (crc ^ 0xffffffff) >>> 0
}

function zipArchive(entries) {
	const localParts = []
	const centralParts = []
	let localOffset = 0

	for (const entry of entries) {
		const name = Buffer.from(entry.name)
		const contents = Buffer.from(entry.contents ?? '')
		const checksum = crc32(contents)
		const mode = entry.mode ?? 0o100644

		const localHeader = Buffer.alloc(30)
		localHeader.writeUInt32LE(0x04034b50, 0)
		localHeader.writeUInt16LE(20, 4)
		localHeader.writeUInt32LE(checksum, 14)
		localHeader.writeUInt32LE(contents.length, 18)
		localHeader.writeUInt32LE(contents.length, 22)
		localHeader.writeUInt16LE(name.length, 26)

		const localEntry = Buffer.concat([localHeader, name, contents])
		localParts.push(localEntry)

		const centralHeader = Buffer.alloc(46)
		centralHeader.writeUInt32LE(0x02014b50, 0)
		centralHeader.writeUInt16LE(0x0314, 4)
		centralHeader.writeUInt16LE(20, 6)
		centralHeader.writeUInt32LE(checksum, 16)
		centralHeader.writeUInt32LE(contents.length, 20)
		centralHeader.writeUInt32LE(contents.length, 24)
		centralHeader.writeUInt16LE(name.length, 28)
		centralHeader.writeUInt32LE((mode << 16) >>> 0, 38)
		centralHeader.writeUInt32LE(localOffset, 42)
		centralParts.push(Buffer.concat([centralHeader, name]))

		localOffset += localEntry.length
	}

	const centralDirectory = Buffer.concat(centralParts)
	const end = Buffer.alloc(22)
	end.writeUInt32LE(0x06054b50, 0)
	end.writeUInt16LE(entries.length, 8)
	end.writeUInt16LE(entries.length, 10)
	end.writeUInt32LE(centralDirectory.length, 12)
	end.writeUInt32LE(localOffset, 16)

	return Buffer.concat([...localParts, centralDirectory, end])
}

function releaseFetch(archive) {
	return async (input) => {
		const url = new URL(input)
		if (url.pathname.endsWith('/releases')) {
			return Response.json([
				{
					draft: false,
					assets: [
						{
							name: 'plugin.zip',
							url: 'https://api.github.com/repos/ChengLabResearch/plugin/releases/assets/1'
						}
					]
				}
			])
		}

		return new Response(archive, { status: 200 })
	}
}

test('GitHub repository URLs require the real HTTPS GitHub origin', () => {
	assert.deepEqual(parseGitHubRepositoryUrl('https://github.com/ChengLabResearch/plugin.git'), {
		owner: 'ChengLabResearch',
		repo: 'plugin'
	})
	assert.deepEqual(
		parseGitHubRepositoryUrl('https://github.com/ChengLabResearch/plugin/releases/latest'),
		{
			owner: 'ChengLabResearch',
			repo: 'plugin'
		}
	)

	for (const url of [
		'http://github.com/ChengLabResearch/plugin',
		'https://github.com.attacker.invalid/ChengLabResearch/plugin',
		'https://user@github.com/ChengLabResearch/plugin',
		'https://github.com/../plugin'
	]) {
		assert.throws(() => parseGitHubRepositoryUrl(url))
	}
})

test('plugin releases extract regular files and clean temporary data', async () => {
	const temporaryDirectory = await mkdtemp(join(tmpdir(), 'ouroboros-plugin-release-test-'))
	let extractedDirectory = ''

	try {
		const archive = zipArchive([
			{ name: 'package.json', contents: '{"name":"test-plugin"}\n' },
			{ name: 'nested/plugin.txt', contents: 'safe\n' }
		])

		const pluginName = await withDownloadedPluginRelease(
			'https://github.com/ChengLabResearch/plugin',
			temporaryDirectory,
			async (outputDirectory) => {
				extractedDirectory = outputDirectory
				assert.equal(
					await readFile(join(outputDirectory, 'nested', 'plugin.txt'), 'utf8'),
					'safe\n'
				)
				return JSON.parse(await readFile(join(outputDirectory, 'package.json'), 'utf8'))
					.name
			},
			{ fetchImplementation: releaseFetch(archive) }
		)

		assert.equal(pluginName, 'test-plugin')
		await assert.rejects(access(extractedDirectory))
		assert.deepEqual(await readdir(temporaryDirectory), [])
	} finally {
		await rm(temporaryDirectory, { recursive: true, force: true })
	}
})

for (const maliciousArchive of [
	{
		name: 'path traversal',
		entries: [{ name: '../outside.txt', contents: 'escaped\n' }],
		error: /(invalid relative path|escapes the extraction directory)/
	},
	{
		name: 'Windows path traversal',
		entries: [{ name: '..\\outside.txt', contents: 'escaped\n' }],
		error: /invalid characters in fileName/
	},
	{
		name: 'symbolic link',
		entries: [{ name: 'link', contents: '../../outside.txt', mode: 0o120777 }],
		error: /symbolic link/
	},
	{
		name: 'output collision',
		entries: [
			{ name: 'package.json', contents: '{}\n' },
			{ name: 'package.json', contents: '{"overwrite":true}\n' }
		],
		error: /EEXIST/
	}
]) {
	test(`plugin release extraction rejects ${maliciousArchive.name}`, async () => {
		const temporaryDirectory = await mkdtemp(join(tmpdir(), 'ouroboros-plugin-release-test-'))
		let consumed = false

		try {
			await assert.rejects(
				withDownloadedPluginRelease(
					'https://github.com/ChengLabResearch/plugin',
					temporaryDirectory,
					async () => {
						consumed = true
					},
					{ fetchImplementation: releaseFetch(zipArchive(maliciousArchive.entries)) }
				),
				maliciousArchive.error
			)

			assert.equal(consumed, false)
			assert.deepEqual(await readdir(temporaryDirectory), [])
			await assert.rejects(access(join(temporaryDirectory, 'outside.txt')))
		} finally {
			await rm(temporaryDirectory, { recursive: true, force: true })
		}
	})
}

test('temporary release data is removed when plugin installation fails', async () => {
	const temporaryDirectory = await mkdtemp(join(tmpdir(), 'ouroboros-plugin-release-test-'))
	const archive = zipArchive([{ name: 'package.json', contents: '{}\n' }])
	let extractedDirectory = ''

	try {
		await assert.rejects(
			withDownloadedPluginRelease(
				'https://github.com/ChengLabResearch/plugin',
				temporaryDirectory,
				async (outputDirectory) => {
					extractedDirectory = outputDirectory
					throw new Error('installation failed')
				},
				{ fetchImplementation: releaseFetch(archive) }
			),
			/installation failed/
		)

		await assert.rejects(access(extractedDirectory))
		assert.deepEqual(await readdir(temporaryDirectory), [])
	} finally {
		await rm(temporaryDirectory, { recursive: true, force: true })
	}
})

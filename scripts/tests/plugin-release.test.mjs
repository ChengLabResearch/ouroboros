import assert from 'node:assert/strict'
import { access, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { downloadRelease } from '../../src/main/plugin-release.mts'

const pluginArchive = Buffer.from(
	'UEsDBBQAAAAAAAAAAAAduIlLFwAAABcAAAAMAAAAcGFja2FnZS5qc29ueyJuYW1lIjoidGVzdC1wbHVnaW4ifQpQSwMEFAAAAAAAAAAAALvz9N0GAAAABgAAABEAAABuZXN0ZWQvcGx1Z2luLnR4dHdvcmtzClBLAQIUABQAAAAAAAAAAAAduIlLFwAAABcAAAAMAAAAAAAAAAAAAAAAAAAAAABwYWNrYWdlLmpzb25QSwECFAAUAAAAAAAAAAAAu/P03QYAAAAGAAAAEQAAAAAAAAAAAAAAAABBAAAAbmVzdGVkL3BsdWdpbi50eHRQSwUGAAAAAAIAAgB5AAAAdgAAAAAA',
	'base64'
)

test('downloads and extracts the latest plugin release assets', async () => {
	const outputDirectory = await mkdtemp(join(tmpdir(), 'ouroboros-plugin-release-test-'))

	const fetchImplementation = async (input) => {
		const url = new URL(input)
		if (url.pathname.endsWith('/releases')) {
			return Response.json([
				{
					assets: [
						{
							name: 'plugin.zip',
							url: 'https://api.github.com/repos/ChengLabResearch/plugin/releases/assets/1'
						}
					]
				}
			])
		}

		return new Response(pluginArchive, { status: 200 })
	}

	try {
		await downloadRelease('ChengLabResearch', 'plugin', outputDirectory, {
			fetchImplementation
		})

		assert.deepEqual(
			JSON.parse(await readFile(join(outputDirectory, 'package.json'), 'utf8')),
			{ name: 'test-plugin' }
		)
		assert.equal(
			await readFile(join(outputDirectory, 'nested', 'plugin.txt'), 'utf8'),
			'works\n'
		)
		await assert.rejects(access(join(outputDirectory, 'plugin.zip')))
	} finally {
		await rm(outputDirectory, { recursive: true, force: true })
	}
})

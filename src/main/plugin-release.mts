import { createWriteStream } from 'node:fs'
import { mkdir, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as NodeReadableStream } from 'node:stream/web'
import StreamZip from 'node-stream-zip'

const USER_AGENT = 'ChengLabResearch/ouroboros'

type GitHubReleaseAsset = {
	name: string
	url: string
}

type GitHubRelease = {
	assets: GitHubReleaseAsset[]
}

type DownloadOptions = {
	fetchImplementation?: typeof fetch
	token?: string
}

function githubHeaders(accept: string, token?: string): Record<string, string> {
	const headers: Record<string, string> = {
		Accept: accept,
		'User-Agent': USER_AGENT
	}

	if (token) {
		headers.Authorization = `token ${token}`
	}

	return headers
}

export async function extractPluginArchive(
	archivePath: string,
	outputDirectory: string
): Promise<void> {
	await mkdir(outputDirectory, { recursive: true })
	const archive = new StreamZip.async({ file: archivePath })

	try {
		await archive.extract(null, outputDirectory)
	} finally {
		await archive.close()
	}
}

async function downloadAsset(
	asset: GitHubReleaseAsset,
	outputDirectory: string,
	fetchImplementation: typeof fetch,
	token?: string
): Promise<void> {
	const response = await fetchImplementation(asset.url, {
		headers: githubHeaders('application/octet-stream', token),
		redirect: 'follow'
	})

	if (!response.ok || !response.body) {
		throw new Error(
			`GitHub release asset download failed for ${asset.name}: ${response.status}`
		)
	}

	const assetPath = join(outputDirectory, asset.name)
	await pipeline(
		Readable.fromWeb(response.body as unknown as NodeReadableStream),
		createWriteStream(assetPath)
	)

	if (/\.zip$/.exec(assetPath)) {
		await extractPluginArchive(assetPath, outputDirectory)
		await rm(assetPath)
	}
}

export async function downloadRelease(
	user: string,
	repo: string,
	outputDirectory: string,
	options: DownloadOptions = {}
): Promise<void> {
	const fetchImplementation = options.fetchImplementation ?? fetch
	const token = options.token ?? process.env.GITHUB_TOKEN
	const response = await fetchImplementation(
		`https://api.github.com/repos/${user}/${repo}/releases`,
		{
			headers: githubHeaders('application/vnd.github+json', token)
		}
	)

	if (!response.ok) {
		throw new Error(`GitHub release lookup failed for ${user}/${repo}: ${response.status}`)
	}

	const releases = (await response.json()) as GitHubRelease[]
	const release = releases.find(({ assets }) => assets.length > 0)
	if (!release) {
		throw new Error(`Could not find a release for ${user}/${repo}`)
	}

	await mkdir(outputDirectory, { recursive: true })
	await Promise.all(
		release.assets.map((asset) =>
			downloadAsset(asset, outputDirectory, fetchImplementation, token)
		)
	)
}

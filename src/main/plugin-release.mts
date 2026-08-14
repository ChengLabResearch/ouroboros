import { createWriteStream } from 'node:fs'
import { mkdir, mkdtemp, open, rm } from 'node:fs/promises'
import { dirname, join, resolve, sep } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as NodeReadableStream } from 'node:stream/web'
import yauzl, { type Entry } from 'yauzl'

const GITHUB_HOST = 'github.com'
const GITHUB_API_HOST = 'api.github.com'
const USER_AGENT = 'ChengLabResearch/ouroboros'
const UNIX_FILE_TYPE_MASK = 0o170000
const UNIX_REGULAR_FILE = 0o100000
const UNIX_DIRECTORY = 0o040000
const UNIX_SYMBOLIC_LINK = 0o120000

export type GitHubRepository = {
	owner: string
	repo: string
}

type GitHubReleaseAsset = {
	name: string
	url: string
}

type DownloadOptions = {
	fetchImplementation?: typeof fetch
	token?: string
}

export function parseGitHubRepositoryUrl(input: string): GitHubRepository {
	let url: URL
	try {
		url = new URL(input)
	} catch {
		throw new Error('Plugin URL must be a valid HTTPS GitHub repository URL.')
	}

	if (
		url.protocol !== 'https:' ||
		url.hostname.toLowerCase() !== GITHUB_HOST ||
		url.username !== '' ||
		url.password !== ''
	) {
		throw new Error('Plugin URL must be an HTTPS github.com repository URL.')
	}

	const segments = url.pathname
		.split('/')
		.filter(Boolean)
		.map((segment) => decodeURIComponent(segment))
	const owner = segments[0]
	const repo = segments[1]?.replace(/\.git$/, '')

	if (
		!owner ||
		!repo ||
		!/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/.test(owner) ||
		!/^[A-Za-z0-9._-]{1,100}$/.test(repo) ||
		repo === '.' ||
		repo === '..'
	) {
		throw new Error('Plugin URL must identify a valid GitHub owner and repository.')
	}

	return { owner, repo }
}

function githubHeaders(accept: string, token?: string): Record<string, string> {
	const headers: Record<string, string> = {
		Accept: accept,
		'User-Agent': USER_AGENT,
		'X-GitHub-Api-Version': '2022-11-28'
	}

	if (token) {
		headers.Authorization = `Bearer ${token}`
	}

	return headers
}

function validatedReleaseAssets(value: unknown): GitHubReleaseAsset[] {
	if (!Array.isArray(value)) {
		throw new Error('GitHub returned an invalid releases response.')
	}

	for (const release of value) {
		if (typeof release !== 'object' || release === null || release.draft === true) continue
		if (!Array.isArray(release.assets) || release.assets.length === 0) continue

		return release.assets.map((asset) => {
			if (
				typeof asset !== 'object' ||
				asset === null ||
				typeof asset.name !== 'string' ||
				typeof asset.url !== 'string'
			) {
				throw new Error('GitHub returned invalid release asset metadata.')
			}

			if (
				asset.name === '' ||
				asset.name === '.' ||
				asset.name === '..' ||
				asset.name.includes('/') ||
				asset.name.includes('\\')
			) {
				throw new Error(`GitHub returned an unsafe release asset name: ${asset.name}`)
			}

			const assetUrl = new URL(asset.url)
			if (assetUrl.protocol !== 'https:' || assetUrl.hostname !== GITHUB_API_HOST) {
				throw new Error(`GitHub returned an unsafe release asset URL for ${asset.name}.`)
			}

			return { name: asset.name, url: assetUrl.toString() }
		})
	}

	throw new Error('The GitHub repository has no published release assets.')
}

function destinationForEntry(root: string, fileName: string): string {
	const destination = resolve(root, fileName)
	if (destination === root || !destination.startsWith(`${root}${sep}`)) {
		throw new Error(`Archive entry escapes the extraction directory: ${fileName}`)
	}
	return destination
}

function unixMode(entry: Entry): number {
	const creatorSystem = entry.versionMadeBy >>> 8
	return creatorSystem === 3 ? entry.externalFileAttributes >>> 16 : 0
}

function validateEntryType(entry: Entry): { isDirectory: boolean; mode: number } {
	const mode = unixMode(entry)
	const fileType = mode & UNIX_FILE_TYPE_MASK
	const isDirectory = entry.fileName.endsWith('/')

	if (fileType === UNIX_SYMBOLIC_LINK) {
		throw new Error(`Archive contains a symbolic link: ${entry.fileName}`)
	}

	if (fileType !== 0 && fileType !== UNIX_REGULAR_FILE && fileType !== UNIX_DIRECTORY) {
		throw new Error(`Archive contains an unsupported special entry: ${entry.fileName}`)
	}

	if ((fileType === UNIX_DIRECTORY) !== isDirectory && fileType !== 0) {
		throw new Error(`Archive entry type does not match its path: ${entry.fileName}`)
	}

	return { isDirectory, mode: mode & 0o777 }
}

export async function extractPluginArchive(
	archivePath: string,
	outputDirectory: string
): Promise<void> {
	const root = resolve(outputDirectory)
	await mkdir(root, { recursive: true })

	const archive = await yauzl.openPromise(archivePath, {
		strictFileNames: true,
		validateEntrySizes: true
	})

	try {
		for await (const entry of archive.eachEntry()) {
			const destination = destinationForEntry(root, entry.fileName)
			const { isDirectory, mode } = validateEntryType(entry)

			if (isDirectory) {
				await mkdir(destination, { recursive: true })
				continue
			}

			await mkdir(dirname(destination), { recursive: true })
			const input = await archive.openReadStreamPromise(entry)
			const outputFile = await open(destination, 'wx', mode || 0o666)

			try {
				await pipeline(
					input,
					createWriteStream(destination, { fd: outputFile.fd, autoClose: false })
				)
			} catch (error) {
				await outputFile.close()
				await rm(destination, { force: true })
				throw error
			}

			await outputFile.close()
		}
	} finally {
		archive.close()
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
	const outputFile = await open(assetPath, 'wx')

	try {
		await pipeline(
			Readable.fromWeb(response.body as unknown as NodeReadableStream),
			createWriteStream(assetPath, { fd: outputFile.fd, autoClose: false })
		)
	} catch (error) {
		await outputFile.close()
		await rm(assetPath, { force: true })
		throw error
	}

	await outputFile.close()

	if (/\.zip$/i.test(asset.name)) {
		try {
			await extractPluginArchive(assetPath, outputDirectory)
		} finally {
			await rm(assetPath, { force: true })
		}
	}
}

async function downloadLatestPluginRelease(
	repository: GitHubRepository,
	outputDirectory: string,
	options: DownloadOptions = {}
): Promise<void> {
	const fetchImplementation = options.fetchImplementation ?? fetch
	const token = options.token ?? process.env.GITHUB_TOKEN
	const releasesUrl = new URL(
		`/repos/${encodeURIComponent(repository.owner)}/${encodeURIComponent(repository.repo)}/releases`,
		'https://api.github.com'
	)
	releasesUrl.searchParams.set('per_page', '100')

	const response = await fetchImplementation(releasesUrl, {
		headers: githubHeaders('application/vnd.github+json', token)
	})
	if (!response.ok) {
		throw new Error(
			`GitHub release lookup failed for ${repository.owner}/${repository.repo}: ${response.status}`
		)
	}

	const assets = validatedReleaseAssets(await response.json())
	for (const asset of assets) {
		await downloadAsset(asset, outputDirectory, fetchImplementation, token)
	}
}

export async function withDownloadedPluginRelease<T>(
	repositoryUrl: string,
	temporaryDirectory: string,
	consume: (outputDirectory: string) => Promise<T>,
	options: DownloadOptions = {}
): Promise<T> {
	const repository = parseGitHubRepositoryUrl(repositoryUrl)
	await mkdir(temporaryDirectory, { recursive: true })
	const outputDirectory = await mkdtemp(join(temporaryDirectory, 'ouroboros-plugin-'))

	try {
		await downloadLatestPluginRelease(repository, outputDirectory, options)
		return await consume(outputDirectory)
	} finally {
		await rm(outputDirectory, { recursive: true, force: true })
	}
}

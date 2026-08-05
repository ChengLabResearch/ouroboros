import { constants, existsSync } from 'node:fs'
import { copyFile, mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { gitOutput } from './lib/git.mjs'
import {
	RELEASE_PLATFORMS,
	loadReleaseLock,
	releaseFingerprint,
	sha256File
} from './lib/release-lock.mjs'

const releaseAssetPattern =
	/(?:\.exe|\.zip|\.dmg|\.AppImage|\.snap|\.deb|\.rpm|\.tar\.gz|\.blockmap)$/

export function isReleaseAssetName(name) {
	return releaseAssetPattern.test(name) || (/^latest/.test(name) && name.endsWith('.yml'))
}

export function parsePlatform(args) {
	const index = args.indexOf('--platform')
	const platform = index === -1 ? null : args[index + 1]
	if (!RELEASE_PLATFORMS.includes(platform)) {
		throw new Error(
			`--platform must be one of ${RELEASE_PLATFORMS.join(', ')}; received ${platform}`
		)
	}
	return platform
}

export async function buildReleaseArtifacts({
	root = process.cwd(),
	platform = parsePlatform(process.argv.slice(2))
} = {}) {
	const startedAt = new Date()
	const startedMs = performance.now()
	const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
	const { lock, sha256: releaseLockSha256 } = await loadReleaseLock(root)
	if (packageJson.version !== lock.releaseVersion) {
		throw new Error(
			`package.json version ${packageJson.version} does not match release lock version ${lock.releaseVersion}`
		)
	}

	const sourceCommit = process.env.GITHUB_SHA ?? (await gitOutput(root, ['rev-parse', 'HEAD']))
	const sourceTree = await gitOutput(root, ['rev-parse', 'HEAD^{tree}'])
	const inputFingerprint = releaseFingerprint({
		lockSha256: releaseLockSha256,
		sourceTree,
		releaseVersion: packageJson.version
	})
	const outputDir = resolve(root, process.env.OUROBOROS_RELEASE_ARTIFACT_DIR ?? 'release-assets')
	const distDir = join(root, 'dist')
	const packageMetrics = []
	const artifacts = []

	await rm(outputDir, { recursive: true, force: true })
	await mkdir(outputDir, { recursive: true })

	const compileStartedMs = performance.now()
	await run(npmCommand(), ['run', 'build'], { cwd: root })
	const compileDurationMs = Math.round(performance.now() - compileStartedMs)
	if (!existsSync(join(root, 'out'))) {
		throw new Error('Electron compilation completed without producing out/')
	}

	await run(process.execPath, [join(root, 'scripts', 'prepare-production-server.mjs')], {
		cwd: root
	})

	for (const flavor of lock.packageFlavors) {
		const packageStartedMs = performance.now()
		const environment = {
			...process.env,
			OUROBOROS_PACKAGE_FLAVOR: flavor
		}

		await run(process.execPath, [join(root, 'scripts', 'prepare-package-flavor.mjs')], {
			cwd: root,
			env: environment
		})
		await rm(distDir, { recursive: true, force: true })
		await run(
			npmCommand(),
			['exec', '--', 'electron-builder', `--${platform}`, '--config', '--publish', 'never'],
			{ cwd: root, env: environment }
		)
		await run(process.execPath, [join(root, 'scripts', 'rename-release-artifacts.mjs')], {
			cwd: root,
			env: environment
		})

		const flavorArtifacts = await collectFlavorArtifacts({
			root,
			distDir,
			outputDir,
			platform,
			flavor
		})
		artifacts.push(...flavorArtifacts)
		packageMetrics.push({
			flavor,
			durationMs: Math.round(performance.now() - packageStartedMs),
			artifactCount: flavorArtifacts.length,
			producedBytes: flavorArtifacts.reduce((total, artifact) => total + artifact.bytes, 0)
		})
	}

	artifacts.sort((left, right) => left.name.localeCompare(right.name))
	const finishedAt = new Date()
	const manifest = {
		schemaVersion: 1,
		releaseVersion: packageJson.version,
		platform,
		flavors: lock.packageFlavors,
		sourceCommit,
		certifiedCommit: sourceCommit,
		sourceTree,
		sourceRef: process.env.GITHUB_REF_NAME ?? null,
		certification: {
			mode: 'built',
			workflowRun: process.env.GITHUB_RUN_ID ?? null
		},
		releaseLockSha256,
		inputFingerprint,
		releaseLock: lock,
		artifacts,
		metrics: {
			startedAt: startedAt.toISOString(),
			finishedAt: finishedAt.toISOString(),
			runnerDurationMs: Math.round(performance.now() - startedMs),
			compileDurationMs,
			packageDurationMs: packageMetrics.reduce(
				(total, metric) => total + metric.durationMs,
				0
			),
			producedBytes: artifacts.reduce((total, artifact) => total + artifact.bytes, 0),
			operations: {
				electronCompilations: 1,
				packageOperations: lock.packageFlavors.length,
				serverWheelBuilds: 0,
				serverImageBuilds: 0
			},
			packages: packageMetrics
		}
	}

	await writeFile(
		join(outputDir, `release-build-${platform}.json`),
		`${JSON.stringify(manifest, null, 2)}\n`
	)
	console.log(
		`Built ${artifacts.length} ${platform} release files for ${lock.packageFlavors.join(', ')} after one Electron compilation.`
	)
	return manifest
}

async function collectFlavorArtifacts({ root, distDir, outputDir, platform, flavor }) {
	const collected = []
	const entries = await readdir(distDir, { withFileTypes: true })
	const releaseEntries = entries.filter(
		(entry) => entry.isFile() && isReleaseAssetName(entry.name)
	)
	if (releaseEntries.length === 0) {
		throw new Error(`electron-builder produced no release assets for ${platform}/${flavor}`)
	}

	for (const entry of releaseEntries) {
		collected.push(
			await copyAndDescribe({
				source: join(distDir, entry.name),
				destination: join(outputDir, entry.name),
				name: entry.name,
				flavor,
				type: 'installer'
			})
		)
	}

	const metadataName = `package-metadata-${platform}-${flavor}.json`
	collected.push(
		await copyAndDescribe({
			source: join(root, 'extra-resources', 'package-flavor.json'),
			destination: join(outputDir, metadataName),
			name: metadataName,
			flavor,
			type: 'package-metadata'
		})
	)
	return collected
}

async function copyAndDescribe({ source, destination, name, flavor, type }) {
	try {
		await copyFile(source, destination, constants.COPYFILE_EXCL)
	} catch (error) {
		if (error.code === 'EEXIST') {
			throw new Error(`Release artifact name collision: ${name}`)
		}
		throw error
	}
	const details = await stat(destination)
	return {
		name,
		flavor,
		type,
		bytes: details.size,
		sha256: await sha256File(destination)
	}
}

function npmCommand() {
	return process.platform === 'win32' ? 'npm.cmd' : 'npm'
}

async function run(command, args, options) {
	await new Promise((resolvePromise, reject) => {
		const child = spawn(command, args, { stdio: 'inherit', ...options })
		child.on('error', reject)
		child.on('close', (code) => {
			if (code === 0) resolvePromise()
			else reject(new Error(`${command} ${args.join(' ')} exited with code ${code}`))
		})
	})
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	await buildReleaseArtifacts()
}

import { constants, existsSync } from 'node:fs'
import { copyFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises'
import { spawn } from 'node:child_process'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { gitOutput } from './lib/git.mjs'
import { npmInvocation } from './lib/npm.mjs'
import { createPackageWorkspace } from './lib/package-workspace.mjs'
import { validatePackageConfiguration } from './preflight-release.mjs'
import { prepareProductionServer } from './prepare-production-server.mjs'
import { preparePackageFlavor } from './prepare-package-flavor.mjs'
import { renameReleaseArtifacts } from './rename-release-artifacts.mjs'
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
	const platform =
		index === -1 && args.includes('--dir')
			? { linux: 'linux', darwin: 'mac', win32: 'win' }[process.platform]
			: index === -1
				? null
				: args[index + 1]
	if (!RELEASE_PLATFORMS.includes(platform)) {
		throw new Error(
			`--platform must be one of ${RELEASE_PLATFORMS.join(', ')}; received ${platform}`
		)
	}
	return platform
}

export async function buildReleaseArtifacts({
	root = process.cwd(),
	platform = parsePlatform(process.argv.slice(2)),
	unpack = process.argv.includes('--dir'),
	tempRoot,
	artifactDir,
	pluginCacheDir,
	environment = process.env,
	runCommand = run,
	prepareFlavor = preparePackageFlavor
} = {}) {
	const startedAt = new Date()
	const startedMs = performance.now()
	const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
	const releaseLock = await loadReleaseLock(root)
	const { lock, sha256: releaseLockSha256 } = releaseLock
	if (packageJson.version !== lock.releaseVersion) {
		throw new Error(
			`package.json version ${packageJson.version} does not match release lock version ${lock.releaseVersion}`
		)
	}

	await validatePackageConfiguration(root)
	const flavors = environment.OUROBOROS_PACKAGE_FLAVOR
		? [environment.OUROBOROS_PACKAGE_FLAVOR]
		: lock.packageFlavors
	if (flavors.some((flavor) => !lock.packageFlavors.includes(flavor)))
		throw new Error('Requested package flavor is not in the release lock')
	const sourceCommit = environment.GITHUB_SHA ?? (await gitOutput(root, ['rev-parse', 'HEAD']))
	const sourceTree = await gitOutput(root, ['rev-parse', 'HEAD^{tree}'])
	const inputFingerprint = releaseFingerprint({
		lockSha256: releaseLockSha256,
		sourceTree,
		releaseVersion: packageJson.version
	})
	const workspace = await createPackageWorkspace({ root, tempRoot, artifactDir, pluginCacheDir })
	const outputDir = workspace.artifactDir
	console.log(`Packaging workspace: ${workspace.directory}`)
	console.log(`Release artifacts: ${outputDir}`)
	try {
		await workspace.stageSource()
		const buildEnvironment = {
			...environment,
			ELECTRON_BUILDER_CACHE: workspace.builderCacheDir
		}
		const packageMetrics = []
		const artifacts = []

		await mkdir(outputDir, { recursive: true })

		const compileStartedMs = performance.now()
		await runNpm(
			['run', 'build'],
			{ cwd: workspace.compiledDir, env: buildEnvironment },
			runCommand
		)
		const compileDurationMs = Math.round(performance.now() - compileStartedMs)
		if (!existsSync(join(workspace.compiledDir, 'out'))) {
			throw new Error('Electron compilation completed without producing out/')
		}

		for (const flavor of flavors) {
			const packageStartedMs = performance.now()
			const project = await workspace.stageFlavor(flavor)
			try {
				const flavorEnvironment = { ...buildEnvironment, OUROBOROS_PACKAGE_FLAVOR: flavor }
				const preparation = {
					project,
					packageJson,
					releaseLock,
					environment: flavorEnvironment
				}
				await prepareProductionServer(preparation)
				await prepareFlavor(preparation)
				await runNpm(
					[
						'exec',
						'--',
						'electron-builder',
						`--${platform}`,
						...(unpack ? ['--dir'] : []),
						'--config',
						`-c.directories.output=${project.distDir}`,
						`-c.electronDownload.cache=${workspace.electronCacheDir}`,
						'--publish',
						'never'
					],
					{ cwd: project.root, env: flavorEnvironment },
					runCommand
				)
				if (unpack) {
					console.log(`Unpacked ${flavor}: ${project.distDir}`)
					continue
				}
				await renameReleaseArtifacts({ project, packageJson, flavor })

				const flavorArtifacts = await collectFlavorArtifacts({
					root: project.root,
					distDir: project.distDir,
					outputDir,
					platform,
					flavor
				})
				artifacts.push(...flavorArtifacts)
				packageMetrics.push({
					flavor,
					durationMs: Math.round(performance.now() - packageStartedMs),
					artifactCount: flavorArtifacts.length,
					producedBytes: flavorArtifacts.reduce(
						(total, artifact) => total + artifact.bytes,
						0
					)
				})
			} finally {
				await project.cleanup({ keepOutput: unpack })
			}
		}

		if (unpack) return { platform, flavors }
		artifacts.sort((left, right) => left.name.localeCompare(right.name))
		const finishedAt = new Date()
		const manifest = {
			schemaVersion: 1,
			releaseVersion: packageJson.version,
			platform,
			flavors,
			sourceCommit,
			certifiedCommit: sourceCommit,
			sourceTree,
			sourceRef: environment.GITHUB_REF_NAME ?? null,
			certification: {
				mode: 'built',
				workflowRun: environment.GITHUB_RUN_ID ?? null
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
					packageOperations: flavors.length,
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
			`Built ${artifacts.length} ${platform} release files for ${flavors.join(', ')} after one Electron compilation.`
		)
		return manifest
	} finally {
		await workspace.cleanup({ keepArtifacts: true, keepOutputs: unpack })
	}
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

async function runNpm(args, options, runCommand) {
	const invocation = npmInvocation(args)
	await runCommand(invocation.command, invocation.args, options)
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

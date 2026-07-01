import { app, BrowserWindow, dialog } from 'electron'
import { existsSync } from 'fs'
import fs from 'fs/promises'
import { join } from 'path'
import { parsePluginPackageJSON, PluginPackageJSON } from './schemas'
import { downloadRelease } from '@terascope/fetch-github-release'
import { fetchFolderContents, readFile } from './helpers'
import {
	buildDockerCompose,
	checkDocker,
	startDockerCompose,
	stopDockerCompose
} from './servers/docker'
import { getPluginFileURL } from './servers/file-server'

export async function getPluginFolder(): Promise<string> {
	const userData = app.getPath('userData')
	const pluginFolder = join(userData, 'plugins')

	// Create the plugin folder if it doesn't exist
	await fs.mkdir(pluginFolder, { recursive: true })

	return pluginFolder
}

export function getPreinstalledPluginFolder(): string {
	return join(__dirname, '../../../extra-resources/preinstalled-plugins')
}

async function readPluginPackage(pluginFolder: string): Promise<PluginPackageJSON | string> {
	const pathToPackageJSON = join(pluginFolder, 'package.json')

	if (!existsSync(pathToPackageJSON)) {
		return `Plugin package.json not found: ${pathToPackageJSON}`
	}

	const packageJSON = await readFile({ folder: pluginFolder, name: 'package.json' })
	return parsePluginPackageJSON(packageJSON)
}

export async function installPreinstalledPlugins(
	pluginFolder: string,
	preinstalledPluginFolder = getPreinstalledPluginFolder()
): Promise<void> {
	if (!existsSync(preinstalledPluginFolder)) {
		console.info(`No preinstalled plugin folder found at ${preinstalledPluginFolder}`)
		return
	}

	const bundledEntries = await fs.readdir(preinstalledPluginFolder, { withFileTypes: true })
	for (const entry of bundledEntries) {
		if (!entry.isDirectory()) continue

		const bundledFolder = join(preinstalledPluginFolder, entry.name)
		const bundledPackage = await readPluginPackage(bundledFolder)
		if (typeof bundledPackage === 'string') {
			console.error(`Skipping bundled plugin ${entry.name}: ${bundledPackage}`)
			continue
		}

		const targetFolder = join(pluginFolder, bundledPackage.name)
		const installAction = await preinstalledPluginInstallAction(targetFolder, bundledPackage)

		if (installAction === 'skip') {
			console.info(
				`Skipping bundled plugin ${bundledPackage.name}; installed version is current or newer`
			)
			continue
		}

		if (installAction === 'skip-unversioned') {
			console.info(
				`Skipping bundled plugin ${bundledPackage.name}; existing install has no comparable version`
			)
			continue
		}

		await fs.rm(targetFolder, { recursive: true, force: true })
		await fs.cp(bundledFolder, targetFolder, { recursive: true })
		console.info(
			`${installAction === 'upgrade' ? 'Upgraded' : 'Installed'} bundled plugin ${bundledPackage.name} ${bundledPackage.version ?? ''}`.trim()
		)
	}
}

async function preinstalledPluginInstallAction(
	targetFolder: string,
	bundledPackage: PluginPackageJSON
): Promise<'install' | 'upgrade' | 'skip' | 'skip-unversioned'> {
	if (!existsSync(targetFolder)) {
		return 'install'
	}

	const installedPackage = await readPluginPackage(targetFolder)
	if (typeof installedPackage === 'string') {
		return 'skip-unversioned'
	}

	const versionComparison = comparePluginVersions(
		bundledPackage.version,
		installedPackage.version
	)
	if (versionComparison === null) {
		return 'skip-unversioned'
	}

	return versionComparison > 0 ? 'upgrade' : 'skip'
}

function comparePluginVersions(bundledVersion?: string, installedVersion?: string): number | null {
	if (!bundledVersion || !installedVersion) return null

	const bundledParts = versionParts(bundledVersion)
	const installedParts = versionParts(installedVersion)
	const length = Math.max(bundledParts.length, installedParts.length)

	for (let index = 0; index < length; index += 1) {
		const bundledPart = bundledParts[index] ?? 0
		const installedPart = installedParts[index] ?? 0
		if (bundledPart !== installedPart) return bundledPart - installedPart
	}

	return bundledVersion.localeCompare(installedVersion)
}

function versionParts(version: string): number[] {
	return version
		.split(/[.-]/)
		.map((part) => Number.parseInt(part, 10))
		.filter((part) => Number.isFinite(part))
}

export async function getPluginList(
	pluginFolder: string,
	includeJSON = false
): Promise<
	{ name: string; id: string; folder: string; folderName: string; json?: PluginPackageJSON }[]
> {
	const result = await fetchFolderContents(pluginFolder)
	const folders = result.files.filter((_, i) => result.isFolder[i])

	const pluginFolderContents: {
		name: string
		id: string
		folder: string
		folderName: string
		json?: PluginPackageJSON
	}[] = []

	for (const folder of folders) {
		const parentFolder = join(pluginFolder, folder)
		const pathToPackageJSON = join(parentFolder, 'package.json')

		// Check if the folder contains a package.json file
		if (!existsSync(pathToPackageJSON)) {
			continue
		}

		const packageJSON = await readFile({ folder: parentFolder, name: 'package.json' })

		// Check if the package.json file is valid
		const parsedJSON = parsePluginPackageJSON(packageJSON)

		if (typeof parsedJSON === 'string') {
			console.error(parsedJSON)
			continue
		}

		// Check if the index html file exists
		const pathToIndex = join(parentFolder, parsedJSON.index)

		if (!existsSync(pathToIndex)) {
			console.error(`Index html file not found: ${pathToIndex}`)
			continue
		}

		// Check if the Docker Compose exists
		if (parsedJSON.dockerCompose) {
			const pathToDockerCompose = join(parentFolder, parsedJSON.dockerCompose)

			if (!existsSync(pathToDockerCompose)) {
				console.error(`Dockerfile not found: ${pathToDockerCompose}`)
				continue
			}
		}

		if (includeJSON) {
			pluginFolderContents.push({
				name: parsedJSON.pluginName,
				id: parsedJSON.name,
				folder: parentFolder,
				folderName: folder,
				json: parsedJSON
			})
		} else {
			pluginFolderContents.push({
				name: parsedJSON.pluginName,
				id: parsedJSON.name,
				folder: parentFolder,
				folderName: folder
			})
		}
	}

	return pluginFolderContents
}

export async function sendPluginFolderContents(
	pluginWindow: BrowserWindow | null,
	pluginFolder: string
): Promise<void> {
	const pluginFolderContents = await getPluginList(pluginFolder)

	pluginWindow?.webContents.send('plugin-folder-contents', pluginFolderContents)
}

export async function deletePlugin(pluginFolder: string): Promise<void> {
	await fs.rm(pluginFolder, { recursive: true })
}

export async function addLocalPlugin(pluginFolder: string): Promise<void> {
	const pluginFolderPath = await getPluginFolder()

	// Get the package.json file
	const pathToPackageJSON = join(pluginFolder, 'package.json')

	// Check if the folder contains a package.json file
	if (!existsSync(pathToPackageJSON)) {
		return
	}

	const packageJSON = await readFile({ folder: pluginFolder, name: 'package.json' })

	// Check if the package.json file is valid
	const parsedJSON = parsePluginPackageJSON(packageJSON)

	if (typeof parsedJSON === 'string') {
		console.error(parsedJSON)
		return
	}

	const pluginFolderName = parsedJSON.name

	// Create the plugin folder if it doesn't exist
	const targetFolder = join(pluginFolderPath, pluginFolderName)
	await fs.mkdir(targetFolder, { recursive: true })

	// Copy the folder from the given path to the plugin folder
	await fs.cp(pluginFolder, targetFolder, { recursive: true })

	// Build the docker container for the plugin
	if (parsedJSON.dockerCompose) {
		await buildDockerCompose({
			cwd: targetFolder,
			config: join(targetFolder, parsedJSON.dockerCompose),
			onError: (err) => {
				console.error(
					`An error occurred while building plugin ${parsedJSON.pluginName}'s Dockerfile:`,
					err
				)
			}
		})
	}
}

/**
 * Downloads the plugin from the given github releases URL
 */
export async function downloadPlugin(url: string): Promise<void> {
	// Make sure the URL is a github repository
	const isGithub = url.includes('github.com')

	if (!isGithub) {
		console.error('URL is not a github repository URL')
		return
	}

	// Get the user and repo from the URL
	const user = url.split('/')[3]
	const repo = url.split('/')[4]

	const outputDir = join(app.getPath('temp'), `${user}-${repo}`)

	// Make sure the output directory is empty
	if (existsSync(outputDir)) {
		await fs.rm(outputDir, { recursive: true })
	}

	const leaveZipped = false
	const disableLogging = false

	try {
		await downloadRelease(
			user,
			repo,
			outputDir,
			() => true,
			() => true,
			leaveZipped,
			disableLogging
		)
	} catch (error) {
		console.error(error)
		return
	}

	if (existsSync(outputDir)) {
		// Add the plugin to the local plugins
		await addLocalPlugin(outputDir)

		// Delete the downloaded release
		await fs.rm(outputDir, { recursive: true })
	}
}

export type PluginDetail = {
	id: string
	name: string
	indexPath: string
	iconPath?: string
}

export async function startAllPlugins(): Promise<PluginDetail[]> {
	const pluginFolder = await getPluginFolder()
	await installPreinstalledPlugins(pluginFolder)
	const plugins = await getPluginList(pluginFolder, true)

	const dockerCheck = await checkDocker()

	if (!dockerCheck.available) {
		console.error(dockerCheck.error)

		dialog.showErrorBox(
			'Docker Not Found',
			'Docker was not found on your system. Start Docker if it is installed, or download it.'
		)
	}

	const pluginDetails: PluginDetail[] = []

	plugins.forEach(async (plugin) => {
		const json = plugin.json

		if (!json) return

		// Try to start the docker container
		if (json.dockerCompose && dockerCheck.available) {
			startDockerCompose({
				cwd: join(plugin.folder),
				config: join(plugin.folder, json.dockerCompose),
				onError: (err) => {
					console.log(
						`An error occurred while starting plugin ${json.pluginName}'s Dockerfile:`,
						err
					)
				}
			})
		}

		const localPluginDetails: PluginDetail = {
			id: json.name,
			name: json.pluginName,
			indexPath: getPluginFileURL(plugin.folderName, json.index)
		}

		// Get the icons
		if (json.icon) {
			const pathToIcon = getPluginFileURL(plugin.folderName, json.icon)
			localPluginDetails.iconPath = pathToIcon
		}

		pluginDetails.push(localPluginDetails)
	})

	return pluginDetails
}

export async function initializePlugins(
	pluginDetails: PluginDetail[],
	mainWindow: BrowserWindow
): Promise<void> {
	startAllPlugins()
		.then((result) => {
			// Clear the plugin paths
			pluginDetails.length = 0

			// Add the plugin paths to the pluginPaths array
			pluginDetails.push(...result)
		})
		.then(() => {
			// Send the plugin paths to the renderer
			mainWindow.webContents.send('plugin-paths', pluginDetails)
		})
}

export async function restartPlugins(
	pluginDetails: PluginDetail[],
	mainWindow: BrowserWindow
): Promise<void> {
	await stopAllPlugins()

	initializePlugins(pluginDetails, mainWindow)
}

export async function stopAllPlugins(): Promise<void> {
	const pluginFolder = await getPluginFolder()
	const plugins = await getPluginList(pluginFolder, true)

	// Stop all plugins at the same time
	await Promise.all(
		plugins.map(async (plugin) => {
			const json = plugin.json

			if (!json) return

			// Try to stop the docker container
			if (json.dockerCompose) {
				try {
					await stopDockerCompose({
						cwd: join(plugin.folder),
						config: join(plugin.folder, json.dockerCompose)
					})
				} catch (err) {
					console.log(
						`An error occurred while stopping plugin ${json.pluginName}'s Dockerfile:`,
						err
					)
				}
			}
		})
	)
}

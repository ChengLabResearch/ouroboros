import { join } from 'path'
import { getPluginFolder } from '../plugins'
import { ChildProcess, fork } from 'child_process'
import { scope } from '../logging'

let pluginFileServer: ChildProcess
// Env override: OUROBOROS_PLUGIN_FILE_SERVER_PORT. Falls back to 3000 when unset,
// empty, or non-numeric (parseInt returns NaN, which || treats as falsy).
const PLUGIN_FILE_SERVER_PORT: number =
	parseInt(process.env.OUROBOROS_PLUGIN_FILE_SERVER_PORT ?? '', 10) || 3000
const pluginFileServerURL: string = `http://127.0.0.1:${PLUGIN_FILE_SERVER_PORT}`

const logger = scope('file-server')

/**
 * Starts a file server to serve plugin files
 */
export async function startPluginFileServer(): Promise<void> {
	const pluginFolder = await getPluginFolder()

	pluginFileServer = fork(join(__dirname, '../../resources/processes/file-server-script.mjs'), [
		pluginFolder,
		`${PLUGIN_FILE_SERVER_PORT}`
	])

	pluginFileServer.on('error', (error) => {
		logger.error(error)
	})
}

export async function stopPluginFileServer(): Promise<void> {
	pluginFileServer.kill()
}

export function getPluginFileServerURL(): string {
	return pluginFileServerURL
}

export function getPluginFileURL(pluginFolder: string, fileRelativePath: string): string {
	const filePath = join(pluginFolder, fileRelativePath)

	const url = new URL(filePath, getPluginFileServerURL())

	return url.toString()
}

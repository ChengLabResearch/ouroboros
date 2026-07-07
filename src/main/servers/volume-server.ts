import { join } from 'path'
import { ChildProcess, fork } from 'child_process'
import { scope } from '../logging'

let volumeServer: ChildProcess
// Env override: OUROBOROS_VOLUME_SERVER_PORT. Falls back to 3001 when unset,
// empty, or non-numeric (parseInt returns NaN, which || treats as falsy).
const VOLUME_SERVER_PORT: number =
	parseInt(process.env.OUROBOROS_VOLUME_SERVER_PORT ?? '', 10) || 3001
const volumeServerURL: string = `http://127.0.0.1:${VOLUME_SERVER_PORT}`

const VOLUME = 'ouroboros-volume'

const logger = scope('volume-server')

/**
 * Starts a serve that facilitates copying files from the host
 * file system to docker volumes and vice versa.
 */
export async function startVolumeServer(): Promise<void> {
	volumeServer = fork(join(__dirname, '../../resources/processes/volume-server-script.mjs'), [
		`${VOLUME_SERVER_PORT}`
	])

	volumeServer.on('error', (error) => {
		logger.error(error)
	})
}

export async function stopVolumeServer(): Promise<void> {
	// Remove all files from the volume
	await clearVolume({ volumeName: VOLUME })

	volumeServer.kill()
}

export function getVolumeServerURL(): string {
	return volumeServerURL
}

export type CopyToVolumeData = {
	volumeName: string
	pluginFolderName: string
	files: { sourcePath: string; targetPath: string }[]
}

export type CopyToHostData = {
	volumeName: string
	pluginFolderName: string
	files: { sourcePath: string; targetPath: string }[]
}

export type ClearPluginFolderData = {
	volumeName: string
	pluginFolderName: string
}

export type ClearVolumeData = {
	volumeName: string
}

export async function copyToVolume(data: CopyToVolumeData): Promise<[boolean, string]> {
	return requestVolumeServer('copy-to-volume', data)
}

export async function copyToHost(data: CopyToHostData): Promise<[boolean, string]> {
	return requestVolumeServer('copy-to-host', data)
}

export async function clearPluginFolder(data: ClearPluginFolderData): Promise<[boolean, string]> {
	return requestVolumeServer('clear-plugin-folder', data)
}

export async function clearVolume(data: ClearVolumeData): Promise<[boolean, string]> {
	return requestVolumeServer('clear-volume', data)
}

export async function requestVolumeServer(path: string, data: object): Promise<[boolean, string]> {
	const url = `${volumeServerURL}/${path}`

	try {
		const result = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify(data)
		})

		if (!result.ok) {
			return [false, `${result.statusText}`]
		} else {
			return [true, '']
		}
	} catch (error) {
		return [false, `${error}`]
	}
}

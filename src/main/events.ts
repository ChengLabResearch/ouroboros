import { BrowserWindow, ipcMain } from 'electron'
import { is } from '@electron-toolkit/utils'
import { PluginDetail } from './plugins'
import { addFSEventHandlers } from './event-handlers/filesystem'
import { addPluginEventHandlers } from './event-handlers/plugins'
import { getMainServerLaunchStatus, onMainServerLaunchStatus } from './servers/main-server'
import {
	MAIN_SERVER_LAUNCH_STATUS_CHANNEL,
	MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL
} from '../shared/main-server-status'

export function handleEvents({
	getMainWindow,
	getPluginDetails
}: {
	getMainWindow: () => BrowserWindow
	getPluginDetails: () => PluginDetail[]
}): void {
	// Add filesystem event handlers
	addFSEventHandlers(ipcMain, getMainWindow)

	ipcMain.on('get-is-dev', () => {
		getMainWindow()?.webContents.send('is-dev', is.dev)
	})

	// Add plugin event handlers
	addPluginEventHandlers(ipcMain, getPluginDetails, getMainWindow)

	// A snapshot also serves renderers that connect after startup has finished.
	ipcMain.handle(MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL, getMainServerLaunchStatus)
	onMainServerLaunchStatus((status) => {
		const window = getMainWindow()
		if (window && !window.isDestroyed() && !window.webContents.isDestroyed()) {
			window.webContents.send(MAIN_SERVER_LAUNCH_STATUS_CHANNEL, status)
		}
	})
}

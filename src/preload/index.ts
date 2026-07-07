import { contextBridge, webUtils } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// The file explorer uses these renderer -> main IPC channels (handled in
// `src/main/event-handlers/filesystem.ts`):
//   - `fetch-folder-contents(root)`    open a root, close all watchers
//   - `expand-folder(subPath)`         start a non-recursive watcher lazily
//   - `collapse-folder(subPath)`       schedule watcher teardown (see
//                                      COLLAPSE_TEARDOWN_DELAY_MS in
//                                      `src/shared/constants.ts`)
//   - `get-folder-contents({ folderPath, recursive, rootPath })`
//                                      one-shot enumeration used by the
//                                      plugin-facing `request-directory-
//                                      contents` message; does not touch
//                                      the watcher manager.
// `electronAPI.ipcRenderer.invoke` accepts arbitrary channel strings, so no
// per-channel type is required here.

// Custom APIs for renderer
const api = {
  getFilePath: (file: File): string => {
    return webUtils.getPathForFile(file)
  }
}
// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
	try {
		// https://www.npmjs.com/package/@electron-toolkit/preload
		contextBridge.exposeInMainWorld('electron', electronAPI)
		contextBridge.exposeInMainWorld('api', api)
	} catch (error) {
		console.error(error)
	}
} else {
	// @ts-ignore (define in dts)
	window.electron = electronAPI
	// @ts-ignore (define in dts)
	window.api = api
}

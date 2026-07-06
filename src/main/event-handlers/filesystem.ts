import { BrowserWindow, IpcMain } from 'electron'
import { rename, rm, stat } from 'fs/promises'
import { basename, join, relative, sep } from 'path'
import Watcher from 'watcher'

import { readFile, saveFile } from '../helpers'

const FILE_EXPLORER_WATCH_DEPTH = 6
const FILE_EXPLORER_WATCH_LIMIT = 100_000
const IGNORED_PATH_SEGMENTS = new Set(['node_modules', '__pycache__', 'venv'])

type FSEvent = {
	event: string
	directoryPath: string
	targetPath: string
	targetPathNext: string
	isDirectory: boolean
	pathParts: string[]
	nextPathParts: string[]
	relativePath: string
	relativePathNext: string
	separator: string
}

type FSError = {
	directoryPath: string
	message: string
	code?: string
	depth: number
	limit: number
}

export const addFSEventHandlers = (ipcMain: IpcMain, getMainWindow: () => BrowserWindow): void => {
	let subscription: Watcher | null = null

	const sendFSError = (error: FSError): void => {
		const mainWindow = getMainWindow?.()

		if (mainWindow && !mainWindow.isDestroyed()) {
			mainWindow.webContents.send('folder-contents-error', error)
		}
	}

	// Fetch the contents of the given folder
	ipcMain.handle('fetch-folder-contents', async (_, folderPath: string) => {
		if (folderPath === '' || folderPath === undefined || folderPath === null) return

		if (subscription) {
			subscription.close()
			subscription = null
		}

		const shouldIgnorePath = (targetPath: string): boolean => {
			const targetRelativePath = relative(folderPath, targetPath)

			if (!targetRelativePath || targetRelativePath === '') return false
			if (targetRelativePath.startsWith('..')) return false

			const pathSegments = targetRelativePath.split(/[\\/]/)
			return pathSegments.some((segment) => {
				if (segment.startsWith('.')) return true
				return IGNORED_PATH_SEGMENTS.has(segment)
			})
		}

		let visibleAddCount = 0
		let sentLimitWarning = false

		const sendLimitWarning = (): void => {
			if (sentLimitWarning) return

			sentLimitWarning = true
			sendFSError({
				directoryPath: folderPath,
				message: `File explorer stopped loading after ${FILE_EXPLORER_WATCH_LIMIT} visible paths. Choose a smaller folder or expand the folder in smaller pieces.`,
				depth: FILE_EXPLORER_WATCH_DEPTH,
				limit: FILE_EXPLORER_WATCH_LIMIT
			})
		}

		// Send updates to the renderer when the folder contents change
		subscription = new Watcher(folderPath, {
			recursive: true,
			renameDetection: true,
			depth: FILE_EXPLORER_WATCH_DEPTH,
			limit: FILE_EXPLORER_WATCH_LIMIT,
			ignore: shouldIgnorePath
		})

		subscription.on('error', (error) => {
			sendFSError({
				directoryPath: folderPath,
				message: error instanceof Error ? error.message : 'Unknown file explorer watcher error',
				code:
					error instanceof Error && 'code' in error && typeof error.code === 'string'
						? error.code
						: undefined,
				depth: FILE_EXPLORER_WATCH_DEPTH,
				limit: FILE_EXPLORER_WATCH_LIMIT
			})
		})

		subscription.on('all', async (event, targetPath, targetPathNext) => {
			try {
				if (shouldIgnorePath(targetPath)) return

				let isDirectory = false

				const isRename = (event === 'rename' || event === 'renameDir') && targetPathNext
				const isAddLike = event === 'add' || event === 'change' || event === 'addDir'

				// Check if the target path is a directory
				if (isRename) {
					const stats = await stat(targetPathNext)
					isDirectory = stats.isDirectory()
				} else if (isAddLike) {
					const stats = await stat(targetPath)
					isDirectory = stats.isDirectory()
				}

				if (event === 'add' || event === 'addDir') {
					visibleAddCount++
					if (visibleAddCount >= FILE_EXPLORER_WATCH_LIMIT) {
						sendLimitWarning()
					}
				}

				// eslint-disable-next-line no-useless-escape
				const relativePath = targetPath.replace(folderPath, '').replace(/^[\/\\]/, '')

				let relativePathNext = ''
				if (targetPathNext) {
					// eslint-disable-next-line no-useless-escape
					relativePathNext = targetPathNext.replace(folderPath, '').replace(/^[\/\\]/, '')
				}

				// Make sure the path is not the root folder
				if (relativePath === '' || relativePath.length === 0) return

				// Split the path by the separator
				const pathParts = relativePath.split(sep)
				const nextPathParts = relativePathNext.split(sep)

				const fsEvent: FSEvent = {
					directoryPath: folderPath,
					event,
					targetPath,
					targetPathNext: targetPathNext ?? '',
					isDirectory,
					pathParts,
					nextPathParts,
					relativePath,
					relativePathNext,
					separator: sep
				}

				if (getMainWindow && getMainWindow()) {
					getMainWindow().webContents.send('folder-contents-update', fsEvent)
				}
			} catch (error) {
				sendFSError({
					directoryPath: folderPath,
					message: error instanceof Error ? error.message : 'Unknown file explorer update error',
					code:
						error instanceof Error && 'code' in error && typeof error.code === 'string'
							? error.code
							: undefined,
					depth: FILE_EXPLORER_WATCH_DEPTH,
					limit: FILE_EXPLORER_WATCH_LIMIT
				})
			}
		})
	})

	// Save a string to a file
	ipcMain.handle('save-file', async (_, args) => {
		return await saveFile(args)
	})

	// Join two paths
	ipcMain.handle('join-path', (_, args: string[]) => {
		return join(...args)
	})

	// Get base name of a path
	ipcMain.handle('basename-path', (_, { folder }) => {
		return basename(folder)
	})

	// Read the contents of a file as a string
	ipcMain.handle('read-file', async (_, args) => {
		return await readFile(args)
	})

	// Delete a file or folder
	ipcMain.handle('delete-fs-item', async (_, path: string) => {
		try {
			await rm(path, {
				recursive: true
			})
		} catch (error) {
			console.error(error)
		}
	})

	// Rename a file or folder
	ipcMain.handle('rename-fs-item', async (_, { oldPath, newPath }) => {
		try {
			await rename(oldPath, newPath)
		} catch (error) {
			console.error(error)
		}
	})
}

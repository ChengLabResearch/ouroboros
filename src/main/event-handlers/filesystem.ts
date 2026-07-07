import { BrowserWindow, IpcMain } from 'electron'
import { rename, rm, stat } from 'fs/promises'
import { basename, join, relative, sep } from 'path'
import Watcher from 'watcher'

import { readFile, saveFile } from '../helpers'
import { COLLAPSE_TEARDOWN_DELAY_MS } from '../../shared/constants'

const FILE_EXPLORER_WATCH_LIMIT = 100_000
const FILE_EXPLORER_UPDATE_BATCH_LIMIT = 500
const FILE_EXPLORER_UPDATE_INTERVAL_MS = 100
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
	limit: number
}

type WatcherEntry = {
	watcher: Watcher
	addCount: number
	pendingTeardownTimer: NodeJS.Timeout | null
}

export const addFSEventHandlers = (ipcMain: IpcMain, getMainWindow: () => BrowserWindow): void => {
	// One non-recursive watcher per currently expanded directory. The root
	// watcher lives here too - root is just a distinguished expanded folder.
	const watchers = new Map<string, WatcherEntry>()

	// Aggregate visible-add count across every open watcher in this session.
	// `FILE_EXPLORER_WATCH_LIMIT` is a session budget, not a per-watcher one.
	let visibleAddCount = 0
	let sentLimitWarning = false

	// The currently open root. `collapse-folder(root)` is a no-op so we can
	// guard against nonsensical collapse-of-root requests.
	let currentRoot: string | null = null

	let pendingFSEvents: FSEvent[] = []
	let pendingFSEventsTimeout: NodeJS.Timeout | null = null

	const sendFSError = (error: FSError): void => {
		const mainWindow = getMainWindow?.()

		if (mainWindow && !mainWindow.isDestroyed()) {
			mainWindow.webContents.send('folder-contents-error', error)
		}
	}

	const flushPendingFSEvents = (): void => {
		if (pendingFSEventsTimeout) {
			clearTimeout(pendingFSEventsTimeout)
			pendingFSEventsTimeout = null
		}

		if (pendingFSEvents.length === 0) return

		const mainWindow = getMainWindow?.()
		const batch = pendingFSEvents
		pendingFSEvents = []

		if (mainWindow && !mainWindow.isDestroyed()) {
			mainWindow.webContents.send('folder-contents-update-batch', batch)
		}
	}

	const clearPendingFSEvents = (): void => {
		if (pendingFSEventsTimeout) {
			clearTimeout(pendingFSEventsTimeout)
			pendingFSEventsTimeout = null
		}

		pendingFSEvents = []
	}

	const queueFSEvent = (fsEvent: FSEvent): void => {
		pendingFSEvents.push(fsEvent)

		if (pendingFSEvents.length >= FILE_EXPLORER_UPDATE_BATCH_LIMIT) {
			flushPendingFSEvents()
			return
		}

		if (!pendingFSEventsTimeout) {
			pendingFSEventsTimeout = setTimeout(
				flushPendingFSEvents,
				FILE_EXPLORER_UPDATE_INTERVAL_MS
			)
		}
	}

	const sendLimitWarning = (folderPath: string): void => {
		if (sentLimitWarning) return

		sentLimitWarning = true
		sendFSError({
			directoryPath: folderPath,
			message: `File explorer stopped loading after ${FILE_EXPLORER_WATCH_LIMIT} visible paths across all open folders. Choose a smaller folder or expand the folder in smaller pieces.`,
			limit: FILE_EXPLORER_WATCH_LIMIT
		})
	}

	// Close every open watcher and reset all associated state. Used when the
	// root changes and on window teardown / app quit.
	const closeAllWatchers = (): void => {
		for (const entry of watchers.values()) {
			if (entry.pendingTeardownTimer) {
				clearTimeout(entry.pendingTeardownTimer)
				entry.pendingTeardownTimer = null
			}
			entry.watcher.close()
		}
		watchers.clear()
		visibleAddCount = 0
		sentLimitWarning = false
	}

	// Tear down a single watcher entry and reclaim its contribution to the
	// aggregate visible-add counter.
	const teardownWatcher = (subPath: string): void => {
		const entry = watchers.get(subPath)
		if (!entry) return

		if (entry.pendingTeardownTimer) {
			clearTimeout(entry.pendingTeardownTimer)
			entry.pendingTeardownTimer = null
		}

		entry.watcher.close()
		visibleAddCount = Math.max(0, visibleAddCount - entry.addCount)
		watchers.delete(subPath)
	}

	// Start a non-recursive watcher on `subPath`. `rootPath` is the root the
	// renderer treats as the top of its tree; every batched event's relative
	// path is computed against it so the renderer's nested `nodes` map lines
	// up regardless of which folder produced the event.
	const startWatcher = (rootPath: string, subPath: string): void => {
		if (watchers.has(subPath)) return

		const shouldIgnorePath = (targetPath: string): boolean => {
			const targetRelativePath = relative(rootPath, targetPath)

			if (!targetRelativePath || targetRelativePath === '') return false
			if (targetRelativePath.startsWith('..')) return false

			const pathSegments = targetRelativePath.split(/[\\/]/)
			return pathSegments.some((segment) => {
				if (segment.startsWith('.')) return true
				return IGNORED_PATH_SEGMENTS.has(segment)
			})
		}

		const watcher = new Watcher(subPath, {
			recursive: false,
			renameDetection: true,
			ignore: shouldIgnorePath
		})

		const entry: WatcherEntry = {
			watcher,
			addCount: 0,
			pendingTeardownTimer: null
		}
		watchers.set(subPath, entry)

		watcher.on('error', (error) => {
			sendFSError({
				directoryPath: subPath,
				message: error instanceof Error ? error.message : 'Unknown file explorer watcher error',
				code:
					error instanceof Error && 'code' in error && typeof error.code === 'string'
						? error.code
						: undefined,
				limit: FILE_EXPLORER_WATCH_LIMIT
			})
		})

		watcher.on('all', async (event, targetPath, targetPathNext) => {
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
					entry.addCount++
					visibleAddCount++
					if (visibleAddCount >= FILE_EXPLORER_WATCH_LIMIT) {
						sendLimitWarning(rootPath)
					}
				}

				if (event === 'unlink' || event === 'unlinkDir') {
					entry.addCount = Math.max(0, entry.addCount - 1)
					visibleAddCount = Math.max(0, visibleAddCount - 1)
				}

				// eslint-disable-next-line no-useless-escape
				const relativePath = targetPath.replace(rootPath, '').replace(/^[\/\\]/, '')

				let relativePathNext = ''
				if (targetPathNext) {
					// eslint-disable-next-line no-useless-escape
					relativePathNext = targetPathNext.replace(rootPath, '').replace(/^[\/\\]/, '')
				}

				// Make sure the path is not the root folder
				if (relativePath === '' || relativePath.length === 0) return

				// Split the path by the separator
				const pathParts = relativePath.split(sep)
				const nextPathParts = relativePathNext.split(sep)

				const fsEvent: FSEvent = {
					directoryPath: rootPath,
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

				queueFSEvent(fsEvent)
			} catch (error) {
				sendFSError({
					directoryPath: subPath,
					message: error instanceof Error ? error.message : 'Unknown file explorer update error',
					code:
						error instanceof Error && 'code' in error && typeof error.code === 'string'
							? error.code
							: undefined,
					limit: FILE_EXPLORER_WATCH_LIMIT
				})
			}
		})
	}

	// The main window does not exist at handler-registration time (see
	// `handleEvents` runs before `createWindow` in `src/main/index.ts`), so
	// we lazily bind the teardown listener once the window becomes available.
	let boundTeardownWindow: BrowserWindow | null = null
	const bindTeardownIfNeeded = (): void => {
		const mainWindow = getMainWindow?.()
		if (!mainWindow || boundTeardownWindow === mainWindow) return
		boundTeardownWindow = mainWindow
		mainWindow.on('closed', () => {
			clearPendingFSEvents()
			closeAllWatchers()
			currentRoot = null
			boundTeardownWindow = null
		})
	}

	// Fetch the contents of the given folder. Also serves as the entry point
	// for root loads: closes any existing watchers, resets the aggregate
	// counter, and delegates to the same lazy-expand path used for subfolders.
	ipcMain.handle('fetch-folder-contents', async (_, folderPath: string) => {
		if (folderPath === '' || folderPath === undefined || folderPath === null) return

		bindTeardownIfNeeded()

		clearPendingFSEvents()
		closeAllWatchers()

		currentRoot = folderPath

		startWatcher(folderPath, folderPath)
	})

	// Lazily expand a subfolder. Open-close-open within the teardown delay is
	// idempotent: it cancels the pending teardown rather than churning the
	// watcher.
	ipcMain.handle('expand-folder', async (_, subPath: string) => {
		if (!subPath || !currentRoot) return

		const existing = watchers.get(subPath)
		if (existing) {
			if (existing.pendingTeardownTimer) {
				clearTimeout(existing.pendingTeardownTimer)
				existing.pendingTeardownTimer = null
			}
			return
		}

		startWatcher(currentRoot, subPath)
	})

	// Lazily collapse a subfolder. The watcher stays alive for
	// COLLAPSE_TEARDOWN_DELAY_MS so quick re-expand costs nothing; only after
	// the delay elapses does the watcher close and its contribution to the
	// aggregate visible-add counter get reclaimed. Collapsing the root is a
	// no-op - root's watcher only closes when `fetch-folder-contents` is
	// called again or on app teardown.
	ipcMain.handle('collapse-folder', async (_, subPath: string) => {
		if (!subPath) return

		if (currentRoot && subPath === currentRoot) {
			// Guard against nonsensical collapse-of-root - see design in #110.
			console.debug('collapse-folder called on root; ignoring', subPath)
			return
		}

		const entry = watchers.get(subPath)
		if (!entry) return

		if (entry.pendingTeardownTimer) {
			// Already scheduled; keep the existing timer rather than resetting
			// the countdown.
			return
		}

		entry.pendingTeardownTimer = setTimeout(() => {
			teardownWatcher(subPath)
		}, COLLAPSE_TEARDOWN_DELAY_MS)
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

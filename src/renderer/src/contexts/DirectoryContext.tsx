import { joinWithSeparator } from '@renderer/interfaces/file'
import { AlertContext } from '@renderer/contexts/AlertContext'
import { JSX, createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { COLLAPSE_TEARDOWN_DELAY_MS } from '../../../shared/constants'

export type DirectoryContextValue = {
	nodes: NodeChildren
	directoryPath: string | null
	directoryName: string | null
	setDirectory: (directory: string) => void
	expandFolder: (subPath: string) => void
	collapseFolder: (subPath: string) => void
}

export const DirectoryContext = createContext<DirectoryContextValue>(null as never)

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

export type FileSystemNode = {
	name: string
	path: string
	children?: NodeChildren
}

export type NodeChildren = {
	[key: string]: FileSystemNode
}

function DirectoryProvider({ children }: { children: React.ReactNode }): JSX.Element {
	const [directoryName, setDirectoryName] = useState<string | null>(null)
	const [directoryPath, setDirectoryPath] = useState<string | null>(null)
	const [nodes, setNodes] = useState<NodeChildren>({})
	const { addAlert } = useContext(AlertContext)

	// Mirror the main-side collapse teardown: renderer-side state for a
	// collapsed subfolder is pruned on the same delay. Re-expanding within
	// the window cancels both timers so the existing slice stays.
	const pendingCollapseTimers = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())

	const setDirectory = useCallback(
		(directory: string): void => {
			// Make sure the directory is not empty
			if (!directory || directory.length === 0) return

			// Make sure the directory is not the same as the current directory
			if (directory === directoryPath) return

			setDirectoryPath(directory)

			// Clean up the directory name
			const directorySplit = directory.split(/[/\\]/)
			setDirectoryName(directorySplit[directorySplit.length - 1])

			// Clear the folder contents and any pending collapse timers -
			// they refer to paths under the old root.
			for (const timer of pendingCollapseTimers.current.values()) {
				clearTimeout(timer)
			}
			pendingCollapseTimers.current.clear()
			setNodes({})
		},
		[directoryPath]
	)

	const expandFolder = useCallback(
		(subPath: string): void => {
			if (!subPath) return

			// Cancel a pending renderer-side prune if we re-expand in time.
			const timer = pendingCollapseTimers.current.get(subPath)
			if (timer) {
				clearTimeout(timer)
				pendingCollapseTimers.current.delete(subPath)
			}

			window.electron.ipcRenderer.invoke('expand-folder', subPath)
		},
		[]
	)

	const collapseFolder = useCallback(
		(subPath: string): void => {
			if (!subPath) return

			// If a prune is already scheduled for this path, do not restart
			// the timer.
			if (pendingCollapseTimers.current.has(subPath)) return

			window.electron.ipcRenderer.invoke('collapse-folder', subPath)

			const timer = setTimeout(() => {
				pendingCollapseTimers.current.delete(subPath)
				setNodes((prev) => pruneSubtree(prev, subPath))
			}, COLLAPSE_TEARDOWN_DELAY_MS)
			pendingCollapseTimers.current.set(subPath, timer)
		},
		[]
	)

	useEffect(() => {
		const clearSelectedFolderListener = window.electron.ipcRenderer.on(
			'selected-folder',
			(_, directory) => {
				setDirectory(directory)
			}
		)

		const clearFolderUpdateListener = window.electron.ipcRenderer.on(
			'folder-contents-update',
			async (_, fsEvent: FSEvent) => {
				setNodes((prev) => handleFSEvent(prev, fsEvent))
			}
		)

		const clearFolderUpdateBatchListener = window.electron.ipcRenderer.on(
			'folder-contents-update-batch',
			async (_, fsEvents: FSEvent[]) => {
				setNodes((prev) => handleFSEvents(prev, fsEvents))
			}
		)

		const clearFolderErrorListener = window.electron.ipcRenderer.on(
			'folder-contents-error',
			(_, fsError: FSError) => {
				console.error('File explorer watcher error', fsError)
				addAlert(fsError.message, 'warning')
			}
		)

		return (): void => {
			clearSelectedFolderListener()
			clearFolderUpdateListener()
			clearFolderUpdateBatchListener()
			clearFolderErrorListener()
		}
	}, [addAlert])

	const refreshDirectory = useCallback(() => {
		window.electron.ipcRenderer.invoke('fetch-folder-contents', directoryPath)
	}, [directoryPath])

	useEffect(() => {
		refreshDirectory()
	}, [directoryPath])

	// Clear any pending collapse timers on provider teardown.
	useEffect(() => {
		return (): void => {
			for (const timer of pendingCollapseTimers.current.values()) {
				clearTimeout(timer)
			}
			pendingCollapseTimers.current.clear()
		}
	}, [])

	return (
		<DirectoryContext.Provider
			value={{
				nodes,
				directoryPath,
				directoryName,
				setDirectory,
				expandFolder,
				collapseFolder
			}}
		>
			{children}
		</DirectoryContext.Provider>
	)
}

export default DirectoryProvider

function handleFSEvents(nodes: NodeChildren, fsEvents: FSEvent[]): NodeChildren {
	return fsEvents.reduce(handleFSEvent, nodes)
}

function handleFSEvent(nodes: NodeChildren, fsEvent: FSEvent): NodeChildren {
	if (!fsEvent) return nodes

	const {
		event,
		targetPath,
		targetPathNext,
		isDirectory,
		pathParts,
		nextPathParts,
		directoryPath,
		separator
	} = fsEvent

	if (event === 'change') return nodes

	// Create a copy of the nodes object
	const nodesCopy = { ...nodes }

	let parts = pathParts
	let currentPath = nodesCopy
	let path = targetPath

	let deletedPath: FileSystemNode | null = null

	const deletePath = (_currentPath: NodeChildren): FileSystemNode | null => {
		for (let i = 0; i < pathParts.length; i++) {
			const part = parts[i]

			if (i === parts.length - 1) {
				const deleted = _currentPath[part]
				delete _currentPath[part]
				return deleted
			}

			if (!_currentPath[part] || _currentPath[part].children === undefined) return null

			_currentPath = _currentPath[part].children!
		}

		return null
	}

	if (['rename', 'renameDir', 'unlink', 'unlinkDir'].includes(event)) {
		// Recursively delete the old path
		deletedPath = deletePath(nodesCopy)

		if (event === 'rename' || event === 'renameDir') {
			parts = nextPathParts
			path = targetPathNext
		} else return nodesCopy
	}

	// Add the path to the nodes object
	for (let i = 0; i < parts.length; i++) {
		const part = parts[i]

		// If this is the last part of the path, add the node
		if (i === parts.length - 1 && !currentPath[part]) {
			if (deletedPath) {
				// Rename the node
				currentPath[part] = deletedPath
				currentPath[part].name = part
				currentPath[part].path = path
			} else {
				// Create the node
				currentPath[part] = {
					name: part,
					path: path,
					children: isDirectory ? {} : undefined
				}
			}
		}

		// If the intermediate part does not exist, create it
		if (!currentPath[part]) {
			currentPath[part] = {
				name: part,
				path: joinWithSeparator(separator, directoryPath, ...parts.slice(0, i + 1)),
				children: {}
			}
		}

		// Move to the next part of the path
		currentPath = currentPath[part].children!
	}

	return nodesCopy
}

/**
 * Drop the children of the folder identified by `subPath` from the nested
 * `nodes` map, leaving the folder itself in place so its collapsed shell can
 * still render. The folder is located by walking the tree and matching each
 * node's absolute `path`; this keeps the reducer independent of the current
 * root because `path` is the absolute filesystem path already.
 */
export function pruneSubtree(nodes: NodeChildren, subPath: string): NodeChildren {
	if (!subPath) return nodes

	let mutated = false
	const walk = (children: NodeChildren): NodeChildren => {
		let next: NodeChildren | null = null
		for (const key of Object.keys(children)) {
			const node = children[key]
			if (node.path === subPath) {
				if (node.children && Object.keys(node.children).length > 0) {
					if (!next) next = { ...children }
					next[key] = { ...node, children: {} }
					mutated = true
				}
				continue
			}
			if (node.children) {
				const walked = walk(node.children)
				if (walked !== node.children) {
					if (!next) next = { ...children }
					next[key] = { ...node, children: walked }
				}
			}
		}
		return next ?? children
	}

	const result = walk(nodes)
	return mutated ? result : nodes
}

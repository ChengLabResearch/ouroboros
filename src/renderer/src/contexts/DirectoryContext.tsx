import { joinWithSeparator } from '@renderer/interfaces/file'
import { JSX, createContext, useCallback, useEffect, useState } from 'react'

export type DirectoryContextValue = {
	nodes: NodeChildren
	directoryPath: string | null
	directoryName: string | null
	setDirectory: (directory: string) => void
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

			// Clear the folder contents
			setNodes({})
		},
		[directoryPath]
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

		return (): void => {
			clearSelectedFolderListener()
			clearFolderUpdateListener()
		}
	}, [])

	const refreshDirectory = useCallback(() => {
		window.electron.ipcRenderer.invoke('fetch-folder-contents', directoryPath)
	}, [directoryPath])

	useEffect(() => {
		refreshDirectory()
	}, [directoryPath])

	return (
		<DirectoryContext.Provider value={{ nodes, directoryPath, directoryName, setDirectory }}>
			{children}
		</DirectoryContext.Provider>
	)
}

export default DirectoryProvider

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

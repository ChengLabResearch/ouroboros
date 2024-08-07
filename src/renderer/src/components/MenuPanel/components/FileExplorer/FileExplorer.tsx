import Header from '@renderer/components/Header/Header'
import styles from './FileExplorer.module.css'
import FileEntry from './components/FileEntry/FileEntry'
import { DragOverlay, useDroppable } from '@dnd-kit/core'
import { MouseEvent, useCallback, useContext, useEffect, useRef, useState } from 'react'
import DraggableEntry from './components/DraggableEntry/DraggableEntry'
import { DragContext } from '@renderer/contexts/DragContext'
import { DirectoryContext, FileSystemNode } from '@renderer/contexts/DirectoryContext'
import useContextMenu from '@renderer/hooks/use-context-menu'
import ContextMenu, { ContextMenuAction } from '@renderer/components/ContextMenu/ContextMenu'
import {
	deleteFSItem,
	join,
	moveFSItem,
	newFolder,
	renameFSItem,
	isPathFile,
	directory
} from '@renderer/interfaces/file'

function FileExplorer(): JSX.Element {
	const { clearDragEvent, parentChildData, active } = useContext(DragContext)
	const { nodes, directoryName, directoryPath, setDirectory } = useContext(DirectoryContext)
	const { point, clicked, data, handleContextMenu } = useContextMenu<FileSystemNode>()

	const [renamePath, setRenamePath] = useState<string | null>(null)

	const handleRenameChange = useCallback(
		(event: InputEvent): void => {
			const value = (event.target as HTMLInputElement).value

			if (value === '') return
			if (!data) return

			const oldPath = data.path
			const newPath = data.path.replace(data.name, value)

			if (oldPath === newPath) {
				// Reset the rename path
				setRenamePath(null)
				return
			}

			renameFSItem(oldPath, newPath)

			// Rename the file
			setRenamePath(null)
		},
		[renamePath, setRenamePath, data]
	)

	const fileEntries = nodes
		? Object.entries(nodes).map(([, node]) => {
				return (
					<DraggableEntry
						node={node}
						key={node.path}
						handleContextMenu={handleContextMenu}
						editPath={renamePath}
						handleChange={handleRenameChange}
					/>
				)
			})
		: null

	const backgroundContextActions: ContextMenuAction[] = [
		{
			label: 'New Folder',
			onClick: async (): Promise<void> => {
				if (!data || data.children == undefined) return

				let name = 'untitled folder'

				const isUsed = (name: string): boolean => {
					return Object.values(data.children!).some((node) => node.name === name)
				}

				// Find a unique name
				let i = 1

				while (isUsed(name)) {
					name = `untitled folder ${i}`
					i++
				}

				const folderPath = await join(data.path, name)

				newFolder(folderPath)
			}
		}
	]

	const fileContextActions: ContextMenuAction[] = [
		{
			label: 'Rename',
			onClick: (): void => {
				if (!data) return

				setRenamePath(data.path)
			}
		},
		{
			label: 'Delete',
			onClick: (): void => {
				if (!data) return

				deleteFSItem(data.path)
			}
		}
	]

	const folderContextActions: ContextMenuAction[] = [
		...backgroundContextActions,
		...fileContextActions
	]

	const contextActions =
		data?.name === directoryName
			? backgroundContextActions
			: data?.children
				? folderContextActions
				: fileContextActions

	const { isOver, setNodeRef: setDropNodeRef } = useDroppable({
		id: 'file-explorer'
	})

	useEffect(() => {
		const handleDrop = async (): Promise<void> => {
			if (
				parentChildData &&
				parentChildData[0].toString() === 'file-explorer' &&
				directoryPath
			) {
				const item = parentChildData[1]

				if (item.data.current?.source === 'file-explorer') {
					if (directoryPath !== item.id.toString()) {
						const oldPath = item.id.toString()
						const newPath = await join(directoryPath, item.data.current.name)

						// Move the file
						moveFSItem(oldPath, newPath)

						// Clear the drag event
						clearDragEvent()
					}
				}
			}
		}

		handleDrop()
	}, [isOver, parentChildData, directoryPath])

	const dropFileRef = useRef<HTMLDivElement>(null)

	// Set the drop node ref
	useEffect(() => {
		setDropNodeRef(dropFileRef.current)
	}, [dropFileRef])

	const [customDragOver, setCustomDragOver] = useState(false)

	useEffect(() => {
		const handleDrop = async (event: DragEvent): Promise<void> => {
			event.preventDefault()

			setCustomDragOver(false)

			if (!event.dataTransfer) return

			const files = event.dataTransfer.files

			// Determine if the dropped item is a folder
			const item = files[0]

			const isFolder = item.type === '' || !isPathFile(item.path)

			if (isFolder) {
				// Send the folder to the main process
				setDirectory(item.path)
			} else {
				// Get the folder path
				const folderPath = directory(item.path)

				setDirectory(folderPath)
			}
		}

		const handleDragOver = (event: DragEvent): void => {
			event.preventDefault()
			setCustomDragOver(true)
		}
		const handleDragLeave = (event: DragEvent): void => {
			event.preventDefault()
			setCustomDragOver(false)
		}

		if (dropFileRef.current) {
			dropFileRef.current.addEventListener('drop', handleDrop)
			dropFileRef.current.addEventListener('dragover', handleDragOver)
			dropFileRef.current.addEventListener('dragleave', handleDragLeave)
		}

		return (): void => {
			if (dropFileRef.current) {
				dropFileRef.current.removeEventListener('drop', handleDrop)
				dropFileRef.current.removeEventListener('dragover', handleDragOver)
				dropFileRef.current.removeEventListener('dragleave', handleDragLeave)
			}
		}
	}, [directoryPath, dropFileRef])

	return (
		<>
			{clicked && <ContextMenu {...point} actions={contextActions} />}
			<div className={styles.fileExplorerPanel}>
				<div
					className={styles.fileExplorerInnerPanel}
					ref={dropFileRef}
					onContextMenu={(e) => {
						if (!directoryName || !directoryPath) return

						const data: FileSystemNode = {
							name: directoryName,
							path: directoryPath,
							children: nodes
						}

						handleContextMenu(e as MouseEvent, data)
					}}
				>
					{directoryName ? (
						<>
							<Header text={directoryName} highlight={isOver || customDragOver} />
							<div>{fileEntries}</div>
						</>
					) : (
						<>
							<Header text={'Files'} highlight={customDragOver} />
							<div className={`poppins-medium ${styles.helpText}`}>
								File &gt; Open Folder
							</div>
						</>
					)}
				</div>
			</div>
			<DragOverlay>
				{active && active.data.current ? (
					<FileEntry
						name={active.data.current.name}
						path={active.data.current.path}
						type={active.data.current.type}
					/>
				) : null}
			</DragOverlay>
		</>
	)
}

export default FileExplorer

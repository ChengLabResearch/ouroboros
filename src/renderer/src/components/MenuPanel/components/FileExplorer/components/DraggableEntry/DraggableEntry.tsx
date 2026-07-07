import { useDraggable, useDroppable } from '@dnd-kit/core'
import FileEntry from '../FileEntry/FileEntry'
import { DirectoryContext, FileSystemNode } from '@renderer/contexts/DirectoryContext'

import styles from './DraggableEntry.module.css'
import { MouseEvent, useContext, useEffect, useState } from 'react'
import EditFileEntry from '../FileEntry/EditFileEntry'
import { DragContext } from '@renderer/contexts/DragContext'
import { join, moveFSItem } from '@renderer/interfaces/file'

function DraggableEntry({
	node,
	handleContextMenu,
	editPath,
	handleChange
}: {
	node: FileSystemNode
	handleContextMenu: (event: MouseEvent, data?: FileSystemNode | undefined) => void
	editPath: string | null
	handleChange: (event: InputEvent) => void
}): JSX.Element {
	// Determine the type of the entry
	const type = node.children ? 'folder' : (node.name.endsWith('.tif') || node.name.endsWith('.tiff')) ? 'image' : 'file'
	const isFolder = type === 'folder'
	const isEmpty = !node.children || Object.keys(node.children).length === 0
	// With lazy expansion an unopened folder starts with an empty children
	// map. We still want to show a chevron so the user can expand it; the
	// contents (or true emptiness) become visible once the main-side watcher
	// streams the initial batch.
	const showChevron = isFolder

	const [isCollapsed, setIsCollapsed] = useState(true)

	const { expandFolder, collapseFolder } = useContext(DirectoryContext)

	// Dispatch expand/collapse to the main-side watcher manager alongside the
	// existing visual toggle. Idempotent by design: main-side handlers no-op
	// when the state already matches, and the visual toggle is instant while
	// the fetch runs asynchronously.
	const toggleCollapsed = (): void => {
		setIsCollapsed((prev) => {
			const next = !prev
			if (isFolder) {
				if (next) {
					collapseFolder(node.path)
				} else {
					expandFolder(node.path)
				}
			}
			return next
		})
	}

	const { attributes, listeners, setNodeRef } = useDraggable({
		id: node.path,
		data: {
			name: node.name,
			path: node.path,
			type,
			source: 'file-explorer'
		}
	})

	const { clearDragEvent, parentChildData, active } = useContext(DragContext)
	const { isOver, setNodeRef: setDropNodeRef } = useDroppable({
		id: node.path
	})

	useEffect(() => {
		const handleDrop = async (): Promise<void> => {
			if (parentChildData) {
				const item = parentChildData[1]

				if (item.data.current?.source === 'file-explorer') {
					if (
						node.path !== item.id.toString() &&
						parentChildData[0].toString() === node.path
					) {
						const oldPath = item.id.toString()
						const newPath = await join(node.path, item.data.current.name)

						// Move the file
						moveFSItem(oldPath, newPath)

						// Clear the drag event
						clearDragEvent()
					}
				}
			}
		}

		handleDrop()
	}, [isOver, parentChildData])

	const isEdited = editPath === node.path

	return (
		<div ref={isFolder ? setDropNodeRef : null}>
			<div
				className={isFolder ? (isEmpty ? styles.emptyFolder : styles.folder) : styles.file}
			>
				{showChevron && (
					<svg
						xmlns="http://www.w3.org/2000/svg"
						viewBox="0 0 320 512"
						className={`${styles.collapseIcon} ${isCollapsed ? '' : styles.collapseIconOpen}`}
						onClick={toggleCollapsed}
					>
						<path
							fill="currentColor"
							d="M137.4 374.6c12.5 12.5 32.8 12.5 45.3 0l128-128c9.2-9.2 11.9-22.9 6.9-34.9s-16.6-19.8-29.6-19.8L32 192c-12.9 0-24.6 7.8-29.6 19.8s-2.2 25.7 6.9 34.9l128 128z"
						/>
					</svg>
				)}
				{!isEdited ? (
					<div
						ref={setNodeRef}
						{...listeners}
						{...attributes}
						onContextMenu={(e) => {
							handleContextMenu(e as MouseEvent, node)
						}}
					>
						<FileEntry
							name={node.name}
							path={node.path}
							type={type}
							bold={
								isOver &&
								active &&
								active.data.current?.source === 'file-explorer' &&
								node.path !== active.id.toString()
									? true
									: false
							}
						/>
					</div>
				) : (
					<EditFileEntry
						name={node.name}
						path={node.path}
						type={type}
						handleChange={handleChange}
					/>
				)}
			</div>
			{!isEmpty && !isCollapsed && node.children && (
				<div className={styles.children}>
					{Object.entries(node.children).map(([, child]) => (
						<DraggableEntry
							node={child}
							key={child.path}
							handleContextMenu={handleContextMenu}
							editPath={editPath}
							handleChange={handleChange}
						/>
					))}
				</div>
			)}
		</div>
	)
}

export default DraggableEntry

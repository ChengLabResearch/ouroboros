import { JSX, useContext, useEffect } from 'react'

import { DragContext } from '@renderer/contexts/DragContext'
import { useDroppable } from '@dnd-kit/core'
import Header from '@renderer/components/Header/Header'

function OptionsHeader({
	onHeaderDrop
}: {
	onHeaderDrop: ((content: string) => void) | undefined
}): JSX.Element {
	const { parentChildData, clearDragEvent } = useContext(DragContext)

	const id = 'options-header'

	const { isOver, setNodeRef } = useDroppable({
		id: id
	})

	useEffect(() => {
		if (parentChildData && parentChildData[0] === id) {
			const childData = parentChildData[1]?.data?.current

			if (childData && childData.source === 'file-explorer') {
				if (onHeaderDrop) {
					onHeaderDrop(childData.path)
					clearDragEvent()
				}
			}
		}
	}, [parentChildData])

	return (
		<div ref={setNodeRef}>
			<Header text={'Options'} highlight={isOver && onHeaderDrop !== undefined} />
		</div>
	)
}

export default OptionsHeader

import { createContext, useState } from 'react'

import { DndContext, DragEndEvent, UniqueIdentifier, Active } from '@dnd-kit/core'

export const DragContext = createContext(null as any)

function Drag({ children }): JSX.Element {
	const [active, setActive] = useState<Active | null>(null)
	const [parentChildData, setParentChildData] = useState<[UniqueIdentifier, Active] | null>(null)

	return (
		<DragContext.Provider value={{ active, parentChildData, setParentChildData }}>
			<DndContext onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
				{children}
			</DndContext>
		</DragContext.Provider>
	)

	function handleDragStart(event: DragEndEvent) {
		if (event.active) {
			setActive(event.active)
		}
	}

	function handleDragEnd(event: DragEndEvent) {
		if (event.over && event.active) {
			setParentChildData([event.over.id, event.active])
		}
		setActive(null)
	}
}

export default Drag

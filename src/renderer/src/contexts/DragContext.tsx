import { JSX, createContext, useState } from 'react'

import { DndContext, DragEndEvent, UniqueIdentifier, Active } from '@dnd-kit/core'

export type DragContextValue = {
	active: Active | null
	parentChildData: [UniqueIdentifier, Active] | null
	clearDragEvent: () => void
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const DragContext = createContext<DragContextValue>(null as any)

function DragProvider({ children }: { children: React.ReactNode }): JSX.Element {
	const [active, setActive] = useState<Active | null>(null)
	const [parentChildData, setParentChildData] = useState<[UniqueIdentifier, Active] | null>(null)

	const clearDragEvent = (): void => setParentChildData(null)

	return (
		<DragContext.Provider value={{ active, parentChildData, clearDragEvent }}>
			<DndContext onDragStart={handleDragStart} onDragEnd={handleDragEnd}>
				{children}
			</DndContext>
		</DragContext.Provider>
	)

	function handleDragStart(event: DragEndEvent): void {
		if (event.active) {
			setActive(event.active)
		}
	}

	function handleDragEnd(event: DragEndEvent): void {
		if (event.over && event.active) {
			setParentChildData([event.over.id, event.active])
		}
		setActive(null)
	}
}

export default DragProvider

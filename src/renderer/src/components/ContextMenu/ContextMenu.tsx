import { JSX, useLayoutEffect, useRef } from 'react'
import styles from './ContextMenu.module.css'

export type ContextMenuAction = {
	label: string
	onClick: () => void
}

function ContextMenu({
	x,
	y,
	actions
}: {
	x: number
	y: number
	actions: ContextMenuAction[]
}): JSX.Element {
	const ref = useRef<HTMLDivElement>(null)

	// Position the context menu so that it doesn't go off screen
	useLayoutEffect(() => {
		if (ref.current) {
			const rect = ref.current.getBoundingClientRect()
			const { innerWidth, innerHeight } = window

			if (rect.right > innerWidth)
				ref.current.style.left = `${x - (rect.right - innerWidth)}px`

			if (rect.bottom > innerHeight)
				ref.current.style.top = `${y - (rect.bottom - innerHeight)}px`
		}
	}, [x, y])

	return (
		<div
			className={`poppins-light ${styles.contextMenu}`}
			style={{ top: `${y}px`, left: `${x}px` }}
			ref={ref}
		>
			<ul className={styles.options}>
				{actions.map((action: ContextMenuAction, index: number) => (
					<li key={index} onClick={action.onClick}>
						{action.label}
					</li>
				))}
			</ul>
		</div>
	)
}

export default ContextMenu

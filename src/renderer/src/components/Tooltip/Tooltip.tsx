import { JSX, useEffect, useRef, useState } from 'react'
import styles from './Tooltip.module.css'

function Tooltip({ x, y, message }: { x: number; y: number; message: string }): JSX.Element {
	const ref = useRef<HTMLDivElement>(null)

	return (
		<div
			className={`poppins-light ${styles.tooltip}`}
			style={{ top: `${y}px`, right: `${x}px`, transform: 'translateY(-50%)' }}
			ref={ref}
		>
			{message}
		</div>
	)
}

export default Tooltip

export function useTooltip(
	ref: React.RefObject<HTMLElement | null>,
	message?: string,
	gap = 10
): JSX.Element {
	if (!message) return <></>

	const [point, setPoint] = useState({ x: 0, y: 0 })
	const [show, setShow] = useState(false)

	useEffect(() => {
		const handleMouseOver = (): void => {
			if (ref.current) {
				const { innerWidth } = window

				// Determine bottom left corner of the element
				const rect = ref.current.getBoundingClientRect()
				const x = innerWidth - rect.left + gap
				const y = (rect.bottom - rect.top) / 2 + rect.top

				setPoint({ x, y })
				setShow(true)

				// If the right side of the tooltip goes off screen, decrease the width to match
				if (x + ref.current.offsetWidth > innerWidth) {
					ref.current.style.width = `${innerWidth - x}px`
				} else {
					ref.current.style.width = '350px'
				}
			}
		}

		const handleMouseOut = (): void => {
			setShow(false)
		}

		ref.current?.addEventListener('mouseover', handleMouseOver)
		ref.current?.addEventListener('mouseout', handleMouseOut)

		return (): void => {
			ref.current?.removeEventListener('mouseover', handleMouseOver)
			ref.current?.removeEventListener('mouseout', handleMouseOut)
		}
	}, [ref])

	return <>{show ? <Tooltip {...point} message={message} /> : null}</>
}

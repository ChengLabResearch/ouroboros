import { useEffect, useState } from 'react'

function useDropFile(
	ref: React.RefObject<HTMLElement | null>,
	handleDrop: (event: DragEvent) => Promise<void>,
	handleDragOver?: (event: DragEvent) => void,
	handleDragLeave?: (event: DragEvent) => void
): { dragOver: boolean } {
	const [dragOver, setDragOver] = useState(false)

	useEffect(() => {
		const _handleDrop = async (event: DragEvent): Promise<void> => {
			event.preventDefault()
			setDragOver(false)
			await handleDrop(event)
		}

		const _handleDragOver = (event: DragEvent): void => {
			event.preventDefault()
			setDragOver(true)
			handleDragOver?.(event)
		}
		const _handleDragLeave = (event: DragEvent): void => {
			event.preventDefault()
			setDragOver(false)
			handleDragLeave?.(event)
		}

		if (ref.current) {
			ref.current.addEventListener('drop', _handleDrop)
			ref.current.addEventListener('dragover', _handleDragOver)
			ref.current.addEventListener('dragleave', _handleDragLeave)
		}

		return (): void => {
			if (ref.current) {
				ref.current.removeEventListener('drop', _handleDrop)
				ref.current.removeEventListener('dragover', _handleDragOver)
				ref.current.removeEventListener('dragleave', _handleDragLeave)
			}
		}
	}, [ref, handleDrop, handleDragOver, handleDragLeave])

	return { dragOver }
}

export default useDropFile

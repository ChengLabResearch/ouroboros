import { useRef, useEffect, useState, useContext, ChangeEvent, FormEventHandler } from 'react'
import styles from './OptionEntry.module.css'
import { useDroppable } from '@dnd-kit/core'
import { ValueType, Entry } from '@renderer/lib/options'
import { DragContext } from '@renderer/contexts/DragContext'
import { DirectoryContext } from '@renderer/contexts/DirectoryContext'
import { join } from '@renderer/lib/file'

const MIN_WIDTH = 25
const LABEL_GAP = 15

function OptionEntry({
	entry,
	minWidth = MIN_WIDTH,
	onEntryChange
}: {
	entry: Entry
	minWidth?: number
	onEntryChange: (entry: Entry) => void
}): JSX.Element {
	const label = entry.label
	const initialValue = entry.value
	const inputType = entry.type
	const inputName = entry.name
	const inputOptions = entry.options

	const inputRef = useRef<HTMLInputElement>(null)
	const labelRef = useRef<HTMLDivElement>(null)
	const [labelWidth, setLabelWidth] = useState(0)

	const [inputValue, setInputValue] = useState(initialValue)

	const { parentChildData, clearDragEvent } = useContext(DragContext)
	const { directoryPath } = useContext(DirectoryContext)

	const { isOver, setNodeRef } = useDroppable({
		id: inputName
	})

	const style = {
		opacity: isOver && inputType == 'filePath' ? 0.5 : 1
	}

	let htmlInputType = 'text'

	switch (inputType) {
		case 'number':
			htmlInputType = 'number'
			break
		case 'string':
			if (inputOptions) {
				htmlInputType = 'select'
			} else htmlInputType = 'text'
			break
		case 'boolean':
			htmlInputType = 'checkbox'
			break
		case 'filePath':
			htmlInputType = 'text'
			break
		default:
			break
	}

	// Receive file paths dropped from FileExplorer
	useEffect(() => {
		;(async () => {
			if (parentChildData && parentChildData[0] === inputName && inputType === 'filePath') {
				const childData = parentChildData[1]?.data?.current

				if (childData && childData.source === 'file-explorer' && directoryPath) {
					if (inputRef.current) {
						const path = await join(directoryPath, childData.path)
						updateValue(path)
						clearDragEvent()

						resizeInput(path)
					}
				}
			}
		})()
	}, [parentChildData])

	function resizeInput(value: ValueType) {
		if (inputType === 'boolean' || htmlInputType === 'select') return

		if (labelRef.current) {
			const label = labelRef.current
			// Measure and store label width
			setLabelWidth(label.offsetWidth + LABEL_GAP)
		}

		if (inputRef.current) {
			const input = inputRef.current

			// Create a temporary span to measure text width
			const tempSpan = document.createElement('span')

			// Match input font size and family
			tempSpan.style.fontSize = getComputedStyle(input).fontSize
			tempSpan.style.fontFamily = getComputedStyle(input).fontFamily
			tempSpan.style.visibility = 'hidden' // Hide span
			tempSpan.textContent = value.toString()
			document.body.appendChild(tempSpan)

			// Set input width based on temp span width, with a minimum width
			input.style.width = `${Math.max(tempSpan.offsetWidth, minWidth)}px`
			document.body.removeChild(tempSpan)
		}
	}

	function updateValue(value: any) {
		switch (inputType) {
			case 'boolean':
				setInputValue(value)
				entry.setValue(value)
				break
			case 'number':
				setInputValue(value)
				entry.setValue(Number(value))
				break
			default:
				setInputValue(value)
				entry.setValue(value)
				break
		}

		onEntryChange(entry)
	}

	useEffect(() => {
		updateValue(initialValue)
		resizeInput(initialValue)
	}, [initialValue, minWidth])

	const onChange = (e: ChangeEvent<HTMLInputElement>) => {
		if (inputType === 'boolean') {
			updateValue(e.target.checked)
			return
		}

		updateValue(e.target.value)

		if (htmlInputType === 'select') return
		if (!inputRef.current) return

		const target = e.target
		const parentWidth = target.parentElement?.offsetWidth || 0
		const maxWidth = parentWidth - labelWidth

		target.style.width = `${minWidth}px` // Reset width to minimum before calculating new width
		target.style.width = `${Math.min(target.scrollWidth, maxWidth)}px`
	}

	const onInput: FormEventHandler<HTMLSelectElement> = (e) => {
		updateValue(e.currentTarget.value)
	}

	return (
		<div ref={setNodeRef} style={style}>
			<div className={`${styles.optionEntry} poppins-medium`}>
				<div
					ref={labelRef}
					className={`${styles.optionLabel} option-font-size ${isOver && inputType == 'filePath' ? 'poppins-bold' : ''}`}
				>
					{label}
				</div>
				{htmlInputType === 'select' ? (
					<select
						name={inputName}
						className={`${styles.optionInput} ${styles.optionSelect}`}
						value={inputValue as string}
						onInput={onInput}
					>
						{inputOptions?.map((option) => (
							<option key={option} value={option}>
								{option}
							</option>
						))}
					</select>
				) : (
					<input
						name={inputName}
						ref={inputRef}
						type={htmlInputType}
						className={styles.optionInput}
						checked={
							inputType == 'boolean'
								? inputValue === 'true' || inputValue === true
								: undefined
						}
						value={inputValue as string}
						onChange={onChange}
					/>
				)}
			</div>
		</div>
	)
}

export default OptionEntry

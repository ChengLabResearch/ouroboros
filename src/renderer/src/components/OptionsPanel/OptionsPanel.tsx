import { JSX, useRef, useState } from 'react'
import { CompoundEntry, Entry } from '@renderer/interfaces/options'
import OptionEntry from './components/OptionEntry/OptionEntry'
import OptionSubmit from './components/OptionSubmit/OptionSubmit'
import styles from './OptionsPanel.module.css'
import CompoundEntryElement from './components/CompoundEntry/CompoundEntry'
import OptionsHeader from './components/OptionsHeader/OptionsHeader'

function OptionsPanel({
	entries,
	onSubmit,
	onEntryChange,
	onHeaderDrop,
	isRunning = false
}: {
	entries: (Entry | CompoundEntry)[]
	onSubmit: () => Promise<void>
	onEntryChange?: (entry: Entry) => void
	onHeaderDrop?: (content: string) => void
	isRunning?: boolean
}): JSX.Element {
	if (onEntryChange === undefined) onEntryChange = (): void => {}
	const submitLock = useRef(false)
	const [isSubmitting, setIsSubmitting] = useState(false)

	const entryElement = entries.flatMap((entryObject) => {
		if (entryObject instanceof Entry) return entryToElement(entryObject, onEntryChange)
		else return compoundEntryToElement(entryObject, onEntryChange, false, false)
	})

	const handleSubmit = async (e: React.FormEvent<HTMLFormElement>): Promise<void> => {
		e.preventDefault()
		if (submitLock.current || isRunning) return

		submitLock.current = true
		setIsSubmitting(true)
		try {
			await onSubmit()
		} finally {
			submitLock.current = false
			setIsSubmitting(false)
		}
	}

	return (
		<div className="panel">
			<div className="inner-panel">
				<form className={styles.form} method="post" onSubmit={handleSubmit}>
					<OptionsHeader onHeaderDrop={onHeaderDrop} />
					{entryElement}
					<OptionSubmit
						state={isSubmitting ? 'starting' : isRunning ? 'running' : 'idle'}
					/>
				</form>
			</div>
		</div>
	)
}

function compoundEntryToElement(
	compoundEntry: CompoundEntry,
	onEntryChange: (entry: Entry) => void,
	indent = true,
	title = true
): JSX.Element {
	if (compoundEntry.hidden) return <span key={compoundEntry.name}></span>

	return (
		<CompoundEntryElement
			key={compoundEntry.name}
			entry={compoundEntry}
			indent={indent}
			title={title}
		>
			{compoundEntry.entries.map((entry) => {
				if (entry instanceof Entry) return entryToElement(entry, onEntryChange)
				else return compoundEntryToElement(entry, onEntryChange)
			})}
		</CompoundEntryElement>
	)
}

function entryToElement(entry: Entry, onEntryChange: (entry: Entry) => void): JSX.Element {
	if (entry.hidden) return <span key={entry.name}></span>

	return <OptionEntry key={entry.name} entry={entry} onEntryChange={onEntryChange} />
}

export default OptionsPanel

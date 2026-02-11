import { JSX } from "react"
import Separator from '@renderer/components/Separator/Separator'
import styles from './CompoundEntry.module.css'
import { CompoundEntry } from '@renderer/interfaces/options'

function CompoundEntryElement({
	entry,
	children,
	title = true,
	indent = true
}: {
	entry: CompoundEntry
	children: React.ReactNode
	title?: boolean
	indent?: boolean
}): JSX.Element {
	return (
		<>
			{title ? (
				<div className={`${styles.title} poppins-bold option-font-size`}>{entry.label}</div>
			) : null}
			<div className={indent ? styles.indent : ''}>{children}</div>
			{entry.separator && <Separator />}
		</>
	)
}

export default CompoundEntryElement

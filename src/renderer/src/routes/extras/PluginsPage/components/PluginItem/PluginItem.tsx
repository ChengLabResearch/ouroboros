import { JSX } from "react"
import styles from './PluginItem.module.css'

function PluginItem({
	name,
	id,
	selectedPlugins,
	setSelectedPlugins
}: {
	name: string
	id: string
	selectedPlugins: Set<string>
	setSelectedPlugins: (selectedPlugins: Set<string>) => void
}): JSX.Element {
	const selected = selectedPlugins.has(id)

	const setSelected = (selected: boolean): void => {
		const newSelectedPlugins = new Set(selectedPlugins)
		if (selected) {
			newSelectedPlugins.add(id)
		} else {
			newSelectedPlugins.delete(id)
		}
		setSelectedPlugins(newSelectedPlugins)
	}

	return (
		<div
			className={`${styles.pluginItem} ${selected ? styles.active : ''}`}
			onClick={() => setSelected(!selected)}
		>
			<div className={`${selected ? 'poppins-bold' : 'poppins-medium'}`}>{name}</div>
		</div>
	)
}

export default PluginItem

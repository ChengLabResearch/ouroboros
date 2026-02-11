import { JSX } from "react"
import styles from './PluginTestInput.module.css'

function PluginTestInput({
	pluginURL,
	setPluginURL,
	setShowPlugin
}: {
	pluginURL: string
	setPluginURL: (value: string) => void
	setShowPlugin: (value: boolean) => void
}): JSX.Element {
	return (
		<div className={`${styles.input} poppins-regular`}>
			<input
				type="text"
				value={pluginURL}
				placeholder="Enter Plugin URL"
				onChange={(event) => setPluginURL(event.target.value)}
			/>
			<div
				onClick={() => {
					setShowPlugin(true)
				}}
			>
				Test Plugin
			</div>
		</div>
	)
}

export default PluginTestInput

import { JSX } from "react"
import styles from './PluginMenuButton.module.css'

function PluginMenuButton({
	icon,
	onClick
}: {
	icon: JSX.Element
	onClick: () => void
}): JSX.Element {
	return (
		<a className={styles.pluginMenuButton} onClick={onClick}>
			{icon}
		</a>
	)
}

export default PluginMenuButton

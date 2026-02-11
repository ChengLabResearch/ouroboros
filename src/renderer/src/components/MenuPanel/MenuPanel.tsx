import { JSX } from "react"
import styles from './MenuPanel.module.css'
import FileExplorer from './components/FileExplorer/FileExplorer'
import Menu from './components/Menu/Menu'

function MenuPanel(): JSX.Element {
	return (
		<div className={styles.menuPanel}>
			<Menu />
			<FileExplorer />
		</div>
	)
}

export default MenuPanel

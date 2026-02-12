import { JSX } from "react"
import styles from './ServerConnectedIndicator.module.css'

// TODO make this respond to changes from the server

function ServerConnectedIndicator({ connected }: { connected: boolean }): JSX.Element {
	return (
		<div className={styles.indicatorParent}>
			<div
				className={`${styles.indicator} ${connected ? styles.indicatorConnected : ''}`}
			></div>
			<div className={styles.indicatorLabel}>
				{connected ? 'Connected' : 'Waiting for Server'}
			</div>
		</div>
	)
}

export default ServerConnectedIndicator

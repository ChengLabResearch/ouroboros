import { JSX } from "react"
import styles from './ProgressBar.module.css'

function ProgressBar({
	progress,
	name,
	duration
}: {
	progress: number
	name: string
	duration?: number
}): JSX.Element {
	return (
		<div className={styles.progressBarParent}>
			<div className={styles.progressBarContainer}>
				<div className={styles.progressBarText}>{name}</div>
				<div className={styles.progressBar} style={{ width: `${progress}%` }}></div>
			</div>
			<div className={styles.progressLabel}>{Math.round(progress)}%</div>
			{duration && (
				<div className={styles.progressLabel}>{roundToDecimals(duration, 1)}s</div>
			)}
		</div>
	)
}

function roundToDecimals(value: number, decimals: number): number {
	const factor = 10 ** decimals
	return Math.round(value * factor) / factor
}

export default ProgressBar

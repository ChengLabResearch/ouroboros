import { JSX } from 'react'
import type { StepStatus } from '../../../../schemas/task-status-schema'
import { STATUS_LABELS } from '../../../../interfaces/status-presentation'
import StatusLabel from '../StatusLabel/StatusLabel'
import styles from './ProgressBar.module.css'

function ProgressBar({
	progress,
	name,
	duration,
	status,
	stage,
	live = true
}: {
	progress: number | null
	name: string
	duration?: number
	status?: StepStatus['status']
	stage?: string
	live?: boolean
}): JSX.Element {
	const displayStatus =
		status === 'pending' ? 'waiting' : status === 'succeeded' ? 'complete' : status
	const indeterminate = progress === null && status === 'running' && live
	return (
		<div className={styles.progressBarParent} data-state={status}>
			<div className={styles.progressHeading}>
				<span>{name}</span>
				{displayStatus && <StatusLabel status={displayStatus} />}
			</div>
			{stage && <div className={styles.stage}>{stage}</div>}
			<div className={styles.progressRow}>
				<div
					className={styles.progressBarContainer}
					role="progressbar"
					aria-label={name}
					aria-valuemin={0}
					aria-valuemax={100}
					aria-valuenow={progress === null ? undefined : progress}
					aria-valuetext={
						stage ?? (displayStatus ? STATUS_LABELS[displayStatus] : undefined)
					}
					aria-busy={indeterminate}
				>
					<div
						className={`${styles.progressBar} ${indeterminate ? styles.indeterminate : ''}`}
						style={progress === null ? undefined : { width: `${progress}%` }}
					/>
				</div>
				{progress !== null && (
					<span className={styles.progressLabel}>{Math.round(progress)}%</span>
				)}
				{duration !== undefined && duration > 0 && (
					<span className={styles.progressLabel}>{roundToDecimals(duration, 1)}s</span>
				)}
			</div>
		</div>
	)
}

function roundToDecimals(value: number, decimals: number): number {
	const factor = 10 ** decimals
	return Math.round(value * factor) / factor
}

export default ProgressBar

import { JSX } from 'react'
import type { MainServerLaunchStatus } from '@shared/main-server-status'
import Alert from '../AlertArea/components/Alert/Alert'
import { StreamResult } from '../../interfaces/task-status-stream'
import {
	activeTaskLabel,
	stageLabel,
	stepLabel,
	taskStatusMessages
} from '../../interfaces/task-status-presentation'
import { TaskStatus } from '../../schemas/task-status-schema'
import Header from '../Header/Header'
import ProgressBar from './components/ProgressBar/ProgressBar'
import BackendStatus from './components/BackendStatus/BackendStatus'
import styles from './Progress.module.css'

/**
 * [name, progress (0-1), duration (seconds)]
 */
export type ProgressType = TaskStatus['progress'][number]

function ProgressPanel({
	operation,
	stream,
	connected,
	launchStatus = null
}: {
	operation: string
	stream: StreamResult
	connected: boolean
	launchStatus?: MainServerLaunchStatus | null
}): JSX.Element {
	const progress = stream.results?.progress ?? []
	const messages = taskStatusMessages(operation, stream.results)
	const live = connected && !stream.error.status
	const phase = activeTaskLabel(stream.results)
	const steps = stream.results?.steps
	const progressBars =
		steps !== undefined
			? steps.map((step, index) => (
					<ProgressBar
						key={index}
						name={stepLabel(step.name)}
						progress={step.progress === null ? null : step.progress * 100}
						duration={step.duration_seconds}
						status={step.status}
						stage={step.stage ? stageLabel(step.stage) : undefined}
						live={live}
					/>
				))
			: progress.map((p: ProgressType, i: number) => {
					const [name, _progress, duration] = p
					return (
						<ProgressBar
							key={i}
							progress={_progress * 100}
							name={stepLabel(name)}
							duration={duration}
						/>
					)
				})

	return (
		<div className="panel">
			<div className="inner-panel">
				<Header text={'Progress'} />
				<BackendStatus connected={connected} launchStatus={launchStatus} />
				{phase && (
					<div className={styles.taskPhase} role="status" aria-live="polite">
						<span>
							{!live && 'Last reported status: '}
							{phase}
						</span>
						{live && <span className={styles.activity} aria-label="Task active" />}
					</div>
				)}
				{!stream.results && !stream.error.status && (
					<div className={styles.empty}>No task status yet.</div>
				)}
				{progressBars}
				{messages.map((message, index) => (
					<Alert key={index} {...message} />
				))}
				{stream.error.status && <Alert type="warning" message={stream.error.message} />}
			</div>
		</div>
	)
}

export default ProgressPanel

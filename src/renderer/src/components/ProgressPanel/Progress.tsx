import { JSX } from 'react'
import Alert from '../AlertArea/components/Alert/Alert'
import { StreamResult } from '../../interfaces/task-status-stream'
import { taskStatusMessages } from '../../interfaces/task-status-presentation'
import { TaskStatus } from '../../schemas/task-status-schema'
import Header from '../Header/Header'
import ProgressBar from './components/ProgressBar/ProgressBar'
import ServerConnectedIndicator from './components/ServerConnectedIndicator/ServerConnectedIndicator'

/**
 * [name, progress (0-1), duration (seconds)]
 */
export type ProgressType = TaskStatus['progress'][number]

function ProgressPanel({
	operation,
	stream,
	connected
}: {
	operation: string
	stream: StreamResult
	connected: boolean
}): JSX.Element {
	const progress = stream.results?.progress ?? []
	const messages = taskStatusMessages(operation, stream.results)
	const progressBars = progress
		? progress.map((p: ProgressType, i: number) => {
				const [name, _progress, duration] = p
				return (
					<ProgressBar
						key={i}
						progress={_progress * 100}
						name={name}
						duration={duration}
					/>
				)
			})
		: null

	return (
		<div className="panel">
			<div className="inner-panel">
				<Header text={'Progress'} />
				<ServerConnectedIndicator connected={connected} />
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

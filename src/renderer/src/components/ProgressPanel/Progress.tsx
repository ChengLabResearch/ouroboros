import { JSX } from "react"
import Header from '../Header/Header'
import ProgressBar from './components/ProgressBar/ProgressBar'
import ServerConnectedIndicator from './components/ServerConnectedIndicator/ServerConnectedIndicator'

/**
 * [name, progress (0-1), duration (seconds)]
 */
export type ProgressType = [string, number, number]

function ProgressPanel({
	progress,
	connected
}: {
	progress: ProgressType[]
	connected: boolean
}): JSX.Element {
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
			</div>
		</div>
	)
}

export default ProgressPanel

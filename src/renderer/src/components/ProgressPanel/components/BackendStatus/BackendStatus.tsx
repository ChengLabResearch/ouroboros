import type { MainServerLaunchStatus } from '@shared/main-server-status'
import type { PresentationStatus } from '../../../../interfaces/status-presentation'
import StatusLabel from '../StatusLabel/StatusLabel'
import Alert from '../../../AlertArea/components/Alert/Alert'
import styles from './BackendStatus.module.css'

const LAUNCH_STATUS: Record<MainServerLaunchStatus['state'], PresentationStatus> = {
	idle: 'waiting',
	starting: 'running',
	started: 'complete',
	failed: 'failed'
}

export default function BackendStatus({
	connected,
	launchStatus
}: {
	connected: boolean
	launchStatus: MainServerLaunchStatus | null
}): React.JSX.Element {
	const launchFailed = launchStatus?.state === 'failed'
	const summary = connected
		? launchFailed
			? 'Connected — local launch failed'
			: 'Connected'
		: launchFailed
			? 'Local launch failed'
			: launchStatus?.state === 'starting'
				? 'Starting backend'
				: 'Waiting for backend'
	const status: PresentationStatus = launchFailed
		? connected
			? 'warning'
			: 'failed'
		: connected
			? 'complete'
			: launchStatus?.state === 'starting'
				? 'running'
				: 'waiting'

	return (
		<section className={styles.backend} aria-label="Backend initialization">
			<div className={styles.summary}>
				<strong>Backend</strong>
				<StatusLabel status={status} label={summary} />
			</div>
			<dl className={styles.steps}>
				{launchStatus && (
					<div>
						<dt>Local backend launch</dt>
						<dd>
							<StatusLabel status={LAUNCH_STATUS[launchStatus.state]} />
						</dd>
					</div>
				)}
				<div>
					<dt>Backend connection</dt>
					<dd>
						<StatusLabel
							status={connected ? 'complete' : 'waiting'}
							label={connected ? 'Connected' : 'Waiting'}
						/>
					</dd>
				</div>
			</dl>
			{launchFailed && launchStatus.error && (
				<Alert
					type="warning"
					message={`Local backend launch failed: ${launchStatus.error}`}
				/>
			)}
		</section>
	)
}

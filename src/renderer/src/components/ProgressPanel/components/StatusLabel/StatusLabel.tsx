import { STATUS_LABELS, type PresentationStatus } from '../../../../interfaces/status-presentation'
import styles from './StatusLabel.module.css'

export default function StatusLabel({
	status,
	label = STATUS_LABELS[status]
}: {
	status: PresentationStatus
	label?: string
}): React.JSX.Element {
	return (
		<span className={styles.status} data-status={status}>
			{label}
		</span>
	)
}

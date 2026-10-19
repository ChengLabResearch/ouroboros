import { JSX } from 'react'
import styles from './Alert.module.css'

function Alert({
	message,
	type,
	onDismiss
}: {
	message: string
	type: 'error' | 'info' | 'success' | 'warning'
	onDismiss?: () => void
}): JSX.Element {
	const style = styles[`alert-${type}`]

	return (
		<div className={`${styles.alert} ${style}`} role={type === 'error' ? 'alert' : 'status'}>
			<div className={styles.message}>{message}</div>
			{onDismiss && (
				<button type="button" className={styles.dismiss} onClick={onDismiss}>
					Dismiss
				</button>
			)}
		</div>
	)
}

export default Alert

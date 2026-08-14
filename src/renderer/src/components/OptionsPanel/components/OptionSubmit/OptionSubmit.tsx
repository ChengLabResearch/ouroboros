import { JSX } from 'react'
import Start from './assets/start.svg?react'
import styles from './OptionSubmit.module.css'

export type SubmitState = 'idle' | 'starting' | 'running'

function OptionSubmit({ state }: { state: SubmitState }): JSX.Element {
	const busy = state !== 'idle'
	const label = state === 'starting' ? 'Starting…' : 'Running…'

	return (
		<button
			aria-busy={busy}
			aria-label={busy ? label : 'Start'}
			className={`${styles.submitButton} poppins-bold`}
			disabled={busy}
			type="submit"
		>
			{busy ? <span className={styles.busyLabel}>{label}</span> : <Start />}
		</button>
	)
}

export default OptionSubmit

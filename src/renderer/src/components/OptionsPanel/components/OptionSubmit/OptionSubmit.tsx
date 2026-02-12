import { JSX } from "react"
import Start from './assets/start.svg?react'
import styles from './OptionSubmit.module.css'

function OptionSubmit(): JSX.Element {
	return (
		<button className={`${styles.submitButton} poppins-bold`} type="submit">
			<Start />
		</button>
	)
}

export default OptionSubmit

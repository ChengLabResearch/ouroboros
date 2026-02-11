import { JSX } from "react"
import styles from './Alert.module.css'

function Alert({
	message,
	type
}: {
	message: string
	type: 'error' | 'info' | 'success' | 'warning'
}): JSX.Element {
	const style = styles[`alert-${type}`]

	// const [visible, setVisible] = useState(true)

	// // Clear the alert after 4 seconds
	// useEffect(() => {
	// 	const timeout = setTimeout(() => {
	// 		setVisible(false)
	// 	}, duration)

	// 	return () => {
	// 		clearTimeout(timeout)
	// 	}
	// }, [])

	// ${visible ? '' : styles.alertHidden}

	return (
		<div className={`${styles.alert} ${style}`}>
			<div className="poppins-medium">{message}</div>
		</div>
	)
}

export default Alert

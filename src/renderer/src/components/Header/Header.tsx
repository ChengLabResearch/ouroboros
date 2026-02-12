import { JSX } from "react"
import styles from './Header.module.css'

function Header({ text, highlight = false }): JSX.Element {
	return (
		<div
			className={`${styles.header} header-font-size ${highlight ? 'poppins-extrabold ' + styles.highlight : 'poppins-bold'}`}
		>
			{text.toUpperCase()}
		</div>
	)
}

export default Header

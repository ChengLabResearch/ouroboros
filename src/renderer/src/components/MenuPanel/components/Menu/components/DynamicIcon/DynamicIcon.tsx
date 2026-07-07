import { useEffect, useRef } from 'react'
import styles from './DynamicIcon.module.css'
import { Location } from 'react-router-dom'

const TRY_AGAIN_TIMEOUT = 100

function DynamicIcon({
	url,
	location,
	path
}: {
	url: string
	location: Location
	path: string
}): JSX.Element {
	const active = location.pathname ? location.pathname.includes(path) : false
	const activeStyle = active ? styles.dynamicIconActive : ''

	const ref = useRef<HTMLDivElement>(null)

	useEffect(() => {
		const setProperties = async (): Promise<void> => {
			if (ref.current) {
				ref.current.style.maskRepeat = 'no-repeat'
				ref.current.style.maskPosition = 'center'
				ref.current.style.maskImage = `url(${url})`

				// Fetch the file from the url and if it fails, set a timeout to try again
				try {
					const response = await fetch(url)
					if (!response.ok) {
						ref.current.style.opacity = '0'
						ref.current.style.maskImage = ''

						setTimeout(() => {
							setProperties()
						}, TRY_AGAIN_TIMEOUT)
					} else {
						ref.current.style.opacity = '1'
					}
				} catch (_error) {
					ref.current.style.opacity = '0'
					ref.current.style.maskImage = ''

					setTimeout(() => {
						setProperties()
					}, TRY_AGAIN_TIMEOUT)
				}
			}
		}

		setProperties()
	}, [url])

	return <div ref={ref} className={`${styles.dynamicIcon} ${activeStyle}`}></div>
}

export default DynamicIcon

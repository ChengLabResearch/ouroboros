import { JSX } from "react"
import AlertArea from '@renderer/components/AlertArea/AlertArea'
import { createContext, useEffect, useRef, useState } from 'react'

export type AlertContextValue = {
	addAlert: (message: string, type: 'error' | 'info' | 'success' | 'warning') => void
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const AlertContext = createContext<AlertContextValue>(null as any)

export type AlertMessageType = {
	id: string
	message: string
	type: 'error' | 'info' | 'success' | 'warning'
}

function AlertProvider({
	children,
	duration = 6000
}: {
	children: React.ReactNode
	duration?: number
}): JSX.Element {
	const [alerts, setAlerts] = useState<AlertMessageType[]>([])
	const timeoutsRef = useRef<NodeJS.Timeout[]>([])

	const addAlert = (message: string, type: 'error' | 'info' | 'success' | 'warning'): void => {
		const id = `${type}-${message}-${Date.now()}`
		setAlerts((currentAlerts) => [...currentAlerts, { id, message, type }])

		const timeout = setTimeout(() => {
			setAlerts((currentAlerts) => currentAlerts.filter((alert) => alert.id !== id))
		}, duration)
		timeoutsRef.current.push(timeout)
	}

	useEffect(() => {
		// Clear all timeouts when the component unmounts
		return (): void => {
			timeoutsRef.current.forEach(clearTimeout)
		}
	}, [])

	return (
		<AlertContext.Provider value={{ addAlert }}>
			<AlertArea alerts={alerts} />
			{children}
		</AlertContext.Provider>
	)
}

export default AlertProvider

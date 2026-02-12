import { JSX, createContext, useEffect, useState } from 'react'

export type TestingPluginContextValue = {
	testingPlugin: boolean
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const TestingPluginContext = createContext<TestingPluginContextValue>(null as any)

function TestingPluginProvider({ children }: { children: React.ReactNode }): JSX.Element {
	const [testingPlugin, setTestingPlugin] = useState<boolean>(false)

	useEffect(() => {
		window.electron.ipcRenderer.send('get-is-dev')
		const removeListener = window.electron.ipcRenderer.on('is-dev', (_, is_dev) => {
			setTestingPlugin(is_dev)
		})

		return (): void => {
			removeListener()
		}
	}, [])

	return (
		<TestingPluginContext.Provider value={{ testingPlugin }}>
			{children}
		</TestingPluginContext.Provider>
	)
}

export default TestingPluginProvider

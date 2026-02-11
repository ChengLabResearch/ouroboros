import { TestingPluginContext } from '@renderer/contexts/TestingPluginContext'
import { JSX, useContext, useState } from 'react'
import PluginTestInput from './components/PluginTestInput'
import PluginDisplay from '@renderer/components/PluginDisplay/PluginDisplay'

function PluginTestPage(): JSX.Element {
	const { testingPlugin } = useContext(TestingPluginContext)

	const [pluginURL, setPluginURL] = useState<string>('')
	const [showPlugin, setShowPlugin] = useState<boolean>(false)

	return (
		<div style={{ position: 'relative' }}>
			{testingPlugin ? (
				!showPlugin ? (
					<PluginTestInput
						pluginURL={pluginURL}
						setPluginURL={setPluginURL}
						setShowPlugin={setShowPlugin}
					/>
				) : (
					<PluginDisplay url={pluginURL} />
				)
			) : null}
		</div>
	)
}

export default PluginTestPage

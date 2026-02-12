import Header from '@renderer/components/Header/Header'

import styles from './PluginsPage.module.css'
import PluginMenuButton from './components/PluginMenu/PluginMenuButton'

import TrashIcon from './assets/trash.svg?react'
import AddIcon from './assets/plus.svg?react'
import PluginItem from './components/PluginItem/PluginItem'
import { JSX, useEffect, useState } from 'react'
import AddPlugin from './components/AddPlugin/AddPlugin'

function PluginsPage(): JSX.Element {
	const [selectedPlugins, setSelectedPlugins] = useState<Set<string>>(new Set())

	const [showAdd, setShowAdd] = useState<boolean>(false)

	const [pluginFolderContents, setPluginFolderContents] = useState<
		{ name: string; id: string; folder: string }[]
	>([])

	useEffect(() => {
		window.electron.ipcRenderer.send('get-plugin-folder-contents')

		const clearPluginFolderContentsListener = window.electron.ipcRenderer.on(
			'plugin-folder-contents',
			(_, arg) => {
				setPluginFolderContents(arg)
			}
		)

		return (): void => {
			clearPluginFolderContentsListener()
		}
	}, [])

	const onAdd = ({ type, content }: { type: string; content: string }): void => {
		if (type === 'release') {
			window.electron.ipcRenderer.send('download-plugin', content)
		} else if (type === 'local') {
			window.electron.ipcRenderer.send('add-local-plugin', content)
		}
	}

	const onDelete = (): void => {
		for (const plugin of selectedPlugins) {
			// Find the folder of the plugin with the given ID
			const pluginFolder = pluginFolderContents.find((folder) => folder.id === plugin)?.folder

			// Send a message to the main process to delete the plugin with the given ID (name attribute)
			window.electron.ipcRenderer.send('delete-plugin', pluginFolder)
		}
	}

	return (
		<div className={styles.mainPluginArea}>
			<Header text={'Plugins'} />
			<div className={styles.pluginMenuArea}>
				<PluginMenuButton icon={<TrashIcon />} onClick={onDelete} />
				<PluginMenuButton
					icon={<AddIcon />}
					onClick={() => setShowAdd((current) => !current)}
				/>
			</div>
			{showAdd ? <AddPlugin onAdd={onAdd} /> : null}
			<div className={styles.pluginItemArea}>
				{pluginFolderContents.map((plugin) => {
					return (
						<PluginItem
							name={plugin.name}
							id={plugin.id}
							key={plugin.id}
							selectedPlugins={selectedPlugins}
							setSelectedPlugins={setSelectedPlugins}
						/>
					)
				})}
			</div>
		</div>
	)
}

export default PluginsPage

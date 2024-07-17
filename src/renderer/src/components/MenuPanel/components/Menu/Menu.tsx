/// <reference types="vite-plugin-svgr/client" />
import SliceIcon from './assets/slice-icon.svg?react'
import BackprojectIcon from './assets/backproject-icon.svg?react'
import DefaultIcon from './assets/segment-icon.svg?react'
import MenuOption from './components/MenuOption/MenuOption'
import { useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import DynamicIcon from './components/DynamicIcon/DynamicIcon'

function Menu(): JSX.Element {
	const location = useLocation()

	const [pluginDetails, setPluginDetails] = useState<
		{
			id: string
			name: string
			mainPath: string
			stylesPath?: string
			iconPath?: string
		}[]
	>([])

	useEffect(() => {
		const clearListener = window.electron.ipcRenderer.on('plugin-paths', (_, paths) => {
			try {
				setPluginDetails(paths)
			} catch (e) {
				console.error(e)
			}
		})

		return (): void => {
			clearListener()
		}
	}, [])

	return (
		<div>
			<MenuOption
				path={'/slice'}
				optionName={'Slice'}
				icon={<SliceIcon />}
				location={location}
			/>
			<MenuOption
				path={'/backproject'}
				optionName={'Backproject'}
				icon={<BackprojectIcon />}
				location={location}
			/>
			{pluginDetails.map((plugin) => {
				const Icon = plugin.iconPath ? (
					<DynamicIcon url={plugin.iconPath} path={`/${plugin.id}`} location={location} />
				) : (
					<DefaultIcon />
				)

				return (
					<MenuOption
						key={plugin.id}
						path={`/${plugin.id}`}
						optionName={plugin.name}
						icon={Icon}
						location={location}
					/>
				)
			})}
		</div>
	)
}

export default Menu

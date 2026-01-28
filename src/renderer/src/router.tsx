/* eslint-disable */
// @ts-nocheck

import { createHashRouter, Navigate } from 'react-router-dom'

import SlicesPage from './routes/SlicesPage/SlicesPage'
import BackprojectPage from './routes/BackprojectPage/BackprojectPage'
import Root from './routes/Root/Root'
import PluginsPage from './routes/extras/PluginsPage/PluginsPage'
import PluginDisplay from './components/PluginDisplay/PluginDisplay'

const pluginDetails: {
	id: string
	name: string
	indexPath: string
	iconPath?: string
}[] = []

window.electron.ipcRenderer.on('plugin-paths', (_, paths) => {
	try {
		pluginDetails.length = 0
		pluginDetails.push(...paths)
	} catch (e) {
		console.error(e)
	}
})

export const router = createHashRouter(
	[
		{
			id: 'root',
			path: '/',
			element: <Root />,
			children: [
				{
					path: '/',
					element: <Navigate to="/slice" />
				},
				{
					path: '/slice',
					element: <SlicesPage />
				},
				{
					path: '/backproject',
					element: <BackprojectPage />
				}
			]
		},
		{
			path: '/extras',
			children: [
				{
					path: 'plugins',
					element: <PluginsPage />
				}
			]
		}
	],
	{
		async patchRoutesOnNavigation({ path, patch }) {
			// Insert the plugin test page, a page that is only available in development
			if (path === '/plugin-test') {
				patch('root', [
					{
						path: path,
						Component: (await import('./routes/PluginTestPage/PluginTestPage')).default
					}
				])
			}

			const errorRoute = {
				path: path,
				element: <></>
			}

			// Try to add the plugin route
			if (pluginDetails.length === 0) {
				patch('root', [errorRoute])
				return
			}

			let route = await getPluginRoute(path)

			if (!route) {
				patch('root', [errorRoute])
				return
			}

			patch('root', [route])
		}
	}
)

const getPluginRoute = async (route: string) => {
	// Find the plugin id from the route
	const pluginId = route.split('/').at(-1)

	// Find the plugin details from the plugin id
	const pluginDetail = pluginDetails.find((plugin) => plugin.id === pluginId)

	if (!pluginDetail) return null

	return {
		path: route,
		element: <PluginDisplay url={pluginDetail.indexPath} />
	}
}

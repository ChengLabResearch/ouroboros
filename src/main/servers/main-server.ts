import { join } from 'path'
import { startDockerCompose, stopDockerCompose } from './docker'

const DEVELOPMENT_PATH = join(__dirname, '../../python/')
const DEVELOPMENT_CONFIG = join(DEVELOPMENT_PATH, 'compose.yml')

const PRODUCTION_PATH = join(__dirname, '../../../extra-resources/server/')
const PRODUCTION_CONFIG = join(PRODUCTION_PATH, 'compose.yml')

export async function startMainServerDevelopment(): Promise<void> {
	try {
		await startDockerCompose({
			cwd: DEVELOPMENT_PATH,
			config: DEVELOPMENT_CONFIG,
			onError: (err) => {
				console.error('An error occurred while starting the main server:', err)
			}
		})
	} catch (error) {
		console.error('An error occurred while starting the main server:', error)
	}
}

export async function stopMainServerDevelopment(): Promise<void> {
	await stopDockerCompose({
		cwd: DEVELOPMENT_PATH,
		config: DEVELOPMENT_CONFIG
	})
}

export async function startMainServerProduction(): Promise<void> {
	startDockerCompose({
		cwd: PRODUCTION_PATH,
		config: PRODUCTION_CONFIG,
		onError: (err) => {
			console.error('An error occurred while starting the main server:', `${err}`)
		}
	}).catch((error) => {
		console.error('An error occurred while trying to start the main server:', `${error}`)
	})
}

export async function stopMainServerProduction(): Promise<void> {
	await stopDockerCompose({
		cwd: PRODUCTION_PATH,
		config: PRODUCTION_CONFIG
	})
}

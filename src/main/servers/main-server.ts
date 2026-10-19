import { join } from 'path'
import { startDockerCompose, stopDockerCompose } from './docker'
import type { MainServerLaunchStatus } from '../../shared/main-server-status'

const DEVELOPMENT_PATH = join(__dirname, '../../python/')
const DEVELOPMENT_CONFIG = join(DEVELOPMENT_PATH, 'compose.dev.yml')

const PRODUCTION_PATH = join(__dirname, '../../../extra-resources/server/')
const PRODUCTION_CONFIG = join(PRODUCTION_PATH, 'compose.yml')

let launchStatus: MainServerLaunchStatus = { state: 'idle', error: null }
let launch: Promise<void> | undefined
const launchListeners = new Set<(status: MainServerLaunchStatus) => void>()

export function getMainServerLaunchStatus(): MainServerLaunchStatus {
	return launchStatus
}

export function onMainServerLaunchStatus(
	listener: (status: MainServerLaunchStatus) => void
): () => void {
	launchListeners.add(listener)
	return () => launchListeners.delete(listener)
}

function setLaunchStatus(status: MainServerLaunchStatus): void {
	launchStatus = status
	for (const listener of launchListeners) listener(status)
}

function launchErrorMessage(error: unknown): string {
	// docker-compose rejects with its captured result rather than an Error when
	// the command exits unsuccessfully. Preserve stderr for actionable diagnostics.
	if (error && typeof error === 'object' && 'err' in error) {
		if (typeof error.err === 'string' && error.err.trim()) return error.err.trim()
	}
	if (error instanceof Error) return error.message
	if (typeof error === 'string' && error.trim()) return error.trim()
	return 'Docker Compose could not start the main server. Check the application logs.'
}

async function runLaunch(cwd: string, config: string): Promise<void> {
	setLaunchStatus({ state: 'starting', error: null })
	try {
		await startDockerCompose({
			cwd,
			config,
			onError: (error) => {
				// The shared helper lets each caller decide how to handle errors.
				// Propagate here so this launcher can record a failed attempt.
				throw error
			}
		})
		setLaunchStatus({ state: 'started', error: null })
		console.info('Started Main Server')
	} catch (error) {
		setLaunchStatus({ state: 'failed', error: launchErrorMessage(error) })
		console.error('An error occurred while starting the main server:', error)
	}
}

async function startMainServer(cwd: string, config: string): Promise<void> {
	// Window recreation must observe the current attempt, not launch another one.
	if (launch) return launch
	if (launchStatus.state === 'started') return
	launch = runLaunch(cwd, config)
	try {
		await launch
	} finally {
		launch = undefined
	}
}

async function stopMainServer(cwd: string, config: string): Promise<void> {
	await stopDockerCompose({ cwd, config })
	setLaunchStatus({ state: 'idle', error: null })
}

export async function startMainServerDevelopment(): Promise<void> {
	await startMainServer(DEVELOPMENT_PATH, DEVELOPMENT_CONFIG)
}

export async function stopMainServerDevelopment(): Promise<void> {
	await stopMainServer(DEVELOPMENT_PATH, DEVELOPMENT_CONFIG)
}

export async function startMainServerProduction(): Promise<void> {
	await startMainServer(PRODUCTION_PATH, PRODUCTION_CONFIG)
}

export async function stopMainServerProduction(): Promise<void> {
	await stopMainServer(PRODUCTION_PATH, PRODUCTION_CONFIG)
}

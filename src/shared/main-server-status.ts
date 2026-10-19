// Compose startup is owned by the main process. HTTP availability is observed
// separately by ServerContext; a completed compose command does not imply readiness.
export type MainServerLaunchStatus = Readonly<{
	state: 'idle' | 'starting' | 'started' | 'failed'
	error: string | null
}>

export const MAIN_SERVER_LAUNCH_STATUS_CHANNEL = 'main-server-launch-status'
export const MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL = 'get-main-server-launch-status'

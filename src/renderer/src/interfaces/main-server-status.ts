import {
	MAIN_SERVER_LAUNCH_STATUS_CHANNEL,
	MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL,
	type MainServerLaunchStatus
} from '@shared/main-server-status'

// Subscribe before requesting the snapshot: a slow initial response must not
// replace a newer launch event, and late responses must not update an unmount.
export function subscribeToMainServerLaunchStatus(
	ipc: Pick<Window['electron']['ipcRenderer'], 'on' | 'invoke'>,
	update: (status: MainServerLaunchStatus) => void
): () => void {
	let active = true
	let receivedEvent = false
	const unsubscribe = ipc.on(MAIN_SERVER_LAUNCH_STATUS_CHANNEL, (_event, status) => {
		receivedEvent = true
		if (active) update(status)
	})
	void ipc
		.invoke(MAIN_SERVER_LAUNCH_STATUS_REQUEST_CHANNEL)
		.then((status) => {
			if (active && !receivedEvent) update(status)
		})
		.catch(() => {
			// Without this IPC capability, the renderer can still observe HTTP reachability.
		})
	return () => {
		active = false
		unsubscribe()
	}
}

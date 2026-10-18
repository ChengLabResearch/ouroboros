import { parseTaskStatus, isTerminalTaskStatus, TaskStatus } from '../schemas/task-status-schema'

export type ServerError = { status: boolean; message: string }
export type StreamResult = {
	results: TaskStatus | null
	error: ServerError
	// Stream closure, not a scientific outcome. Transport failure also closes it.
	done: boolean
}

export function emptyStreamResult(): StreamResult {
	return { results: null, error: { status: false, message: '' }, done: false }
}

export function openTaskStatusStream(
	url: string,
	taskId: string,
	update: (state: Partial<StreamResult>) => void
): { close: () => void; finished: Promise<void> } {
	const source = new EventSource(url)
	let closed = false
	let resolve: () => void
	const finished = new Promise<void>((done) => {
		resolve = done
	})
	const close = (): void => {
		if (closed) return
		closed = true
		source.close()
		resolve()
	}
	const communicationFailure = (message: string): void => {
		if (closed) return
		update({ error: { status: true, message }, done: true })
		close()
	}
	update(emptyStreamResult())

	const receive = (event: MessageEvent<string>): void => {
		if (closed || event.lastEventId !== taskId) return
		let payload: object
		try {
			payload = JSON.parse(event.data) as object
		} catch {
			communicationFailure(
				'Invalid JSON from the status stream. The task outcome could not be determined.'
			)
			return
		}
		const { result, error } = parseTaskStatus(payload)
		if (error) {
			communicationFailure(`${error} The task outcome could not be determined.`)
			return
		}
		// get_status uses this legacy response when the task is no longer known.
		// It describes a lookup failure, not Task's scientific outcome.
		if (
			result.status === 'error' &&
			result.error === 'Item ID Not Found' &&
			result.statistics === undefined
		) {
			communicationFailure(
				'The server cannot find this task. Its outcome is unknown; the backend may have restarted. Last reported progress is shown.'
			)
			return
		}
		const done = isTerminalTaskStatus(result)
		// error_event can carry successful science with failed report publication.
		update({ results: result, error: { status: false, message: '' }, done })
		if (done) close()
	}
	for (const event of ['update_event', 'done_event', 'error_event']) {
		source.addEventListener(event, receive as EventListener)
	}
	source.addEventListener('error', () => {
		communicationFailure(
			'Connection to the status stream was lost. The task outcome is unknown; it may still be running. Last reported progress is shown.'
		)
	})
	return { close, finished }
}

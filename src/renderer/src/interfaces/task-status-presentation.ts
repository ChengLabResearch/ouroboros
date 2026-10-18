import type { AlertMessageType } from '../contexts/AlertContext'
import { isTerminalTaskStatus, TaskStatus } from '../schemas/task-status-schema'

export function taskStatusMessages(
	operation: string,
	status: TaskStatus | null
): Pick<AlertMessageType, 'type' | 'message'>[] {
	if (!status) return []
	const messages: Pick<AlertMessageType, 'type' | 'message'>[] = []
	if (isTerminalTaskStatus(status)) {
		if (status.status === 'done') {
			messages.push({ type: 'success', message: `${operation} completed.` })
		} else {
			// The API's aggregate error gives Task.error priority for scientific failures.
			messages.push({
				type: 'error',
				message: `${operation} ${status.status === 'interrupted' ? 'interrupted' : 'failed'}.${status.error ? ` ${status.error}` : ''}`
			})
		}
	}

	const publication = status.statistics
	if (publication && publication.status !== 'disabled') {
		const details: string[] = []
		const failed = ['failed', 'interrupted', 'unavailable'].includes(publication.status)
		if (failed) {
			details.push(
				publication.status === 'unavailable'
					? 'Statistics report unavailable.'
					: `Statistics delivery ${publication.status}.`
			)
			if (publication.error) details.push(publication.error)
		}
		if (publication.presentation_error) {
			details.push(
				`Statistics log presentation ${publication.presentation_interrupted ? 'interrupted' : 'failed'}: ${publication.presentation_error}`
			)
		}
		if (publication.path) {
			details.push(
				publication.status === 'published'
					? `Statistics report: ${publication.path}`
					: `Intended statistics destination (not published): ${publication.path}`
			)
		}
		if (publication.recovery_path)
			details.push(`Statistics recovery location: ${publication.recovery_path}`)
		if (details.length)
			messages.push({
				type: failed || publication.presentation_error ? 'warning' : 'info',
				message: details.join(' ')
			})
	}
	if (status.progress_error)
		messages.push({
			type: 'warning',
			message: `Progress details unavailable: ${status.progress_error}`
		})
	return messages
}

// Display vocabulary only. Execution and connection facts stay with their owners.
// Reference: documentation/docs/development/status-presentation.md.
export const STATUS_LABELS = {
	waiting: 'Waiting',
	running: 'Running',
	complete: 'Complete',
	warning: 'Warning',
	failed: 'Failed',
	interrupted: 'Interrupted',
	unknown: 'Unknown'
} as const

export type PresentationStatus = keyof typeof STATUS_LABELS

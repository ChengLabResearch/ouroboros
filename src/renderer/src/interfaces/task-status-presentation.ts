import type { AlertMessageType } from '../contexts/AlertContext'
import { isTerminalTaskStatus, TaskStatus } from '../schemas/task-status-schema'

// Presentation labels only: identifiers and execution state come from the server.
const STEP_LABELS: Record<string, string> = {
	ParseJSONPipelineStep: 'Read annotations and source',
	SlicesGeometryPipelineStep: 'Prepare slice geometry',
	VolumeCachePipelineStep: 'Prepare volume data',
	SliceParallelPipelineStep: 'Slicing',
	BackprojectPipelineStep: 'Backprojection',
	SaveConfigPipelineStep: 'Save configuration',
	LoadConfigPipelineStep: 'Load configuration'
}

const STAGE_LABELS: Record<string, string> = {
	downloading_and_slicing: 'Downloading data and slicing',
	normalizing_and_writing: 'Finalizing TIFF output',
	preparing_input: 'Preparing input',
	backprojecting_and_writing: 'Backprojecting and writing output',
	exporting_tiff: 'Exporting TIFF',
	cleaning_intermediates: 'Cleaning intermediate files'
}

const PHASE_LABELS: Record<string, string> = {
	queued: 'Waiting in queue',
	loading_input: 'Loading input',
	processing: 'Processing',
	returning_output: 'Returning output',
	cleanup: 'Cleaning up',
	publishing_statistics: 'Publishing statistics report'
}

export function stepLabel(name: string): string {
	return (
		STEP_LABELS[name] ?? name.replace(/PipelineStep$/, '').replace(/([a-z])([A-Z])/g, '$1 $2')
	)
}

export function stageLabel(stage: string): string {
	return STAGE_LABELS[stage] ?? stage.replaceAll('_', ' ')
}

export function phaseLabel(phase: string): string {
	return PHASE_LABELS[phase] ?? phase.replaceAll('_', ' ')
}

export function activeTaskLabel(status: TaskStatus | null): string | null {
	if (!status || isTerminalTaskStatus(status)) return null
	if (status.phase && status.phase !== 'complete') return phaseLabel(status.phase)
	if (status.status === 'enqueued') return phaseLabel('queued')
	// Fractions from older servers cannot distinguish computation from output delivery.
	return 'Working'
}

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

	if (status.failure_phase) {
		messages.push({
			type: 'info',
			message: `Failure stage: ${phaseLabel(status.failure_phase)}.`
		})
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

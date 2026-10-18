import { 
    array, boolean, InferOutput, nullable, number, object, optional, picklist, string, tuple
} from 'valibot'
import { baseParse } from './schema-helpers'

const PublicationSchema = object({
	status: picklist(['pending', 'published', 'disabled', 'unavailable', 'failed', 'interrupted']),
	path: nullable(string()),
	recovery_path: nullable(string()),
	error: nullable(string()),
	presentation_error: optional(nullable(string())),
	presentation_interrupted: optional(boolean())
})

const TaskStatusSchema = object({
	status: picklist(['enqueued', 'started', 'done', 'error', 'interrupted']),
	progress: array(tuple([string(), number(), number()])),
	error: nullable(string()),
	progress_error: optional(nullable(string())),
	// Older servers and unknown-task responses may omit this descriptor.
	statistics: optional(PublicationSchema)
})

export type TaskStatus = InferOutput<typeof TaskStatusSchema>
export const parseTaskStatus = baseParse(TaskStatusSchema, 'Invalid task status response.')

export function isTerminalTaskStatus(status: TaskStatus): boolean {
	return (
		status.statistics?.status !== 'pending' &&
		['done', 'error', 'interrupted'].includes(status.status)
	)
}

import type { ServerContextValue } from '../contexts/ServerContext'

// Both automatic terminal cleanup and the next submission's retry use this order.
export async function deleteOperation(
	deleteURL: string,
	optionsPath: string | undefined,
	performFetch: ServerContextValue['performFetch']
): Promise<boolean> {
	const result = await performFetch(deleteURL, {}, { method: 'POST' })
	if (
		result.error.status ||
		!result.results ||
		!('success' in result.results) ||
		result.results.success !== true
	) {
		return false
	}

	if (optionsPath) {
		try {
			await window.electron.ipcRenderer.invoke('evict-output-cache', optionsPath)
		} catch (error) {
			// Cache hygiene cannot change the server's scientific outcome.
			console.warn('Could not evict operation outputs from the host cache:', error)
		}
	}
	return true
}

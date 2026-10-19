import { useContext } from 'react'
import { ServerContext } from '../../../../contexts/ServerContext'
import Alert from '../../../AlertArea/components/Alert/Alert'

export type TaskRequestURLs = { start: string; delete: string | null }

export default function RequestNotices({ urls }: { urls: TaskRequestURLs }): React.JSX.Element {
	const { useFetchListener, clearFetchError } = useContext(ServerContext)
	const { start: startURL, delete: deleteURL } = urls
	const { error: startError } = useFetchListener(startURL)
	const { error: deleteError } = useFetchListener(deleteURL ?? '')

	return (
		<>
			{startError.status && (
				<Alert
					type="error"
					message={`Task submission failed: ${startError.message}`}
					onDismiss={() => clearFetchError(startURL)}
				/>
			)}
			{deleteURL && deleteError.status && (
				<Alert
					type="error"
					message={`Previous task could not be deleted: ${deleteError.message}`}
					onDismiss={() => clearFetchError(deleteURL)}
				/>
			)}
		</>
	)
}

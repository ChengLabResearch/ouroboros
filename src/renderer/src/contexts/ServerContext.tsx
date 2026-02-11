/* eslint-disable @typescript-eslint/no-explicit-any */
import { JSX, createContext, useCallback, useEffect, useState } from 'react'

const DEFAULT_SERVER_URL = 'http://127.0.0.1:8000'

export type ServerError = {
	status: boolean
	message: string
}

export type FetchResult = {
	results: object | null
	error: ServerError
}

export type StreamResult = {
	results: object | null
	error: ServerError
	done: boolean
}

export type ServerContextValue = {
	baseURL: string
	connected: boolean
	performFetch: (
		relativeURL: string,
		query?: Record<string, any>,
		options?: RequestInit
	) => Promise<void>
	performStream: (relativeURL: string, query?: Record<string, any>) => void
	clearFetch: (relativeURL: string) => void
	clearStream: (relativeURL: string) => void
	useFetchListener: (relativeURL: string) => FetchResult
	useStreamListener: (relativeURL: string) => StreamResult
}

export const ServerContext = createContext<ServerContextValue>(null as any)

function useServerContextProvider(baseURL = DEFAULT_SERVER_URL): ServerContextValue {
	const [connected, setConnected] = useState(false)
	const retryDelay = 5000 // Delay between checks in milliseconds

	const [fetchStates, setFetchStates] = useState<Map<string, FetchResult>>(new Map())
	const [streamStates, setStreamStates] = useState<Map<string, StreamResult>>(new Map())
	const [abortControllers, setAbortControllers] = useState<Map<string, AbortController>>(
		new Map()
	)
	const [fetchQueue, setFetchQueue] = useState<unknown[][]>([])

	const setFetchStatesHelper = useCallback(
		({
			relativeURL,
			results,
			error
		}: {
			relativeURL: string
			results?: object | null
			error?: ServerError
		}) => {
			setFetchStates(
				(prev) =>
					new Map(
						prev.set(relativeURL, {
							results:
								results == undefined
									? prev.get(relativeURL)?.results ?? null
									: results,
							error:
								error == undefined
									? prev.get(relativeURL)?.error ?? {
											status: false,
											message: ''
										}
									: error
						})
					)
			)
		},
		[]
	)

	const setStreamStatesHelper = useCallback(
		({
			relativeURL,
			results,
			error,
			done
		}: {
			relativeURL: string
			results?: object | null
			error?: ServerError
			done?: boolean
		}) => {
			setStreamStates(
				(prev) =>
					new Map(
						prev.set(relativeURL, {
							results:
								results == undefined
									? prev.get(relativeURL)?.results ?? null
									: results,
							error:
								error == undefined
									? prev.get(relativeURL)?.error ?? {
											status: false,
											message: ''
										}
									: error,
							done: done == undefined ? prev.get(relativeURL)?.done ?? false : done
						})
					)
			)
		},
		[]
	)

	const getFullURL = useCallback(
		(relativeURL: string, query = {}) => {
			// Append query parameters to the URL
			const searchParams = Object.keys(query)
				.map((key) => {
					const value = query[key]
					return `${key}=${value}`
				})
				.join('&')

			if (searchParams.toString().length > 0) {
				relativeURL += '?' + searchParams.toString()
			}

			return new URL(relativeURL, baseURL).toString()
		},
		[baseURL]
	)

	const performFetch = useCallback(
		async (relativeURL: string, query: Record<string, any> = {}, options: RequestInit = {}) => {
			// Enqueue the fetch request if the server is not connected
			if (!connected) {
				setFetchQueue((prev) => [...prev, [relativeURL, query, options]])
				return
			}

			const fullURL = getFullURL(relativeURL, query)

			setFetchStatesHelper({
				relativeURL,
				error: { status: false, message: '' }
			})

			try {
				const abortController = new AbortController()
				const signal = abortController.signal

				const response = await fetch(fullURL, { ...options, signal })
				const data = await response.json()

				// Add the abort controller to the state
				setAbortControllers((prev) => new Map(prev.set(relativeURL, abortController)))

				setFetchStatesHelper({
					relativeURL,
					results: data
				})
			} catch (error) {
				const message =
					error instanceof Error
						? error.message
						: 'Unknown error occurred while fetching data.'
				setFetchStatesHelper({
					relativeURL,
					error: { status: true, message: message }
				})
			}
		},
		[getFullURL, connected, fetchQueue]
	)

	const clearFetch = useCallback(
		(relativeURL: string) => {
			setFetchStatesHelper({
				relativeURL,
				results: null,
				error: { status: false, message: '' }
			})

			const abortController = abortControllers.get(relativeURL)

			// Abort the fetch request if it is still pending
			if (abortController) {
				abortController.abort()
				setAbortControllers((prev) => {
					prev.delete(relativeURL)
					return new Map(prev)
				})
			}
		},
		[abortControllers]
	)

	const performStream = useCallback(
		(relativeURL: string, query: Record<string, any> = {}): Promise<void> => {
			const fullURL = getFullURL(relativeURL, query)
			const eventSource = new EventSource(fullURL)

			eventSource.addEventListener('open', () => {
				setStreamStatesHelper({
					relativeURL,
					done: false,
					error: { status: false, message: '' }
				})
			})

			eventSource.addEventListener('update_event', (event) => {
				const data = JSON.parse(event.data)
				setStreamStatesHelper({
					relativeURL,
					results: data
				})
			})

			eventSource.addEventListener('done_event', (event) => {
				const data = JSON.parse(event.data)
				setStreamStatesHelper({
					relativeURL,
					results: data,
					done: true
				})
				eventSource.close()
			})

			eventSource.addEventListener('error_event', (event) => {
				const data = JSON.parse(event.data)
				setStreamStatesHelper({
					relativeURL,
					results: data
				})

				let error = 'Unknown error occurred while streaming data.'

				if ('error' in data && data.error && typeof data.error === 'string') {
					error = data.error
				}

				setStreamStatesHelper({
					relativeURL,
					error: { status: true, message: error },
					done: true
				})
				eventSource.close()
			})

			eventSource.addEventListener('error', (error) => {
				const message =
					error instanceof Error
						? error.message
						: 'Unknown error occurred while streaming data.'
				setStreamStatesHelper({
					relativeURL,
					error: { status: true, message: message }
				})
				eventSource.close()
			})

			return new Promise((resolve) => {
				// Resolve the promise when the done event is received
				eventSource.addEventListener('done_event', () => {
					resolve()
				})

				// Resolve the promise when the error event is received
				eventSource.addEventListener('error_event', () => {
					resolve()
				})
			})
		},
		[getFullURL]
	)

	const clearStream = useCallback((relativeURL: string) => {
		setStreamStatesHelper({
			relativeURL,
			results: null,
			error: { status: false, message: '' },
			done: false
		})
	}, [])

	const useFetchListener = (
		relativeURL: string
	): { results: object | null; error: ServerError } => {
		const [results, setResults] = useState<object | null>(null)
		const [error, setError] = useState<ServerError>({ status: false, message: '' })

		useEffect(() => {
			const state = fetchStates.get(relativeURL)
			if (state) {
				setResults(state.results)
				setError(state.error)
			}
		}, [relativeURL, fetchStates])

		return { results, error }
	}

	const useStreamListener = (
		relativeURL: string
	): { results: object | null; error: ServerError; done: boolean } => {
		const [results, setResults] = useState<object | null>(null)
		const [error, setError] = useState<ServerError>({ status: false, message: '' })
		const [done, setDone] = useState(false)

		useEffect(() => {
			const state = streamStates.get(relativeURL)
			if (state) {
				setResults(state.results)
				setError(state.error)
				setDone(state.done)
			}
		}, [relativeURL, streamStates])

		return { results, error, done }
	}

	// Process the fetch queue
	useEffect(() => {
		if (fetchQueue.length === 0 || !connected) return

		const [relativeURL, query, options] = fetchQueue[0] as [
			string,
			Record<string, any>,
			RequestInit
		]

		// Make sure the request is not already enqueued
		if (fetchQueue.slice(1).some((item) => item[0] === relativeURL)) {
			setFetchQueue((prev) => prev.slice(1))
			return
		}

		performFetch(relativeURL, query, options).finally(() => {
			setFetchQueue((prev) => prev.slice(1))
		})
	}, [connected, fetchQueue, performFetch])

	// Check server connection status
	useEffect(() => {
		const delay = (ms: number): Promise<void> =>
			new Promise((resolve) => setTimeout(resolve, ms))

		let isMounted = true // Flag to manage cleanup

		const checkServerStatus = async (): Promise<void> => {
			while (isMounted) {
				try {
					const response = await fetch(baseURL)
					if (response.ok) setConnected(true)
					else setConnected(false)
				} catch (error) {
					setConnected(false) // Ensure disconnected state on error
				}
				await delay(retryDelay) // Wait before next check
			}
		}

		checkServerStatus()

		// Cleanup function to stop polling when component unmounts
		return (): void => {
			isMounted = false
		}
	}, [baseURL, retryDelay])

	return {
		baseURL,
		connected,
		performFetch,
		performStream,
		clearFetch,
		clearStream,
		useFetchListener,
		useStreamListener
	}
}

function ServerProvider({
	baseURL = DEFAULT_SERVER_URL,
	children
}: {
	baseURL?: string
	children: React.ReactNode
}): JSX.Element {
	const serverContextValue = useServerContextProvider(baseURL)

	return <ServerContext.Provider value={serverContextValue}>{children}</ServerContext.Provider>
}

export default ServerProvider

/* eslint-disable @typescript-eslint/no-explicit-any */
import { JSX, createContext, useCallback, useEffect, useRef, useState } from 'react'

const DEFAULT_SERVER_URL = 'http://127.0.0.1:8000'

export type ServerError = {
	status: boolean
	message: string
}

export type FetchResult = {
	results: object | null
	error: ServerError
	pending: boolean
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
	) => Promise<FetchResult>
	performStream: (relativeURL: string, query?: Record<string, any>) => Promise<void>
	clearFetch: (relativeURL: string) => void
	clearStream: (relativeURL: string) => void
	useFetchListener: (relativeURL: string) => FetchResult
	useStreamListener: (relativeURL: string) => StreamResult
}

export const ServerContext = createContext<ServerContextValue>(null as any)

type ActiveStream = {
	eventSource: EventSource
	taskId: string
	resolve: () => void
}

function emptyFetchResult(): FetchResult {
	return {
		results: null,
		error: { status: false, message: '' },
		pending: false
	}
}

function responseErrorMessage(data: object, status: number): string {
	if ('detail' in data && typeof data.detail === 'string') return data.detail
	if ('error' in data && typeof data.error === 'string') return data.error
	return `Request failed with status ${status}.`
}

function useServerContextProvider(baseURL = DEFAULT_SERVER_URL): ServerContextValue {
	const [connected, setConnected] = useState(false)
	const retryDelay = 5000 // Delay between checks in milliseconds

	const [fetchStates, setFetchStates] = useState<Map<string, FetchResult>>(new Map())
	const [streamStates, setStreamStates] = useState<Map<string, StreamResult>>(new Map())
	const abortControllers = useRef<Map<string, AbortController>>(new Map())
	const activeFetches = useRef<Map<string, Promise<FetchResult>>>(new Map())
	const activeStreams = useRef<Map<string, ActiveStream>>(new Map())
	const [fetchQueue, setFetchQueue] = useState<unknown[][]>([])

	const setFetchStatesHelper = useCallback(
		({
			relativeURL,
			results,
			error,
			pending
		}: {
			relativeURL: string
			results?: object | null
			error?: ServerError
			pending?: boolean
		}) => {
			setFetchStates((prev) => {
				const previous = prev.get(relativeURL) ?? emptyFetchResult()
				return new Map(prev).set(relativeURL, {
					results: results === undefined ? previous.results : results,
					error: error === undefined ? previous.error : error,
					pending: pending === undefined ? previous.pending : pending
				})
			})
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
									? (prev.get(relativeURL)?.results ?? null)
									: results,
							error:
								error == undefined
									? (prev.get(relativeURL)?.error ?? {
											status: false,
											message: ''
										})
									: error,
							done: done == undefined ? (prev.get(relativeURL)?.done ?? false) : done
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
		async (
			relativeURL: string,
			query: Record<string, any> = {},
			options: RequestInit = {}
		): Promise<FetchResult> => {
			// Enqueue the fetch request if the server is not connected
			if (!connected) {
				setFetchQueue((prev) => [...prev, [relativeURL, query, options]])
				return emptyFetchResult()
			}

			const fullURL = getFullURL(relativeURL, query)
			const requestKey = `${(options.method ?? 'GET').toUpperCase()} ${fullURL}`
			const activeFetch = activeFetches.current.get(requestKey)
			if (activeFetch) return activeFetch

			const request = (async (): Promise<FetchResult> => {
				const abortController = new AbortController()
				setFetchStatesHelper({
					relativeURL,
					error: { status: false, message: '' },
					pending: true
				})
				abortControllers.current.set(relativeURL, abortController)

				try {
					const response = await fetch(fullURL, {
						...options,
						signal: abortController.signal
					})
					const data = (await response.json()) as object

					const result: FetchResult = response.ok
						? {
								results: data,
								error: { status: false, message: '' },
								pending: false
							}
						: {
								results: null,
								error: {
									status: true,
									message: responseErrorMessage(data, response.status)
								},
								pending: false
							}

					setFetchStatesHelper({ relativeURL, ...result })
					return result
				} catch (error) {
					if (abortController.signal.aborted) return emptyFetchResult()

					const result: FetchResult = {
						results: null,
						error: {
							status: true,
							message:
								error instanceof Error
									? error.message
									: 'Unknown error occurred while fetching data.'
						},
						pending: false
					}
					setFetchStatesHelper({ relativeURL, ...result })
					return result
				} finally {
					if (abortControllers.current.get(relativeURL) === abortController) {
						abortControllers.current.delete(relativeURL)
					}
				}
			})()

			activeFetches.current.set(requestKey, request)
			void request.finally(() => {
				if (activeFetches.current.get(requestKey) === request) {
					activeFetches.current.delete(requestKey)
				}
			})
			return request
		},
		[getFullURL, connected, setFetchStatesHelper]
	)

	const clearFetch = useCallback(
		(relativeURL: string) => {
			setFetchStatesHelper({
				relativeURL,
				results: null,
				error: { status: false, message: '' },
				pending: false
			})

			const abortController = abortControllers.current.get(relativeURL)

			// Abort the fetch request if it is still pending
			if (abortController) {
				abortController.abort()
				abortControllers.current.delete(relativeURL)
			}
		},
		[setFetchStatesHelper]
	)

	const closeStream = useCallback((relativeURL: string): void => {
		const activeStream = activeStreams.current.get(relativeURL)
		if (!activeStream) return

		activeStreams.current.delete(relativeURL)
		activeStream.eventSource.close()
		activeStream.resolve()
	}, [])

	const performStream = useCallback(
		(relativeURL: string, query: Record<string, any> = {}): Promise<void> => {
			closeStream(relativeURL)

			const fullURL = getFullURL(relativeURL, query)
			const eventSource = new EventSource(fullURL)
			return new Promise((resolve) => {
				const activeStream = {
					eventSource,
					taskId: typeof query.task_id === 'string' ? query.task_id : '',
					resolve
				}
				activeStreams.current.set(relativeURL, activeStream)

				const isCurrent = (event?: MessageEvent<string>): boolean => {
					if (activeStreams.current.get(relativeURL) !== activeStream) return false
					return event === undefined || event.lastEventId === activeStream.taskId
				}

				const finish = (): void => {
					if (activeStreams.current.get(relativeURL) === activeStream) {
						closeStream(relativeURL)
					}
				}

				setStreamStatesHelper({
					relativeURL,
					results: null,
					error: { status: false, message: '' },
					done: false
				})

				eventSource.addEventListener('open', () => {
					if (!isCurrent()) return
					setStreamStatesHelper({
						relativeURL,
						done: false,
						error: { status: false, message: '' }
					})
				})

				eventSource.addEventListener('update_event', (event) => {
					if (!isCurrent(event)) return
					setStreamStatesHelper({
						relativeURL,
						results: JSON.parse(event.data) as object
					})
				})

				eventSource.addEventListener('done_event', (event) => {
					if (!isCurrent(event)) return
					setStreamStatesHelper({
						relativeURL,
						results: JSON.parse(event.data) as object,
						done: true
					})
					finish()
				})

				eventSource.addEventListener('error_event', (event) => {
					if (!isCurrent(event)) return
					const data = JSON.parse(event.data) as { error?: unknown }
					const error =
						typeof data.error === 'string'
							? data.error
							: 'Unknown error occurred while streaming data.'

					setStreamStatesHelper({
						relativeURL,
						results: data,
						error: { status: true, message: error },
						done: true
					})
					finish()
				})

				eventSource.addEventListener('error', (error) => {
					if (!isCurrent()) return
					const message =
						error instanceof Error
							? error.message
							: 'Unknown error occurred while streaming data.'
					setStreamStatesHelper({
						relativeURL,
						error: { status: true, message },
						done: true
					})
					finish()
				})
			})
		},
		[closeStream, getFullURL, setStreamStatesHelper]
	)

	const clearStream = useCallback(
		(relativeURL: string) => {
			closeStream(relativeURL)
			setStreamStatesHelper({
				relativeURL,
				results: null,
				error: { status: false, message: '' },
				done: false
			})
		},
		[closeStream, setStreamStatesHelper]
	)

	useEffect(() => {
		return (): void => {
			for (const relativeURL of activeStreams.current.keys()) {
				closeStream(relativeURL)
			}
		}
	}, [closeStream])

	const useFetchListener = (relativeURL: string): FetchResult => {
		const [results, setResults] = useState<object | null>(null)
		const [error, setError] = useState<ServerError>({ status: false, message: '' })
		const [pending, setPending] = useState(false)

		useEffect(() => {
			const state = fetchStates.get(relativeURL)
			if (state) {
				setResults(state.results)
				setError(state.error)
				setPending(state.pending)
			}
		}, [relativeURL, fetchStates])

		return { results, error, pending }
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
				} catch (_error) {
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

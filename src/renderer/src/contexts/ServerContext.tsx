/* eslint-disable @typescript-eslint/no-explicit-any */
import { JSX, createContext, useCallback, useEffect, useRef, useState } from 'react'
import type { MainServerLaunchStatus } from '@shared/main-server-status'
import { subscribeToMainServerLaunchStatus } from '../interfaces/main-server-status'
import { deleteOperation } from '../interfaces/operation-cleanup'
import { parseSliceResult } from '../schemas/slice-result-schema'
import { parseBackprojectResult } from '../schemas/backproject-result-schema'
import { isTerminalTaskStatus } from '../schemas/task-status-schema'

import {
	emptyStreamResult,
	openTaskStatusStream,
	ServerError,
	StreamResult
} from '../interfaces/task-status-stream'
export type { ServerError, StreamResult } from '../interfaces/task-status-stream'

const DEFAULT_SERVER_URL = 'http://127.0.0.1:8000'

export type FetchResult = {
	results: object | null
	error: ServerError
	pending: boolean
	// Saved at submission, alongside the outputs; never read from the editable form.
	optionsPath?: string
}

export type ServerContextValue = {
	baseURL: string
	connected: boolean
	launchStatus: MainServerLaunchStatus | null
	performFetch: (
		relativeURL: string,
		query?: Record<string, any>,
		options?: RequestInit
	) => Promise<FetchResult>
	deleteTask: (streamURL: string, deleteURL: string, optionsPath?: string) => Promise<boolean>
	clearFetch: (relativeURL: string) => void
	clearFetchError: (relativeURL: string) => void
	clearStream: (relativeURL: string) => void
	useFetchListener: (relativeURL: string) => FetchResult
	useStreamListener: (relativeURL: string) => StreamResult
}

export const ServerContext = createContext<ServerContextValue>(null as any)

type ActiveStream = ReturnType<typeof openTaskStatusStream>

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
	const [launchStatus, setLaunchStatus] = useState<MainServerLaunchStatus | null>(null)
	const retryDelay = 5000 // Delay between checks in milliseconds

	const [fetchStates, setFetchStates] = useState<Map<string, FetchResult>>(new Map())
	const [streamStates, setStreamStates] = useState<Map<string, StreamResult>>(new Map())
	const abortControllers = useRef<Map<string, AbortController>>(new Map())
	const activeFetches = useRef<Map<string, Promise<FetchResult>>>(new Map())
	const activeStreams = useRef<Map<string, ActiveStream>>(new Map())
	const [fetchQueue, setFetchQueue] = useState<unknown[][]>([])

	useEffect(() => {
		// A custom URL is an external backend; the local Docker launcher does not own it.
		if (baseURL !== DEFAULT_SERVER_URL || !window.electron?.ipcRenderer) return
		return subscribeToMainServerLaunchStatus(window.electron.ipcRenderer, setLaunchStatus)
	}, [baseURL])

	const setFetchStatesHelper = useCallback(
		({
			relativeURL,
			results,
			error,
			pending,
			optionsPath
		}: {
			relativeURL: string
			results?: object | null
			error?: ServerError
			pending?: boolean
			optionsPath?: string
		}) => {
			setFetchStates((prev) => {
				const previous = prev.get(relativeURL) ?? emptyFetchResult()
				return new Map(prev).set(relativeURL, {
					results: results === undefined ? previous.results : results,
					error: error === undefined ? previous.error : error,
					pending: pending === undefined ? previous.pending : pending,
					optionsPath: results === undefined ? previous.optionsPath : optionsPath
				})
			})
		},
		[]
	)

	const setStreamStatesHelper = useCallback(
		({ relativeURL, ...update }: Partial<StreamResult> & { relativeURL: string }) => {
			setStreamStates((previous) =>
				new Map(previous).set(relativeURL, {
					...(previous.get(relativeURL) ?? emptyStreamResult()),
					...update
				})
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

	const closeStream = useCallback((relativeURL: string): void => {
		const activeStream = activeStreams.current.get(relativeURL)
		if (!activeStream) return
		activeStreams.current.delete(relativeURL)
		activeStream.close()
	}, [])

	const cleanupOperation = useCallback(
		async (
			streamURL: string,
			deleteURL: string,
			optionsPath: string | undefined,
			request: ServerContextValue['performFetch']
		): Promise<boolean> => {
			const stream = activeStreams.current.get(streamURL)
			setStreamStatesHelper({ relativeURL: streamURL, cleanupPending: true })
			try {
				return await deleteOperation(deleteURL, optionsPath, request)
			} finally {
				if (activeStreams.current.get(streamURL) === stream) {
					setStreamStatesHelper({ relativeURL: streamURL, cleanupPending: false })
				}
			}
		},
		[setStreamStatesHelper]
	)

	const performStream = useCallback(
		(relativeURL: string, taskId: string, onTerminal: () => Promise<void>): Promise<void> => {
			closeStream(relativeURL)
			let terminal = false
			const stream = openTaskStatusStream(
				getFullURL(relativeURL, { task_id: taskId }),
				taskId,
				(update) => {
					terminal = !!update.results && isTerminalTaskStatus(update.results)
					setStreamStatesHelper({ relativeURL, ...update, cleanupPending: terminal })
				}
			)
			activeStreams.current.set(relativeURL, stream)
			return stream.finished.then(async () => {
				try {
					// A disconnect, unknown task, or provider shutdown is not a terminal outcome.
					if (terminal) await onTerminal()
				} finally {
					if (activeStreams.current.get(relativeURL) === stream) {
						activeStreams.current.delete(relativeURL)
					}
				}
			})
		},
		[closeStream, getFullURL, setStreamStatesHelper]
	)

	const performFetch = useCallback(
		async function performFetch(
			relativeURL: string,
			query: Record<string, any> = {},
			options: RequestInit = {}
		): Promise<FetchResult> {
			// Enqueue the fetch request if the server is not connected
			if (!connected) {
				setFetchQueue((prev) => [...prev, [relativeURL, query, options]])
				return emptyFetchResult()
			}

			const fullURL = getFullURL(relativeURL, query)
			const requestKey = `${(options.method ?? 'GET').toUpperCase()} ${fullURL}`
			const activeFetch = activeFetches.current.get(requestKey)
			if (activeFetch) return activeFetch
			const optionsPath = typeof query.options === 'string' ? query.options : undefined

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

					const operation =
						options.method?.toUpperCase() === 'POST'
							? relativeURL === '/slice/'
								? 'slice'
								: relativeURL === '/backproject/'
									? 'backproject'
									: null
							: null
					if (response.ok && operation) {
						result.optionsPath = optionsPath
					}
					setFetchStatesHelper({ relativeURL, ...result })
					if (response.ok && operation) {
						const task = (
							operation === 'slice' ? parseSliceResult : parseBackprojectResult
						)(data)
						if (!task.error) {
							const deleteURL = `/delete/?task_id=${encodeURIComponent(task.result.task_id)}`
							void performStream(
								`/${operation}_status_stream/`,
								task.result.task_id,
								async () => {
									if (
										await cleanupOperation(
											`/${operation}_status_stream/`,
											deleteURL,
											result.optionsPath,
											performFetch
										)
									) {
										// Keep the terminal stream snapshot, but no longer offer a deleted task.
										setFetchStates((previous) => {
											if (
												previous.get(relativeURL)?.results !==
												result.results
											)
												return previous
											return new Map(previous).set(
												relativeURL,
												emptyFetchResult()
											)
										})
									}
								}
							).catch((error) => console.warn('Operation cleanup failed:', error))
						}
					}
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
		[getFullURL, connected, setFetchStatesHelper, performStream, cleanupOperation]
	)

	const deleteTask = useCallback(
		(streamURL: string, deleteURL: string, optionsPath?: string): Promise<boolean> =>
			cleanupOperation(streamURL, deleteURL, optionsPath, performFetch),
		[cleanupOperation, performFetch]
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

	const clearFetchError = useCallback(
		(relativeURL: string) => {
			setFetchStatesHelper({ relativeURL, error: { status: false, message: '' } })
		},
		[setFetchStatesHelper]
	)

	const clearStream = useCallback(
		(relativeURL: string) => {
			closeStream(relativeURL)
			setStreamStatesHelper({
				relativeURL,
				results: null,
				error: { status: false, message: '' },
				done: false,
				cleanupPending: false
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
		return fetchStates.get(relativeURL) ?? emptyFetchResult()
	}

	const useStreamListener = (relativeURL: string): StreamResult => {
		return streamStates.get(relativeURL) ?? emptyStreamResult()
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
		launchStatus: baseURL === DEFAULT_SERVER_URL ? launchStatus : null,
		performFetch,
		deleteTask,
		clearFetch,
		clearFetchError,
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

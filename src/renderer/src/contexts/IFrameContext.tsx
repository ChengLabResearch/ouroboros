import {
	IFrameMessage,
	IFrameMessageSchema,
	PluginNodeChildren,
	ReadFileRequestSchema,
	ReadFileResponse,
	RegisterIFrameSchema,
	RequestDirectoryContentsSchema,
	SaveFileRequestSchema,
	SendDirectoryContents,
	SendDirectoryContentsResponse
} from '@renderer/schemas/iframe-message-schema'
import { safeParse } from 'valibot'
import { readFile, writeFile } from '../interfaces/file'
import { JSX, createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import { DirectoryContext } from '@renderer/contexts/DirectoryContext'

export type IFrameContextValue = {
	broadcast: (message: IFrameMessage) => void
}

export const IFrameContext = createContext<IFrameContextValue>(null as never)

export function IFrameProvider({ children }: { children: React.ReactNode }): JSX.Element {
	// Allowed iframe host names. Plugin content is served locally by the plugin
	// file server (see `src/main/servers/file-server.ts`), so all legitimate
	// origins are loopback. We compare against `new URL(event.origin).hostname`
	// for exact equality rather than a `startsWith` prefix match, because
	// a prefix match would accept attacker-controlled hosts such as
	// `http://localhost.evil.example`. Any loopback port is accepted; the
	// plugin file server port itself is not fixed and can be overridden via
	// OUROBOROS_PLUGIN_FILE_SERVER_PORT in the main process.
	const allowedHostnames: string[] = ['localhost', '127.0.0.1', '0.0.0.0']

	// Store references to iframes
	const [iframes, setIframes] = useState(new Map<string, MessageEventSource>())

	const updateIframes = async (
		source: MessageEventSource,
		data: IFrameMessage
	): Promise<void> => {
		const result = safeParse(RegisterIFrameSchema, data)

		if (!result.success) return

		const { pluginName } = result.output.data

		setIframes((prev) => new Map([...prev, [pluginName, source]]))
	}

	// Access the selected directory without broadcasting the full recursive file tree.
	const { directoryPath, directoryName } = useContext(DirectoryContext)

	// The request-directory-contents handler needs the *current* directory
	// root at message time, not the value captured when the listener was
	// registered. A ref keeps the closure stable while the listener effect
	// runs once, matching the pattern used for the message listener below.
	const directoryPathRef = useRef<string | null>(directoryPath)
	useEffect(() => {
		directoryPathRef.current = directoryPath
	}, [directoryPath])

	// Define the handlers for the different message types
	const handlers: {
		[key: string]: (source: MessageEventSource, data: IFrameMessage) => Promise<void>
	} = {
		'read-file': handleReadFileRequest,
		'save-file': handleSaveFileRequest,
		'register-plugin': updateIframes,
		'request-directory-contents': (source, data) =>
			handleRequestDirectoryContentsRequest(source, data, directoryPathRef.current)
	}

	const broadcast = useCallback(
		(message: IFrameMessage): void => {
			iframes.forEach((iframe) => {
				iframe.postMessage(message, {
					targetOrigin: '*'
				})
			})
		},
		[iframes]
	)

	const broadcastDirectoryContext = useCallback((): void => {
		const message: SendDirectoryContents = {
			type: 'send-directory-contents',
			data: {
				directoryPath,
				directoryName,
				nodes: {}
			}
		}

		broadcast(message)
	}, [broadcast, directoryName, directoryPath])

	useEffect(() => {
		broadcastDirectoryContext()
	}, [broadcastDirectoryContext])

	useEffect(() => {
		const listener = (event: MessageEvent): void => {
			const origin = event.origin

			// Validate the origin of the request via exact hostname equality.
			if (!isAllowedOrigin(allowedHostnames, origin)) return

			const request = event.data

			// Validate the request format
			const messageParse = safeParse(IFrameMessageSchema, request)

			if (!messageParse.success) return

			const message = messageParse.output

			if (!event.source) return

			// Asynchronously handle the message
			if (message.type in handlers) {
				handlers[message.type](event.source, message).catch(console.error)
			}
		}

		// Create listener for messages from any iframe
		window.addEventListener('message', listener)

		return (): void => {
			window.removeEventListener('message', listener)
		}
	}, [])

	return <IFrameContext.Provider value={{ broadcast }}>{children}</IFrameContext.Provider>
}

async function handleReadFileRequest(
	source: MessageEventSource,
	data: IFrameMessage
): Promise<void> {
	// Validate the data format
	const parseResult = safeParse(ReadFileRequestSchema, data)

	if (!parseResult.success) return

	const readFileRequest = parseResult.output

	// Read the file contents
	const fileName = readFileRequest.data.fileName
	const folder = readFileRequest.data.folder
	let contents = ''

	try {
		contents = await readFile(folder, fileName)
	} catch (e) {
		console.error(e)
		return
	}

	// Send the file contents back to the iframe
	const response: ReadFileResponse = {
		type: 'read-file-response',
		data: {
			fileName,
			contents
		}
	}

	// Send the response to the iframe
	source.postMessage(response, {
		targetOrigin: '*'
	})
}

async function handleSaveFileRequest(_: MessageEventSource, data: IFrameMessage): Promise<void> {
	// Validate the data format
	const parseResult = safeParse(SaveFileRequestSchema, data)

	if (!parseResult.success) return

	const readFileRequest = parseResult.output

	const folder = readFileRequest.data.folder
	const fileName = readFileRequest.data.fileName
	const fileContents = readFileRequest.data.contents

	try {
		await writeFile(folder, fileName, fileContents)
	} catch (e) {
		console.error(e)
		return
	}
}

async function handleRequestDirectoryContentsRequest(
	source: MessageEventSource,
	data: IFrameMessage,
	directoryPath: string | null
): Promise<void> {
	const parseResult = safeParse(RequestDirectoryContentsSchema, data)

	// If the request itself is malformed, we cannot correlate a requestId
	// or trust the requested path. Report as `denied` with a null requestId
	// so the plugin still learns something went wrong.
	if (!parseResult.success) {
		sendDirectoryContentsResponse(source, {
			path: '',
			nodes: {},
			requestId: null,
			error: { code: 'denied', message: 'Malformed request-directory-contents payload.' }
		})
		return
	}

	const { path: requestedPath, recursive, requestId } = parseResult.output.data
	const correlationId = requestId ?? generateRequestId()

	// No root selected: plugins that request before the user opens a folder
	// get `denied` rather than a hanging promise.
	if (!directoryPath) {
		sendDirectoryContentsResponse(source, {
			path: requestedPath,
			nodes: {},
			requestId: correlationId,
			error: { code: 'denied', message: 'No directory is currently open.' }
		})
		return
	}

	// Path-traversal guard. `path.relative` returns `..`-prefixed results
	// when the target escapes the base, and absolute results when the
	// candidate is rooted on a different volume (Windows). Both are denied.
	if (!isPathUnderRoot(directoryPath, requestedPath)) {
		sendDirectoryContentsResponse(source, {
			path: requestedPath,
			nodes: {},
			requestId: correlationId,
			error: { code: 'denied', message: 'Path is outside the current directory root.' }
		})
		return
	}

	try {
		const { nodes, truncated } = (await window.electron.ipcRenderer.invoke(
			'get-folder-contents',
			{ folderPath: requestedPath, recursive: recursive === true, rootPath: directoryPath }
		)) as { nodes: PluginNodeChildren; truncated: boolean }

		sendDirectoryContentsResponse(source, {
			path: requestedPath,
			nodes,
			requestId: correlationId,
			error: truncated
				? {
						code: 'limit',
						message: `Enumeration truncated at aggregate cap; partial results returned.`,
						truncated: true
					}
				: undefined
		})
	} catch (e) {
		const err = e as { code?: string; message?: string }
		const isNotFound = err?.code === 'ENOENT' || err?.code === 'ENOTDIR'

		sendDirectoryContentsResponse(source, {
			path: requestedPath,
			nodes: {},
			requestId: correlationId,
			error: {
				code: isNotFound ? 'not-found' : 'internal',
				message: err?.message ?? 'Failed to read folder contents.'
			}
		})
	}
}

function sendDirectoryContentsResponse(
	source: MessageEventSource,
	data: SendDirectoryContentsResponse['data']
): void {
	const response: SendDirectoryContentsResponse = {
		type: 'send-directory-contents-response',
		data
	}
	source.postMessage(response, { targetOrigin: '*' })
}

function isPathUnderRoot(rootPath: string, candidate: string): boolean {
	// Reject empty candidates outright.
	if (!candidate) return false

	// A request for the root itself is allowed.
	if (candidate === rootPath) return true

	// Cross-volume checks on Windows: two absolute paths on different drives
	// yield an absolute result from `path.relative`. Node's `path.posix` and
	// `path.win32` behave the same way for this check on their respective
	// separators; the renderer runs against the platform-native module.
	// We use browser-safe string checks rather than importing 'path' in the
	// renderer bundle.
	const normalizedRoot = rootPath.replace(/[\\/]+$/, '')
	const rel = getPathRelative(normalizedRoot, candidate)
	if (rel === null) return false
	if (rel === '' || rel === '.') return true
	if (rel.startsWith('..')) return false
	// Absolute-looking segment means Windows drive change (e.g. `D:\foo`).
	if (/^[A-Za-z]:[\\/]/.test(rel)) return false
	if (rel.startsWith('/') || rel.startsWith('\\')) return false
	return true
}

// Minimal string-based relative computation that avoids pulling Node's
// `path` module into the renderer. Handles both POSIX and Windows-style
// separators. Returns `null` for pathological inputs (e.g. differing drive
// letters), which the caller treats as a deny.
function getPathRelative(rootPath: string, candidate: string): string | null {
	const rootNorm = rootPath.replace(/\\/g, '/')
	const candNorm = candidate.replace(/\\/g, '/')

	// Windows drive-letter mismatch: reject.
	const rootDrive = /^([A-Za-z]:)/.exec(rootNorm)?.[1]?.toUpperCase()
	const candDrive = /^([A-Za-z]:)/.exec(candNorm)?.[1]?.toUpperCase()
	if (rootDrive && candDrive && rootDrive !== candDrive) return null

	if (candNorm === rootNorm) return ''
	if (candNorm.startsWith(rootNorm + '/')) {
		return candNorm.slice(rootNorm.length + 1)
	}
	return '..'
}

function generateRequestId(): string {
	// Best-effort correlation id when the plugin didn't supply one. Not
	// security-sensitive; a short random suffix is enough.
	const g = globalThis as unknown as { crypto?: { randomUUID?: () => string } }
	if (g.crypto?.randomUUID) return g.crypto.randomUUID()
	return `req-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function isAllowedOrigin(allowedHostnames: string[], origin: string): boolean {
	// Parse the origin URL and compare the hostname exactly. Invalid URLs
	// (empty string for `postMessage` from opaque origins, malformed values)
	// fall through to a deny.
	let hostname: string
	try {
		hostname = new URL(origin).hostname
	} catch {
		return false
	}
	for (const item of allowedHostnames) {
		if (hostname === item) {
			return true
		}
	}

	return false
}

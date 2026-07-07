import {
	IFrameMessage,
	IFrameMessageSchema,
	ReadFileRequestSchema,
	ReadFileResponse,
	RegisterIFrameSchema,
	SaveFileRequestSchema,
	SendDirectoryContents
} from '@renderer/schemas/iframe-message-schema'
import { safeParse } from 'valibot'
import { readFile, writeFile } from '../interfaces/file'
import { JSX, createContext, useCallback, useContext, useEffect, useState } from 'react'
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

	// Define the handlers for the different message types
	const handlers: {
		[key: string]: (source: MessageEventSource, data: IFrameMessage) => Promise<void>
	} = {
		'read-file': handleReadFileRequest,
		'save-file': handleSaveFileRequest,
		'register-plugin': updateIframes
	}

	// Access the selected directory without broadcasting the full recursive file tree.
	const { directoryPath, directoryName } = useContext(DirectoryContext)

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

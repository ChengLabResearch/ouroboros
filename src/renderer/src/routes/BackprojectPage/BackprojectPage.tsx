import { JSX } from 'react'
import type { MainServerLaunchStatus } from '@shared/main-server-status'
import OptionsPanel from '@renderer/components/OptionsPanel/OptionsPanel'
import styles from './BackprojectPage.module.css'
import VisualizePanel from '@renderer/components/VisualizePanel/VisualizePanel'
import ProgressPanel from '@renderer/components/ProgressPanel/Progress'
import { ServerContext, StreamResult } from '@renderer/contexts/ServerContext'
import {
	CompoundEntry,
	Entry,
	BackprojectOptionsFile,
	CompoundValueType,
	findPathsToType
} from '@renderer/interfaces/options'
import { useContext, useEffect, useState } from 'react'
import { DirectoryContext } from '@renderer/contexts/DirectoryContext'
import { join, readFile, writeFile } from '@renderer/interfaces/file'
import { AlertContext } from '@renderer/contexts/AlertContext'
import { safeParse } from 'valibot'
import { parseBackprojectResult } from '@renderer/schemas/backproject-result-schema'

const BACKPROJECT_STREAM = '/backproject_status_stream/'

function BackprojectPage(): JSX.Element {
	const { stream, connected, launchStatus, entries, onSubmit, onHeaderDrop, isRunning } =
		useBackprojectPageState()

	return (
		<div className={styles.backprojectPage}>
			<VisualizePanel></VisualizePanel>
			<ProgressPanel
				operation="Backprojection"
				stream={stream}
				connected={connected}
				launchStatus={launchStatus}
			/>
			<OptionsPanel
				entries={entries}
				onSubmit={onSubmit}
				onHeaderDrop={onHeaderDrop}
				isRunning={isRunning}
			/>
		</div>
	)
}

type BackprojectPageState = {
	stream: StreamResult
	connected: boolean
	launchStatus: MainServerLaunchStatus | null
	entries: (Entry | CompoundEntry)[]
	onSubmit: () => Promise<void>
	onHeaderDrop: (content: string) => Promise<void>
	isRunning: boolean
}

function useBackprojectPageState(): BackprojectPageState {
	const {
		connected,
		launchStatus,
		performFetch,
		performStream,
		useFetchListener,
		useStreamListener,
		clearFetch,
		clearStream
	} = useContext(ServerContext)
	const { directoryPath } = useContext(DirectoryContext)

	const [entries, setEntries] = useState<(Entry | CompoundEntry)[]>([
		new BackprojectOptionsFile()
	])

	const { addAlert } = useContext(AlertContext)

	const {
		results: backprojectResults,
		error: backprojectRequestError,
		pending: backprojectRequestPending
	} = useFetchListener('/backproject/')
	const stream = useStreamListener(BACKPROJECT_STREAM)
	const streamDone = stream.done

	// Listen to the status stream for the active task
	useEffect(() => {
		const { result, error } = parseBackprojectResult(backprojectResults)

		if (!error) {
			void performStream(BACKPROJECT_STREAM, result)
		}
	}, [backprojectResults, performStream])

	useEffect(() => {
		if (backprojectRequestError.status) {
			addAlert(backprojectRequestError.message, 'error')
		}
	}, [backprojectRequestError])

	useEffect(() => {
		return (): void => clearStream(BACKPROJECT_STREAM)
	}, [clearStream])

	////// HANDLE OPTIONS PANEL FORM SUBMISSION //////
	const onSubmit = async (): Promise<void> => {
		if (!connected || !directoryPath) {
			return
		}

		const { result, error } = parseBackprojectResult(backprojectResults)

		// Delete the previous task if it exists
		if (!error) {
			const deleteResult = await performFetch('/delete/', result, { method: 'POST' })
			if (deleteResult.error.status) {
				addAlert(deleteResult.error.message, 'error')
				return
			}

			// Clear the task once it has been deleted
			clearFetch('/backproject/')
			clearStream(BACKPROJECT_STREAM)
		}

		const optionsObject = entries[0].toObject()

		const pathsToFilePathType = findPathsToType(entries[0], 'filePath')

		// Make all file paths absolute
		for (const path of pathsToFilePathType) {
			let current = optionsObject

			// Traverse the object to find the entry with the path
			for (let i = 0; i < path.length - 1; i++) current = current[path[i]]

			const name = path[path.length - 1]

			// Convert relative paths to absolute paths if necessary
			const filePathValue = current[name].startsWith('.')
				? await join(directoryPath, current[name])
				: current[name]

			// Add the absolute path to the options object
			current[name] = filePathValue
		}

		const outputFolder = optionsObject['output_file_folder']

		const outputName = optionsObject['output_file_name']

		// Validate options
		if (
			!optionsObject['output_file_folder'] ||
			!outputName ||
			!optionsObject['straightened_volume_path'] ||
			!optionsObject['slice_options_path'] ||
			optionsObject['output_file_folder'] === '' ||
			outputName === '' ||
			optionsObject['straightened_volume_path'] === '' ||
			optionsObject['slice_options_path'] === ''
		) {
			return
		}

		const modifiedName = `${outputName}-backproject-options.json`

		// Save options to file
		await writeFile(outputFolder, modifiedName, JSON.stringify(optionsObject, null, 4))

		const outputOptions = await join(outputFolder, modifiedName)

		// Run the backprojection with this submission's saved reporting setting.
		await performFetch(
			'/backproject/',
			{
				options: outputOptions,
				reporting_enabled: optionsObject['reporting_enabled']
			},
			{ method: 'POST' }
		)
	}

	////// HANDLE FILE DROP ON HEADER IN OPTIONS PANEL //////
	const onHeaderDrop = async (content: string): Promise<void> => {
		if (!directoryPath || !content || content === '') return

		const fileContent = await readFile('', content)

		let jsonContent = null

		try {
			jsonContent = JSON.parse(fileContent)
		} catch (_e) {
			addAlert('Invalid JSON file', 'error')
			return
		}

		const schema = entries[0].toSchema()

		const parseResult = safeParse(schema, jsonContent)

		if (!parseResult.success) {
			addAlert(
				'Wrong JSON file format. Make sure you provide a backprojection options JSON file.',
				'error'
			)
			return
		}

		// Update the entries with the new values from the file
		entries[0].setValue(parseResult.output as CompoundValueType)
		setEntries([...entries])
	}

	const backprojectTaskActive = !parseBackprojectResult(backprojectResults).error && !streamDone
	const isRunning = backprojectRequestPending || backprojectTaskActive

	return { stream, connected, launchStatus, entries, onSubmit, onHeaderDrop, isRunning }
}

export default BackprojectPage

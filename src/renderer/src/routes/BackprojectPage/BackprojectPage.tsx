import { JSX } from "react"
import OptionsPanel from '@renderer/components/OptionsPanel/OptionsPanel'
import styles from './BackprojectPage.module.css'
import VisualizePanel from '@renderer/components/VisualizePanel/VisualizePanel'
import ProgressPanel, { ProgressType } from '@renderer/components/ProgressPanel/Progress'
import { ServerContext } from '@renderer/contexts/ServerContext'
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
import { parseBackprojectStatusResult } from '@renderer/schemas/backproject-status-result-schema'

const BACKPROJECT_STREAM = '/backproject_status_stream/'

function BackprojectPage(): JSX.Element {
	const { progress, connected, entries, onSubmit, onHeaderDrop } = useBackprojectPageState()

	return (
		<div className={styles.backprojectPage}>
			<VisualizePanel></VisualizePanel>
			<ProgressPanel progress={progress} connected={connected} />
			<OptionsPanel entries={entries} onSubmit={onSubmit} onHeaderDrop={onHeaderDrop} />
		</div>
	)
}

type BackprojectPageState = {
	progress: ProgressType[]
	connected: boolean
	entries: (Entry | CompoundEntry)[]
	onSubmit: () => Promise<void>
	onHeaderDrop: (content: string) => Promise<void>
}

function useBackprojectPageState(): BackprojectPageState {
	const {
		connected,
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

	const [progress, setProgress] = useState<ProgressType[]>([])

	const { results: backprojectResults } = useFetchListener('/backproject/')
	const {
		results: streamResults,
		error: streamError,
		done: streamDone
	} = useStreamListener(BACKPROJECT_STREAM)

	// Listen to the status stream for the active task
	useEffect(() => {
		const { result, error } = parseBackprojectResult(backprojectResults)

		if (!error) {
			performStream(BACKPROJECT_STREAM, result)
		}
	}, [backprojectResults])

	// Update the progress state when new data is received
	useEffect(() => {
		const { result, error } = parseBackprojectStatusResult(streamResults)

		if (!error && !result.error) {
			setProgress(result.progress)
		}
	}, [streamResults])

	// Refresh the file list when the task is done
	useEffect(() => {
		if (streamError?.status) {
			addAlert(streamError.message, 'error')
		}
	}, [streamDone, streamError])

	////// HANDLE OPTIONS PANEL FORM SUBMISSION //////
	const onSubmit = async (): Promise<void> => {
		if (!connected || !directoryPath) {
			return
		}

		const { result, error } = parseBackprojectResult(backprojectResults)

		// Delete the previous task if it exists
		if (!error) {
			performFetch('/delete/', result, { method: 'POST' }).then(() => {
				// Clear the task once it has been deleted
				clearFetch('/slice/')
				clearStream(BACKPROJECT_STREAM)
			})
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

		// Run the slice generation
		performFetch(
			'/backproject/',
			{
				options: outputOptions
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
		} catch (e) {
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

	return { progress, connected, entries, onSubmit, onHeaderDrop }
}

export default BackprojectPage

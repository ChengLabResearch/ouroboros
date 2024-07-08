import OptionsPanel from '@renderer/components/OptionsPanel/OptionsPanel'
import styles from './SlicesPage.module.css'
import VisualizePanel from '@renderer/components/VisualizePanel/VisualizePanel'
import ProgressPanel from '@renderer/components/ProgressPanel/Progress'
import { ServerContext } from '@renderer/contexts/ServerContext/ServerContext'
import { CompoundEntry, Entry, OptionsFile } from '@renderer/lib/options'
import { useContext, useEffect, useState } from 'react'
import { DirectoryContext } from '@renderer/contexts/DirectoryContext/DirectoryContext'
import { join, writeFile } from '@renderer/lib/file'
import { AlertContext } from '@renderer/contexts/AlertContext/AlertContext'
import VisualizeSlicing from './components/VisualizeSlicing/VisualizeSlicing'

const SLICE_STREAM = '/slice_status_stream/'

function SlicesPage(): JSX.Element {
	const { progress, connected, entries, onSubmit, visualizationResults } = useSlicePageState()

	const visualizationData =
		visualizationResults && 'data' in visualizationResults
			? visualizationDataToProps(
					visualizationResults.data as {
						rects: number[][][]
						bounding_boxes: { min: number[]; max: number[] }[]
						link_rects: number[]
					}
				)
			: null

	return (
		<div className={styles.slicePage}>
			<VisualizePanel>
				{visualizationData ? (
					<VisualizeSlicing {...visualizationData} useEveryNthRect={10} />
				) : null}
			</VisualizePanel>
			<ProgressPanel progress={progress} connected={connected} />
			<OptionsPanel entries={entries} onSubmit={onSubmit} />
		</div>
	)
}

function visualizationDataToProps(
	visualizationData: {
		rects: number[][][]
		bounding_boxes: { min: number[]; max: number[] }[]
		link_rects: number[]
	} | null
) {
	if (!visualizationData) {
		return null
	}

	const rects = visualizationData.rects.map((rect) => {
		return { topLeft: rect[0], topRight: rect[1], bottomRight: rect[2], bottomLeft: rect[3] }
	})
	const boundingBoxes = visualizationData.bounding_boxes
	const linkRects = visualizationData.link_rects

	return { rects, boundingBoxes, linkRects }
}

function useSlicePageState() {
	const { connected, performFetch, useFetchListener, performStream, useStreamListener } =
		useContext(ServerContext)
	const { directoryPath, refreshDirectory } = useContext(DirectoryContext)

	const [entries] = useState<(Entry | CompoundEntry)[]>([
		new Entry('neuroglancer_json', 'Neuroglancer JSON', '', 'filePath'),
		new OptionsFile()
	])

	const onSubmit = async () => {
		if (!connected) {
			return
		}

		const optionsObject = entries[1].toObject()

		const outputFolder = await join(directoryPath, optionsObject['output_file_folder'])

		// Add the absolute output folder to the options object
		optionsObject['output_file_folder'] = outputFolder

		const outputName = optionsObject['output_file_name']
		const neuroglancerJSON = await join(directoryPath, entries[0].toObject() as string)

		// Validate options
		if (
			!optionsObject['output_file_folder'] ||
			!outputName ||
			!entries[0].toObject() ||
			optionsObject['output_file_folder'] === '' ||
			outputName === '' ||
			entries[0].toObject() === ''
		) {
			return
		}

		const modifiedName = `${outputName}-options-slice.json`

		// Save options to file
		await writeFile(outputFolder, modifiedName, JSON.stringify(optionsObject, null, 4))

		refreshDirectory()

		const outputOptions = await join(outputFolder, modifiedName)

		// Run the slice generation
		performFetch(
			'/slice/',
			{ neuroglancer_json: neuroglancerJSON, options: outputOptions },
			{ method: 'POST' }
		)
	}

	const { addAlert } = useContext(AlertContext)

	const [progress, setProgress] = useState<any>([])

	const { results: fetchResults } = useFetchListener('/slice/')
	const {
		results: streamResults,
		error: streamError,
		done: streamDone
	} = useStreamListener(SLICE_STREAM)

	const { results: visualizationResults } = useFetchListener('/slice_visualization/')

	// Listen to the status stream for the active task
	useEffect(() => {
		if (fetchResults && 'task_id' in fetchResults) {
			performStream(SLICE_STREAM, fetchResults)
		}
	}, [fetchResults])

	// Update the progress state when new data is received
	useEffect(() => {
		if (streamResults && 'progress' in streamResults) {
			if (!('error' in streamResults && streamResults.error)) {
				setProgress(streamResults.progress)
			}
		}
	}, [streamResults])

	// Refresh the file list when the task is done
	useEffect(() => {
		refreshDirectory()

		if (streamError?.status) {
			addAlert(streamError.message, 'error')
		}

		if (streamDone && fetchResults && 'task_id' in fetchResults) {
			// Get the visualization data
			performFetch('/slice_visualization/', fetchResults)
		}

		// if (streamDone && !streamError?.status) {
		// 	addAlert('Task completed successfully!', 'success')
		// 	refreshDirectory()

		// 	// Delete the task from the server
		// 	if (fetchResults && 'task_id' in fetchResults) {
		// 		performFetch('/delete/', fetchResults, { method: 'POST' })
		// 	}
		// } else if (streamError?.status) {
		// 	addAlert(streamError.message, 'error')
		// 	refreshDirectory()

		// 	// Delete the task from the server
		// 	if (fetchResults && 'task_id' in fetchResults) {
		// 		performFetch('/delete/', fetchResults, { method: 'POST' })
		// 	}
		// }
	}, [streamDone, streamError])

	return { progress, connected, entries, onSubmit, visualizationResults }
}

export default SlicesPage

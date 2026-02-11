import OptionsPanel from '@renderer/components/OptionsPanel/OptionsPanel'
import styles from './SlicesPage.module.css'
import VisualizePanel from '@renderer/components/VisualizePanel/VisualizePanel'
import ProgressPanel, { ProgressType } from '@renderer/components/ProgressPanel/Progress'
import { ServerContext } from '@renderer/contexts/ServerContext'
import {
	CompoundEntry,
	CompoundValueType,
	Entry,
	findPathsToType,
	SliceOptionsFile
} from '@renderer/interfaces/options'
import { JSX, useContext, useEffect, useState } from 'react'
import { DirectoryContext } from '@renderer/contexts/DirectoryContext'
import { join, readFile, writeFile } from '@renderer/interfaces/file'
import { AlertContext } from '@renderer/contexts/AlertContext'
import VisualizeSlicing, {
	VisualizationOutput
} from './components/VisualizeSlicing/VisualizeSlicing'
import { parseSliceResult } from '@renderer/schemas/slice-result-schema'
import { safeParse } from 'valibot'
import { parseSliceStatusResult } from '@renderer/schemas/slice-status-result-schema'
import { parseNeuroglancerJSON } from '@renderer/schemas/neuroglancer-json-schema'
import { parseSliceVisualizationToOutputFormat } from '@renderer/schemas/slice-visualization-result-schema'

const SLICE_RENDER_PROPORTION = 0.008

const SLICE_STREAM = '/slice_status_stream/'

const SLICE_STEP_NAME = 'SliceParallelPipelineStep'

function SlicesPage(): JSX.Element {
	const {
		progress,
		boundingBoxProgress,
		connected,
		entries,
		onSubmit,
		visualizationData,
		isNewVisualization,
		onEntryChange,
		onHeaderDrop
	} = useSlicePageState()

	return (
		<div className={styles.slicePage}>
			<VisualizePanel>
				{visualizationData ? (
					<VisualizeSlicing
						isNew={isNewVisualization}
						{...visualizationData}
						useEveryNthRect={Math.floor(
							visualizationData.rects.length * SLICE_RENDER_PROPORTION
						)}
						bboxPercent={boundingBoxProgress}
					/>
				) : null}
			</VisualizePanel>
			<ProgressPanel progress={progress} connected={connected} />
			<OptionsPanel
				entries={entries}
				onSubmit={onSubmit}
				onEntryChange={onEntryChange}
				onHeaderDrop={onHeaderDrop}
			/>
		</div>
	)
}

type SlicePageState = {
	progress: ProgressType[]
	boundingBoxProgress: number
	connected: boolean
	entries: (Entry | CompoundEntry)[]
	onSubmit: () => Promise<void>
	visualizationData: VisualizationOutput | null
	isNewVisualization: boolean
	onEntryChange: (entry: Entry) => Promise<void>
	onHeaderDrop: (content: string) => Promise<void>
}

function useSlicePageState(): SlicePageState {
	const {
		connected,
		performFetch,
		useFetchListener,
		performStream,
		useStreamListener,
		clearFetch,
		clearStream
	} = useContext(ServerContext)
	const { directoryPath } = useContext(DirectoryContext)

	const [entries, setEntries] = useState<(Entry | CompoundEntry)[]>([new SliceOptionsFile()])

	const { addAlert } = useContext(AlertContext)

	const [progress, setProgress] = useState<ProgressType[]>([])

	const { results: sliceResults } = useFetchListener('/slice/')
	const {
		results: streamResults,
		error: streamError,
		done: streamDone
	} = useStreamListener(SLICE_STREAM)

	const [isNewVisualization, setIsNewVisualization] = useState(true)
	const [visualizationData, setVisualizationData] = useState<VisualizationOutput | null>(null)

	const { results: onDemandVisualizationResults } = useFetchListener(
		'/create_slice_visualization/'
	)

	// Update the visualization data when new data is received
	useEffect(() => {
		const { result, error } = parseSliceVisualizationToOutputFormat(
			onDemandVisualizationResults
		)

		if (!error) {
			setVisualizationData(result)
		}
	}, [onDemandVisualizationResults])

	// Listen to the status stream for the active task
	useEffect(() => {
		const { result, error } = parseSliceResult(sliceResults)

		if (!error) {
			performStream(SLICE_STREAM, result)
		}
	}, [sliceResults])

	// Update the progress state when new data is received
	useEffect(() => {
		const { result, error } = parseSliceStatusResult(streamResults)

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

	const saveOptionsToFile = async (): Promise<string | undefined> => {
		if (!directoryPath) return

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
			!optionsObject['neuroglancer_json'] ||
			optionsObject['output_file_folder'] === '' ||
			outputName === '' ||
			optionsObject['neuroglancer_json'] === ''
		) {
			return
		}

		const modifiedName = `${outputName}-slice-options.json`

		// Save options to file
		await writeFile(outputFolder, modifiedName, JSON.stringify(optionsObject, null, 4))

		const outputOptions = await join(outputFolder, modifiedName)

		return outputOptions
	}

	const requestVisualization = async (): Promise<void> => {
		if (!directoryPath) return

		if (onDemandVisualizationResults) clearFetch('/create_slice_visualization/')

		// Save the options to a file
		const outputOptions = await saveOptionsToFile()

		if (!outputOptions) return

		// Run the visualization
		performFetch('/create_slice_visualization/', { options: outputOptions })
	}

	////// HANDLE ENTRY CHANGES IN OPTIONS PANEL /////////
	const onEntryChange = async (entry: Entry): Promise<void> => {
		if (entry.name === 'neuroglancer_json' && directoryPath) {
			if (entry.value === '') return

			const neuroglancerJSONContent = await readFile('', entry.value as string)

			const { result, error } = parseNeuroglancerJSON(neuroglancerJSONContent)

			if (error) {
				addAlert(error, 'error')
				return
			}

			const imageLayers: { type: string; name: string }[] = []
			const annotationLayers: { type: string; name: string }[] = []

			// Read all image and annotation layers from the Neuroglancer JSON
			for (const layer of result['layers']) {
				if (layer.type === 'image' && layer.name !== '') {
					imageLayers.push(layer)
				} else if (layer.type === 'annotation' && layer.name !== '') {
					annotationLayers.push(layer)
				}
			}

			// Update the options for the image and annotation layer entries
			if (entries[0] instanceof CompoundEntry) {
				entries[0].getEntries().forEach((entry) => {
					if (entry.name === 'neuroglancer_image_layer' && entry instanceof Entry) {
						entry.options = imageLayers.map((layer) => layer.name)

						if (imageLayers.length > 0) entry.value = imageLayers[0].name
					} else if (
						entry.name === 'neuroglancer_annotation_layer' &&
						entry instanceof Entry
					) {
						entry.options = annotationLayers.map((layer) => layer.name)

						if (annotationLayers.length > 0) entry.value = annotationLayers[0].name
					}
				})

				setEntries([...entries])
			}
		}

		// Request visualization when the Neuroglancer JSON, slice dimensions, or bounding box params are changed
		const visualizationEntries = new Set([
			'neuroglancer_json',
			'neuroglancer_annotation_layer',
			'slice_width',
			'slice_height',
			'dist_between_slices',
			'use_adaptive_slicing',
			'adaptive_slicing_ratio',
			'bounding_box',
			'max_depth',
			'target_slices_per_box'
		])

		const streamInProgress = !streamDone && progress.length > 0

		if (visualizationEntries.has(entry.name) && entry.value !== '' && !streamInProgress) {
			requestVisualization()

			// Mark the visualization as new only if the Neuroglancer JSON is changed
			const isNew =
				entry.name === 'neuroglancer_json' || entry.name === 'neuroglancer_annotation_layer'
			setIsNewVisualization(isNew)
		}
	}

	//////// HANDLE OPTIONS FORM SUBMISSION /////////
	const onSubmit = async (): Promise<void> => {
		if (!connected || !directoryPath) {
			return
		}

		const { result, error } = parseSliceResult(sliceResults)

		// Delete the previous task if it exists
		if (!error) {
			performFetch('/delete/', result, { method: 'POST' }).then(() => {
				// Clear the task once it has been deleted
				clearFetch('/slice/')
				clearStream(SLICE_STREAM)
			})
		}

		const outputOptions = await saveOptionsToFile()

		if (!outputOptions) return

		// Run the slice generation
		performFetch('/slice/', { options: outputOptions }, { method: 'POST' })
	}

	////// HANDLE FILE DROP ONTO HEADER OF OPTIONS PANEL //////
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
				'Wrong JSON file format. Make sure you provide a slice options JSON file.',
				'error'
			)
			return
		}

		// Update the entries with the new values from the file
		entries[0].setValue(parseResult.output as CompoundValueType)
		setEntries([...entries])

		// Update the neuroglancer JSON entry
		if (entries[0] instanceof CompoundEntry) {
			const neuroglancerJSONEntry = entries[0].findEntry('neuroglancer_json')

			if (neuroglancerJSONEntry instanceof Entry)
				onEntryChange(entries[0].findEntry('neuroglancer_json') as Entry)
		}
	}

	// Synchronize the visualization with the slice progress
	const boundingBoxProgress = visualizationData
		? (progress.find(([name]) => name === SLICE_STEP_NAME) ?? [SLICE_STEP_NAME, 0.0])[1]
		: 0.0

	return {
		progress,
		boundingBoxProgress,
		connected,
		entries,
		onSubmit,
		visualizationData,
		isNewVisualization,
		onEntryChange,
		onHeaderDrop
	}
}

export default SlicesPage

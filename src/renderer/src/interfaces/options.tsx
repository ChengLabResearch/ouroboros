import { any, boolean, number, object, string } from 'valibot'

import { Schema } from '@renderer/schemas/schema-helpers'

export type CompoundValueType = { [key: string]: CompoundValueType | ValueType } | ValueType
export type ValueType = number | string | boolean
export type EntryValueType = 'number' | 'string' | 'boolean' | 'filePath'

export class Entry {
	name: string
	label: string
	value: ValueType
	type: EntryValueType
	options?: string[]
	hidden: boolean = false
	description?: string
	separator: boolean = false

	constructor(
		name: string,
		label: string,
		value: ValueType,
		type: EntryValueType,
		options?: string[]
	) {
		this.name = name
		this.label = label
		this.value = value
		this.type = type
		this.options = options
	}

	withHidden(): Entry {
		this.hidden = true
		return this
	}

	withDescription(description: string): Entry {
		this.description = description
		return this
	}

	withSeparator(): Entry {
		this.separator = true
		return this
	}

	setValue(value: CompoundValueType): void {
		if (typeof value === 'object') return

		if (typeof value !== typeof this.value) return

		this.value = value
	}

	setValueFromEntry(entry: Entry | CompoundEntry): void {
		if (entry instanceof Entry) this.setValue(entry.value)
	}

	toObject(): ValueType {
		return this.value
	}

	toSchema(): Schema {
		switch (this.type) {
			case 'number':
				return number()
			case 'string':
			case 'filePath':
				return string()
			case 'boolean':
				return boolean()
			default:
				return any()
		}
	}
}

export class CompoundEntry {
	name: string
	label: string
	entries: (Entry | CompoundEntry)[]
	entryMap: { [key: string]: Entry | CompoundEntry } = {}
	schema: Schema | null = null
	hidden: boolean = false
	separator: boolean = false

	constructor(name: string, label: string, entries: (Entry | CompoundEntry)[]) {
		this.name = name
		this.label = label
		this.entries = entries

		// Create the entry map
		for (const entry of entries) {
			this.entryMap[entry.name] = entry
		}
	}

	withHidden(): CompoundEntry {
		this.hidden = true
		return this
	}

	withSeparator(): CompoundEntry {
		this.separator = true
		return this
	}

	setValue(value: CompoundValueType): void {
		if (typeof value !== 'object') return

		for (const [name, val] of Object.entries(value)) {
			if (!(name in this.entryMap)) return

			this.entryMap[name].setValue(val)
		}
	}

	setValueFromEntry(entry: Entry | CompoundEntry): void {
		if (entry instanceof Entry) return

		for (const _entry of entry.entries) {
			if (!(_entry.name in this.entryMap)) return

			this.entryMap[_entry.name].setValueFromEntry(_entry)
		}
	}

	toObject(includeSelf = false): CompoundValueType {
		const result = {}

		for (const entry of this.entries) {
			result[entry.name] = entry.toObject()
		}

		if (includeSelf) {
			const includeSelfResult = {}
			includeSelfResult[this.name] = result
			return includeSelfResult
		}

		return result
	}

	toSchema(): Schema {
		if (this.schema) return this.schema

		const result = {}

		// Create a schema for each entry
		for (const entry of this.entries) {
			result[entry.name] = entry.toSchema()
		}

		this.schema = object(result)

		return this.schema
	}

	getEntries(): (Entry | CompoundEntry)[] {
		return this.entries
	}

	findEntry(name: string): Entry | CompoundEntry | null {
		if (name in this.entryMap) return this.entryMap[name]

		for (const entry of this.entries) {
			if (entry instanceof CompoundEntry) {
				const result = entry.findEntry(name)
				if (result) return result
			}
		}

		return null
	}
}

export class SliceOptionsFile extends CompoundEntry {
	constructor(values: CompoundValueType = {}) {
		super('options', 'Options File', [
			new Entry('neuroglancer_json', 'Neuroglancer JSON', '', 'filePath').withDescription(
				'Path to the Neuroglancer configuration JSON file (exported from Neuroglancer website or Plugin)'
			),
			new Entry(
				'neuroglancer_image_layer',
				'Neuroglancer Image Layer',
				'',
				'string'
			).withDescription('Select Neuroglancer image layer to slice from.'),
			new Entry(
				'neuroglancer_annotation_layer',
				'Neuroglancer Annotation Layer',
				'',
				'string'
			)
				.withDescription('Select Neuroglancer annotation layer to slice from.')
				.withSeparator(),
			new Entry('slice_width', 'Slice Width', 120, 'number').withDescription(
				'The output width of each slice image.'
			),
			new Entry('slice_height', 'Slice Height', 120, 'number').withDescription(
				'The output height of each slice image.'
			),
			new Entry('output_file_folder', 'Output File Folder', './', 'filePath').withDescription(
				'The folder to save all the resulting files into.'
			),
			new Entry('output_file_name', 'Output File Name', 'sample', 'string').withDescription(
				'Base name for all output files.'
			),
			new Entry('annotation_mip_level', 'Annotation MIP Level', 0, 'number').withDescription(
				"The annotation layer's MIP level. 0 is the highest resolution."
			),
			new Entry('output_mip_level', 'Output MIP Level', 0, 'number').withDescription(
				'The MIP level to output slices in (essentially a downsample option). 1 is a good starting point.'
			),
			new CompoundEntry('slicing_params', 'Slicing Parameters', [
				new Entry(
					'dist_between_slices',
					'Distance Between Slices',
					1,
					'number'
				).withDescription('The distance between each slice along the annotation path.'),
				new Entry(
					'use_adaptive_slicing',
					'Use Adaptive Slicing',
					true,
					'boolean'
				).withDescription(
					'Rather than just using equidistant slices, add more slices in more curved areas.'
				),
				new Entry(
					'adaptive_slicing_ratio',
					'Adaptive Slicing Ratio',
					0.5,
					'number'
				).withDescription(
					'1 indicates to consider distance and curvature equally, 0.5 is biased towards distance, and 2 is biased towards curvature.'
				)
			]),
			new Entry('make_single_file', 'Output Single File', true, 'boolean').withDescription(
				'Whether to output one tiff stack file or a folder of files.'
			),
			new Entry('connect_start_and_end', 'Connect Endpoints', false, 'boolean').withHidden(),
			new Entry('normalize_output', 'Normalize Output', true, 'boolean').withDescription(
				'Whether to normalize the output data.'
			),
			new Entry('zeroguard_output', 'Zero-Guard Output', true, 'boolean').withDescription(
				'Whether to fix sub-zero slice values that may cause overflow artifacts.'
			),
			new Entry('flush_cache', 'Flush CloudVolume Cache', false, 'boolean').withHidden(),
			new CompoundEntry('bounding_box_params', 'Bounding Box Parameters', [
				new Entry('max_depth', 'Max Depth', 12, 'number').withDescription(
					'The maximum depth for binary space partitioning. It is not recommended to change this option unless you encounter RAM issues.'
				),
				new Entry(
					'target_slices_per_box',
					'Target Slices per Box',
					128,
					'number'
				).withDescription(
					'If you are running on a low-RAM system, or you are taking very large slices, you may want to decrease this.'
				)
			])
		])

		this.setValue(values)
	}
}

export class BackprojectOptionsFile extends CompoundEntry {
	constructor(values: CompoundValueType = {}) {
		super('options', 'Options File', [
			new Entry(
				'straightened_volume_path',
				'Straightened Volume File',
				'',
				'filePath'
			).withDescription(
				'Path to the volume of slices to backproject (e.g. the output tif of the slicing step).'
			),
			new Entry('slice_options_path', 'Slice Options File', '', 'filePath')
				.withDescription(
					'Path to the `-slice-options.json` file which includes the information needed for backprojection.'
				)
				.withSeparator(),
			new Entry('output_file_folder', 'Output File Folder', './', 'filePath').withDescription(
				'The folder to save all the resulting files into.'
			),
			new Entry('output_file_name', 'Output File Name', 'sample', 'string').withDescription(
				'Base name for all output files.'
			),
			new Entry('output_mip_level', 'Output MIP Level', 0, 'number').withDescription(
				'The MIP level to output the backprojection in (essentially an upsample option). Use this if you downsampled in the slicing step.  MIP levels that would downsample are ignored currently.'
			),
			new Entry('upsample_order', 'Upsample Order (2 = Cubic)', 2, 'number').withDescription(
				'The interpolation order Ouroboros uses to interpolate values from a lower MIP level (matches opencv interpolation parameter). If you check the binary option, feel free to set this to 0.'
			),
			new Entry(
				'backprojection_compression',
				'Backprojection Compression',
				'zlib',
				'string'
			).withDescription(
				'The compression option to use for the backprojected tiff(s). Recommended options: `none`, `zlib`, `zstd`.'
			),
			new Entry('make_single_file', 'Output Single File', false, 'boolean').withDescription(
				'Whether to output one tiff stack file or a folder of files.'
			),
			new Entry(
				'backproject_min_bounding_box',
				'Output Min Bounding Box',
				true,
				'boolean'
			).withDescription(
				'Save only the minimum volume needed to contain the backprojected slices. The offset will be stored in the `-configuration.json` file under `backprojection_offset`. This value is the (x_min, y_min, z_min).'
			),
			new Entry(
				'make_backprojection_binary',
				'Binary Backprojection',
				false,
				'boolean'
			).withDescription(
				'Whether or not to binarize all the values of the backprojection. Enable this to backproject a single-part segmentation.'
			),
			new Entry(
				'make_backprojection_discrete',
				'Discrete Backprojection',
				false,
				'boolean'
			).withDescription(
				'Whether to limit backprojected values to discrete integers. Enable this to backproject a multi-part segmentation.'
			),
			new Entry('offset_in_name', 'Offset in Filename', true, 'boolean').withDescription(
				'Whether or not to include the (x_min, y_min, z_min) offset for min bounding box in the output file name. Only applies if `Output Min Bounding Box` is true.'
			),
			new Entry('flush_cache', 'Flush CloudVolume Cache', false, 'boolean').withHidden(),
			new Entry(
				'process_count',
				'Process Count',
				navigator.hardwareConcurrency,
				'number'
			).withDescription(
				'Number of parallel processes to use during the backprojection process.'
			),
			new Entry('chunk_size', 'Chunk Size', 160, 'number').withDescription(
				'Size of each dimension (x/y/z) of the processing chunk.  Larger chunks will use more memory and (after a certain point) slow down calculation.'
			)
		])

		this.setValue(values)
	}
}

/**
 * Finds all paths to a given type in a compound entry.
 *
 * @param entry The entry to search
 * @param targetType The type to search for
 * @returns An array of paths to the target type
 */
export function findPathsToType(
	entry: CompoundEntry | Entry,
	targetType: EntryValueType
): string[][] {
	const queue: { node: CompoundEntry | Entry; path: string[] }[] = [{ node: entry, path: [] }]
	const paths: string[][] = []

	while (queue.length > 0) {
		const { node, path } = queue.shift()!

		if (node instanceof Entry && node.type === targetType) {
			paths.push(path)
		} else if (node instanceof CompoundEntry) {
			for (const entry of node.getEntries()) {
				queue.push({ node: entry, path: [...path, entry.name] })
			}
		}
	}

	return paths
}

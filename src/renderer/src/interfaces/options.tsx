import { any, BaseIssue, BaseSchema, boolean, number, object, string } from 'valibot'

export type CompoundValueType = { [key: string]: CompoundValueType | ValueType } | ValueType
export type ValueType = number | string | boolean
export type EntryValueType = 'number' | 'string' | 'boolean' | 'filePath'

export type Schema = BaseSchema<unknown, unknown, BaseIssue<unknown>>

export class Entry {
	name: string
	label: string
	value: ValueType
	type: EntryValueType
	options?: string[]
	hidden: boolean = false

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
			new Entry('neuroglancer_json', 'Neuroglancer JSON', '', 'filePath'),
			new Entry('neuroglancer_image_layer', 'Neuroglancer Image Layer', '', 'string'),
			new Entry(
				'neuroglancer_annotation_layer',
				'Neuroglancer Annotation Layer',
				'',
				'string'
			),
			new Entry('slice_width', 'Slice Width', 120, 'number'),
			new Entry('slice_height', 'Slice Height', 120, 'number'),
			new Entry('output_file_folder', 'Output File Folder', './', 'filePath'),
			new Entry('output_file_name', 'Output File Name', 'sample', 'string'),
			new Entry('annotation_mip_level', 'Annotation MIP Level', 0, 'number'),
			new Entry('output_mip_level', 'Output MIP Level', 0, 'number'),
			new Entry('dist_between_slices', 'Distance Between Slices', 1, 'number'),
			new Entry('make_single_file', 'Output Single File', true, 'boolean'),
			new Entry('connect_start_and_end', 'Connect Endpoints', false, 'boolean'),
			new Entry('flush_cache', 'Flush CloudVolume Cache', false, 'boolean').withHidden(),
			new CompoundEntry('bounding_box_params', 'Bounding Box Parameters', [
				new Entry('max_depth', 'Max Depth', 12, 'number'),
				new Entry('target_slices_per_box', 'Target Slices per Box', 128, 'number')
			]),
			new Entry('max_ram_gb', 'Max RAM (GB) (0 = no limit)', 0, 'number')
		])

		this.setValue(values)
	}
}

export class BackprojectOptionsFile extends CompoundEntry {
	constructor(values: CompoundValueType = {}) {
		super('options', 'Options File', [
			new Entry('straightened_volume_path', 'Straightened Volume File', '', 'filePath'),
			new Entry('config_path', 'Slice Configuration File', '', 'filePath'),
			new Entry('output_file_folder', 'Output File Folder', './', 'filePath'),
			new Entry('output_file_name', 'Output File Name', 'sample', 'string'),
			new Entry('output_mip_level', 'Output MIP Level', 0, 'number'),
			new Entry('backprojection_compression', 'Backprojection Compression', 'zlib', 'string'),
			new Entry('make_single_file', 'Output Single File', false, 'boolean'),
			new Entry('backproject_min_bounding_box', 'Output Min Bounding Box', true, 'boolean'),
			new Entry('make_backprojection_binary', 'Binary Backprojection', false, 'boolean'),
			new Entry('flush_cache', 'Flush CloudVolume Cache', false, 'boolean').withHidden(),
			new Entry('max_ram_gb', 'Max RAM (GB) (0 = no limit)', 0, 'number')
		])

		this.setValue(values)
	}
}

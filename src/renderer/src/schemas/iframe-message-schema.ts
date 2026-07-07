import {
	any,
	boolean,
	includes,
	lazy,
	object,
	optional,
	picklist,
	pipe,
	record,
	string,
	InferOutput,
	nullable,
	GenericSchema
} from 'valibot'

export const IFrameMessageSchema = object({
	type: string(),
	data: any()
})

export type IFrameMessage = InferOutput<typeof IFrameMessageSchema>

export const RegisterIFrameSchema = object({
	type: string(),
	data: object({
		pluginName: string('Plugin name is required')
	})
})

export type RegisterIFrame = InferOutput<typeof RegisterIFrameSchema>

export const SendDirectoryContentsSchema = object({
	type: pipe(string(), includes('send-directory-contents')),
	data: object({
		directoryPath: nullable(string('Directory path is required')),
		directoryName: nullable(string('Directory name is required')),
		nodes: object({})
	})
})

export type SendDirectoryContents = InferOutput<typeof SendDirectoryContentsSchema>

export const ReadFileRequestSchema = object({
	type: pipe(string(), includes('read-file')),
	data: object({
		folder: string('Folder path is required'),
		fileName: string('File name is required')
	})
})

export type ReadFileRequest = InferOutput<typeof ReadFileRequestSchema>

export const ReadFileResponseSchema = object({
	type: pipe(string(), includes('read-file-response')),
	data: object({
		fileName: string('File name is required'),
		contents: string()
	})
})

export type ReadFileResponse = InferOutput<typeof ReadFileResponseSchema>

export const SaveFileRequestSchema = object({
	type: pipe(string(), includes('save-file')),
	data: object({
		folder: string('Folder path is required'),
		fileName: string('File name is required'),
		contents: string('File contents are required')
	})
})

export type SaveFileRequest = InferOutput<typeof SaveFileRequestSchema>

// A path-keyed nested map of `FileSystemNode` matching the renderer's
// `DirectoryContext.NodeChildren` shape. Defined recursively via a lazy
// object schema (valibot cannot self-reference statically).
export type PluginFileSystemNode = {
	name: string
	path: string
	children?: { [key: string]: PluginFileSystemNode }
}

export type PluginNodeChildren = { [key: string]: PluginFileSystemNode }

const PluginFileSystemNodeSchema: GenericSchema<PluginFileSystemNode> = object({
	name: string(),
	path: string(),
	children: optional(
		lazy(() => record(string(), PluginFileSystemNodeSchema))
	)
})

const PluginNodeChildrenSchema = record(string(), PluginFileSystemNodeSchema)

export const RequestDirectoryContentsSchema = object({
	type: pipe(string(), includes('request-directory-contents')),
	data: object({
		path: string('Path is required'),
		recursive: optional(boolean()),
		requestId: optional(string())
	})
})

export type RequestDirectoryContents = InferOutput<typeof RequestDirectoryContentsSchema>

export const SendDirectoryContentsResponseSchema = object({
	type: pipe(string(), includes('send-directory-contents-response')),
	data: object({
		path: string(),
		nodes: PluginNodeChildrenSchema,
		requestId: nullable(string()),
		error: optional(
			object({
				code: picklist(['denied', 'not-found', 'limit', 'internal'] as const),
				message: string(),
				truncated: optional(boolean())
			})
		)
	})
})

export type SendDirectoryContentsResponse = InferOutput<typeof SendDirectoryContentsResponseSchema>

export const SendNeuroglancerJSONSchema = object({
	type: pipe(string(), includes('send-neuroglancer-json')),
	data: object({
		contents: string()
	})
})

export type SendNeuroglancerJSON = InferOutput<typeof SendNeuroglancerJSONSchema>

import './helpers/renderer-loader.mjs'
import assert from 'node:assert/strict'
import test from 'node:test'
import { safeParse } from 'valibot'

const schemas = await import('../../src/renderer/src/schemas/iframe-message-schema.ts')

for (const [schemaName, message] of [
	['IFrameMessageSchema', { type: 'plugin-message', data: { pluginValue: 7 } }],
	[
		'RegisterIFrameSchema',
		{ type: 'register-plugin', data: { pluginName: 'annotation-editor' } }
	],
	[
		'SendDirectoryContentsSchema',
		{
			type: 'send-directory-contents',
			data: { directoryPath: null, directoryName: null, nodes: {} }
		}
	],
	[
		'ReadFileRequestSchema',
		{ type: 'read-file', data: { folder: '/project', fileName: 'options.json' } }
	],
	[
		'ReadFileResponseSchema',
		{ type: 'read-file-response', data: { fileName: 'options.json', contents: '' } }
	],
	[
		'SaveFileRequestSchema',
		{ type: 'save-file', data: { folder: '/project', fileName: 'empty.txt', contents: '' } }
	],
	[
		'RequestDirectoryContentsSchema',
		{ type: 'request-directory-contents', data: { path: '/project' } }
	],
	[
		'SendNeuroglancerJSONSchema',
		{ type: 'send-neuroglancer-json', data: { contents: '{"layers":[]}' } }
	]
]) {
	test(`iframe schema ${schemaName}: accepts its current protocol payload`, () => {
		const result = safeParse(schemas[schemaName], message)
		assert.equal(result.success, true, `${schemaName} rejected a valid ${message.type} payload`)
		assert.deepEqual(
			result.output,
			message,
			'Parsing must retain all fields in the declared payload'
		)
	})
}

for (const [schemaName, message, field, reason] of [
	[
		'RegisterIFrameSchema',
		{ type: 'register-plugin', data: {} },
		'pluginName',
		'Plugin name is required'
	],
	[
		'ReadFileRequestSchema',
		{ type: 'read-file', data: { fileName: 'options.json' } },
		'folder',
		'Folder path is required'
	],
	[
		'ReadFileRequestSchema',
		{ type: 'read-file', data: { folder: '/project' } },
		'fileName',
		'File name is required'
	],
	[
		'SaveFileRequestSchema',
		{ type: 'save-file', data: { folder: '/project', fileName: 'options.json' } },
		'contents',
		'File contents are required'
	],
	[
		'RequestDirectoryContentsSchema',
		{ type: 'request-directory-contents', data: {} },
		'path',
		'Path is required'
	]
]) {
	test(`iframe schema ${schemaName}: invalid ${field} identifies the field and reason`, () => {
		const missing = safeParse(schemas[schemaName], message)
		assert.equal(missing.success, false, `Missing ${field} must fail validation`)
		assert.deepEqual(
			missing.issues[0].path.map(({ key }) => key),
			['data', field],
			'The missing-key diagnostic must locate the absent payload field'
		)
		assert.match(
			missing.issues[0].message,
			new RegExp(field),
			'The missing-key reason must identify the absent field'
		)
		const result = safeParse(schemas[schemaName], {
			...message,
			data: { ...message.data, [field]: 7 }
		})
		assert.equal(result.success, false, `Non-string ${field} must fail validation`)
		assert.deepEqual(
			result.issues.map((issue) => ({
				path: issue.path.map(({ key }) => key),
				message: issue.message
			})),
			[{ path: ['data', field], message: reason }],
			'The diagnostic must name the invalid payload field'
		)
	})
}

test('iframe directory request schema preserves recursion and request correlation', () => {
	const message = {
		type: 'request-directory-contents',
		data: { path: '/project/images', recursive: true, requestId: 'list-images' }
	}
	const result = safeParse(schemas.RequestDirectoryContentsSchema, message)
	assert.equal(
		result.success,
		true,
		'Recursive directory requests with a string request ID are valid'
	)
	assert.deepEqual(
		result.output,
		message,
		'Recursion and correlation must survive schema parsing'
	)
	for (const [field, value] of [
		['recursive', 'true'],
		['requestId', 7]
	]) {
		const malformed = safeParse(schemas.RequestDirectoryContentsSchema, {
			...message,
			data: { ...message.data, [field]: value }
		})
		assert.equal(malformed.success, false, `${field} must reject the wrong scalar type`)
		assert.deepEqual(
			malformed.issues[0].path.map(({ key }) => key),
			['data', field],
			`The invalid ${field} diagnostic must point to that field`
		)
	}
})

function directoryResponse() {
	return {
		type: 'send-directory-contents-response',
		data: {
			path: '/project',
			nodes: {
				'/project/images': {
					name: 'images',
					path: '/project/images',
					children: {
						'/project/images/slice.tif': {
							name: 'slice.tif',
							path: '/project/images/slice.tif'
						}
					}
				}
			},
			requestId: 'list-images'
		}
	}
}

test('iframe directory response schema validates a nested file tree without dropping children', () => {
	const message = directoryResponse()
	const result = safeParse(schemas.SendDirectoryContentsResponseSchema, message)
	assert.equal(result.success, true, 'A nested directory/file response must validate')
	assert.deepEqual(result.output, message, 'Nested children and their paths must survive parsing')
	message.data.nodes['/project/images'].children['/project/images/slice.tif'].path = 7
	const malformed = safeParse(schemas.SendDirectoryContentsResponseSchema, message)
	assert.equal(malformed.success, false, 'Nested file paths must be strings at every depth')
	assert.deepEqual(
		malformed.issues[0].path.map(({ key }) => key),
		['data', 'nodes', '/project/images', 'children', '/project/images/slice.tif', 'path'],
		'The diagnostic must locate the nested file path that failed validation'
	)
})

for (const code of ['denied', 'not-found', 'limit', 'internal']) {
	test(`iframe directory response schema preserves the ${code} error reason`, () => {
		const message = directoryResponse()
		message.data.requestId = null
		message.data.error = { code, message: `Directory enumeration failed: ${code}` }
		if (code === 'limit') message.data.error.truncated = true
		const result = safeParse(schemas.SendDirectoryContentsResponseSchema, message)
		assert.equal(
			result.success,
			true,
			`${code} is an established directory response error code`
		)
		assert.deepEqual(
			result.output,
			message,
			'Keep the error reason, truncation flag and nullable ID'
		)
	})
}

test('iframe directory response schema rejects an unrecognized error category', () => {
	const message = directoryResponse()
	message.data.error = { code: 'unrecognized', message: 'Unknown error category' }
	const result = safeParse(schemas.SendDirectoryContentsResponseSchema, message)
	assert.equal(
		result.success,
		false,
		'Unknown error categories must not cross the plugin protocol'
	)
	assert.deepEqual(
		result.issues[0].path.map(({ key }) => key),
		['data', 'error', 'code'],
		'The diagnostic must point to the unsupported error category'
	)
})

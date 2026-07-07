import { array, InferOutput, object, safeParse, string } from 'valibot'
import { makeErrorResult, makeSuccessResult, ParseResult } from './schema-helpers'

const NeuroglancerJSONSchema = object({
	layers: array(
		object({
			type: string('Type is required'),
			name: string('Name is required')
		})
	)
})

export type NeuroglancerJSON = InferOutput<typeof NeuroglancerJSONSchema>

export const parseNeuroglancerJSON = (jsonString: string | null): ParseResult<NeuroglancerJSON> => {
	const errorResult = makeErrorResult<NeuroglancerJSON>('Invalid Neuroglancer JSON')

	if (!jsonString || jsonString === '') return errorResult

	let parsedJSON = ''

	try {
		parsedJSON = JSON.parse(jsonString)
	} catch (_e) {
		return errorResult
	}

	const jsonResult = safeParse(NeuroglancerJSONSchema, parsedJSON)

	if (jsonResult.success) {
		return makeSuccessResult(jsonResult.output)
	} else {
		return errorResult
	}
}

export default NeuroglancerJSONSchema

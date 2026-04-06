import { array, InferOutput, nullable, number, object, string } from 'valibot'
import { baseParse, makeErrorResult, makeSuccessResult, ParseResult } from './schema-helpers'
import { VisualizationOutput } from '@renderer/routes/SlicesPage/components/VisualizeSlicing/VisualizeSlicing'

const SliceVisualizationResultSchema = object(
	{
		data: nullable(
			object({
				rects: array(array(array(number()))),
				bounding_boxes: array(
					object({
						min: array(number()),
						max: array(number())
					})
				),
				link_rects: array(number())
			})
		),
		error: nullable(string())
	},
	'The data provided for slice visualization is invalid.'
)

export type VisualizationInput = Exclude<
	InferOutput<typeof SliceVisualizationResultSchema>['data'],
	null
>

export const parseSliceVisualizationResult = baseParse(
	SliceVisualizationResultSchema,
	'Invalid slice visualization result'
)

export const parseSliceVisualizationToOutputFormat = (
	input: object | null
): ParseResult<VisualizationOutput> => {
	const { result, error } = parseSliceVisualizationResult(input)
	if (error) {
		return makeErrorResult(error)
	} else if (!result.data) {
		return makeErrorResult('No data provided for slice visualization')
	} else if (result.error) {
		return makeErrorResult(result.error)
	}

	return makeSuccessResult(visualizationDataToOutputFormat(result.data as VisualizationInput))
}

export function visualizationDataToOutputFormat(
	visualizationData: VisualizationInput
): VisualizationOutput {
	const rects = visualizationData.rects.map((rect) => {
		return { topLeft: rect[0], topRight: rect[1], bottomRight: rect[2], bottomLeft: rect[3] }
	})
	const boundingBoxes = visualizationData.bounding_boxes
	const linkRects = visualizationData.link_rects

	return { rects, boundingBoxes, linkRects }
}

export default SliceVisualizationResultSchema

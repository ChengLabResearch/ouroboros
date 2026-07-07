import { InferOutput, object, optional, safeParse, string } from 'valibot'

export const PluginPackageJSONSchema = object({
	name: string('Plugin name is required'),
	pluginName: string('Readable plugin name is required'),
	version: optional(string()),
	icon: optional(string()),
	index: string('Plugin index.html file is required'),
	dockerCompose: optional(string())
})

export type PluginPackageJSON = InferOutput<typeof PluginPackageJSONSchema>

export const parsePluginPackageJSON = (data: string): PluginPackageJSON | string => {
	try {
		const result = JSON.parse(data)

		const parseResult = safeParse(PluginPackageJSONSchema, result)

		if (parseResult.success) {
			return parseResult.output
		}

		return `Unable to parse plugin package.json file: ${parseResult.issues.join(', ')}`
	} catch (_error) {
		return 'Could not parse plugin package.json file'
	}
}

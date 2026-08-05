import { writeFile } from 'node:fs/promises'

const repository = required('OUROBOROS_SERVER_IMAGE_REPOSITORY')
const digest = required('OUROBOROS_SERVER_IMAGE_DIGEST')
const sourceCommit = required('OUROBOROS_SERVER_SOURCE_COMMIT')
const tag = `sha-${sourceCommit}`

await writeFile(
	'server-image-manifest.json',
	`${JSON.stringify(
		{
			schemaVersion: 1,
			image: `${repository}@${digest}`,
			repository,
			digest,
			tag,
			sourceCommit,
			workflowRun: process.env.GITHUB_RUN_ID ?? null
		},
		null,
		2
	)}\n`
)

function required(name) {
	const value = process.env[name]
	if (!value) throw new Error(`${name} is required`)
	return value
}

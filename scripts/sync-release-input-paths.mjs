import { readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { loadArtifactInputDefinition } from './lib/release-inputs.mjs'

const workflowPath = '.github/workflows/build-and-release.yml'
const generatedBlock =
	/^([ \t]*)# BEGIN GENERATED ARTIFACT INPUT PATHS\r?\n[\s\S]*?^\1# END GENERATED ARTIFACT INPUT PATHS/gm

export async function syncReleaseInputPaths({ root = process.cwd(), check = true } = {}) {
	const definition = await loadArtifactInputDefinition(root)
	const path = join(root, workflowPath)
	const workflow = await readFile(path, 'utf8')
	const newline = workflow.includes('\r\n') ? '\r\n' : '\n'
	let blocks = 0
	const generated = workflow.replace(generatedBlock, (_block, indent) => {
		blocks += 1
		return [
			`${indent}# BEGIN GENERATED ARTIFACT INPUT PATHS`,
			...definition.paths.map((input) => `${indent}- ${input}`),
			`${indent}# END GENERATED ARTIFACT INPUT PATHS`
		].join(newline)
	})
	if (blocks !== 2) {
		throw new Error(`Expected two generated artifact input path blocks in ${workflowPath}`)
	}
	if (generated === workflow) return
	if (check) {
		throw new Error(
			`Artifact workflow paths differ from release/artifact-inputs.json; run node scripts/sync-release-input-paths.mjs --write`
		)
	}
	await writeFile(path, generated)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	await syncReleaseInputPaths({ check: !process.argv.includes('--write') })
}

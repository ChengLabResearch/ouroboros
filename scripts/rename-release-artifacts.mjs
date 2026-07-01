import { readFile, readdir, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

const root = process.cwd()
const dist = join(root, 'dist')
const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const flavor = process.env.OUROBOROS_PACKAGE_FLAVOR

if (!flavor) {
	console.log('OUROBOROS_PACKAGE_FLAVOR is not set; leaving release artifact names unchanged.')
	process.exit(0)
}

const entries = await readdir(dist, { withFileTypes: true })
const renameMap = new Map()

for (const entry of entries) {
	if (!entry.isFile()) continue

	const nextName = artifactNameForFlavor(entry.name)
	if (!nextName || nextName === entry.name) continue

	renameMap.set(entry.name, nextName)
}

for (const [oldName, newName] of renameMap) {
	await rename(join(dist, oldName), join(dist, newName))
	console.log(`Renamed ${oldName} -> ${newName}`)
}

const updatedEntries = await readdir(dist, { withFileTypes: true })
for (const entry of updatedEntries) {
	if (!entry.isFile() || !entry.name.endsWith('.yml')) continue

	const path = join(dist, entry.name)
	let text = await readFile(path, 'utf8')

	for (const [oldName, newName] of renameMap) {
		text = text.split(oldName).join(newName)
	}

	await writeFile(path, text)
}

function artifactNameForFlavor(fileName) {
	if (fileName.includes(`-${flavor}`) || fileName.includes(`_${flavor}`)) return null

	if (fileName.startsWith('latest') && fileName.endsWith('.yml')) {
		return `${fileName.slice(0, -'.yml'.length)}-${flavor}.yml`
	}

	const dashedPrefix = `${packageJson.name}-${packageJson.version}`
	if (fileName.startsWith(dashedPrefix)) {
		return `${dashedPrefix}-${flavor}${fileName.slice(dashedPrefix.length)}`
	}

	const underscoredPrefix = `${packageJson.name}_${packageJson.version}`
	if (fileName.startsWith(underscoredPrefix)) {
		return `${underscoredPrefix}_${flavor}${fileName.slice(underscoredPrefix.length)}`
	}

	return null
}

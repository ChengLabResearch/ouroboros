import { appendFile, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { gitOutput } from '../../lib/git.mjs'

// Use the real application, lock, and packaging inputs. Only the history and
// changes under test are manufactured; this exercises the repository's actual
// artifact boundary instead of a second, test-only definition of that boundary.
export async function releaseRepository(t, sourceRoot) {
	const directory = await mkdtemp(join(tmpdir(), 'ouroboros-release-history-'))
	t.after(() => rm(directory, { recursive: true, force: true }))
	const root = join(directory, 'repository')
	await mkdir(root)
	const paths = await gitOutput(sourceRoot, [
		'ls-files',
		'--cached',
		'--others',
		'--exclude-standard',
		'-z'
	])
	for (const name of new Set(paths.split('\0').filter(Boolean))) {
		const destination = join(root, name)
		await mkdir(dirname(destination), { recursive: true })
		try {
			await copyFile(join(sourceRoot, name), destination)
		} catch (error) {
			if (error.code !== 'ENOENT') throw error
		}
	}
	await gitOutput(root, ['init', '--initial-branch=main'])
	await gitOutput(root, ['config', 'user.name', 'Release fixture'])
	await gitOutput(root, ['config', 'user.email', 'release-fixture@example.invalid'])
	await gitOutput(root, ['config', 'commit.gpgsign', 'false'])
	await gitOutput(root, ['config', 'core.autocrlf', 'false'])
	const commit = async (message) => {
		await gitOutput(root, ['add', '--all'])
		await gitOutput(root, ['commit', '--quiet', '-m', message])
		return await gitOutput(root, ['rev-parse', 'HEAD'])
	}
	const built = await commit('Actual repository release inputs')
	const pythonProjectPath = join(root, 'python', 'pyproject.toml')
	const pythonProject = await readFile(pythonProjectPath, 'utf8')
	await writeFile(
		pythonProjectPath,
		pythonProject.replace(/^version = "[^"]+"/m, 'version = "0.0.0+release.fixture"')
	)
	const pythonOnly = await commit(
		'Change Python project without changing the pinned server image'
	)
	const runtimePath = join(root, 'resources', 'processes', 'volume-server-script.mjs')
	const originalRuntime = await readFile(runtimePath)
	await appendFile(runtimePath, '\n// Runtime release fixture change\n')
	const artifactChanged = await commit('Change packaged runtime source')
	await writeFile(runtimePath, originalRuntime)
	const artifactReverted = await commit('Revert packaged runtime source')
	await gitOutput(root, ['update-ref', 'refs/remotes/origin/main', artifactReverted])
	await gitOutput(root, ['checkout', '--quiet', '-b', 'side', built])
	await appendFile(join(root, 'python', 'pyproject.toml'), '\n# Unmerged branch fixture change\n')
	const side = await commit('Unmerged branch with matching release inputs')
	await gitOutput(root, ['checkout', '--quiet', 'main'])
	return { root, directory, built, pythonOnly, artifactChanged, artifactReverted, side }
}

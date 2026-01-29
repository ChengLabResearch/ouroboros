import { buildAll, downAll, upAll } from 'docker-compose'
import { execPromise } from '../helpers'

export async function checkDocker(): Promise<{ available: boolean; error: string | null }> {
	const commands = [
		{
			cmd: 'docker --version',
			successMsg: 'Docker is installed',
			failureMsg: 'Docker is not installed'
		},
		{ cmd: 'docker info', successMsg: 'Docker is running', failureMsg: 'Docker is not running' }
	]

	const results = await commands.reduce(async function (p, command) {
		const results: { success: boolean; message: string }[] = await p
		try {
			const stdout = await execPromise(command.cmd)
			results.push({ success: true, message: command.successMsg + ': ' + stdout.trim() })
		} catch (err) {
			results.push({ success: false, message: command.failureMsg + ': ' + `${err}` })
		}
		return results
	}, Promise.resolve<{ success: boolean; message: string }[]>([]))

	// If any of the commands failed, return the error message
	const failed = results.filter((result) => !result.success)

	if (failed.length > 0) {
		return { available: false, error: failed[0].message }
	}

	return { available: true, error: null }
}

export async function startDockerCompose({
	cwd,
	config,
	onError,
	log = false,
	build = false
}: {
	cwd: string
	config: string
	onError: (err) => void
	log?: boolean
	build?: boolean
}): Promise<void> {
	await upAll({
		cwd,
		log,
		config,
		commandOptions: build ? ['--build'] : []
	}).then(() => {}, onError)
}

export async function stopDockerCompose({
	cwd,
	config,
	log = false
}: {
	cwd: string
	config: string
	log?: boolean
}): Promise<void> {
	await downAll({
		cwd,
		log,
		config
	})
}

export async function buildDockerCompose({
	cwd,
	config,
	onError
}: {
	cwd: string
	config: string
	onError: (err) => void
}): Promise<void> {
	await buildAll({
		cwd,
		log: false,
		config
	}).then(() => {}, onError)
}

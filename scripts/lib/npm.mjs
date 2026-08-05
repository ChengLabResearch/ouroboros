export function npmInvocation(
	args,
	{
		nodeExecutable = process.execPath,
		npmExecutable = process.env.npm_execpath,
		platform = process.platform
	} = {}
) {
	if (npmExecutable) {
		return {
			command: nodeExecutable,
			args: [npmExecutable, ...args]
		}
	}
	if (platform === 'win32') {
		throw new Error(
			'npm_execpath is required on Windows so npm can be launched without a command shell'
		)
	}
	return { command: 'npm', args }
}

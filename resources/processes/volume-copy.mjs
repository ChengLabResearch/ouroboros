import path from 'node:path'

// Docker receives argument data directly. Only the fixed script is shell syntax;
// the paths following its $0 placeholder remain quoted positional parameters.
function copyCommand(sourceFolder, volumeName, script, ...parameters) {
	return [
		'run',
		'--rm',
		'-v',
		`${sourceFolder}:/host`,
		'-v',
		`${volumeName}:/volume`,
		'-w',
		'/host',
		'alpine',
		'sh',
		'-c',
		script,
		'sh',
		...parameters
	]
}

/** Copy a host input into the requested folder in the volume. */
export function copyFileToVolumeCommand(sourceFolder, fileName, volumeName, destFolder) {
	const innerFilePath = path.posix.join('/host/', fileName)
	const newDestFolder = path.posix.join('/volume/', destFolder)
	const destFile = path.posix.join(newDestFolder, fileName)
	return copyCommand(
		sourceFolder,
		volumeName,
		'mkdir -p "$1" && cp -r "$2" "$3"',
		newDestFolder,
		innerFilePath,
		destFile
	)
}

/** Copy a volume output to the host, assigning private reports to the host user. */
export function copyFileToHostCommand(
	sourceFolder,
	fileName,
	volumeName,
	destFolder,
	pluginFolderName
) {
	const innerFilePath = path.posix.join('/host/', fileName)
	const destFile = path.posix.join('/volume/', destFolder, fileName)
	// The host process owns identity; clients cannot supply an ownership override.
	const hostOwner =
		pluginFolderName === 'main-statistics' &&
		typeof process.getuid === 'function' &&
		typeof process.getgid === 'function'
			? `${process.getuid()}:${process.getgid()}`
			: undefined
	const script = 'rm -rf "$1" && cp -r "$2" "$1"'
	return copyCommand(
		sourceFolder,
		volumeName,
		hostOwner === undefined ? script : `${script} && chown "$3" "$1"`,
		innerFilePath,
		destFile,
		...(hostOwner === undefined ? [] : [hostOwner])
	)
}

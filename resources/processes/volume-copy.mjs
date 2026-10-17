import path from 'node:path'

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
	const ownership = hostOwner === undefined ? '' : ` && chown ${hostOwner} '${innerFilePath}'`

	return `
        docker run --rm -v "${sourceFolder}":/host -v ${volumeName}:/volume -w /host alpine
        sh -c "rm -rf '${innerFilePath}' && cp -r '${destFile}' '${innerFilePath}'${ownership}"
    `
		.replace(/\s+/g, ' ')
		.trim()
}

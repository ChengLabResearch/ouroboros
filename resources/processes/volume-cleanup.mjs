import path from 'node:path'

/**
 * The shared cleanup command for a plugin folder or the entire scratch volume.
 * Undelivered run reports in main-statistics survive desktop shutdown. As with
 * the original wildcard cleanup, top-level hidden entries remain untouched.
 */
export function deleteFilesFromVolumeFolder(volumeName, targetFolder) {
	const targetFolderPath = path.posix.join('/volume/', targetFolder, '/*')
	const cleanup =
		targetFolder === '/'
			? "find /volume/ -mindepth 1 -maxdepth 1 ! -name '.*' ! -name 'main-statistics' -exec rm -rf {} +"
			: `rm -rf ${targetFolderPath}`

	return `docker run --rm -v ${volumeName}:/volume alpine sh -c "${cleanup}"`
}

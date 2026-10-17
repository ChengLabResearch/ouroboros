import express from 'express'
import cors from 'cors'
import path, { basename, dirname } from 'path'
import { exec } from 'child_process'
import bodyParser from 'body-parser'
import { deleteFilesFromVolumeFolder } from './volume-cleanup.mjs'

const app = express()
app.use(cors())
app.use(bodyParser.json())

const PORT = process.argv[2] || 3001

function runCommand(command) {
	return new Promise((resolve, reject) => {
		exec(command, (error, stdout, stderr) => {
			if (error) {
				reject(stderr)
			} else {
				resolve(stdout)
			}
		})
	})
}

const copyHandler = (toVolume) => async (req, res) => {
	const { volumeName, pluginFolderName, files } = req.body
	try {
		await Promise.all(
			files.map(async (file) => {
				const { sourcePath, targetPath } = file
				const sourceFolder = dirname(sourcePath)
				const sourceFileName = basename(sourcePath)
				const targetDir = path.posix.join(pluginFolderName, targetPath)

				let command

				if (toVolume)
					command = copyFileToVolumeCommand(
						sourceFolder,
						sourceFileName,
						volumeName,
						targetDir
					)
				else
					command = copyFileToHostCommand(
						sourceFolder,
						sourceFileName,
						volumeName,
						targetDir
					)

				await runCommand(command)
			})
		)
		res.status(200).send('Files copied successfully.')
	} catch (error) {
		res.status(500).send(`Error copying files: ${error}`)
	}
}

app.post('/copy-to-volume', copyHandler(true))

app.post('/copy-to-host', copyHandler(false))

app.post('/clear-plugin-folder', async (req, res) => {
	const { volumeName, pluginFolderName } = req.body
	try {
		const command = deleteFilesFromVolumeFolder(volumeName, pluginFolderName)
		await runCommand(command)

		res.status(200).send('Plugin folder cleared successfully.')
	} catch (error) {
		res.status(500).send(`Error clearing plugin folder: ${error}`)
	}
})

app.post('/clear-volume', async (req, res) => {
	const { volumeName } = req.body
	try {
		const command = deleteFilesFromVolumeFolder(volumeName, '/')
		await runCommand(command)

		res.status(200).send('Volume cleared successfully.')
	} catch (error) {
		res.status(500).send(`Error clearing volume: ${error}`)
	}
})

app.listen(PORT)

/**
 * Creates a command to copy a file from the host file system into a volume
 * @param {string} sourceFolder The folder of the file on the host
 * @param {string} fileName The name of the file
 * @param {string} volumeName The volume to copy into
 * @param {string} destFolder The target folder in the volume
 * @returns An executable command
 */
function copyFileToVolumeCommand(sourceFolder, fileName, volumeName, destFolder) {
	const innerFilePath = path.posix.join('/host/', fileName)
	const newDestFolder = path.posix.join('/volume/', destFolder)
	const destFile = path.posix.join(newDestFolder, fileName)

	// Construct the Docker command
	const command = `
    docker run --rm -v "${sourceFolder}":/host -v ${volumeName}:/volume -w /host alpine sh -c "
    mkdir -p '${newDestFolder}' && cp -r '${innerFilePath}' '${destFile}'"
`
		.replace(/\s+/g, ' ')
		.trim()

	return command
}

/**
 * Creates a command to copy a file from a volume to the host file system
 * @param {string} sourceFolder The folder of the file on the host
 * @param {string} fileName The name of the file
 * @param {string} volumeName The volume to copy into
 * @param {string} destFolder The target folder in the volume
 * @returns An executable command
 */
function copyFileToHostCommand(sourceFolder, fileName, volumeName, destFolder) {
	const innerFilePath = path.posix.join('/host/', fileName)
	const destFile = path.posix.join('/volume/', destFolder, fileName)

	// Construct the Docker command
	const command = `
        docker run --rm -v "${sourceFolder}":/host -v ${volumeName}:/volume -w /host alpine 
        sh -c "rm -rf '${innerFilePath}' && cp -r '${destFile}' '${innerFilePath}'"
    `
		.replace(/\s+/g, ' ')
		.trim()

	return command
}

import express from 'express'
import cors from 'cors'
import path, { basename, dirname } from 'path'
import { execFile } from 'child_process'
import bodyParser from 'body-parser'
import { deleteFilesFromVolumeFolder } from './volume-cleanup.mjs'
import { copyFileToHostCommand, copyFileToVolumeCommand } from './volume-copy.mjs'

const app = express()
app.use(cors())
app.use(bodyParser.json())

const PORT = process.argv[2] || 3001

function runCommand(args) {
	return new Promise((resolve, reject) => {
		execFile('docker', args, (error, stdout, stderr) => {
			if (error) {
				reject(stderr || error.message)
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
						targetDir,
						pluginFolderName
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

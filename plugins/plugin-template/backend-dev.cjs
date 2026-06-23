const { upAll, downAll } = require('docker-compose')
const { join } = require('path')

const containerFolder = join(__dirname, 'backend')
const composeFile = join(containerFolder, 'compose.dev.yml')

// Start docker compose
console.log('Starting Docker Container')
upAll({ cwd: containerFolder, log: false, config: composeFile }).catch(() => {
	console.error(
		'Failed to start plugin docker environment. Make sure Docker is installed and running.'
	)
})

async function cleanup() {
	try {
		console.log('Shutting Down Docker Container')
		await downAll({ cwd: containerFolder, log: false, config: composeFile })
	} catch (e) {
		console.error(e)
	}

	process.exit()
}

// Setup signal handler for Ctrl+C (SIGINT)
process.on('SIGINT', async () => {
	try {
		await cleanup()
	} catch (e) {
		console.error(e)
	}
})

// Maintain the process
setInterval(() => {}, 1000)

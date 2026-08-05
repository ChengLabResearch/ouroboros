import { spawn } from 'node:child_process'

export async function gitOutput(root, args) {
	return await new Promise((resolve, reject) => {
		const child = spawn('git', args, { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] })
		let stdout = ''
		let stderr = ''
		child.stdout.setEncoding('utf8')
		child.stderr.setEncoding('utf8')
		child.stdout.on('data', (chunk) => {
			stdout += chunk
		})
		child.stderr.on('data', (chunk) => {
			stderr += chunk
		})
		child.on('error', reject)
		child.on('close', (code) => {
			if (code === 0) resolve(stdout.trim())
			else reject(new Error(`git ${args.join(' ')} failed: ${stderr.trim()}`))
		})
	})
}

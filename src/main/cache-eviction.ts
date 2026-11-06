import { constants } from 'fs'
import { lstat, open, readdir } from 'fs/promises'
import { dirname, join } from 'path'

// Node does not expose O_DIRECT on Windows. This is libuv's UV_FS_O_DIRECT,
// translated by uv_fs_open to FILE_FLAG_NO_BUFFERING (not the Win32 flag itself).
const WINDOWS_O_DIRECT = 0x02000000

export async function evictFromCache(file: string): Promise<void> {
	if (process.platform !== 'win32') return

	try {
		// Opening without buffering and then closing releases the cached pages.
		// Read-only access preserves file contents and requires no elevation.
		const handle = await open(file, constants.O_RDONLY | WINDOWS_O_DIRECT)
		await handle.close()
	} catch (error) {
		console.warn(`Unable to evict file from the Windows cache: ${file}`, error)
	}
}

export async function evictTree(dir: string): Promise<void> {
	if (process.platform !== 'win32') return

	try {
		// lstat also excludes a root or child directory that is a junction.
		const info = await lstat(dir)
		if (info.isSymbolicLink() || !info.isDirectory()) return

		const entries = await readdir(dir, { withFileTypes: true })
		for (const entry of entries) {
			const path = join(dir, entry.name)
			if (entry.isDirectory()) await evictTree(path)
			else if (entry.isFile()) await evictFromCache(path)
		}
	} catch (error) {
		// A missing or inaccessible branch must not stop its parent's siblings.
		console.warn(`Unable to enumerate output for Windows cache eviction: ${dir}`, error)
	}
}

export async function evictOperationOutput(optionsPath: string): Promise<void> {
	if (process.platform !== 'win32') return

	// Both operation pages save their submitted options alongside the outputs.
	// That submitted path remains authoritative if form values or JSON change.
	await evictTree(dirname(optionsPath))
}

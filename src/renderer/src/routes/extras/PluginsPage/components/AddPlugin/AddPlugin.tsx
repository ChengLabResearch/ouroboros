/* eslint-disable */
// @ts-nocheck

import { JSX, useState } from 'react'
import styles from './AddPlugin.module.css'

function AddPlugin({ onAdd }: { onAdd: ({ type: string, content: string }) => void }): JSX.Element {
	const [releaseURL, setReleaseURL] = useState('')
	const [folderName, setFolderName] = useState('Choose Folder')
	const [folderPath, setFolderPath] = useState('')

	const handleFolderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
		// Display the folder name
		const files = event.target.files
		if (files && files.length > 0) {
			// Set the folder name
			// Note: the folder name is the first part of the webkitRelativePath
			// e.g. if the webkitRelativePath is "folder/file.txt", the folder name is "folder"
			// The regex is needed to match on posix and windows paths
			const name = files[0].webkitRelativePath.split(/[/\\]/)[0]
			setFolderName(name)

			// Set the folder path
			const path = files[0].path.match(/(.*)[\/\\]/)[1] || ''
			setFolderPath(path)
		}
	}

	return (
		<div className={styles.addPlugin}>
			<div className={styles.addPluginOption}>
				<div>GitHub Release</div>
				<input
					className={styles.addPluginInput}
					value={releaseURL}
					onChange={(event) => setReleaseURL(event.target.value)}
					type="text"
					placeholder="Paste URL"
				/>
				<div
					className={styles.addPluginButton}
					onClick={() => {
						onAdd({ type: 'release', content: releaseURL })
					}}
				>
					DOWNLOAD
				</div>
			</div>
			<div className={styles.addPluginOption}>
				<div>Local Folder</div>
				<label htmlFor="folderInput" className={`${styles.addPluginInput} ${styles.label}`}>
					{folderName}
				</label>
				<input
					type="file"
					id="folderInput"
					webkitdirectory="" // Enables directory selection in WebKit browsers
					directory="" // Enables directory selection in other browsers
					style={{ display: 'none' }}
					onChange={handleFolderChange}
				/>
				<div
					className={styles.addPluginButton}
					onClick={() => {
						onAdd({ type: 'local', content: folderPath })
					}}
				>
					ADD
				</div>
			</div>
		</div>
	)
}

export default AddPlugin

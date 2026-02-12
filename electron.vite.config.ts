import { resolve } from 'path'
import { defineConfig } from 'electron-vite'
import react from '@vitejs/plugin-react'
import webfontDownload from 'vite-plugin-webfont-dl'
import svgr from 'vite-plugin-svgr'

export default defineConfig({
	main: {
		build: {
			externalizeDeps: true 
		}
	},
	preload: {
		build: {
			externalizeDeps: true 
		}
	},
	renderer: {
		resolve: {
			alias: {
				'@renderer': resolve('src/renderer/src')
			}
		},
		plugins: [
			react(),
			webfontDownload([
				'https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap'
			]),
			svgr()
		]
	}
})
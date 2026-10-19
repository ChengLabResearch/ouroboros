// Compile renderer TypeScript/JSX with the project's existing compiler for Node tests.
// CSS module class names are retained so server-rendered component assertions can use them.
import { registerHooks } from 'node:module'
import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const renderer = new URL('../../../src/renderer/src/', import.meta.url)
const shared = new URL('../../../src/shared/', import.meta.url)
registerHooks({
	resolve(specifier, context, nextResolve) {
		if (specifier.startsWith('@renderer/')) {
			specifier = new URL(specifier.slice('@renderer/'.length), renderer).href
		}
		if (specifier.startsWith('@shared/')) {
			specifier = new URL(specifier.slice('@shared/'.length), shared).href
		}
		if (specifier.startsWith('.') || specifier.startsWith('file:')) {
			const url = new URL(specifier, context.parentURL)
			for (const suffix of ['', '.ts', '.tsx']) {
				const candidate = new URL(url.href + suffix)
				if (/\.(tsx?|css)$/.test(candidate.pathname) && existsSync(candidate)) {
					return { url: candidate.href, shortCircuit: true }
				}
			}
		}
		return nextResolve(specifier, context)
	},
	load(url, context, nextLoad) {
		if (url.endsWith('.css')) {
			return {
				format: 'module',
				source: 'export default new Proxy({}, { get: (_, key) => key })',
				shortCircuit: true
			}
		}
		if (/\.tsx?$/.test(url)) {
			const { outputText } = ts.transpileModule(readFileSync(new URL(url), 'utf8'), {
				fileName: fileURLToPath(url),
				compilerOptions: { module: ts.ModuleKind.ESNext, jsx: ts.JsxEmit.ReactJSX }
			})
			return { format: 'module', source: outputText, shortCircuit: true }
		}
		return nextLoad(url, context)
	}
})

// Execute a real TypeScript module with only its OS, Electron and Docker boundaries replaced.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import * as path from 'node:path'
import { fileURLToPath } from 'node:url'
import { runInThisContext } from 'node:vm'
import ts from 'typescript'

export function loadCommonJsModule(source, imports, scope = {}) {
	const url = new URL(`../../../${source}`, import.meta.url)
	const filename = fileURLToPath(url)
	const { outputText } = ts.transpileModule(readFileSync(url, 'utf8'), {
		fileName: filename,
		compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 }
	})
	const module = { exports: {} }
	const names = Object.keys(scope)
	const evaluate = runInThisContext(
		`(function(require, module, exports, __dirname, ${names.join(', ')}) { ${outputText}\n})`,
		{ filename }
	)
	evaluate(
		(name) => {
			if (name in imports) return imports[name]
			if (name === 'path') return path
			assert.fail(`Unexpected dependency: ${name}`)
		},
		module,
		module.exports,
		path.dirname(filename),
		...names.map((name) => scope[name])
	)
	return module.exports
}

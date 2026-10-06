import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import test from 'node:test'

const validator = fileURLToPath(new URL('../validate-npm-audit.mjs', import.meta.url))
const sprintfAdvisory = 'GHSA-hp3w-g68c-fv3c'
const routerAdvisory = 'GHSA-qwww-vcr4-c8h2'

function advisory(name, id) {
	return { name, url: `https://github.com/advisories/${id}` }
}

// npm propagates this single source advisory through a cyclic packaging graph.
function packagingReport() {
	return {
		vulnerabilities: {
			'electron-builder': { via: ['app-builder-lib', 'dmg-builder'] },
			'app-builder-lib': {
				via: ['dmg-builder', 'electron-builder-squirrel-windows', '@electron/get']
			},
			'dmg-builder': { via: ['app-builder-lib'] },
			'electron-builder-squirrel-windows': { via: ['app-builder-lib'] },
			'@electron/get': { via: ['global-agent'] },
			'global-agent': { via: ['roarr'] },
			roarr: { via: ['sprintf-js'] },
			'sprintf-js': { via: [advisory('sprintf-js', sprintfAdvisory)] }
		}
	}
}

function runAudit(report, { status = 1, routerVersion = '7.18.2' } = {}) {
	const directory = mkdtempSync(join(tmpdir(), 'ouroboros-audit-test-'))
	try {
		for (const name of ['react-router', 'react-router-dom']) {
			const packageDirectory = join(directory, 'node_modules', name)
			mkdirSync(packageDirectory, { recursive: true })
			writeFileSync(
				join(packageDirectory, 'package.json'),
				JSON.stringify({ version: routerVersion })
			)
		}
		const npm = join(directory, 'npm.cjs')
		writeFileSync(
			npm,
			`console.log(${JSON.stringify(typeof report === 'string' ? report : JSON.stringify(report))}); process.exit(${status});`
		)
		const result = spawnSync(process.execPath, [validator], {
			cwd: directory,
			env: { ...process.env, npm_execpath: npm },
			encoding: 'utf8'
		})
		assert.ifError(result.error)
		return result
	} finally {
		rmSync(directory, { recursive: true, force: true })
	}
}

test('accepts the reviewed sprintf advisory and its cyclic packaging dependents', () => {
	const result = runAudit(packagingReport())
	assert.equal(result.status, 0, result.stderr)
	assert.match(result.stderr, new RegExp(sprintfAdvisory))
	assert.match(result.stderr, /fixed logging format strings/)
})

test('rejects another advisory even on an approved source or downstream package', () => {
	for (const name of ['sprintf-js', 'app-builder-lib']) {
		const report = packagingReport()
		report.vulnerabilities[name].via.push(advisory(name, 'GHSA-test-test-test'))
		assert.equal(runAudit(report).status, 1)
	}
})

test('rejects an unreviewed dependent and a new edge between reviewed packages', () => {
	const extraDependent = packagingReport()
	extraDependent.vulnerabilities['other-tool'] = { via: ['sprintf-js'] }
	assert.equal(runAudit(extraDependent).status, 1)
	const newEdge = packagingReport()
	newEdge.vulnerabilities['electron-builder'].via.push('sprintf-js')
	assert.equal(runAudit(newEdge).status, 1)
})

test('rejects cycles without an advisory and missing dependency findings', () => {
	assert.equal(
		runAudit({
			vulnerabilities: {
				'app-builder-lib': { via: ['dmg-builder'] },
				'dmg-builder': { via: ['app-builder-lib'] }
			}
		}).status,
		1
	)
	const report = packagingReport()
	delete report.vulnerabilities['sprintf-js']
	assert.equal(runAudit(report).status, 1)
})

test('matches advisory URLs exactly and only at the approved source package', () => {
	const report = packagingReport()
	report.vulnerabilities['sprintf-js'].via[0].url += '-different-advisory'
	assert.equal(runAudit(report).status, 1)
	const wrongSource = packagingReport()
	wrongSource.vulnerabilities.roarr.via = [advisory('roarr', sprintfAdvisory)]
	assert.equal(runAudit(wrongSource).status, 1)
})

test('preserves the React Router version guard alone and with the packaging exception', () => {
	const router = {
		'react-router': { via: [advisory('react-router', routerAdvisory)] },
		'react-router-dom': { via: ['react-router'] }
	}
	for (const report of [
		{ vulnerabilities: router },
		{ vulnerabilities: { ...packagingReport().vulnerabilities, ...router } }
	]) {
		assert.equal(runAudit(report).status, 0)
		assert.equal(runAudit(report, { routerVersion: '7.18.1' }).status, 1)
	}
})

test('allows a clean audit and rejects unavailable, malformed, or unexplained failed audits', () => {
	assert.equal(runAudit({ vulnerabilities: {} }, { status: 0 }).status, 0)
	assert.equal(runAudit({ error: { code: 'E503' } }).status, 1)
	assert.equal(runAudit('invalid JSON').status, 1)
	assert.equal(runAudit({ vulnerabilities: {} }).status, 1)
	assert.equal(runAudit(packagingReport(), { status: 2 }).status, 1)
})

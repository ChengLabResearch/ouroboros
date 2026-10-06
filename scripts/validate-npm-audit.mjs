import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { npmInvocation } from './lib/npm.mjs'

const exceptions = [
	{
		advisory: 'GHSA-qwww-vcr4-c8h2',
		sourcePackage: 'react-router',
		via: { 'react-router': [], 'react-router-dom': ['react-router'] },
		versions: { 'react-router': '7.18.2', 'react-router-dom': '7.18.2' },
		reason: 'Both React Router packages are pinned to the patched v7 release 7.18.2.'
	},
	{
		advisory: 'GHSA-hp3w-g68c-fv3c',
		sourcePackage: 'sprintf-js',
		// Reviewed 2026-10-06: global-agent's roarr logging uses fixed format
		// strings. Application input cannot supply sprintf precision specifiers.
		// Limit this exception to the reviewed Electron packaging chain, including
		// npm's cyclic propagated findings. Reassess if the input boundary changes.
		via: {
			'sprintf-js': [],
			roarr: ['sprintf-js'],
			'global-agent': ['roarr'],
			'@electron/get': ['global-agent'],
			'app-builder-lib': [
				'@electron/get',
				'dmg-builder',
				'electron-builder-squirrel-windows'
			],
			'dmg-builder': ['app-builder-lib'],
			'electron-builder-squirrel-windows': ['app-builder-lib'],
			'electron-builder': ['app-builder-lib', 'dmg-builder']
		},
		reason: 'Electron packaging uses fixed logging format strings; application input cannot supply them.'
	}
]

const npm = npmInvocation(['audit', '--json'])
const audit = spawnSync(npm.command, npm.args, {
	encoding: 'utf8',
	stdio: ['ignore', 'pipe', 'inherit']
})

if (audit.error) {
	throw audit.error
}

let report
try {
	report = JSON.parse(audit.stdout)
} catch {
	console.error('npm audit did not return valid JSON.')
	process.exit(1)
}

if (audit.status === 0) {
	console.log('npm audit found no vulnerabilities.')
	process.exit(0)
}

if (audit.status !== 1 || report.error || !report.vulnerabilities) {
	console.error(report.message || 'npm audit failed before producing a vulnerability report.')
	process.exit(1)
}

const vulnerabilities = report.vulnerabilities
const reportedPackages = Object.keys(vulnerabilities)
if (reportedPackages.length === 0) {
	console.error('npm audit failed without reporting a vulnerability.')
	process.exit(1)
}

const appliedExceptions = new Set()
const unapprovedPackages = reportedPackages.filter((packageName) => {
	const exception = exceptions.find((candidate) => validateFinding(packageName, candidate))
	if (!exception) return true
	appliedExceptions.add(exception)
	return false
})

if (unapprovedPackages.length > 0) {
	console.error(`npm audit reported unapproved findings for: ${unapprovedPackages.join(', ')}`)
	process.exit(1)
}

for (const exception of appliedExceptions) {
	for (const [packageName, version] of Object.entries(exception.versions ?? {})) {
		const packageJson = JSON.parse(
			readFileSync(join(process.cwd(), 'node_modules', packageName, 'package.json'), 'utf8')
		)
		if (packageJson.version !== version) {
			console.error(
				`${packageName} ${packageJson.version} is not the approved version ${version}.`
			)
			process.exit(1)
		}
	}
	console.warn(`Accepted reviewed npm audit finding ${exception.advisory}: ${exception.reason}`)
}

function validateFinding(packageName, exception) {
	const seen = new Set()
	let foundAdvisory = false

	function visit(name) {
		if (!Object.hasOwn(exception.via, name) || !Object.hasOwn(vulnerabilities, name))
			return false
		if (seen.has(name)) return true
		seen.add(name)
		const finding = vulnerabilities[name]
		if (!finding || !Array.isArray(finding.via) || finding.via.length === 0) return false

		return finding.via.every((source) => {
			if (typeof source === 'string') {
				return exception.via[name].includes(source) && visit(source)
			}
			const approved =
				name === exception.sourcePackage &&
				source !== null &&
				typeof source === 'object' &&
				source.name === name &&
				source.url === `https://github.com/advisories/${exception.advisory}`
			if (approved) foundAdvisory = true
			return approved
		})
	}

	// Cycles in npm's propagated findings are allowed only when every edge is
	// reviewed and the traversal actually reaches the approved source advisory.
	return visit(packageName) && foundAdvisory
}

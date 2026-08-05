import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { npmInvocation } from './lib/npm.mjs'

const approvedAdvisory = 'GHSA-qwww-vcr4-c8h2'
const patchedVersion = '7.18.2'
const approvedPackages = new Set(['react-router', 'react-router-dom'])

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
const unexpectedPackages = reportedPackages.filter((name) => !approvedPackages.has(name))

if (unexpectedPackages.length > 0) {
	console.error(
		`npm audit reported unapproved vulnerable packages: ${unexpectedPackages.join(', ')}`
	)
	process.exit(1)
}

if (reportedPackages.length === 0) {
	console.error('npm audit failed without reporting a vulnerability.')
	process.exit(1)
}

for (const packageName of approvedPackages) {
	const packageJson = JSON.parse(
		readFileSync(join(process.cwd(), 'node_modules', packageName, 'package.json'), 'utf8')
	)
	if (packageJson.version !== patchedVersion) {
		console.error(
			`${packageName} ${packageJson.version} is not the approved patched v7 release ${patchedVersion}.`
		)
		process.exit(1)
	}
}

const seen = new Set()

function validateFinding(packageName) {
	if (seen.has(packageName)) return true
	seen.add(packageName)

	const finding = vulnerabilities[packageName]
	if (!finding || !Array.isArray(finding.via) || finding.via.length === 0) return false

	return finding.via.every((source) => {
		if (typeof source === 'string') {
			return approvedPackages.has(source) && validateFinding(source)
		}

		return (
			typeof source === 'object' &&
			source !== null &&
			typeof source.url === 'string' &&
			source.url.includes(approvedAdvisory)
		)
	})
}

if (!reportedPackages.every(validateFinding)) {
	console.error(`npm audit reported an advisory other than ${approvedAdvisory}.`)
	process.exit(1)
}

console.warn(
	`npm audit still reports ${approvedAdvisory}, but both React Router packages are pinned to its patched v7 release ${patchedVersion}.`
)
console.warn(
	'See https://github.com/remix-run/react-router/security/advisories/GHSA-qwww-vcr4-c8h2'
)

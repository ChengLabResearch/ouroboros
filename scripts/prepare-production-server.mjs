import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { loadReleaseLock } from './lib/release-lock.mjs'

const root = process.cwd()
const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const outputDir = join(root, 'extra-resources', 'server')
const { lock, sha256: releaseLockSha256 } = await loadReleaseLock(root)

if (packageJson.version !== lock.releaseVersion) {
	throw new Error(
		`package.json version ${packageJson.version} does not match release lock version ${lock.releaseVersion}`
	)
}

const image = `${lock.serverImage.repository}@${lock.serverImage.digest}`
// shm_size is an artificial Docker limit: without it, Docker caps
// /dev/shm at 64 MB, which the pipeline exceeds immediately. Setting it too
// large only shifts OOM from the container-side limit to actual host OOM.
// Override at packaging time via OUROBOROS_SERVER_SHM_SIZE (matches the
// compose-file substitution ${OUROBOROS_SERVER_SHM_SIZE:-64gb}).
const shmSize = process.env.OUROBOROS_SERVER_SHM_SIZE ?? '64gb'

await mkdir(outputDir, { recursive: true })
await writeFile(join(outputDir, 'compose.yml'), composeForImage(image))
await writeFile(
	join(outputDir, 'server-image.json'),
	`${JSON.stringify(
		{
			image,
			repository: lock.serverImage.repository,
			digest: lock.serverImage.digest,
			sourceCommit: lock.serverImage.sourceCommit,
			packageCommit: process.env.GITHUB_SHA ?? null,
			packageRef: process.env.GITHUB_REF_NAME ?? null,
			releaseLockSha256
		},
		null,
		2
	)}\n`
)

function composeForImage(serverImage) {
	return `services:
  ouroboros-server:
    image: ${serverImage}
    container_name: ouroboros-server
    ports:
      - '8000:8000'
    volumes:
      - ouroboros-volume:/volume
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - OUR_ENV=docker
    shm_size: ${shmSize}
volumes:
  ouroboros-volume:
    name: ouroboros-volume
`
}

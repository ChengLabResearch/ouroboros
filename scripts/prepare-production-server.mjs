import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

const root = process.cwd()
const packageJson = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
const outputDir = join(root, 'extra-resources', 'server')

const imageRepository =
	process.env.OUROBOROS_SERVER_IMAGE_REPOSITORY ??
	'ghcr.io/chenglabresearch/ouroboros-server'
const explicitImage = process.env.OUROBOROS_SERVER_IMAGE ?? null
const imageTag = process.env.OUROBOROS_SERVER_IMAGE_TAG ?? process.env.GITHUB_REF_NAME ?? `v${packageJson.version}`
const imageDigest = process.env.OUROBOROS_SERVER_IMAGE_DIGEST ?? null
const image = explicitImage ?? imageReference()

await mkdir(outputDir, { recursive: true })
await writeFile(join(outputDir, 'compose.yml'), composeForImage(image))
await writeFile(
	join(outputDir, 'server-image.json'),
	`${JSON.stringify(
		{
			image,
			repository: imageRepository,
			tag: explicitImage || imageDigest ? null : imageTag,
			digest: imageDigest,
			commit: process.env.GITHUB_SHA ?? null,
			ref: process.env.GITHUB_REF_NAME ?? null
		},
		null,
		2
	)}\n`
)

function imageReference() {
	if (imageDigest) return `${imageRepository}@${imageDigest}`
	return `${imageRepository}:${imageTag}`
}

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
    shm_size: 64gb
volumes:
  ouroboros-volume:
    name: ouroboros-volume
`
}

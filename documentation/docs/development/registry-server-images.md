# Registry Server Images

Ouroboros can publish the Python server Docker image to GHCR during release-tag workflows. This avoids turning every production startup into a local image build once release packaging is ready to reference immutable image tags.

## Published Image

The `Publish Server Image` workflow builds the Python wheel, builds `python/Dockerfile-prod`, and publishes:

- `ghcr.io/chenglabresearch/ouroboros-server:<release-tag>` for release tags such as `v1.4.0`
- `ghcr.io/chenglabresearch/ouroboros-server:sha-<commit>` for the exact source revision

Both tags are produced from the same wheel artifact that the Dockerfile installs.

## Release Compose Usage

Release packaging uses `npm run prepare:production-server` to write
`extra-resources/server/compose.yml` and `extra-resources/server/server-image.json`.
For a tag build, the app package workflow sets:

```yaml
OUROBOROS_SERVER_IMAGE_TAG=v1.4.0
```

The generated compose file preserves the production ports, `ouroboros-volume`
mount, `host.docker.internal` mapping, `OUR_ENV=docker`, and `shm_size` settings,
but uses `image:` instead of `build:`. Production first launch therefore pulls
the published GHCR image instead of building the server locally.

For digest-pinned packages, set:

```bash
OUROBOROS_SERVER_IMAGE_REPOSITORY=ghcr.io/chenglabresearch/ouroboros-server
OUROBOROS_SERVER_IMAGE_DIGEST=sha256:<digest>
npm run prepare:production-server
```

Development compose files still build locally from source so local edits do not
depend on registry state.

## Manual Runs

The workflow also supports `workflow_dispatch` for validating or re-publishing the image from the current branch. Release consumers should still use tag or SHA image references rather than mutable names.

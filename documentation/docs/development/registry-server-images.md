# Registry Server Images

Ouroboros can publish the Python server Docker image to GHCR during release-tag workflows. This avoids turning every production startup into a local image build once release packaging is ready to reference immutable image tags.

## Published Image

The `Publish Server Image` workflow builds the Python wheel, builds `python/Dockerfile-prod`, and publishes:

- `ghcr.io/chenglabresearch/ouroboros-server:<release-tag>` for release tags such as `v1.4.0`
- `ghcr.io/chenglabresearch/ouroboros-server:sha-<commit>` for the exact source revision

Both tags are produced from the same wheel artifact that the Dockerfile installs.

## Release Compose Usage

Release packaging should prefer an immutable image tag when the corresponding image exists:

```yaml
services:
  ouroboros-server:
    image: ghcr.io/chenglabresearch/ouroboros-server:v1.4.0
```

Keep the bundled wheel and `Dockerfile-prod` path available as a fallback until installers consistently ship a release-specific compose file. Development compose files should keep building locally from source so local edits do not depend on registry state.

## Manual Runs

The workflow also supports `workflow_dispatch` for validating or re-publishing the image from the current branch. Release consumers should still use tag or SHA image references rather than mutable names.

# Registry Server Images

The Python server image has a separate build lifecycle from desktop installers.
Production packages use the image digest pinned in
[`release/release-lock.json`](https://github.com/ChengLabResearch/ouroboros/blob/main/release/release-lock.json),
so application startup pulls the image instead of building it locally.

## Published Image

The `Build and Certify Server Image` workflow tests Python changes, builds the
wheel, and builds `python/Dockerfile-prod` on pull requests. PR runs do not publish
an image. On `main`, the workflow publishes `sha-<full-commit>` and records the
registry repository, image digest, and source commit in a certified server
manifest. A later release-lock update selects that image for desktop packages.
Application version tags do not rebuild the wheel or server image.

## Base Image Pin

Both Python Dockerfiles keep the readable `thehale/python-poetry` tag paired
with an immutable digest. When upgrading that base, review the upstream image,
resolve the new tag's digest, and update the tag and digest in
`python/Dockerfile` and `python/Dockerfile-prod` together in the same reviewed
change. Do not refresh the digest without an intentional base-image review.

## Release Compose Usage

Release packaging derives `extra-resources/server/compose.yml` and
`extra-resources/server/server-image.json` from the release lock inside its
external workspace. These generated files are package outputs, not editable
production image pins. The source checkout remains unchanged.

The generated compose file preserves the production ports, `ouroboros-volume`
mount, `host.docker.internal` mapping, `OUR_ENV=docker`, and `shm_size` settings,
but uses `image:` instead of `build:`. Production first launch therefore pulls
the published GHCR image instead of building the server locally.

To change the packaged server, update the lock's `serverImage` entry using the
certified server manifest. There are no per-image workflow inputs or preparation
commands that override the lock. Use the
[package build entrypoint](production-package-flavors.md#local-checks) to stage
and build desktop packages.

Development compose files still build locally from source so local edits do not
depend on registry state.

## Manual Runs

The server workflow also supports `workflow_dispatch` for testing and publishing
the selected revision. Desktop releases still consume its certified digest
through the release lock; dispatching the server workflow does not change that
lock or publish a desktop release.

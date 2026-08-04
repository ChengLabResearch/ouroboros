# Production package flavors

The release workflow builds three production package flavors for each supported
desktop runner:

- `core`: Ouroboros only, with the production server compose file pointing at
  the registry-published server image.
- `with-plugins-cpu`: the core package plus bundled Neuroglancer and Automatic
  Segmentation plugin artifacts. The automatic segmentation plugin uses its CPU
  backend image.
- `with-plugins-cuda`: the core package plus bundled Neuroglancer and Automatic
  Segmentation plugin artifacts. The automatic segmentation plugin uses its
  CUDA backend image and Docker Compose GPU reservation.

Plugin flavors stage release artifacts into
`extra-resources/preinstalled-plugins` before Electron Builder runs. The app
installs those bundled plugin folders into the user plugin directory on startup,
so production packages can ship with plugins already available.

## Release inputs

Tag releases default the server image to the Ouroboros release tag and default
bundled plugins to explicit production pins. Manual workflow runs can override
these inputs:

- `server_image_tag` or `server_image_digest`
- `neuroglancer_plugin_tag` (default `v1.1.1`)
- `neuroglancer_plugin_artifact` (default `neuroglancer-plugin-v1.1.1.zip`)
- `autoseg_plugin_tag` (default `v0.4.0-beta.2`)
- `autoseg_cpu_plugin_artifact` (default `auto-segmentation-v0.4.0-beta.2-cpu.zip`)
- `autoseg_cuda_plugin_artifact` (default `auto-segmentation-v0.4.0-beta.2-cuda.zip`)

The current plugin pins are:

- Neuroglancer plugin: `ChengLabResearch/neuroglancer-plugin` tag `v1.1.1`,
  asset `neuroglancer-plugin-v1.1.1.zip`
- Automatic segmentation plugin: `ChengLabResearch/ouroboros_autoseg_plugin`
  tag `v0.4.0-beta.2`, assets `auto-segmentation-v0.4.0-beta.2-cpu.zip` and
  `auto-segmentation-v0.4.0-beta.2-cuda.zip`

`extra-resources/package-flavor.json` records the selected package flavor,
server image metadata, and exact plugin release tag/artifact inputs. When a
plugin archive includes `plugin-release.json`, its release metadata is copied
into `package-flavor.json` as well.

If plugin release repositories are private, set
`OUROBOROS_RELEASE_ASSET_TOKEN` to a token that can read those release assets.
The workflow falls back to `GITHUB_TOKEN` when that secret is not set.

## Dependency audit

The release workflow runs `npm run audit:release` against the locked dependency
tree. npm currently combines the React Router 7 and 8 ranges for
[`GHSA-qwww-vcr4-c8h2`](https://github.com/remix-run/react-router/security/advisories/GHSA-qwww-vcr4-c8h2)
and incorrectly reports React Router 7.18.2. The audit validator permits only
that advisory while both `react-router` and `react-router-dom` resolve to the
patched v7 release 7.18.2. It fails for any other advisory or resolved version.

## Local checks

Prepare the core package resources with:

```sh
npm run prepare:production-server
OUROBOROS_PACKAGE_FLAVOR=core npm run prepare:package-flavor
```

Plugin flavors can be checked against local release zips by placing them in
`.package-plugin-artifacts` or setting `OUROBOROS_PLUGIN_ARTIFACT_DIR`.
Use `OUROBOROS_PACKAGE_FLAVOR=with-plugins-cpu` or
`OUROBOROS_PACKAGE_FLAVOR=with-plugins-cuda` to stage bundled plugin packages.

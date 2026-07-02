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
- `neuroglancer_plugin_tag` (default `v1.0.1`)
- `neuroglancer_plugin_artifact` (default `neuroglancer-plugin-v1.0.1.zip`)
- `autoseg_plugin_tag` (default `v0.4.0-beta.1`)
- `autoseg_cpu_plugin_artifact` (default `auto-segmentation-v0.4.0-beta.1-cpu.zip`)
- `autoseg_cuda_plugin_artifact` (default `auto-segmentation-v0.4.0-beta.1-cuda.zip`)

The current plugin pins are:

- Neuroglancer plugin: `ChengLabResearch/neuroglancer-plugin` tag `v1.0.1`,
  asset `neuroglancer-plugin-v1.0.1.zip`
- Automatic segmentation plugin: `ChengLabResearch/ouroboros_autoseg_plugin`
  tag `v0.4.0-beta.1`, assets `auto-segmentation-v0.4.0-beta.1-cpu.zip` and
  `auto-segmentation-v0.4.0-beta.1-cuda.zip`

`extra-resources/package-flavor.json` records the selected package flavor,
server image metadata, and exact plugin release tag/artifact inputs. When a
plugin archive includes `plugin-release.json`, its release metadata is copied
into `package-flavor.json` as well.

If plugin release repositories are private, set
`OUROBOROS_RELEASE_ASSET_TOKEN` to a token that can read those release assets.
The workflow falls back to `GITHUB_TOKEN` when that secret is not set.

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

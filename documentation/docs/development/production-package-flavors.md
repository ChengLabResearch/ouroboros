# Production package flavors

The release workflow builds three production package flavors for each supported
desktop runner:

- `core`: Ouroboros only, with the production server compose file pointing at
  the registry-published server image.
- `plugins-cpu`: the core package plus bundled Neuroglancer and Automatic
  Segmentation plugin artifacts. The automatic segmentation plugin uses its CPU
  backend image.
- `plugins-cuda`: the core package plus bundled Neuroglancer and Automatic
  Segmentation plugin artifacts. The automatic segmentation plugin uses its
  CUDA backend image and Docker Compose GPU reservation.

Plugin flavors stage release artifacts into
`extra-resources/preinstalled-plugins` before Electron Builder runs. The app
installs those bundled plugin folders into the user plugin directory on startup,
so production packages can ship with plugins already available.

## Release inputs

Tag releases default every image and plugin artifact to the Ouroboros release
tag. Manual workflow runs can override these inputs:

- `server_image_tag` or `server_image_digest`
- `neuroglancer_plugin_tag`
- `neuroglancer_plugin_artifact`
- `autoseg_plugin_tag`
- `autoseg_cpu_plugin_artifact`
- `autoseg_cuda_plugin_artifact`

The default plugin artifact names are:

- `neuroglancer-plugin-<tag>.zip`
- `auto-segmentation-<tag>-cpu.zip`
- `auto-segmentation-<tag>-cuda.zip`

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

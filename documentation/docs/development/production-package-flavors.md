# Production package flavors

The release workflow builds the flavors selected by
[`release/release-lock.json`](https://github.com/ChengLabResearch/ouroboros/blob/main/release/release-lock.json)
for each supported desktop runner. The available flavors are:

- `core`: Ouroboros only, with the production server compose file pointing at
  the registry-published server image.
- `with-plugins-cpu`: the core package plus bundled Neuroglancer and Automatic
  Segmentation plugin artifacts. The automatic segmentation plugin uses its CPU
  backend image.
- `with-plugins-cuda`: the core package plus bundled Neuroglancer and Automatic
  Segmentation plugin artifacts. The automatic segmentation plugin uses its
  CUDA backend image and Docker Compose GPU reservation.

Plugin flavors stage bundled assets in separate resource directories inside the
external packaging workspace. Electron Builder maps each flavor's resources into
the installed `extra-resources` directory, including
`extra-resources/preinstalled-plugins`. The app installs those bundled plugin
folders into the user plugin directory on startup.

## Release inputs

The release lock is the only source of production pins and selected flavors.
Update it to change the server image or bundled plugins; there are no separate
per-image or per-plugin workflow inputs. The application version must match the
lock's `releaseVersion`.

The server is pinned by registry repository, immutable image digest, and source
commit. Plugin inputs are pinned by repository, release tag, asset filename,
SHA-256, and source commit, with separate CPU and CUDA backend identities.
Packaging verifies downloaded archive hashes and requires `plugin-release.json`
to match the locked release identity. Backend image verification follows the
corresponding lock entry. See the lock for current versions and digests.

The packaged `extra-resources/package-flavor.json` records the selected flavor,
release-lock hash, server image metadata, and exact plugin identities, including
the verified plugin release metadata. This file describes the package; the
release lock remains authoritative.

If plugin release repositories are private, set
`OUROBOROS_RELEASE_ASSET_TOKEN` to a token that can read those release assets.
The workflow falls back to `GITHUB_TOKEN` when that secret is not set.

## Build and publication

Pull requests run release preflight checks without building native installers.
Native builds run on `main`, compiling Electron once per operating system and
packaging each selected flavor. On CI, build workspaces, generated resources,
dependency trees, plugin downloads, and final artifacts live under `RUNNER_TEMP`,
outside `GITHUB_WORKSPACE`; the source checkout is read-only input. Electron
Builder uses an explicit application-file allowlist rather than packaging the
checkout.

Source files and installed dependencies are copied once into an external tree
where Electron is compiled. Every flavor packages that same compiled application
and dependency tree, with `npmRebuild: false` enforced. Flavor preparation and
cleanup affect only that flavor's resources and output directory. The shared
compiled tree remains available across flavors and is removed at the end of the
run; there are no links back to checkout dependencies.

A successful `main` build records the exact source commit, input lock, artifact
sizes, and SHA-256 hashes. The tag-triggered publication workflow verifies those
prebuilt artifacts and creates a draft release without rebuilding Electron or the
server. See the [release pipeline guide](https://github.com/ChengLabResearch/ouroboros/blob/main/release/README.md).

Dependency auditing uses `npm run audit:release` against locked dependencies.
Any reviewed audit exceptions are defined in the
[audit validator](https://github.com/ChengLabResearch/ouroboros/blob/main/scripts/validate-npm-audit.mjs).

## Local checks

Run the fast release preflight from the repository root:

```sh
npm run preflight:release
```

It validates the lock, application version, and packaging configuration, and
exercises disposable external staging without compiling Electron, downloading
plugin assets, or producing installers.

Build all locked flavors on the corresponding native platform:

```sh
npm run build:release-artifacts -- --platform linux
```

Use `mac` on macOS or `win` on Windows. Scratch files and final artifacts default
to `RUNNER_TEMP`, or the operating system's temporary directory when running
locally without that variable. The command prints the final output location.
Set `OUROBOROS_RELEASE_ARTIFACT_DIR` to choose another destination beneath that
temporary root. It must be an absolute path outside the checkout and must be new
or empty; existing artifacts are not cleared. Scratch files are removed after
the run, while final artifacts are retained.

The existing `build:linux`, `build:mac`, and `build:win` commands use the same
staging and packaging entrypoint. For a local single-flavor build, select a flavor
already present in the lock, for example:

```sh
OUROBOROS_PACKAGE_FLAVOR=core npm run build:linux
```

`npm run build:unpack` creates an unpacked application for the current platform
through that entrypoint and prints its external location. Preparation and
renaming scripts are internal build steps, not standalone checkout commands.

Plugin downloads use the workspace's cache. To seed it with local release
archives, set `OUROBOROS_PLUGIN_ARTIFACT_DIR` to an external directory that does
not overlap the checkout or build inputs and outputs. Its contents are copied
into the owned workspace cache; the original directory is never modified.
Copied archives must match the filenames and SHA-256 values in the release lock.
Stale copies are replaced within the workspace, without changing the originals
or overriding the locked plugin identities.

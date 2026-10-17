# Release pipeline

[`release-lock.json`](release-lock.json) is the production source of truth for the server image, bundled plugin assets, and package flavors. The packaging scripts and workflows do not carry fallback production pins.

For each release:

1. Update the application version in `package.json`, `package-lock.json`, and `release-lock.json`.
2. Pin the certified Ouroboros server image by digest and source commit. A plugin-only release keeps the existing digest and does not touch `python/`.
3. Pin each plugin release asset by repository, tag, filename, SHA-256, source commit, and recorded backend image identity. CPU and CUDA autoseg inputs are independent.
4. Set `packageFlavors` to the affected installers. A full release selects all three flavors; a CUDA-only hotfix selects only `with-plugins-cuda`.
5. Merge only after release preflight succeeds. Pull requests validate the release inputs and packaging code without building native installers. On `main`, the native build compiles the Electron application once per operating system, then performs final packaging for each selected flavor.
6. The successful `main` run certifies the artifacts for the exact merge commit and creates the matching version tag. `Publish Prebuilt Release` fetches that exact run's artifacts, verifies the certification, source tree, release lock, manifest, sizes, and SHA-256 values, and creates the draft GitHub release. The tag workflow does not compile Electron or build a server image.

Release builds treat the source checkout as read-only input. On CI, the application tree, copied dependencies, plugin downloads, generated resources, and final artifacts are staged under `RUNNER_TEMP`, outside `GITHUB_WORKSPACE`. Local builds fall back to the operating system's temporary directory.

Source files and installed dependencies are copied once; every flavor packages the same compiled application and dependency tree with `npmRebuild: false` enforced. Each flavor has separate resources and outputs, and its resources are mapped into the installed `extra-resources` directory. Flavor cleanup never removes the shared compiled tree, which is cleaned at the end of the run. Final artifacts and requested unpacked applications are retained.

Electron Builder includes only explicitly allowed application files and staged production resources. There are no dependency links back to the checkout, and changing a flavor does not copy dependencies again or write generated files into the checkout. See the [package-flavor guide](../documentation/docs/development/production-package-flavors.md#local-checks) for local preflight, build, and output-directory commands.

Each OS artifact set includes a build manifest with the complete input lock, source identity, artifact hashes, byte counts, runner timing, and operation counts. Each installer flavor also includes package metadata recording the exact server and plugin identities bundled into it.

Server images have a separate lifecycle. Changes under `python/` build and test the server on pull requests, then publish only an immutable `sha-<full-commit>` image on `main`. The resulting digest manifest is the input used for a later release-lock update; application version tags never build the wheel or Docker image.

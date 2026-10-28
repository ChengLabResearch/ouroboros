# Release pipeline

[`release-lock.json`](release-lock.json) is the production source of truth for the server image, bundled plugin assets, and package flavors. The packaging scripts and workflows do not carry fallback production pins.

## Prepare an application release

1. Update the application version in `package.json`, `package-lock.json` (both its top-level version and `packages[""].version`), and `release-lock.json`. Preflight checks all four values. The Python wheel has an independent version; see the server release steps below.
2. Pin the certified Ouroboros server image by digest and source commit. A plugin-only release keeps the existing server pin. Editing `python/pyproject.toml` alone does not change the server shipped with the application.
3. Pin each plugin release asset by repository, tag, filename, SHA-256, source commit, and recorded backend image identity. CPU and CUDA autoseg inputs are independent.
4. Set `packageFlavors` to the affected installers. A full release selects all three flavors; a CUDA-only hotfix selects only `with-plugins-cuda`.
5. Run `npm ci` and `npm run preflight:release` before merging. Pull requests validate release inputs and packaging code without building native installers. A push that changes artifact inputs on `main` compiles the Electron application once per operating system, then packages each selected flavor. A manual `Build Release Artifacts` dispatch on `main` can create a fresh eligible build.
6. Wait for a successful build with all three platform artifacts. Run the pre-tag check below before manually creating and pushing the matching version tag. Builds do not create tags automatically.
7. `Publish Prebuilt Release` selects an eligible ancestor build, downloads its artifacts, verifies the certification, artifact-input fingerprint, release lock, manifests, sizes, and SHA-256 values, and creates the draft GitHub release. Tagging never compiles Electron or builds a server image.

## Check before tagging

Use Node.js 24 and GitHub CLI authenticated with access to this repository's Actions artifacts (`GH_TOKEN` or `gh auth login`). Fetch `origin/main` and its full history before checking the candidate. If the checkout is shallow, run `git fetch --unshallow origin` first.

```sh
git fetch origin main
node scripts/check-release.mjs --commit origin/main
```

`--commit` also accepts a specific candidate commit. Add `--ref v1.6.0-beta.1` with the intended tag name to check version agreement explicitly. The command selects, downloads, and verifies the same artifacts used by publishing; it creates no tag or release. After it succeeds, create and push the tag at the exact candidate commit that was checked.

Eligible builds are successful `Build Release Artifacts` runs from a `push` or `workflow_dispatch` on `main`. The build commit must be an ancestor of the candidate, and the candidate must be on `origin/main`. The candidate and build must have the same canonical artifact-input fingerprint, application version, and release lock. A whole-tree hash or an exact commit match is not the reuse gate. For example, later changes under `documentation/` or `python/` can reuse an earlier application build when they do not change its artifact inputs. An application, packaging, dependency, plugin-pin, or server-pin change requires a new build.

The canonical artifact-input definition is [`artifact-inputs.json`](artifact-inputs.json). It owns the file patterns used both to calculate committed input identity and to generate the build workflow's pull-request and push path filters. After changing that definition, run `node scripts/sync-release-input-paths.mjs --write`; CI checks the generated filters. The complete source tree remains recorded as provenance for the actual build commit.

Release artifacts are retained for **14 days**. A successful old run is not releasable once its required artifacts have expired or disappeared. If no eligible build has a complete, unexpired artifact set, dispatch `Build Release Artifacts` on `main`, wait for success, and rerun the check. Manual dispatch is also the recovery path when a build failed or a relevant input filter was previously missing. Do not tag until verification succeeds. Artifacts from the older certification schema cannot satisfy the new input-fingerprint checks: the initial migration requires one new successful build.

## Ship a server change

The server wheel version in `python/pyproject.toml` describes the Python package at the pinned server source commit. It is independent of the Electron application version. Bump it deliberately when releasing a changed Python package; an application-only or plugin-only release does not require a wheel bump. The current checkout's wheel version is not evidence of which wheel an application bundles.

Server images retain a two-commit release flow:

1. **Commit A: server source.** Update and test the server, including its wheel version when releasing a new Python package, then merge to `main`. `Build and Certify Server Image` tests the source, builds the wheel, and publishes the immutable `sha-<full-A-commit>` image. It uploads `server-image-manifest.json` in the `server-image-<full-A-commit>` artifact. That server manifest is retained for 30 days.
2. Download the manifest from A's successful server workflow. Record its repository, digest, and source commit in `release/release-lock.json`; all three must describe the same published image. A wheel-version edit or a new image tag cannot substitute for this repin.
3. **Commit B: application lock.** Update the application versions if required for the application release, and include the server repin. Before merging B, fetch A's source history and run the server-aware preflight using the downloaded manifest:

    ```sh
    npm run preflight:release -- --server-manifest /path/server-image-manifest.json
    ```

    This option is required when preparing a release intended to ship that server change. It rejects an omitted or mismatched repin, validates the manifest's digest image reference and immutable tag, and prints the wheel version from A's `python/pyproject.toml`. It does not compare the wheel version to the application version or require later Python files to equal A's files. Ordinary preflight cannot infer an intention to ship an unpinned server change and intentionally accepts an existing pin.

4. Merge B, wait for its application build, and run the pre-tag check. The `release/` lock change triggers the native build even though the preceding Python-only A commit does not. Tag the checked B candidate, or a later `main` candidate with identical artifact inputs. The application ships A's certified digest.

Keep the downloaded manifest with the release preparation evidence. The explicit server preflight verifies it against the lock and local pinned source; obtain it from the successful server workflow to establish its origin. A missing local A commit produces a fetch instruction. Application version tags never rebuild the wheel or Docker image.

## Decisions for the release-gating failures

| Issue #189 failure                                                      | Decision                                                                                                                                           |
| ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Python wheel version omitted from the app checklist                  | Wheel and application versions are independent; server releases check the version at the pinned source commit.                                     |
| 2. Editing the current wheel version does not change the bundled image  | The lock's repository, digest, and source commit remain authoritative; server release preparation checks the certified manifest against all three. |
| 3. Python-only changes do not trigger native application builds         | Keep that boundary. A server change reaches the application through the later release-lock repin, which triggers a native build.                   |
| 4. Manual dispatch builds were rejected                                 | Successful dispatches on `main` are eligible alongside successful pushes.                                                                          |
| 5. Exact commit and whole-tree matching rejected harmless later commits | Require build ancestry and equal canonical artifact inputs; retain the build source tree only as provenance.                                       |
| 6. The `reused-pr-build` path was dead                                  | Remove it. Pull requests validate inputs; native builds run on `main`.                                                                             |
| 7. Automatic tag documentation described a disabled workflow            | Remove the disabled tag workflow and use an explicit pre-tag check followed by manual tagging.                                                     |
| 8. Successful runs could outlive retained artifacts                     | Keep 14-day retention and check complete, unexpired artifact availability before tagging; dispatch a fresh build when needed.                      |
| 9. Publishing a server image and repinning it need separate commits     | Retain the A-publish, B-repin flow and check the certified manifest before B is merged.                                                            |

## Packaging boundary

Release builds treat the source checkout as read-only input. On CI, the application tree, copied dependencies, plugin downloads, generated resources, and final artifacts are staged under `RUNNER_TEMP`, outside `GITHUB_WORKSPACE`. Local builds fall back to the operating system's temporary directory.

The release artifact producer reads its release lock and identity from the checked-out commit. Commit intended release inputs before building: `OUROBOROS_RELEASE_LOCK` overrides are rejected, and a supplied `GITHUB_SHA` must match the checked-out commit. The producer and verifier use the same committed input definition and version checks.

Source files and installed dependencies are copied once; every flavor packages the same compiled application and dependency tree with `npmRebuild: false` enforced. Each flavor has separate resources and outputs, and its resources are mapped into the installed `extra-resources` directory. Flavor cleanup never removes the shared compiled tree, which is cleaned at the end of the run. Final artifacts and requested unpacked applications are retained.

Electron Builder includes only explicitly allowed application files and staged production resources. There are no dependency links back to the checkout, and changing a flavor does not copy dependencies again or write generated files into the checkout. See the [package-flavor guide](../documentation/docs/development/production-package-flavors.md#local-checks) for local preflight, build, and output-directory commands.

Each OS artifact set includes a build manifest with the complete input lock, source identity, artifact hashes, byte counts, runner timing, and operation counts. Each installer flavor also includes package metadata recording the exact server and plugin identities bundled into it.

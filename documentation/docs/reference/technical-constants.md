# Technical Constants

This page documents hardcoded implementation limits that affect Ouroboros
behavior at runtime. Values here are extracted directly from the shipping
source. If you change them locally, update this page in the same commit.

The constants below were introduced or touched by the [#51][issue-51]
out-of-memory mitigation stack. Historical context:

- [#104][pr-104] added the file explorer watcher depth, path-count, and
  ignored-segment guardrails.
- [#105][pr-105] added the file explorer update batch size and flush
  interval.
- [#106][pr-106] stopped broadcasting the recursive file tree to plugin
  iframes on every directory change.

## File explorer

All file explorer limits live in
[`src/main/event-handlers/filesystem.ts`][filesystem-ts] at the top of the
file. They apply to the recursive watcher that backs the File Explorer
panel in the Electron app.

### Watcher depth limit

- Symbol: `FILE_EXPLORER_WATCH_DEPTH`
- Value: `6`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 8
- Also passed to the underlying `watcher` package as its `depth` option and
  reported inside every `folder-contents-error` payload.

**Why it exists.** The file explorer used to open folders with an
unbounded recursive watcher. Deep dependency trees, build outputs, or
scientific datasets could push watcher and renderer state well past the
V8 heap and trigger the crash described in [#51][issue-51].

**User-visible behavior when reached.** Directories nested more than six
levels below the selected root are simply not walked, so their contents do
not appear in the File Explorer panel. The panel still works normally at
shallower depths.

**When to reconsider.** Raise this if you routinely need to see files
deeper than six levels from the folder you open in Ouroboros and you have
already excluded large auxiliary trees (see the ignored segments below).
Prefer opening a lower-level folder before increasing the constant.

### Watcher path-count limit

- Symbol: `FILE_EXPLORER_WATCH_LIMIT`
- Value: `100_000`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 9
- Passed to the underlying `watcher` package as its `limit` option and
  reported inside every `folder-contents-error` payload.

**Why it exists.** The watcher mirrors every discovered path into
renderer state. Very large folders can therefore multiply main-process
watcher memory with renderer heap. One hundred thousand visible paths is
the ceiling picked in [#104][pr-104] as a balance between usefulness for
typical scan/output folders and keeping the renderer well under its
default 4 GB heap.

**User-visible behavior when reached.** Once the visible add count crosses
the limit, the main process emits `folder-contents-error` with a message
of the form `File explorer stopped loading after 100000 visible paths.
Choose a smaller folder or expand the folder in smaller pieces.` The
existing renderer alert system surfaces this as a warning toast. The
watcher stops delivering further add events for that session; already
loaded paths remain visible.

**When to reconsider.** Raise this only after profiling shows headroom in
both the main watcher and the renderer heap for your typical workload.
Lower it if you routinely work on constrained machines and want the
warning to appear sooner.

### Ignored path segments

- Symbol: `IGNORED_PATH_SEGMENTS`
- Value: `new Set(['node_modules', '__pycache__', 'venv'])`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 12

**Why it exists.** These directories are almost always noise in an
Ouroboros workflow (JavaScript dependencies, Python bytecode caches,
Python virtual environments) and are the fastest way for a project folder
to blow past the watcher path limit above.

Note that in addition to the explicit set, the watcher's `shouldIgnorePath`
helper also skips any path segment that begins with `.`, so dotfiles and
dot-prefixed directories such as `.git`, `.venv`, and `.cache` are
excluded even though they are not listed by name.

**User-visible behavior when reached.** Matching paths never appear in
the File Explorer panel and do not count toward the path-count limit. No
warning is shown, because ignoring these segments is the intended default.

**When to reconsider.** Add a segment here if you find another commonly
large directory that most users would never want to browse from
Ouroboros. Remove one only if a real workflow genuinely needs it to be
visible, since exposing these trees is the fastest way back to the
out-of-memory behavior tracked in [#51][issue-51].

### Update batch size

- Symbol: `FILE_EXPLORER_UPDATE_BATCH_LIMIT`
- Value: `500`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 10

**Why it exists.** [#105][pr-105] introduced batching so that a large
initial folder scan no longer produces one IPC message and one React
state update per discovered path. Five hundred events per batch amortizes
the IPC round-trip and lets the renderer apply the whole batch inside a
single `setNodes` call.

**User-visible behavior when reached.** As soon as five hundred watcher
events have accumulated in the main process, the pending queue is flushed
immediately over the `folder-contents-update-batch` IPC channel instead
of waiting for the flush interval below. The File Explorer panel then
updates once for the whole batch.

**When to reconsider.** Increase the batch size if profiling shows IPC or
React reconciliation is still the bottleneck on very large folders.
Decrease it if you see visible latency between file system changes and
the first batch appearing in the panel.

### Update flush interval

- Symbol: `FILE_EXPLORER_UPDATE_INTERVAL_MS`
- Value: `100` (milliseconds)
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 11

**Why it exists.** The batch flush also runs on a timer so that small
bursts of changes still reach the renderer promptly, even when the
five-hundred-event threshold above is never hit.

**User-visible behavior when reached.** Any queued events are flushed at
most one hundred milliseconds after the first event in the queue was
enqueued. The panel therefore feels responsive to individual file
create/delete events while still coalescing large scans.

**When to reconsider.** Shorten the interval if the panel feels laggy
during light file activity. Lengthen it if renderer CPU during heavy
scans is a problem and you would rather trade responsiveness for
throughput.

## Plugin directory broadcast

- Broadcast payload: `{ directoryPath, directoryName, nodes: {} }`
- Source:
  [`src/renderer/src/contexts/IFrameContext.tsx`][iframe-context]
  (`broadcastDirectoryContext`)

**Why it exists.** Before [#106][pr-106], every directory context change
posted the full recursive `nodes` tree into each plugin iframe. For the
same large folders that motivated [#51][issue-51] this cloned the entire
file tree on every update and stacked on top of the watcher and renderer
costs above.

**User-visible behavior.** Plugins still receive the selected directory
path and name through the existing `send-directory-contents` message.
The `nodes` field is present for shape compatibility but is always an
empty object. Plugins that need to browse the directory tree should not
depend on this broadcast; the on-demand plugin file API tracked by
[#102][issue-102] is the intended replacement.

**When to reconsider.** Do not restore the full-tree broadcast without
first landing a memory-safe, on-demand alternative. If you need a
minimal listing for a plugin, prefer adding a targeted IPC call over
resurrecting the broadcast.

## Main process (Electron)

### Plugin file server port

- Symbol: `PLUGIN_FILE_SERVER_PORT`
- Value: `3000` (default)
- Source: [`src/main/servers/file-server.ts`][file-server-ts] line 7
- Environment override: `OUROBOROS_PLUGIN_FILE_SERVER_PORT` (integer;
  unset, empty, or non-numeric values fall back to `3000`)

**Why it exists.** The main process forks a local Express + `serve-static`
worker (see [`resources/processes/file-server-script.mjs`][file-server-script])
that serves every installed plugin's static assets over
`http://127.0.0.1:3000/` by default. The renderer and plugin iframes load
their `index.html`, icons, and any bundled JS from this server.

**User-visible behavior.** If another process on the host is already
listening on the chosen port the fork fails and plugins load with broken
assets. There is no automatic fallback beyond the env override.

**When to reconsider.** Set `OUROBOROS_PLUGIN_FILE_SERVER_PORT` when 3000
conflicts with a different local service. The renderer discovers the URL
through `getPluginFileServerURL()`, so the port change is picked up
without touching any user-facing setting. Landed in [PR #109][pr-109].

### Docker volume server port

- Symbol: `VOLUME_SERVER_PORT`
- Value: `3001` (default)
- Source: [`src/main/servers/volume-server.ts`][volume-server-ts] line 7
- Environment override: `OUROBOROS_VOLUME_SERVER_PORT` (integer;
  unset, empty, or non-numeric values fall back to `3001`)

**Why it exists.** The volume server is a second forked Node process
(see [`resources/processes/volume-server-script.mjs`][volume-server-script])
that shells out to `docker run --rm` commands so that host files can be
copied into and out of the `ouroboros-volume` Docker named volume. The
main server container reaches back to it through
`http://host.docker.internal:3001` (see [Python server](#python-server)
below).

**User-visible behavior.** Copy-to-volume and copy-to-host requests fail
if the chosen port is busy or if Docker is not available. Plugin data
transfer into the shared volume breaks in that case.

**When to reconsider.** Set `OUROBOROS_VOLUME_SERVER_PORT` when 3001
conflicts locally. Keep the Python-side default in sync
(`VOLUME_SERVER_URL` in
[`python/ouroboros/common/volume_server_interface.py`][volume-server-interface-py]);
the Python side is still hard-coded to `3001` because the container
reaches the host over `host.docker.internal` where the port is
independent of any host-side conflict. Landed in [PR #109][pr-109].

### Docker volume name

- Symbol: `VOLUME`
- Value: `'ouroboros-volume'`
- Source: [`src/main/servers/volume-server.ts`][volume-server-ts] line 9,
  matched by the `volumes.ouroboros-volume.name` entry in
  [`python/compose.yml`][compose-yml],
  [`python/compose.dev.yml`][compose-dev-yml], and the compose file
  written by [`scripts/prepare-production-server.mjs`][prepare-production-server].

**Why it exists.** The Python server, the volume server helper, and the
main process all address the same Docker named volume so that plugin
uploads land where the pipeline can read them.

**Invariant.** The literal string `'ouroboros-volume'` is duplicated across
four call sites: `src/main/servers/volume-server.ts`,
`python/ouroboros/common/volume_server_interface.py`, `python/compose.yml`,
`python/compose.dev.yml`, and the compose file written by
`scripts/prepare-production-server.mjs`. Docker Compose YAML is required to
be static, so a single source of truth would need a code-generation step
the team has not opted into; the invariant lives in this documentation
and in code review discipline. If you rename this value in one place,
every listed file must move with it in the same commit.

**User-visible behavior.** On app shutdown the volume server helper
issues `docker run --rm ... rm -rf /volume/*` against this named volume.
Anything else stored under the same volume name is deleted.

**When to reconsider.** Rename it if you need to run multiple Ouroboros
installs against the same Docker daemon without stomping each other's
volumes. All call sites listed above must agree.

### Main server locations (main process)

- Symbols: `DEVELOPMENT_PATH`, `DEVELOPMENT_CONFIG`, `PRODUCTION_PATH`,
  `PRODUCTION_CONFIG`
- Values (relative to the compiled `out/main/index.js`):
    - `DEVELOPMENT_PATH` = `../../python/`
    - `DEVELOPMENT_CONFIG` = `../../python/compose.dev.yml`
    - `PRODUCTION_PATH` = `../../../extra-resources/server/`
    - `PRODUCTION_CONFIG` = `../../../extra-resources/server/compose.yml`
- Source: [`src/main/servers/main-server.ts`][main-server-ts] lines 4-8

**Why it exists.** These paths tell the Electron main process which
`docker compose` file to bring up when starting the FastAPI server. In
development the compose file lives in the source tree; in production it
is written into `extra-resources/server/compose.yml` by
[`scripts/prepare-production-server.mjs`][prepare-production-server]
during packaging.

**User-visible behavior.** A packaged app that ships without a valid
`extra-resources/server/compose.yml` cannot start the main server, and
the app comes up in a disconnected state.

**When to reconsider.** Change these together with any restructure of
`extra-resources/` or the packaging scripts.

## Renderer

### Server connection settings

- Symbols: `DEFAULT_SERVER_URL`, `retryDelay`
- Values:
    - `DEFAULT_SERVER_URL` = `'http://127.0.0.1:8000'`
    - `retryDelay` = `5000` (milliseconds)
- Source: [`src/renderer/src/contexts/ServerContext.tsx`][server-context]
  lines 4 and 41

**Why it exists.** `DEFAULT_SERVER_URL` is the base URL the renderer
uses for all REST and SSE calls to the Python server. `retryDelay` is
the interval at which the `checkServerStatus` loop pings that base URL
to update the `connected` flag driving the ProgressPanel status.

**User-visible behavior.** The "connected" indicator can lag reality by
up to about five seconds. Any fetch or stream started while the app
thinks the server is down is placed on `fetchQueue` and replayed once
the health poll flips to `connected`.

**When to reconsider.** Lower `retryDelay` for a snappier connection
indicator at the cost of extra HTTP traffic. Change `DEFAULT_SERVER_URL`
if you point Ouroboros at a remote server, but keep it aligned with the
Python-side `HOST`/`PORT` defaults documented below.

### IFrame plugin allowed origins

- Symbol: `allowedHostnames`
- Value: `['localhost', '127.0.0.1', '0.0.0.0']`
- Source: [`src/renderer/src/contexts/IFrameContext.tsx`][iframe-context]
  line 30

**Why it exists.** The renderer only dispatches `read-file`,
`save-file`, and `register-plugin` messages from iframes whose
`event.origin` parses to a URL whose `hostname` is exactly one of these
values. Plugin content is served locally through the plugin file server
(see [`PLUGIN_FILE_SERVER_PORT`](#plugin-file-server-port), default port
3000), so all legitimate origins are loopback. Any port on those hosts
is accepted; the port itself is env-configurable.

**Security note.** The previous implementation used
`text.startsWith(item)`, which would accept attacker-controlled hosts
such as `http://localhost.evil.example`. The current implementation
parses `event.origin` with `new URL(...)` and compares `hostname` for
exact equality; malformed origins fall through to a deny. Landed in
[PR #109][pr-109].

**User-visible behavior.** Any plugin iframe served from a non-loopback
origin is silently ignored: its messages never trigger file reads,
writes, or registration.

**When to reconsider.** Add an entry only if you deliberately host
plugin content off-device (for example, over `https://` to a controlled
host). Broadening this list widens the file-read and file-write surface
exposed to third-party pages, and non-loopback additions should use
exact origin equality against a full URL, not just a hostname.

### Slice visualization sampling ratio

- Symbol: `SLICE_RENDER_PROPORTION`
- Value: `0.008`
- Source: [`src/renderer/src/routes/SlicesPage/SlicesPage.tsx`][slices-page]
  line 26

**Why it exists.** The 3D slice preview draws only every `Nth` rectangle
where `N = floor(rects.length * 0.008)`. This keeps very long paths
(tens of thousands of rects) from stalling the WebGL scene.

**User-visible behavior.** For a 10 000-rect path this renders roughly
125 rectangles: enough to convey the shape of the sweep without freezing
the visualization. Fewer rects than about 125 render every rect.

**When to reconsider.** Raise it to see more slices in the preview at
the cost of frame rate; lower it if the preview lags on very long paths.

### Stream endpoint paths

- Symbols: `SLICE_STREAM`, `BACKPROJECT_STREAM`, `SLICE_STEP_NAME`
- Values:
    - `SLICE_STREAM` = `'/slice_status_stream/'`
    - `BACKPROJECT_STREAM` = `'/backproject_status_stream/'`
    - `SLICE_STEP_NAME` = `'SliceParallelPipelineStep'`
- Sources:
  [`src/renderer/src/routes/SlicesPage/SlicesPage.tsx`][slices-page] lines 28-30
  and [`src/renderer/src/routes/BackprojectPage/BackprojectPage.tsx`][backproject-page]
  line 22

**Why it exists.** These are the FastAPI SSE endpoints the renderer
subscribes to for slice and backproject progress, and the pipeline step
name that the slice page uses to identify its progress row. They must
stay in sync with the routes exposed by `create_api` in the Python
server (see [`python/ouroboros/common/server_api.py`][server-api-py])
and with the pipeline step class in the slice pipeline.

**Canonical source.** The Python side exposes the pipeline step names
as an enum in `python/ouroboros/common/step_names.py`
(`StepName.SLICE_PARALLEL.value` = `'SliceParallelPipelineStep'`) and
serves them through `GET /step-names` on the FastAPI server. The
renderer keeps `SLICE_STEP_NAME` as a local constant with a
`TODO(#107)` marker; a dev-mode runtime drift check that fetches
`/step-names` and warns on mismatch is deferred until the renderer's
`ServerContext` exposes its base URL to non-fetch-hook callers.

**User-visible behavior.** A rename on either side (server or renderer)
without the other silently breaks progress updates and the visualization
panel. New pipeline steps added to `StepName` show up as additional
keys in the `/step-names` response without breaking existing clients.

**When to reconsider.** Update this file whenever the corresponding
FastAPI route names or Python pipeline step class names change. Landed
in [PR #109][pr-109].

## Python server

### Server host and port

- Symbols: `HOST`, `PORT`, `DOCKER_HOST`, `DOCKER_PORT`
- Values (defaults):
    - `HOST` = `'127.0.0.1'`
    - `PORT` = `8000`
    - `DOCKER_HOST` = `'0.0.0.0'`
    - `DOCKER_PORT` = `8000`
- Source: [`python/ouroboros/common/server.py`][py-server] lines 10-14
- Environment overrides:
    - `OUROBOROS_SERVER_HOST` -> `HOST`
    - `OUROBOROS_SERVER_PORT` -> `PORT` (integer)
    - `OUROBOROS_DOCKER_SERVER_HOST` -> `DOCKER_HOST`
    - `OUROBOROS_DOCKER_SERVER_PORT` -> `DOCKER_PORT` (integer)

**Why it exists.** The desktop-only server entry point
(`ouroboros-server`) binds Uvicorn to loopback, while the containerised
server entry point (`ouroboros-docker-server`) binds to all interfaces
so the host can reach the container through the compose `8000:8000`
port mapping.

**User-visible behavior.** The renderer's `DEFAULT_SERVER_URL` above
assumes port 8000. Changing `PORT` (or `OUROBOROS_SERVER_PORT`) or the
compose port mapping without updating the renderer breaks the client.

**When to reconsider.** Set the env overrides if you need to expose the
desktop server to another machine (rare) or if 8000 conflicts with
another service. Update `DEFAULT_SERVER_URL` in the renderer and the
compose port mapping in the same commit. Landed in [PR #109][pr-109].

### Volume server URL (Python side)

- Symbols: `VOLUME_SERVER_URL`, `PLUGIN_NAME`
- Values:
    - `VOLUME_SERVER_URL` = `'http://host.docker.internal:3001'`
    - `PLUGIN_NAME` = `'main'`
- Source: [`python/ouroboros/common/volume_server_interface.py`][volume-server-interface-py]
  lines 3-4

**Why it exists.** From inside the server container, the volume server
running on the host is reached through the Docker special DNS name
`host.docker.internal`, mapped in each compose file via
`extra_hosts: - "host.docker.internal:host-gateway"`. `PLUGIN_NAME` is
the fixed subfolder on the shared volume used by the main pipeline.

**Why 'main'.** Ouroboros is the *main* application of its own plugin
ecosystem, so the Python side treats itself as the always-`main` plugin
when addressing paths on the shared volume. There is no planned
per-plugin path scheme on the Python side; plugin content is scoped by
the Electron-side plugin folder, not by a Python-visible slug.
`PLUGIN_NAME` therefore stays a constant rather than a parameter.

**User-visible behavior.** If the compose file omits the
`host.docker.internal` extra host, or if the Electron main process is
not running the volume server, requests time out and plugin/data
transfer into the volume fails.

**When to reconsider.** Change the URL only in coordination with the
Electron-side `VOLUME_SERVER_PORT` in
[`src/main/servers/volume-server.ts`][volume-server-ts].

### Bounding box defaults

- Symbols: `DEFAULT_SPLIT_THRESHOLD`, `DEFAULT_MAX_DEPTH`,
  `DEFAULT_TARGET_SLICES_PER_BOX`
- Values:
    - `DEFAULT_SPLIT_THRESHOLD` = `0.9`
    - `DEFAULT_MAX_DEPTH` = `10`
    - `DEFAULT_TARGET_SLICES_PER_BOX` = `128`
- Source: [`python/ouroboros/helpers/bounding_boxes.py`][bounding-boxes-py]
  lines 6-8
- User override: `bounding_box_params.max_depth` and
  `bounding_box_params.target_slices_per_box` in the slice options JSON
  (see [`python/ouroboros/helpers/options.py`][options-py] and
  [`python/ouroboros/helpers/bounding_boxes.py`][bounding-boxes-py])

**Why it exists.** The slice pipeline uses binary space partitioning to
group slice rectangles into cloud-volume fetch boxes.
`DEFAULT_MAX_DEPTH` caps recursion depth, `DEFAULT_TARGET_SLICES_PER_BOX`
is the leaf-box size heuristic, and `DEFAULT_SPLIT_THRESHOLD` decides
whether a box is empty enough to be worth dividing (a box is split when
its utilised volume is below `1 - 0.9 = 10%` of its total volume).

**User-visible behavior.** Larger `target_slices_per_box` or smaller
`max_depth` produces fewer, larger cloud-volume fetches; smaller
`target_slices_per_box` produces many smaller fetches. Tuning these
trades network round-trips against downloaded-but-unused voxels.

**When to reconsider.** Adjust `bounding_box_params` in the slice
options JSON per-run rather than editing this file. Only change the
compiled defaults if you have measured a better global balance.

**User surfacing.** `DEFAULT_SPLIT_THRESHOLD` is intentionally *not*
exposed through `BoundingBoxParams`: only `max_depth` and
`target_slices_per_box` are user-tunable, and the split threshold is
documented here for completeness. A future change would need to add a
matching field on `BoundingBoxParams` and thread it through the slice
options schema before it could be reached from the options JSON.

### Backprojection chunk and process defaults

- Symbols: `chunk_size` and `process_count` on `BackprojectOptions`
- Values:
    - `chunk_size` = `160`
    - `process_count` = `os.cpu_count()`
- Source: [`python/ouroboros/helpers/options.py`][options-py] lines 77-78
- User override: fields of the same name in the backproject options JSON

**Why it exists.** `chunk_size` is the per-dimension size of the
processing chunk used by the backproject pipeline (see
`DataRange(FPShape.make_with(0), FPShape, FPShape.make_with(config.chunk_size))`
in [`python/ouroboros/pipeline/backproject_pipeline.py`][backproject-pipeline]).
`process_count` sets the parallel worker pool for the same pipeline;
executor and writer processes are derived as fractions of it
(`process_count // 4 * 2` or `// 4 * 3`).

**Why 160 specifically.** The primary purpose of the chunk sizing is to
enable clean backprojection - specifically clean backprojection of the
flattened xyz stack. `160` is the judgment-call result for the best
combination of memory and speed on standard pieces of data. Users
tuning this for very different data should measure peak memory and
iteration time on their own volumes rather than picking a "round"
alternative.

**User-visible behavior.** Larger `chunk_size` reduces per-chunk
overhead but raises peak memory per worker. On very small hosts,
`process_count = cpu_count()` may over-subscribe and thrash; lowering it
in the options JSON trades throughput for stability.

**When to reconsider.** Tune per-run through the backproject options
JSON. Edit the defaults only if a different global tuning is measured
to be better across typical workloads.

### Volume-cache flush default

- Symbol: `FLUSH_CACHE`
- Value: `False`
- Source: [`python/ouroboros/helpers/volume_cache.py`][volume-cache-py]
  line 13
- User override: `flush_cache` field on `CommonOptions` in
  [`python/ouroboros/helpers/options.py`][options-py]

**Why it exists.** Controls whether the cloud-volume cache is flushed
after each pipeline run. Keeping it off keeps downloaded chunks around
for the next run.

**User-visible behavior.** With the default off, repeated runs against
the same source volume reuse locally cached tiles, which can hide
upstream data changes but greatly speeds re-runs.

**When to reconsider.** Set `flush_cache: true` in the options JSON for
one-shot runs or when the upstream volume may have changed.

## Docker and packaging

### Server container shared-memory size

- Symbol: `shm_size`
- Value: `64gb` (default)
- Source: [`python/compose.yml`][compose-yml] line 15,
  [`python/compose.dev.yml`][compose-dev-yml] line 28, and the compose
  file written by
  [`scripts/prepare-production-server.mjs`][prepare-production-server]
  (line 58)
- Environment override (packaging time):
  `OUROBOROS_SERVER_SHM_SIZE` (read by
  `scripts/prepare-production-server.mjs` and by the
  `${OUROBOROS_SERVER_SHM_SIZE:-64gb}` substitution in both compose
  files)

**Why it exists.** `shm_size` is an artificial Docker limit. Without a
`shm_size` override, Docker aggressively caps container `/dev/shm` at
`64 MB`, which the pipeline exceeds immediately. Setting it too large
does not "cause" OOM: it just shifts OOM from the container-side
artificial limit to actual host OOM. The FastAPI server uses shared
memory for inter-process numpy arrays (`SharedMemoryManager`,
`SharedNPArray` in [`python/ouroboros/helpers/mem.py`][mem-py]) and for
cloud-volume caches, so the container is given 64 GB of tmpfs headroom
by default.

**User-visible behavior.** On hosts with less than 64 GB of RAM plus
swap, Docker still accepts the setting (`/dev/shm` is a tmpfs and
allocates on demand), but attempts to actually use that much shared
memory will trigger the host OOM killer. On high-RAM hosts you can
raise the ceiling; on conservative shared hosts you can lower it to
prefer the artificial limit hitting first.

**When to reconsider.** Set `OUROBOROS_SERVER_SHM_SIZE` at packaging or
runtime (compose reads the same variable via `${OUROBOROS_SERVER_SHM_SIZE:-64gb}`
substitution in both `python/compose.yml` and `python/compose.dev.yml`).
Lower it on constrained hosts where you know the pipelines you run stay
well under the limit; raise it only if you routinely see shared-memory
`ENOMEM` errors from the server. Landed in [PR #109][pr-109].

### Python base image

- Symbol: base image tag
- Value: `thehale/python-poetry:2.1.3-py3.11-slim`
- Source: [`python/Dockerfile`][py-dockerfile] line 1 and
  [`python/Dockerfile-prod`][py-dockerfile-prod] line 5

**Why it exists.** Pins Poetry to 2.1.3 and Python to 3.11 for the
container build. `python/pyproject.toml` declares a matching
`python = ">=3.11,<3.13"` range.

**User-visible behavior.** Container builds are reproducible against a
known Poetry and Python version. Bumping the tag without updating
`pyproject.toml`'s Python range risks a resolver mismatch at build time.

**Policy.** The pin is intentional and driven by transitive-dependency
stability. Do **not** bump reactively (for example, in response to a
new Poetry release or a routine security scan). Only bump when a
concrete downstream requirement forces the move, and land the
`pyproject.toml` update in the same commit.

**When to reconsider.** Update this tag when you deliberately move to a
newer Poetry or Python release, and keep `pyproject.toml` in sync.

### Server image reference defaults

- Symbols: `imageRepository`, `imageTag`
- Values:
    - `imageRepository` (default) = `ghcr.io/chenglabresearch/ouroboros-server`
    - `imageTag` (default) = `v${packageJson.version}` (falls back to
      `GITHUB_REF_NAME` in CI)
- Source: [`scripts/prepare-production-server.mjs`][prepare-production-server]
  lines 10-14
- Environment overrides: `OUROBOROS_SERVER_IMAGE_REPOSITORY`,
  `OUROBOROS_SERVER_IMAGE_TAG`, `OUROBOROS_SERVER_IMAGE_DIGEST`,
  `OUROBOROS_SERVER_IMAGE` (fully qualified reference)

**Why it exists.** The prepare script writes an
`extra-resources/server/compose.yml` that pulls the server image by
name so packaged app installs do not need to build the container
locally.

**User-visible behavior.** With no environment overrides, packaging
pulls `ghcr.io/chenglabresearch/ouroboros-server:v<appVersion>`. A
release that has not yet published a matching image tag will fail at
first startup with a `docker pull` error.

**When to reconsider.** Set `OUROBOROS_SERVER_IMAGE_DIGEST` to pin a
specific build for a release, or point `OUROBOROS_SERVER_IMAGE` at a
private registry for internal builds.

### Preinstalled plugin pins

- Symbol: `productionPluginPins`
- Value:
    - `neuroglancer.tag` = `v1.0.1`
    - `neuroglancer.artifact` = `neuroglancer-plugin-v1.0.1.zip`
    - `autoseg.tag` = `v0.4.0-beta.1`
    - `autoseg.cpuArtifact` = `auto-segmentation-v0.4.0-beta.1-cpu.zip`
    - `autoseg.cudaArtifact` = `auto-segmentation-v0.4.0-beta.1-cuda.zip`
- Source: [`scripts/prepare-package-flavor.mjs`][prepare-package-flavor]
  lines 7-16
- Environment overrides: `OUROBOROS_NEUROGLANCER_PLUGIN_TAG`,
  `OUROBOROS_NEUROGLANCER_PLUGIN_ARTIFACT`,
  `OUROBOROS_AUTOSEG_PLUGIN_TAG`,
  `OUROBOROS_AUTOSEG_CPU_PLUGIN_ARTIFACT`,
  `OUROBOROS_AUTOSEG_CUDA_PLUGIN_ARTIFACT`

**Why it exists.** These are the exact plugin release versions that the
`with-plugins-cpu` and `with-plugins-cuda` package flavors ship. The
`core` flavor ships no preinstalled plugins.

**User-visible behavior.** A packaged app pins the listed plugin
versions in `extra-resources/preinstalled-plugins/`. First launch copies
each into the user's plugin folder if the installed version is older or
missing.

**Bump workflow.** `productionPluginPins` and the fallback
`neuroglancerArtifactForTag` / `autosegArtifactForTag` builders in
[`scripts/prepare-package-flavor.mjs`][prepare-package-flavor] move
together on every plugin release. The single-file edit lives in
`scripts/prepare-package-flavor.mjs`: update the tags in
`productionPluginPins`, and the fallback builders pick up the new
filename shape automatically. The environment overrides above take
precedence over both.

**When to reconsider.** Update these pins whenever a new upstream plugin
release becomes the intended default. Bump both the tag and the artifact
filename together, since the artifact filename embeds the tag.

### Supported package flavors

- Symbol: `supportedFlavors`
- Value: `new Set(['core', 'with-plugins-cpu', 'with-plugins-cuda'])`
- Source: [`scripts/prepare-package-flavor.mjs`][prepare-package-flavor]
  line 6
- Environment selector: `OUROBOROS_PACKAGE_FLAVOR` (defaults to
  `'core'`)

**Why it exists.** Whitelists the packaging-time flavor names. Anything
else raises `Unsupported OUROBOROS_PACKAGE_FLAVOR ...` and aborts.

**User-visible behavior.** Only these three flavors have artifact
renaming, preinstalled-plugin selection, and a release-time build
pipeline. See
[Production Package Flavors](../development/production-package-flavors.md)
for the operator-facing view of these three flavors.

**When to reconsider.** Add a new flavor here together with matching
artifact handling in the same file and a documentation update.

## Where the values come from

Every value above was read directly from the tree at the time this page
was written. The five file explorer constants live together as a block at
the top of [`src/main/event-handlers/filesystem.ts`][filesystem-ts]
(lines 8-12). If you edit any of them, update this page in the same
commit so operators and plugin authors can rely on it.

[issue-51]: https://github.com/ChengLabResearch/ouroboros/issues/51
[issue-102]: https://github.com/ChengLabResearch/ouroboros/issues/102
[pr-104]: https://github.com/ChengLabResearch/ouroboros/pull/104
[pr-105]: https://github.com/ChengLabResearch/ouroboros/pull/105
[pr-106]: https://github.com/ChengLabResearch/ouroboros/pull/106
[pr-109]: https://github.com/ChengLabResearch/ouroboros/pull/109
[filesystem-ts]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/main/event-handlers/filesystem.ts
[iframe-context]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/renderer/src/contexts/IFrameContext.tsx
[file-server-ts]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/main/servers/file-server.ts
[file-server-script]: https://github.com/ChengLabResearch/ouroboros/blob/main/resources/processes/file-server-script.mjs
[volume-server-ts]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/main/servers/volume-server.ts
[volume-server-script]: https://github.com/ChengLabResearch/ouroboros/blob/main/resources/processes/volume-server-script.mjs
[main-server-ts]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/main/servers/main-server.ts
[server-context]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/renderer/src/contexts/ServerContext.tsx
[slices-page]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/renderer/src/routes/SlicesPage/SlicesPage.tsx
[backproject-page]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/renderer/src/routes/BackprojectPage/BackprojectPage.tsx
[py-server]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/common/server.py
[server-api-py]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/common/server_api.py
[volume-server-interface-py]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/common/volume_server_interface.py
[bounding-boxes-py]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/helpers/bounding_boxes.py
[options-py]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/helpers/options.py
[backproject-pipeline]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/pipeline/backproject_pipeline.py
[mem-py]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/helpers/mem.py
[volume-cache-py]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/helpers/volume_cache.py
[compose-yml]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/compose.yml
[compose-dev-yml]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/compose.dev.yml
[py-dockerfile]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/Dockerfile
[py-dockerfile-prod]: https://github.com/ChengLabResearch/ouroboros/blob/main/python/Dockerfile-prod
[prepare-production-server]: https://github.com/ChengLabResearch/ouroboros/blob/main/scripts/prepare-production-server.mjs
[prepare-package-flavor]: https://github.com/ChengLabResearch/ouroboros/blob/main/scripts/prepare-package-flavor.mjs

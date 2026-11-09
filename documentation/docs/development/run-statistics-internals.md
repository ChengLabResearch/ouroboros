# Run statistics internals

Implementation contracts behind the [pipeline run statistics](../reference/run-statistics.md). Report fields and metric definitions are documented there.

## Desktop status handling

The HTTP/SSE contract is unchanged: `/status/{task_id}` can return HTTP 500 with
`status: done`, and a stream can emit `error_event` for successful science with a
reporting problem. The desktop interprets the validated status and `statistics`
descriptor, rather than treating the event name or aggregate `error` as the
scientific outcome. For failed/interrupted science the aggregate error retains
`Task.error` priority; publication and log-presentation errors are displayed
separately from that descriptor. No report files are read to reconstruct status.

Publication pending keeps the server status active and retains the existing
submission/deletion guards. Terminal status closes its stream; events for other
task IDs and events from closed/replaced streams are ignored. Connection failures
or invalid status payloads close the stream and show a communication warning with
the last reported progress, without declaring a scientific failure or resubmitting
the task. Status messages remain in the progress panel until the stream is cleared
or a new task starts, so a recovery location does not disappear with a timed alert.

The existing unknown-task response (`status: error`, `error: Item ID Not Found`,
and no `statistics` descriptor) is also treated as an outcome-unknown warning.
The stream closes and retains the last reported progress: the backend may have
restarted or dropped the task, which does not establish scientific failure.
This exact legacy response is recognized at the stream boundary; other scientific
errors retain their normal presentation and the API contract is unchanged.

The short desktop regression tests run through `npm run test:release`, using the
Node test runner and TypeScript compiler. For
individual case output, run `node scripts/tests/task-status.test.mjs`. They feed
HTTP-shaped status bodies and simulated SSE events through the production parser,
stream handler, and React progress panel; they do not exercise a full Electron GUI
or perform slicing/backprojection workloads.

## Ownership

`Task` owns operation identity, outcome, and the selected `LoggingMode`;
pipelines and steps receive that same enum object. Report and step-statistics
gates derive from its `reporting_enabled` property; RAM collection derives from
`memory_enabled`. Legacy CLI/API/configuration representations are input
adapters, not additional runtime settings. The report's `logging_mode` is a
one-way serialization of the task selection. `RunStatistics` owns operation timing
and report publication. It references the existing options and pipeline owners.
`Pipeline` and `PipelineStep` own all pipeline/worker timing samples and progress.
Summary dictionaries are one-way output adapters: they never replace or mutate
the owner's samples. Server progress is read from that owner without a second
cached progress state. Actual executor counts also come from dispatch-owned
resource fields rather than a separate calculation in the reporter.

`VolumeCache`, `CloudVolumeInterface`, `BoundingBox`, and the native TIFF inspector
own geometry and effective volume metadata. Downloads and reporting share
`download_volume_geometry`. The volume reporting snapshot is a one-way
serialization boundary before cleanup, never an input to scientific execution.
Output naming, atomic file publication, Docker export, and recovery extend the
existing file/volume facilities. The memory collector extends this same lifecycle.

## Memory collector

A small separate interpreter performs collection, avoiding a sampler thread in
the parent before Linux forks workers. The manager's existing `_segments`
registry remains the sole owner of live allocation handles and release policy.
At allocation, registration and release boundaries, it publishes a derived
snapshot of names and sizes, plus its own mapping addresses on macOS. Overlapping
runs share one observation session. Manager startup receives that session through
its existing initializer seam; workers and array attachments need no observation
context, mapping registry or TIFF wrappers.

Snapshots and sampler-control files are one-way observation/IPC adapters. They
never supply allocation or lifecycle state back to the manager. POSIX residency
queries temporarily open a read-only mapping without faulting in the buffer or
acquiring unlink authority. A manager started before an observation session is
not retroactively observed. The scientific pipelines start their managers inside
the collection window. Off and Standard create no sampler or observation
workspace.

The narrow identity adapter in `common/memory_ipc.py` serializes the validated
private `Process._ident` value together with the psutil version and platform. It
matches incoming identities against live `Process` objects at the IPC boundary;
serialized values do not become a second runtime identity registry. Missing,
unsupported, or unverified identity data makes the affected measurement
unavailable rather than falling back to a PID or public creation timestamp, or
treating an unverified manager as owning no segments.

Psutil is pinned to **7.1.3** in package metadata as well as the Poetry lock,
covering pip/wheel installations and both Docker build paths. This private-API
dependency must be reviewed before changing that pin: Linux, Windows and macOS
identity/IPC compatibility tests and the clock-adjustment regressions must pass
for the proposed version. Supported runtime identity comes from psutil itself;
Ouroboros does not maintain an alternative platform identity calculation.
Liveness checks likewise validate a fresh psutil identity before comparing it
with the retained `Process`: unreadable identity is unknown coverage, not proof
that the worker exited. Discovery still uses psutil's `children()` implementation;
in 7.1.3 a backwards clock adjustment within one enumeration can omit a previously
unseen child because psutil itself filters by public creation timestamps. Retained
workers are checked independently, but sampling is not a complete process census.

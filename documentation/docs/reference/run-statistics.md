# Pipeline run statistics

Use `--verbose` with either Python CLI operation to print and save the same
formatted statistics JSON:

```sh
ouroboros-cli slice slice-options.json --verbose
ouroboros-cli backproject backproject-options.json --verbose
```

The report is written beside the configured outputs, using
`<output_file_name>-<operation>-<run_id>-statistics.json`. Each invocation gets a
new run identifier, so repeated runs retain their reports. This works with a
single TIFF or a directory of TIFF planes. JSON numeric values retain their
precision; byte counts are integers. Console headings and progress bars are
outside the JSON payload.

Without `--verbose`, the CLI skips the pipeline's accumulation of detailed timing
samples, memory sampler, volume statistics, report formatting and report writes. Existing worker
phase clocks still run; basic step duration and progress remain available. The
pipeline factory's `verbose` argument continues to control its progress bar; it
is separate from run reporting.

## Server and Docker

Server tasks report statistics by default. Pass `reporting_enabled=false` to
`POST /slice/` or `POST /backproject/` to disable detailed reporting for that
task. Both CLI and server use the same operation lifecycle and reporting code.

The existing status response and status events include `statistics`:

```json
{
    "status": "published",
    "path": "/configured/output/sample-slice-run-id-statistics.json",
    "recovery_path": null,
    "error": null
}
```

Publication status is `pending`, `published`, `disabled`, `unavailable`,
`failed`, or `interrupted`. The API keeps a task active through required cleanup
and publication; duplicate submission and deletion remain blocked while it is
active. Cleanup remains part of that lifecycle when reporting is disabled. A
progress read failure is exposed as `progress_error` without changing the
scientific outcome.

Docker copies scientific outputs to the host and cleans its working folder
before finalizing the operation timer. It then stages and transfers the report
through the existing volume server into the configured host output directory.
Reports retain private file permissions (`0600` on POSIX). For Docker exports
on POSIX hosts, the volume server assigns the copied report to its host user and
group so that user can read it. Windows retains Docker Desktop's host mapping.
Undelivered reports are retained under `/volume/main-statistics/`, outside the
scientific working folder. The status response gives the recovery path.
Successful delivery removes the staging copy. Normal operation and desktop
shutdown cleanup preserve undelivered statistics; explicitly deleting the Docker
volume also deletes those recovery copies.

Scientific outcome and report-publication outcome are distinct. A failed
scientific-output transfer makes the operation fail. A failed statistics write
or transfer leaves a successful scientific outcome intact but exposes the
publication error through status, an error event, and server logging. A console
or log presentation failure is exposed separately as `presentation_error`; an
example is `BrokenPipeError` when a consumer such as `head` closes stdout while
`--verbose` prints the saved report. `presentation_interrupted` distinguishes an
interruption during this final output. Neither changes the scientific outcome
or an already-published file's delivery status and path. The CLI exits with
status 1 on failure and 130 on interruption, including interrupted report
delivery or presentation. When options cannot be loaded and no output
destination is known, the report is
explicitly unavailable; no report is silently written to the current directory.

## Desktop status presentation

The slicing and backprojection progress panels show scientific outcome and
statistics publication separately. A completed task retains its final progress
when report delivery fails, is interrupted, or is unavailable. The panel shows a
reporting warning with the publication error and any recovery location; an
unpublished `path` is labeled as the intended destination. Only `published`
identifies a delivered report. Disabled reporting produces no missing-report
warning. A scientific failure or interruption appears first, even when its
statistics report was delivered successfully.

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
existing Node test runner and TypeScript compiler without new dependencies. For
individual case output, run `node scripts/tests/task-status.test.mjs`. They feed
HTTP-shaped status bodies and simulated SSE events through the production parser,
stream handler, and React progress panel; they do not exercise a full Electron GUI
or perform slicing/backprojection workloads.

## Schema version 1

The record contains these groups:

| Field | Meaning |
| --- | --- |
| `schema_version`, `run_id`, `operation` | Format version, invocation identity, and `slice` or `backproject` |
| `started_at`, `finished_at` | ISO timestamps with timezone |
| `outcome`, `error` | `succeeded`, `failed`, or `interrupted`, and the operation error |
| `operation_elapsed_seconds` | Whole operation wall time with the boundary below |
| `pipeline_elapsed_seconds`, `steps` | Measurements owned by the pipeline and its steps |
| `phase_wall_spans` | `null`: worker start/end intervals are not collected |
| `volumes` | Native source, geometry, input/output headers, and storage comparisons |
| `inputs`, `effective_configuration` | Input identifiers and the effective options, including container path remapping |
| `software`, `environment`, `execution` | Package/build/revision, OS/Python, observed thread settings, and actual dispatched worker/thread counts |
| `cache` | Observed cache configuration; unknown cache warmth/network conditions remain explicit |
| `collection_errors` | Instrumentation failures that prevented part of the record from being collected |
| `memory` | Platform-specific sampled process-tree memory, coverage and collection cost |

Source checkouts record their Git revision and whether the working tree is dirty.
Packaged deployments can provide `OUROBOROS_BUILD_REVISION` and
`OUROBOROS_BUILD_ID`; missing revision/build metadata remains unknown. The
execution OS describes the Python backend. A Docker host's OS is not inferred
from the Linux container. Thread environment variables are included only when
set; actual executor counts are recorded separately. Cache warmth covers local,
OS, and remote caches and remains unknown without observation.

The schema defines timing boundaries, units, axis conventions, and unavailable
values here rather than repeating explanatory text in every record. The
effective slicing configuration is included separately for backprojection; a
slicing run already has it in `effective_configuration.operation`. Under Docker,
`volumes.*.path` and paths in `effective_configuration` describe effective
container paths; scratch files may already have been removed by cleanup.
`inputs` retains original host identifiers before Docker remapping, while
`statistics.path` in the status response identifies the published host report.
Annotation and layer names remain in the effective configuration, and the
resolved dataset URL is `volumes.source.url`.

## Timing definitions

All elapsed measurements use `time.perf_counter()`.

| Measurement | Start and end |
| --- | --- |
| Operation | Execution entry before options loading/input preparation through scientific-output finalization, host transfer where applicable, and required cleanup |
| Pipeline | Entry to `Pipeline.process()` through completion, returned failure, or exception |
| Step | Step processing and progress updates; finalized on failure and catchable interruption |
| Worker phase | The existing instrumented phase within one worker; accumulated across its invocations |

Operation time excludes queue wait, application/container startup, and statistics
serialization/publication. The performance-overhead benchmark separately times
`Pipeline.process()` with detailed instrumentation enabled and disabled. The
real-data comparison uses the existing data-step duration to exclude earlier
preparation; both boundaries remain in the raw benchmark record. Metadata
collection, formatting, printing, and report delivery are outside the pipeline
interval.
Volume metadata is captured before cleanup while output headers are still
available, so that work is included in operation time.

Each step retains its own status. Steps that were never reached are `pending`,
not successful zero-duration measurements. Successful and failed steps have
final durations. Catchable interruptions, including `KeyboardInterrupt`, finalize
records; hard process termination cannot guarantee final publication.

`steps[].custom_times` reports mean, standard deviation, minimum, maximum,
sample count (`loops`), spread (`gap`), and `accumulated_elapsed_seconds`. Each
entry identifies its unit, worker/coordinator scope, and included subphases for
nested totals. Downloads overlap slice processing; backprojection overlaps
writing. These sums are neither CPU time nor a partition that adds to operation
wall time. They can exceed it. Phase wall spans are explicitly unavailable
because the workers currently return durations, not start/end intervals.

Backprojection intermediate payload bytes are in `steps[].byte_counts`, never
in timing summaries. System-available-memory samples have been removed from
timing statistics.

## Volume and storage definitions

`volumes.source` distinguishes full-resolution metadata (mip 0, when available)
from the mip used by the operation. Source axes are XYZ; dimensions exclude the
separate channel count. Logical bytes equal the product of dimensions, channels,
and dtype item size.

`volumes.straightened` describes the actual slicing output or backprojection
input stack. Its axes are DVU: depth/slice count, section height, section width.
Backprojection input image/segmentation classification is unknown unless there
is an explicit declaration; dtype alone is not used to guess it.

`volumes.bounding_box` uses the native enclosing-box facility and records the
integer-rounded envelope in source voxel indices at the reported mip.
`BoundingBox.approx_bounds` floors minima and ceils maxima;
`BoundingBox.get_shape()` includes both rounded endpoints.

Slicing additionally reports a compact `download_partitions` summary: count,
total allocated logical bytes, total source-clipped logical bytes, and the mip
whose source bounds were used for clipping. The native download API casts
requested coordinates to integers and uses half-open maxima. Allocation matches
that requested box size; clipping intersects the request with source bounds.
Totals count every partition request, including overlapping voxels repeatedly.
They are neither the enclosing volume size nor peak RAM. Backprojection does
not download these boxes and omits this summary. Reporting does not alter
scientific rounding, clipping, or geometry.

`volumes.backprojected_output` uses actual output TIFF headers, including
channels, dtype, and emitted plane count. For a directory with numeric filenames,
the report also records its minimum and maximum Z index. Dimensions count
emitted planes, so gaps are visible by comparison with the index span. Logical
bytes count emitted planes, not hypothetical filled gaps. Header spacing is in
micrometres, with explicit axis order and `null` for missing axes. It is recorded
separately from the requested mip, so metadata inconsistencies are visible.
The backprojected output's final nonzero extent is `null`: the pipeline does not
retain it, and reporting does not read pixels to calculate it.

Header inspection reuses the existing native stack inspector: first-plane
shape/dtype/channels and page/file count. Physical file byte counts are separate
from logical uncompressed bytes. No extra source download or full-volume read
is performed. For curved/adaptively sampled stacks, stored TIFF spacing is not
presented as proof of a uniform Cartesian lattice.

Storage reduction is `100 * (1 - straightened_bytes / reference_bytes)`.
The record identifies numerator/denominator bytes and the reference field; the
referenced volume owns its mip and spacing. Full-source,
used-mip-source, and bounding-box-relative comparisons are separately named;
they measure logical storage, not compression or equal information content.
GB means decimal gigabytes and GiB means binary gibibytes; the structured
measurements use bytes directly.

## Verification and ownership

`Task` owns operation identity and outcome; `RunStatistics` owns operation timing
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

The fixed-local-source benchmark and reproducible commands are documented in
[the Python benchmark guide](https://github.com/ChengLabResearch/ouroboros/blob/main/python/benchmarks/README.md).
The owner-approved 1% combined in-operation engineering target applies to longer
runs, assessed alongside inter-run variability and fixed startup/shutdown costs.
For the recorded #139 real-data evidence, acceptance is scoped to the long
Salivary Duct path; short-path costs remain reported and are not a general 1%
guarantee. RAM collection shares the budget with timing and volume reporting.
The real-data comparison uses the canonical data-step duration, retaining worker/
manager startup and shutdown inside that step. Earlier preparation, sampler
startup/shutdown and final reporting are excluded. These measurements do not
isolate every cause of a timing difference; negative changes reflect variation,
not free instrumentation. Two pairs do not establish a statistical upper bound.

## Memory measurements

The collector belongs to `RunStatistics` and runs only around `Pipeline.process`.
It stops before volume inspection, output transfer, cleanup and report delivery.
Each operation starts a fresh sampling window and uses the same run identifier,
status and report delivery as its timing statistics. No container-lifetime peak
or host free-memory delta is attributed to a run.

| Backend execution OS | Metric | Source and coverage |
| --- | --- | --- |
| Linux, including Docker on any host | `process_tree_nonshmem_pss_plus_managed_resident_bytes` | Sum of `/proc/<pid>/smaps_rollup` `Pss - Pss_Shmem`, plus `mincore` residency of each live managed shared segment once |
| Native Windows (64-bit) | `process_tree_private_resident_plus_managed_allocated_bytes` | Sum of `PROCESS_MEMORY_COUNTERS_EX2.PrivateWorkingSetSize`, plus the size of each live managed allocation once |
| Native macOS | `process_tree_private_plus_managed_resident_bytes` | `libproc` private resident pages, excluding the manager's owned segment ranges, plus `mincore` residency of each live managed segment once |

Linux removes the process counter's entire shared-memory PSS contribution before
adding managed residency. This avoids counting a managed page again when it has
only one reader. Reading the backing object's residency also includes resident
managed buffers queued after their producer closes, before another process has
accessed them. Shared-memory segments outside the manager's ownership are not
added back. Other shared memory, such as fork-shared anonymous pages and mapped
files, retains the OS's proportional accounting.

Windows uses one private working-set counter per process instead of enumerating
its working-set pages. Its managed term is **allocated bytes, not resident
bytes**: unused, paged-out or otherwise nonresident parts of live allocations
remain included. The metric name makes this distinction explicit. The counter
requires 64-bit Python and Windows 10 or 11 version 22H2 with the September 2023
update or later. Missing counter support makes the metric unavailable; it does
not silently fall back to the old working-set scan.

macOS retains a `libproc` region walk for private residency. A segment can appear
private when the manager is its only mapper, so the manager's ranges are excluded
before managed residency is added. Compressed/swapped pages are not included.
Native CI exercises the macOS reader and short collector lifecycle. Native
macOS slice/backprojection performance and macOS-host Docker delivery are deferred
on #139; focused CI does not replace those workflow checks.

There is no separate TIFF or resident file-cache accounting. Ordinary mapped-file
pages may still contribute to the native process counter; cached pages absent
from process mappings are not added separately. The platform metrics differ in
how they treat shared pages and, on Windows, managed allocation residency. They
must not be treated as interchangeable measurements of exactly the same RAM.

The scope includes the backend parent, discovered workers and helpers, including
the shared-memory manager. It excludes the sampler itself. Parent state can
predate the operation. Concurrent operations in the same server share the parent
and descendants, so overlapping records describe overlapping process trees,
**not isolated per-job allocations**. A descendant already observed remains in
scope if it is subsequently reparented. Children born and gone between samples
can be missed. Psutil `Process` equality and hashing are the sole runtime process
identity, including PID-reuse protection; public `create_time()` values are not
identity keys. On Linux and macOS those wall-clock timestamps can change after a clock
adjustment even while the same process is running. The host OS remains unknown
from inside Docker; the selected metric follows the execution OS in
`environment.os`.

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
the collection window. Disabled reporting creates no sampler or observation
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

`status` is `sampled` when all admitted sweeps succeeded, `partial` when some
failed, `unavailable` when none completed, or `not_started` if the pipeline was
never reached. Process or allocation changes during a sweep make that sweep
incomplete. Manifest reads and publication reuse the control-file retry for
Windows sharing violations (up to 0.5 seconds, 10 ms spacing). Persistent
failures remain explicit. A failed manager-manifest publication invalidates the
metric rather than silently omitting managed allocations.

The peak is the maximum **complete sampled tree total**, not an OS high-water
mark or a sum of independently observed process peaks. Reads within a sweep are
sequential, not an atomic tree snapshot. Sampling may miss short-lived peaks;
external processes sharing pages affect Linux PSS. Sweeps crossing the end of the
operation are discarded. Requested and observed intervals, sweep cost, complete
and incomplete counts and errors accompany the result. No complete sample means
an unavailable value, never a successful zero. The default interval is one second;
collection cost is assessed against combined reporting overhead.

`last_complete_sample_bytes` is the last admitted complete sample, not a live
reading at report publication. `sampling` gives the window length, first and last
admitted sweep offsets, requested interval, observed start-to-start interval
range, and discarded/incomplete/complete counts. `collector_cpu_seconds` includes
polling and completion of an in-flight sweep discarded at stop;
`max_sweep_seconds` includes discarded sweeps too. These costs diagnose the
observer; only an external pipeline timer measures in-operation overhead.

The native CI matrix covers Linux, Windows and Apple Silicon macOS on Python
3.11 and 3.12. The current simplification is checked with focused ownership,
accounting and reporting tests; it has not repeated the long real-data benchmark
campaign. Historical Linux and Windows pipeline results, component diagnostics,
and Docker/frozen-executable checks are retained in
[the Python benchmark guide](https://github.com/ChengLabResearch/ouroboros/blob/main/python/benchmarks/README.md).
Those results describe earlier collectors. In particular, the earlier Windows
long-path overhead finding applies to the removed working-set enumeration and
mapping-tracking implementation, not a measurement of this replacement.

Focused tests do not establish a 1% overhead bound or native macOS pipeline
acceptance. No cross-platform 1% guarantee is claimed.

The clock-dependent identity defect in
[#162](https://github.com/ChengLabResearch/ouroboros/issues/162) invalidates the
affected RAM results; fixing identity does not retroactively repair those
reports. Fresh long-run overhead acceptance is tracked separately in
[#163](https://github.com/ChengLabResearch/ouroboros/issues/163), and replacement
publication measurements remain under
[#121](https://github.com/ChengLabResearch/ouroboros/issues/121). The native macOS
pipeline and macOS-host Docker deferrals on #139 remain unchanged.

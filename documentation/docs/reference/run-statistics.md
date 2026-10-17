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
`Pipeline.process()` with detailed instrumentation enabled and disabled; metadata
collection, formatting, printing, and report delivery are outside that interval.
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
The target is at most 1% median pipeline elapsed-time overhead for instrumentation
that runs during the data operation. Setup and final reporting costs are assessed
separately. RAM collection shares that same in-operation budget with timing and
volume reporting. Report measured variability alongside overhead; a negative observed
overhead reflects noise, not free instrumentation.

## Memory measurements

The collector belongs to `RunStatistics` and runs only around `Pipeline.process`.
It stops before volume inspection, output transfer, cleanup and report delivery.
Each operation starts a fresh sampling window and uses the same run identifier,
status and report delivery as its timing statistics. No container-lifetime peak
or host free-memory delta is attributed to a run.

| Backend execution OS | Metric | Source and coverage |
| --- | --- | --- |
| Linux, including Docker on any host | `process_tree_pss_bytes` | Sum of current PSS from psutil's `/proc` memory reader; shared resident pages contribute each process's proportional share |
| Native Windows (64-bit) | `process_tree_private_plus_tracked_shared_resident_bytes` | One `QueryWorkingSet` snapshot per process; unique resident pages outside tracked mappings plus shared backing pages deduplicated by allocation and offset inside them; private copy-on-write pages remain private |
| Native macOS | `process_tree_private_plus_tracked_backing_resident_bytes` | `libproc` private resident pages outside tracked mappings, plus `mincore` residency of tracked shared backing objects, counted once |

Windows and macOS observe the existing `SharedNPArray`, manager-held shared-memory
handles, and TIFF mapping owners.
They exclude **untracked shared mappings**. Neither uses the capped Windows share
count to apportion a known shared buffer. macOS measures backing-object residency:
file-cache pages can be resident without currently appearing in a tracked
process's working set. Compressed/swapped pages are excluded. Unsupported mapping
modes, access failures and unavailable APIs make coverage incomplete; they never
silently substitute RSS, commit charge or zero. These three metrics have different
semantics and must not be compared as interchangeable RAM measurements.

**Open accounting limitation (#139):** Linux PSS and Windows working-set scans
can omit resident shared backing after the producer closes if no remaining
process has faulted those pages into its page tables. An untouched manager
mapping preserves the allocation but does not fix this accounting gap. Queued
downloaded chunks can occupy this state. The draft collector therefore has not
yet met shared-memory coverage acceptance; a successful sample is not proof of
complete backing residency. A revised metric needs explicit semantics and
validation before adoption.

The scope includes the backend parent, discovered workers and helpers, including
the shared-memory manager. It excludes the sampler itself. Parent state can
predate the operation. Concurrent operations in the same server share the parent
and descendants, so overlapping records describe overlapping process trees,
**not isolated per-job allocations**. A descendant already observed remains in
scope if it is subsequently reparented. Children born and gone between samples
can be missed; process identities include creation times to guard against PID
reuse. The host OS remains unknown from inside Docker; the selected metric follows
the execution OS in `environment.os`.

A small separate interpreter performs collection, avoiding a sampler thread in
the parent before Linux forks workers. Windows/macOS worker initializers share
observation context through the existing executor setup. Workers already running
before any observation session cannot retroactively publish their mappings;
their untracked shared pages remain outside native Windows/macOS coverage.
Weak references retain no scientific allocations; inactive native sessions retain
backing identities but do not write manifests.
Ephemeral manifests and sampler-control files are one-way observation/IPC
adapters, never alternative owners of allocations, task status or run identity.
The final summary enters the existing report and the temporary workspace is
removed. Disabled reporting creates no sampler or sampling workspace.

`status` is `sampled` when all admitted sweeps succeeded, `partial` when some
failed, `unavailable` when none completed, or `not_started` if the pipeline was
never reached. Manifest reads and publication reuse the control-file retry for Windows sharing
violations (up to 0.5 seconds, 10 ms spacing). Persistent failures remain explicit.
A failed mapping-manifest publication invalidates the native
metric because the sampler cannot reconstruct its missing coverage.

The peak is the maximum **complete sampled tree total**, not an OS high-water
mark or a sum of independently observed process peaks. Reads within a sweep are
sequential, not an atomic tree snapshot. Sampling may miss short-lived peaks;
external processes sharing pages affect Linux PSS. Sweeps crossing the end of the
operation are discarded. Requested and observed intervals, sweep cost, complete
and incomplete counts and errors accompany the result. No complete sample means
an unavailable value, never a successful zero. The default interval is one second;
collection cost is assessed against combined reporting overhead. The real-data
benchmark compares the existing slicing/backprojection step durations, excluding
preceding source/cache setup and final reporting; it retains whole-pipeline times
separately.

`last_complete_sample_bytes` is the last admitted complete sample, not a live
reading at report publication. `sampling` gives the window length, first and last
admitted sweep offsets, requested interval, observed start-to-start interval
range, and discarded/incomplete/complete counts. `collector_cpu_seconds` includes
polling and completion of an in-flight sweep discarded at stop;
`max_sweep_seconds` includes discarded sweeps too. These costs diagnose the
observer; only the external pipeline timer measures in-operation overhead.

Native Linux, Windows and Apple Silicon macOS reader/controller tests run in CI
on Python 3.11 and 3.12. Local verification also exercised a frozen Linux
executable and real slicing/backprojection CLI runs inside the default Docker
image on a Windows/WSL host. This does not constitute a new GUI/volume-server
delivery test or validation on every Docker desktop host.

The fixed-local Linux pipeline benchmark observed negative median differences
for both operations. Subsequent real-data Linux tests ran two paired trials on
three annotation paths, slicing and backprojecting each sliced intensity volume.
The data-step differences varied in sign between pairs; positive median
differences were smaller than at least one treatment's MAD. All enabled runs
returned usable RAM samples. These small samples do not establish a 1% bound.

The fresh native Windows rerun after PR #146 and manager-observer integration
completed all 12 slicing/backprojection workflows (24 operations). All 12 enabled
operations returned usable partial peaks; the previous transfer and manifest
publication failures did not recur. The isolated CloudFiles workaround remained
identical in both treatments and is still not a shipped dependency fix.

Long-path median in-operation overhead was +6.498% for slicing and +3.851% for
backprojection. Both operations were slower in both pairs and their median
differences exceeded either treatment's MAD. This is an unresolved performance
finding, not negligible overhead or completion of the 1% engineering target.
The short-path positive median differences were below at least one treatment's
MAD. Two pairs on a shared host do not establish a strict bound. The queued,
untouched shared-backing gap described above also remains open.

The earlier failed Windows batch is retained, as is the separate component stress
diagnostic; neither substitutes for the fresh real-data evidence. Native macOS
pipeline overhead has not been measured. See the benchmark guide for every
attempt, timing, variability, sampling cost and remaining deployment limitation.
No cross-platform 1% guarantee is claimed.

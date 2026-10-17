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

Without `--verbose`, the CLI skips detailed timing samples, volume statistics,
report formatting and report writes. Basic step duration and progress remain
available. The pipeline factory's `verbose` argument continues to control its
progress bar; it is separate from run reporting.

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
already-published file keeps its delivery status and path. The CLI exits with
status 1 on failure and 130 on interruption, including interrupted report
delivery or presentation. When
options cannot be loaded and no output destination is known, the report is
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
| `memory` | `null`: RAM accounting is deferred to issue #139 |

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
slicing run already has it in `effective_configuration.operation`. Input paths
retain their original host identifiers before Docker remapping. Annotation and
layer names remain in the effective configuration, and the resolved dataset URL
is `volumes.source.url`.

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
existing file/volume facilities. No memory sampler is added by this change.

The fixed-local-source benchmark and reproducible commands are documented in
[the Python benchmark guide](https://github.com/ChengLabResearch/ouroboros/blob/main/python/benchmarks/README.md).
The target is at most 1% median pipeline elapsed-time overhead for instrumentation
that runs during the data operation. Setup and final reporting costs are assessed
separately. RAM collection must use that same in-operation budget when #139 is
implemented. Report measured variability alongside overhead; a negative observed
overhead reflects noise, not free instrumentation.

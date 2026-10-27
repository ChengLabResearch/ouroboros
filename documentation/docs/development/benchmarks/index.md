# Benchmarking

Runnable benchmark scripts live in `python/benchmarks/`. Run the commands below
from the repository's `python/` directory with the project environment installed.
Keep raw reports, logs, profiles, and specimen inputs in a work directory outside
the source tree. Record important findings and validation evidence in the
associated pull request or issue.

## Bounding-box partitioning

The [bounding-box reference](../../reference/bounding-box-partitioning.md)
describes the production estimator, partition selection, and external annotation
manifest. Regenerate complete recorded paths and compare their geometry without
fetching specimen voxels:

```sh
.venv/bin/python benchmarks/benchmark_partitioning.py \
  --fixtures /path/to/local-inputs --repeats 2 --output /tmp/partitioning.json
```

The report includes allocation components, slice assignments and bounds,
cumulative requested source bytes, chunk coverage, and partitioner elapsed time.
Code-derived payloads and logical chunk bytes are separate from measured RAM,
network traffic, and slicing throughput. Full-path runtime comparisons require
the original source datasets and complete recorded options.

## Run-statistics overhead

The real-data harness compares reporting disabled and enabled around the
existing `SliceParallelPipelineStep` or `BackprojectPipelineStep` duration. This
includes downloads, computation, output writes, cleanup, and worker/manager
startup and shutdown inside that step. Earlier geometry/source setup, sampler
startup/shutdown, and final report formatting and delivery are excluded.

The separate `Pipeline.process()` timer covers the broader pipeline. Memory
collection also covers that broader window, so collector CPU or sweep costs
must not be divided by the shorter data-step duration and called overhead.
See the [run-statistics reference](../../reference/run-statistics.md) for metric
definitions, platform differences, sampling coverage, and report boundaries.

From `python/`, with the project's virtual environment installed, reuse saved
slice/backprojection options and choose a new work directory:

```sh
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --real-slice-options /path/to/slice-options.json \
  --real-backproject-options /path/to/backproject-options.json \
  --work-dir /tmp/ouroboros-real-data --workers 12 --pairs 2 --timeout 3600
```

This copies original annotations, preserves the image source/scientific settings,
and visits every annotation path from shortest list to longest. Use
`--annotation-layer "Exact path name"` to select a path without editing the
annotations; repeat the option for multiple paths. Selection is recorded and
checked when resuming. Both the data-operation and pipeline timing
boundaries are recorded without adding production timers. Each pair runs
slicing then backprojection of that exact output. Output headers, times, logs,
options and reports remain; scientific TIFFs are removed after matching
backprojection succeeds. No pixel hashing is performed.

Failed trials remain recorded and dependent backprojections are marked skipped;
remaining independent trials still run. A missing usable RAM sample is retained
as a coverage failure, not silently treated as a disabled collector. An incomplete
pair receives no aggregate overhead estimate. `--resume-real-data` completes an
interrupted batch without repeating completed/failed attempts, after checking
production code, annotations, settings and dependency versions. Retain all results;
do not retry/select favorable trials or enlarge a workload merely to conceal costs.
`--workers` must be a multiple of four. Sandbox execution must permit the existing
multiprocessing manager sockets.

### Other harness modes

These modes use broader timing windows than the data-operation comparison.
Validate Docker report delivery separately from native timing diagnostics.

```sh
# Existing local fixture; time Pipeline.process(), including preparation.
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --scope pipeline --reuse-fixture /tmp/ouroboros-local-fixture \
  --work-dir /tmp/ouroboros-pipeline-overhead --pairs 3

# Generate a local fixture and time the full CLI handler.
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --work-dir /tmp/ouroboros-statistics-benchmark --pairs 7
```

The default deterministic fixture is a 384³ uint16 precomputed volume
with 64³ raw chunks / 1 µm isotropic voxels. A curved 32-point path yields 192²
slices at 0.5-voxel spacing, with 32 target slices per box. A thresholded warmup
segmentation is reused for binary backprojection using 64³ chunks/zlib. Both
modes use four workers and one BLAS/OpenMP thread. `--size 128 --pairs 3` gives a
short smoke check; `--timeout` bounds each child operation (default 600 seconds).

Pipeline mode externally times the same `Pipeline.process()` call in both
treatments, with one disabled warmup per operation and alternating pair order.
It excludes options loading, report collection/delivery and sampler startup/
shutdown; it includes scientific preparation, worker timing aggregation, RAM
collection, output writing and pool shutdown. Reused fixtures are not regenerated
or pixel-hashed (`pixel_equivalence: null`).

Full-handler mode includes option loading, pipeline setup/execution/cleanup,
report formatting, stdout flush and statistics delivery. Imports/process startup
are excluded from that timer and recorded separately. Each operation receives
both disabled/enabled warmups and alternating measured pairs. Source creation,
segmentation preparation, inspection and removal are outside the measured interval;
no remote source is accessed. It hashes output pixels and requires equivalence,
checks successful enabled statistics against stdout and absence of disabled
sidecars, and fails on missing usable memory samples. Per-run images are removed
after inspection. Source/harness digests, hardware, versions, TIFF dimensions,
pixel/storage sizes, every individual time and statistical summaries are retained.

## Memory identity diagnostic

On Linux, the identity probe exercises the production sampler against an
isolated process tree while changing psutil's wall-clock conversion. It does
not change the host clock or measure slicing overhead:

```sh
.venv/bin/python benchmarks/probe_memory_identity.py \
  --output /tmp/memory-identity.json
```

Use `--mid-sweep-shift` to also change the conversion during a sweep. Identity
checks and JSON serialization occur outside each timed sampler call. Choose a
new output path for each attempt so earlier observations remain available.

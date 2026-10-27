# Run-statistics overhead benchmark

Benchmark reports and raw evidence live in this documentation directory. The
runnable benchmark scripts remain in `python/benchmarks/`; reproduction commands
below run from the repository's `python/` directory.

## Bounding-box partitioning

The [bounding-box reference](../../reference/bounding-box-partitioning.md)
describes the implementation and geometry comparison using external annotation
inputs. The earlier reduced-workload runtime report has been withdrawn; it did
not establish performance on the complete recorded paths. Full-path runtime
comparisons use the original source datasets and retain raw evidence outside the
repository.

## Docker cost and coverage after the identity fix (#163)

The [#163 campaign report](results/issue-163/index.md) records Linux Docker on
Windows/WSL2 at merged #162 revision `e81c4d4`, including all paired timings,
collector coverage, controlled-clock identity evidence, and source-cache failures.
It separates the data-operation timer from the broader pipeline timer and retains
the historical native reference runs below. It does not change the metric or
certify native macOS/macOS-host Docker workflows.

## Historical native reference measurements and metric semantics (#139)

The historical native real-data evidence measures revision
[`10145cd`](https://github.com/ChengLabResearch/ouroboros/commit/10145cd34ecaff138fedd21336046cf989ca6ff4),
using manager-owned shared-buffer observations. It does not measure the removed
worker/TIFF registry or Windows working-set enumeration. Later review cleanup
removes formatting-only pipeline changes; these recorded runs are still explicitly
at `10145cd`, not a new benchmark of the later revision.

| Backend | Reported metric | Composition |
| --- | --- | --- |
| Linux / Docker | `process_tree_nonshmem_pss_plus_managed_resident_bytes` | Sum of process-tree `Pss - Pss_Shmem`, plus residency of live managed shared segments once |
| Native Windows | `process_tree_private_resident_plus_managed_allocated_bytes` | Sum of process-tree `PrivateWorkingSetSize`, plus live managed allocation sizes once |
| Native macOS | `process_tree_private_plus_managed_resident_bytes` | Process-tree private residency, excluding managed mappings, plus managed shared-segment residency once |

The [Linux raw results](results/issue-139-simple-10145cd-linux.json) and
[Windows raw results](results/issue-139-simple-10145cd-windows.json) retain every
paired time, all memory reports (including per-run process counts), coverage,
errors, options, dependency versions and output headers. The
[run provenance](results/issue-139-simple-10145cd-provenance.json) and native Windows
[preflight](results/issue-139-simple-10145cd-windows-before.json) /
[postflight](results/issue-139-simple-10145cd-windows-after.json) record revision,
source/harness/annotation digests, imported source location and validation checks.
Both result files have `production_code_unchanged: true` and production digest
`66ec3675faf7554d423c6b8a6b3b0da98dc43fa1f119e90a0a6611b0520df634`.

The Windows test runtime uses the disclosed
[CloudFiles zero-width-interval patch](results/issue-139-cloudfiles-zero-interval.patch)
with [dependency provenance](results/issue-139-simple-10145cd-windows-workaround.json).
CloudFiles 5.8.2 rejects transfers whose start/end timestamps coincide at the
Windows Python 3.11 clock resolution; the isolated workaround represents such
intervals as one microsecond. It is identical for off/on treatments and changes
only the temporary test dependency, not a shipped dependency.

### Method and timing boundary

The original `PosteriorSalivaryDuct_cut_ng.json` supplies the PSU Octo9 16-bit
image layer at mip 0. Each run uses 120 × 120 uint16 slices, adaptive spacing 1 /
ratio 0.5, 128 target slices per box, 12 workers, and one BLAS/OpenMP thread.
Short 2 (`Posterior Salivary Duct 2`) has 10 annotation points / 875 slices;
Short 3 (`Posterior Salivary Duct 3`) has 16 points / 887 slices; Long
(`Posterior Salivary Duct`) has 50 points / 4,901 slices. Each treatment
backprojects its own sliced intensity TIFF, with 160³ chunks, zlib, minimum
bounding box and separate output planes; the data is not binarized.

Linux/WSL2 (Python 3.11.13) and native Windows (Python 3.11.9, build 22631) ran
sequentially on the same shared development host. Each path received one disabled
slicing warmup, then two pairs: off → on, then on → off. Caches were shared;
remote metadata requests can remain. On enables reporting and RAM collection
together through `--verbose`; off omits both. No pixels were hashed. All 48
measured operations succeeded, all 24 enabled operations returned non-null memory
peaks, and all six excluded warmups succeeded. No attempts were selected out.

Overhead uses the existing `SliceParallelPipelineStep` or
`BackprojectPipelineStep` duration. It includes data downloads, computation,
output writes, intermediate cleanup, and worker/manager startup and shutdown
inside that step. Earlier parsing, geometry and source/cache setup, sampler
startup/shutdown, and final report formatting/writing/delivery are excluded.
The separate `Pipeline.process()` timer is retained in the raw files. RAM
collection covers that broader pipeline window, so collector CPU/sweep costs
must not be divided by the shorter data-step duration and called overhead.

### Owner-approved acceptance scope

The owner limited the combined **1% in-operation engineering target to longer
runs**, assessed alongside inter-run variability and fixed startup/shutdown costs.
For this evidence, acceptance is scoped to the measured **Long** path; no general
minimum-duration threshold or blanket short-operation guarantee is established.
Startup/shutdown outside the data step remains excluded. Fixed lifecycle work
inside the step can be significant for short runs, but these measurements do not
isolate it or establish that it explains every positive difference.

Long-path median changes were **−1.764% slicing / −6.737% backprojection on
Linux**, and **−7.260% / +0.288% on Windows**. Windows long backprojection's
+0.208 s difference is below both MADs (1.806 s off / 0.769 s on) and changes sign
between pairs. These results satisfy the owner-approved engineering assessment
for the measured long workloads; two pairs on a shared host do not prove a
statistical 1% upper bound. Negative changes indicate measurement variation,
not an instrumentation-induced speedup.

Short-run costs remain findings: Linux Short 2 slicing/backprojection and Short 3
slicing were slower in both pairs, with differences exceeding both treatment
MADs; Windows Short 3 slicing has the same pattern. These observations are
accepted outside the scoped target, not recast as negligible overhead. Native
macOS pipeline performance and macOS-host Docker delivery are explicitly deferred;
native unit-test CI is not a substitute for those workflow runs.

### All timing summaries

Change = `100 × (median(on) / median(off) - 1)`; Δ = median(on) − median(off).
MAD is each treatment's median absolute deviation. The tables derive from the
canonical harness summaries in the linked raw files.

**Linux / WSL2**

| Path | Operation | Off median s | On median s | Δ s | Change | MAD off / on s |
| --- | --- | --- | --- | --- | --- | --- |
| Short 2 | Slicing | 4.598 | 4.976 | +0.379 | +8.237% | 0.235 / 0.339 |
| Short 2 | Backprojection | 4.011 | 4.318 | +0.306 | +7.634% | 0.250 / 0.297 |
| Short 3 | Slicing | 4.261 | 4.759 | +0.498 | +11.686% | 0.019 / 0.176 |
| Short 3 | Backprojection | 4.286 | 4.230 | -0.056 | -1.299% | 0.543 / 0.065 |
| Long | Slicing | 19.528 | 19.183 | -0.344 | -1.764% | 0.398 / 0.732 |
| Long | Backprojection | 26.198 | 24.433 | -1.765 | -6.737% | 1.123 / 0.617 |

**Native Windows**

| Path | Operation | Off median s | On median s | Δ s | Change | MAD off / on s |
| --- | --- | --- | --- | --- | --- | --- |
| Short 2 | Slicing | 21.450 | 21.597 | +0.147 | +0.684% | 0.433 / 0.625 |
| Short 2 | Backprojection | 18.895 | 18.547 | -0.348 | -1.844% | 0.347 / 0.017 |
| Short 3 | Slicing | 21.461 | 22.781 | +1.320 | +6.151% | 1.010 / 0.092 |
| Short 3 | Backprojection | 18.447 | 19.135 | +0.688 | +3.729% | 0.901 / 1.052 |
| Long | Slicing | 38.122 | 35.355 | -2.768 | -7.260% | 4.843 / 2.052 |
| Long | Backprojection | 72.310 | 72.518 | +0.208 | +0.288% | 1.806 / 0.769 |

### Every enabled report: memory and process counts

C/I/D = complete/incomplete/discarded sweeps; only complete in-window samples
contribute to peaks. `Max processes` is the maximum observed process-tree count,
not the configured worker count; the sampler itself is excluded. Every interval
was requested at one second. Partial reports retain process exits or changing
manager observations/tree errors. Complete samples include preparation and do
not guarantee coverage of a short data operation's peak.

| Platform | Path | Operation | Pair | Status | Peak MiB | Max processes | C/I/D | Window s | Max sweep ms | CPU s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Linux / WSL2 | Short 2 | Slicing | 1 | partial | 1203.50 | 15 | 19/2/0 | 20.564 | 193.137 | 1.237 |
| Linux / WSL2 | Short 2 | Backprojection | 1 | sampled | 1174.27 | 14 | 20/0/0 | 19.806 | 249.948 | 1.219 |
| Linux / WSL2 | Short 2 | Slicing | 2 | partial | 1140.77 | 15 | 18/2/0 | 19.830 | 312.985 | 1.318 |
| Linux / WSL2 | Short 2 | Backprojection | 2 | sampled | 1155.16 | 14 | 20/0/0 | 19.207 | 181.110 | 1.148 |
| Linux / WSL2 | Short 3 | Slicing | 1 | partial | 655.05 | 15 | 17/4/0 | 20.158 | 233.689 | 1.375 |
| Linux / WSL2 | Short 3 | Backprojection | 1 | sampled | 1233.23 | 14 | 20/0/0 | 19.500 | 189.160 | 1.089 |
| Linux / WSL2 | Short 3 | Slicing | 2 | partial | 610.39 | 15 | 17/3/0 | 19.801 | 338.074 | 1.235 |
| Linux / WSL2 | Short 3 | Backprojection | 2 | sampled | 1230.33 | 14 | 20/0/0 | 19.366 | 237.971 | 1.209 |
| Linux / WSL2 | Long | Slicing | 1 | partial | 1173.04 | 15 | 22/13/0 | 34.433 | 271.326 | 3.466 |
| Linux / WSL2 | Long | Backprojection | 1 | sampled | 2141.53 | 14 | 41/0/0 | 40.794 | 517.144 | 5.270 |
| Linux / WSL2 | Long | Slicing | 2 | partial | 1124.07 | 15 | 20/16/0 | 35.651 | 400.087 | 4.297 |
| Linux / WSL2 | Long | Backprojection | 2 | sampled | 2093.96 | 14 | 40/0/0 | 39.569 | 512.914 | 4.687 |
| Native Windows | Short 2 | Slicing | 1 | partial | 1667.56 | 14 | 21/3/0 | 23.223 | 190.308 | 1.359 |
| Native Windows | Short 2 | Backprojection | 1 | partial | 1189.23 | 10 | 17/3/0 | 19.458 | 163.151 | 0.906 |
| Native Windows | Short 2 | Slicing | 2 | partial | 1793.07 | 14 | 20/2/0 | 21.930 | 149.340 | 0.953 |
| Native Windows | Short 2 | Backprojection | 2 | partial | 1641.47 | 10 | 19/1/0 | 19.589 | 205.263 | 0.953 |
| Native Windows | Short 3 | Slicing | 1 | partial | 1739.83 | 14 | 22/2/0 | 23.792 | 172.136 | 1.359 |
| Native Windows | Short 3 | Backprojection | 1 | partial | 1635.38 | 10 | 18/1/0 | 19.035 | 119.063 | 0.734 |
| Native Windows | Short 3 | Slicing | 2 | partial | 1689.23 | 14 | 21/3/0 | 23.655 | 215.457 | 1.453 |
| Native Windows | Short 3 | Backprojection | 2 | partial | 1591.86 | 10 | 20/1/1 | 21.182 | 192.728 | 1.359 |
| Native Windows | Long | Slicing | 1 | partial | 1926.03 | 14 | 31/8/0 | 39.093 | 256.975 | 2.906 |
| Native Windows | Long | Backprojection | 1 | partial | 2527.54 | 13 | 73/2/0 | 74.775 | 472.958 | 3.625 |
| Native Windows | Long | Slicing | 2 | partial | 1823.94 | 15 | 31/4/0 | 35.016 | 192.024 | 2.141 |
| Native Windows | Long | Backprojection | 2 | partial | 2581.66 | 13 | 72/1/0 | 73.336 | 646.981 | 4.562 |

Collector CPU includes polling and completion of discarded in-flight work; it is
not additive elapsed-time overhead. The raw reports also retain observed interval
ranges and exact error kinds/counts.

### Memory coverage limits

One-second sampling can miss brief peaks or short-lived children. Segments
released by, or allocated outside, `SharedNPManager` can be missed while still
attached elsewhere. A deliberately retained-buffer fixture demonstrated a 32 MiB
omission, but real-data probes after #147's worker-startup fix attributed the large
observed gaps to TIFF backing, not persistent retained shared buffers. Normal
pipeline allocations use the manager; this does not prove retention impossible.

The collector does not separately count resident TIFF/file cache after its
process mappings disappear. Earlier sparse Linux diagnostics observed additional
TIFF residency of 262–263 MiB slicing / 95.05 MiB backprojection for long Salivary,
and 240–256 MiB / 35.12 MiB for corrected Daphnia. These are diagnostic observations,
not peak-error bounds or Windows accuracy measurements. Ouroboros controls file
writes/closure, but cache retention and eviction are OS-controlled.

Linux PSS includes other mapped/shared pages proportionally. Windows excludes
other shared pages and counts managed **allocated**, rather than resident, bytes;
untouched or paged-out managed memory can therefore be overcounted. Do not interpret
the difference between Windows and Linux peaks as an actual physical-RAM difference.

### Short Docker delivery check after review

[Raw Docker results and provenance](results/issue-139-docker-review.json) record one
Short 2 slice → backprojection pair through the existing FastAPI server and the
Electron-hosted volume server. Both operations returned `done`, both reports were
`published` to the configured host folders with owner UID 1000 / mode 0600, and
both reported `process_tree_nonshmem_pss_plus_managed_resident_bytes`.

| Operation | Data step s | Memory status | Peak MiB | Max processes | C/I/D | Collector CPU s |
| --- | --- | --- | --- | --- | --- | --- |
| Slicing | 29.409 | partial | 1448.38 | 8 | 29/1/0 | 2.741 |
| Backprojection | 8.873 | sampled | 1751.84 | 10 | 10/0/0 | 0.829 |

The run produced one 875-plane slice TIFF and 571 backprojection TIFF planes.
Slicing's incomplete sweep is retained in the raw report. `/proc` exposes
`Pss_Shmem`; successful managed-residency samples exercise the Linux collector
inside the container. Host and container production digests match. All production
Python ASTs match `10145cd`; the changed byte digest reflects only the three
formatting-cleanup files. Scientific scratch and report staging were empty after
delivery, with no managed `psm_` segments left in `/dev/shm` (two loky semaphore
files remained in the persistent server container).

Deployment: Docker Desktop on Windows through WSL2 Ubuntu, Linux/Python 3.11.12
container, `python/compose.dev.yml`, current source bind mount, default
`ouroboros-volume`, and 64 GiB shared-memory limit. The API was called directly;
this was not an Electron GUI-driven test or a newly built release-image test.
The copy-to-volume/copy-to-host HTTP endpoints and Docker copy commands were
unmodified. This short delivery check is not an overhead benchmark.

Native macOS slice/backprojection and macOS-host Docker delivery are deferred on
#139. A separate native-Linux-host Docker run was not performed; the Windows/WSL2
check must not be labeled a complete three-host deployment matrix. #139 remains
open for those outstanding workflow checks.

## Reproducing the benchmark

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
checked when resuming. Both timing
boundaries above are recorded without adding production timers. Each pair runs
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

These modes remain useful diagnostics; their timing boundary differs from the
real-data acceptance comparison. Neither substitutes for Docker report delivery.

```sh
# Existing local fixture; time Pipeline.process(), including preparation.
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --scope pipeline --reuse-fixture /tmp/ouroboros-statistics-final \
  --work-dir /tmp/ouroboros-pipeline-overhead --pairs 3

# Generate a local fixture and time the full CLI handler.
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --work-dir /tmp/ouroboros-statistics-benchmark --pairs 7
```

The default deterministic seed-138 fixture is a 384³ uint16 precomputed volume
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

## Historical evidence

Superseded #139 artifacts were removed from the current tree. The
[previous artifact directory at 10145cd](https://github.com/ChengLabResearch/ouroboros/tree/10145cd34ecaff138fedd21336046cf989ca6ff4/python/benchmarks/results)
and [previous benchmark README](https://github.com/ChengLabResearch/ouroboros/blob/10145cd34ecaff138fedd21336046cf989ca6ff4/python/benchmarks/README.md)
retain the earlier Linux pipeline/preliminary/real-data measurements, Windows
stress diagnostic, failed transfer-lifetime batches, post-#145 Windows rerun, and
their provenance. They describe former collectors and are not current performance
evidence. Their failures and unfavorable results remain in history.

### Accepted #138 pipeline-only measurement
The [pipeline-only result](results/issue-138-pipeline-only.json) was refreshed after
removing the per-clock switches and worker reporting arguments. Both treatments
retain the pre-existing worker clocks; the existing `PipelineStep` methods
control accumulation. The run reused the same 384³ local source and segmentation
with four workers and one BLAS/OpenMP thread, using three alternating pairs per
operation after one disabled warmup. No data was regenerated or pixel-hashed.
Production code remained unchanged throughout that test. Its source digest
identifies the #138 timing-only implementation, not the current RAM collector.

| Operation | Disabled median | Enabled median | Difference | Observed overhead | MAD, disabled / enabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Slicing | 16.893485 s | 16.726067 s | -167.418 ms | -0.991% | 401.156 / 155.981 ms |
| Backprojection | 9.516044 s | 9.521981 s | +5.937 ms | +0.062% | 105.412 / 96.389 ms |

Backprojection's +5.937 ms difference (+0.062%) is much smaller than its enabled-run MAD of 96.389 ms; slicing's difference is negative. These results remain consistent with negligible in-operation overhead at this measurement resolution. A negative difference is measurement variation, not a speedup claim, and three pairs do not prove a strict statistical upper bound. All samples are retained; no further runs were selected to improve the result.

The development Electron preview was using CPU rendering during this run (a process snapshot showed about 149% CPU). Background activity was not controlled, so absolute runtimes should not be compared with the earlier session, and this is not an idle-machine microbenchmark. The within-run enabled/disabled comparison and its variability are reported together.

Earlier full-handler artifacts and summaries were removed because they measured
superseded implementations. The original accepted pipeline measurement and those
older artifacts remain in Git history; #138 links its original acceptance numbers
directly to that revision rather than to this refreshed file.

# Run-statistics overhead benchmark

**Collector revision (2026-10-17):** PR #144 now uses manager-only shared-buffer
observations, Linux non-shmem PSS plus managed residency, Windows private
working-set bytes plus managed allocation size, and macOS private residency plus
managed residency. Worker/TIFF tracking and Windows working-set enumeration have
been removed. The recorded pipeline benchmarks below describe earlier
collectors; their overhead is not a measurement of this replacement. The Linux
candidate probes informed the new accounting, but isolated query costs likewise
are not pipeline overhead. This revision uses focused native accounting,
ownership and reporting tests, without another long real-data benchmark campaign.
Native macOS pipeline overhead and a cross-platform 1% overhead bound are not established.

The current real-data comparison applies the overhead budget to instrumentation
**during the actual slicing or backprojection step**, excluding preceding source
and cache setup. It reuses the existing step timers in both treatments and retains
whole-pipeline times separately. See the real-data section below.

The older local-fixture mode times `Pipeline.process()`, including its preparation
steps. To reproduce that earlier measurement:

```sh
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --scope pipeline --reuse-fixture /tmp/ouroboros-statistics-final \
  --work-dir /tmp/ouroboros-pipeline-overhead --pairs 3
```

This mode uses an external timer around the same pipeline call for enabled and
disabled reporting. It includes scientific preparation, worker instrumentation,
transport/aggregation of worker measurements, concurrent RAM collection, output writing, and pool shutdown.
Options loading, volume-statistics collection, JSON construction, printing, and
report-file delivery and sampler startup/shutdown are outside that timer. Both treatments retain the basic
duration/progress measurements needed by the application. It performs one
disabled warmup per operation, alternates treatment order across pairs, and
skips output pixel hashing. A reused fixture is neither regenerated nor hashed.
`pixel_equivalence: null` means this run did not perform that check.

The full-handler mode below also measures setup and final reporting. Its numbers
describe total user-visible latency, not the in-operation overhead budget.

From `python/`, with the project's virtual environment installed:

```sh
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --work-dir /tmp/ouroboros-statistics-benchmark --pairs 7
```

Use a new work directory for each invocation. The directory retains the local
source, fixed segmentation, every options file, stdout/stderr, delivered report,
and `results.json`. Per-run image outputs are hashed and removed after inspection
to limit disk usage. Source creation, segmentation preparation, output inspection,
and removal are outside the measured interval. No remote source is accessed.

The default fixture is a deterministic, seed-138, 384³ `uint16` precomputed
CloudVolume with 64³ raw chunks and 1 µm isotropic voxels. A curved 32-point path
produces 192 × 192 slices at 0.5-voxel spacing, with 32 target slices per download
box. A thresholded segmentation is generated once from the warmup slice output
and reused for all binary backprojections. Backprojection uses 64³ chunks and zlib
output compression. The benchmark controls the existing slicing process-count
seam and backprojection `process_count` to use four processes; BLAS/OpenMP use one
thread. Production geometry, download, multiprocessing, interpolation,
backprojection, and TIFF writing all execute normally.

Each operation gets a disabled and enabled warmup, then seven pairs with
alternating order. Both treatments share warm filesystem and CloudVolume caches.
The enabled treatment invokes the real CLI with `--verbose`; the disabled
treatment omits it. A timer outside `cli.main()` includes option loading, pipeline
setup/execution/cleanup, report collection, formatting, printing and flushing
stdout, and statistics-file delivery. Imports and process startup are excluded
from that timer; full subprocess elapsed time is recorded separately. Output is
redirected to local files consistently for both treatments. Docker transfer is
**not measured** by this native benchmark.

`results.json` records the source and output pixel digests, actual TIFF page
dimensions and pixel/storage bytes, delivered statistics paths, hardware,
dependency versions, production Python source digest, every individual time,
medians, median absolute deviations, ranges, and per-pair overhead. Identical
output pixels are required across all repetitions of each operation. Enabled
runs must deliver successful statistics matching stdout without collection
errors; disabled runs must not deliver a sidecar. The production source digest
must remain unchanged throughout the run.

The in-operation target is at most **1% median pipeline elapsed-time overhead**, calculated
as `(median(enabled) / median(disabled) - 1) × 100` separately for slicing and
backprojection. The result records whether the observed median meets the target;
it deliberately does not turn a noisy performance measurement into a CI test.
Always report the variability alongside the median. A negative observed overhead
means run-to-run variation exceeded the measured reporting cost, not that
reporting makes processing faster. If variation is comparable to the target,
repeat on an idle machine and retain all results; do not select only favorable
runs or increase the workload merely to hide overhead.

For a quick scientific-path smoke check, use `--size 128 --pairs 3`. This reduced
fixture is useful for correctness checks, not the default acceptance measurement.
`--workers` must be a multiple of four. `--timeout` limits each child operation
(default 600 seconds). Linux sandbox environments must permit local
multiprocessing manager sockets for the existing shared-memory pipeline.

Enabled runs include RAM sampling and retain the full memory summary in each
measurement: metric, interval, process counts, complete/incomplete samples and
collector cost. Missing usable memory samples fail the benchmark rather than
silently benchmarking a disabled collector. The combined instrumentation uses
the same 1% target. Manager snapshot publication is included in
the measured pipeline interval on every backend; the separate sampler excludes its own memory.

## Existing real-data paths

To measure every annotation path in an existing Neuroglancer file, reuse saved
slice and backprojection options:

```sh
.venv/bin/python benchmarks/benchmark_run_statistics.py \
  --real-slice-options /path/to/slice-options.json \
  --real-backproject-options /path/to/backproject-options.json \
  --work-dir /tmp/ouroboros-real-data --workers 12 --pairs 2 --timeout 3600
```

This mode retains the external `Pipeline.process()` timer and reads the existing
`PipelineStep` duration for the actual slicing/backprojection step in both
treatments. Its overhead summary uses the latter: parsing, geometry, and
source/cache initialization precede data processing and are excluded. No new
production timers are added. It copies the original annotations,
preserves the selected image source and scientific settings, and runs the paths
from shortest annotation list to longest. One disabled slicing warmup per path
populates the shared CloudVolume cache; remote metadata requests can remain.
Each pair runs slicing followed by backprojection of that exact sliced intensity
volume. It does not threshold the data into a synthetic segmentation. The first
pair runs disabled/enabled and the second enabled/disabled. Both operations use
the requested worker count; BLAS/OpenMP use one thread in both treatments.

Output headers, all timings, logs, options, and reports are retained. Scientific
TIFFs are removed after their matching backprojection completes. No pixels are
hashed. Unlike the synthetic acceptance mode, a missing usable RAM sample is
retained in the real-data results instead of aborting the remaining trials; it
must be reported as a coverage failure. Scientific execution and report delivery are checked. Failed trials are retained,
and their dependent backprojections are marked skipped. Remaining independent
trials still run. `--resume-real-data` finishes an interrupted batch without
repeating completed or failed attempts; it verifies unchanged production code,
annotations, settings and dependency versions. An operation with incomplete
pairs receives no aggregate overhead estimate. Two pairs describe the observed variation;
they cannot establish a statistical upper bound on overhead.

## Real-data measurements (#139)

[Linux results](results/issue-139-real-linux.json) use the user's original
`PosteriorSalivaryDuct_cut_ng.json`, the PSU Octo9 16-bit image layer, mip 0,
120 × 120 slices, adaptive spacing 1 / ratio 0.5, 128 target slices per box,
12 processes, and one BLAS/OpenMP thread. Backprojection uses each run's own
sliced uint16 intensity TIFF, 160³ chunks, zlib compression, minimum bounding box,
and separate output planes. It does not binarize the data. The original
`perfdatatest` options supplied these scientific settings.

Two pairs per path ran off/on, then on/off, after one disabled slicing cache
warmup. Only the existing `SliceParallelPipelineStep` or
`BackprojectPipelineStep` duration is compared below. Source/cache setup took
about 15 seconds per invocation and is excluded, along with parsing, geometry,
sampler startup/shutdown, final report collection and delivery. No pixel hashing
was performed. Production source remained unchanged. All measurements are
retained; no trials were selected or repeated to improve the result.

| Path | Operation | Off median (s) | On median (s) | Difference (s) | Observed difference | MAD off / on (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Posterior Salivary Duct 2 | slice | 4.776743 | 4.937809 | +0.161066 | +3.372% | 0.360739 / 0.146882 |
| Posterior Salivary Duct 2 | backproject | 4.317144 | 3.875092 | -0.442052 | -10.239% | 0.756395 / 0.020286 |
| Posterior Salivary Duct 3 | slice | 4.425853 | 4.471380 | +0.045526 | +1.029% | 0.071063 / 0.130346 |
| Posterior Salivary Duct 3 | backproject | 3.889799 | 3.841992 | -0.047807 | -1.229% | 0.412733 / 0.177347 |
| Posterior Salivary Duct | slice | 19.442347 | 20.999911 | +1.557565 | +8.011% | 0.117172 / 1.606434 |
| Posterior Salivary Duct | backproject | 23.300999 | 23.828389 | +0.527390 | +2.263% | 0.361705 / 0.567634 |

The short paths contain 10 and 16 annotation points and produce 875 and 887
slices respectively; the long path contains 50 points and produces 4,901 slices.
No operation was slower with reporting in both pairs. Every positive median
difference is smaller than at least one treatment's MAD. This is consistent with
variation dominating the measured cost, but two pairs cannot establish a 1%
upper bound. The shared development host was not isolated; a process snapshot
included active Electron CPU rendering. Linux and Windows workloads did not run
concurrently.

All 12 enabled Linux operations returned usable PSS peaks. Some sweeps were
incomplete because workers exited or the process tree changed; those reports
retain `partial` status and use only complete sweeps for the peak. Exact peaks,
coverage and collector costs are in the result artifact.

The [preliminary Linux attempt](results/issue-139-real-linux-preliminary.json)
was deliberately stopped when its whole-pipeline timer was found to include
about 15 seconds of source/cache setup and it had not retained the actual data
step times. It is preserved but excluded from this table. The corrected run
repeated the two pairs for every path with both timing boundaries captured and
reused the downloaded chunks. Windows dependency installation overlapped only
that preliminary attempt, not the corrected measured run.

### Historical native Windows results before #145

[Windows results](results/issue-139-real-windows.json) use the same dataset,
settings and two off/on pairs per path, with native Python 3.11.9 and
Windows-local source, cache and output directories. Linux was measured before
the cleanup prerequisite; Windows was measured after `cb3bb3e`. Production
sources remained unchanged within each platform's measurements. The source
digests are retained; absolute Linux and Windows runtimes are not compared.

| Path | Successful slice → backprojection workflows | Failed slices | Dependent backprojections skipped |
| --- | ---: | ---: | ---: |
| Posterior Salivary Duct 2 | 4 | 0 | 0 |
| Posterior Salivary Duct 3 | 1 | 3 | 3 |
| Posterior Salivary Duct | 0 | 4 | 4 |

All 12 scheduled slicing trials were attempted: five succeeded and their five
backprojections succeeded; seven failed and their backprojections were skipped.
The long-path cache warmup also failed. Failed trials were not replaced by
successful retries. After the original fail-fast harness stopped, the batch was
resumed with failure recording; all prior successful records were verified
unchanged. Both benchmark script digests are retained. Timing boundaries and
production code were unchanged; only orchestration gained resumption and
explicit failure/skip outcomes.

Only `Posterior Salivary Duct 2` has complete timing pairs:

| Operation | Off median (s) | On median (s) | Difference (s) | Observed difference | MAD off / on (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Slicing | 21.901489 | 22.240072 | +0.338584 | +1.546% | 0.132245 / 0.380839 |
| Backprojection | 18.436838 | 18.656084 | +0.219246 | +1.189% | 0.194656 / 0.186755 |

Neither operation was slower with reporting in both pairs. However, two of the
four completed enabled operations returned **unavailable RAM** because mapping
manifest publication failed: the first slicing trial and second backprojection
trial. The other two returned partial coverage with usable peaks. These timings
are descriptive, not acceptance evidence for a working collector or a 1% bound.
No overhead estimate is calculated from incomplete pairs on the other paths.

The original Windows warmup failed before slicing in CloudFiles 5.8.2: its
[transfer monitor](https://github.com/seung-lab/cloud-files/blob/master/cloudfiles/monitoring.py)
rejects zero-width intervals. Native Python 3.11.9 uses a 15.625 ms monotonic
clock tick, so a short transfer can receive identical start/end times. The
[isolated dependency workaround](results/issue-139-cloudfiles-zero-interval.patch)
represents such an interval as one microsecond. It only changes the dependency
in the temporary Windows test runtime, not the repository lock or production
code. Both reporting treatments use it. [Provenance and the original failure](results/issue-139-windows-workaround.json)
are retained.

The next warmup exposed the pre-existing `SharedNPManager.remove_termed`
cleanup ordering failure. The owner approved fixing it: acquire the cleanup
handle before closing the supplied view, unlink while that handle is open,
close it in `finally`, and retire the existing registry entry after cleanup.
An unregistered array is now rejected before its view is closed. The existing
manager/list remain authoritative; no second allocation registry was added.
Both new regressions failed on old native Windows code and passed after the
fix; all seven Linux memory-helper tests passed. The focused native CI matrix
now includes those regressions. [The original failure](results/issue-139-real-windows-blocked.json)
is retained as history, not an outstanding cleanup blocker.

A subsequent attempt stopped on an intermittent download-result worker-pool
failure before any measured operation completed. A diagnostic rerun succeeded;
both are retained separately. The final scheduled batch encountered this failure
in both reporting modes. A further long-path diagnostic with full chained
tracebacks confirmed `FileNotFoundError` while deserializing `SharedMemory` from
the result pipe: the allocation no longer exists when the receiver reopens its
name. This is a separate transfer-lifetime problem from the fixed final cleanup.
At that revision it remained unresolved, as did mapping-manifest publication
failures. PR #146 subsequently fixed the handoff lifecycle; the rebased results
are recorded separately below.
[All failure and regression provenance](results/issue-139-windows-run-provenance.json)
is retained. These historical failures are not replaced or removed by the new
batch. Overall #139 acceptance still requires the remaining accounting and
deployment work.

### Native Windows rerun after rebasing onto #145

[Fresh results](results/issue-139-real-windows-rebased.json) and
[provenance](results/issue-139-windows-rebased-provenance.json) record production
`be87951`, rebased onto merged PR #146 (`f566276`). The manager retains allocation
handles; the observer derives mappings from those handles, and manifest reads
and replacement reuse the bounded Windows IPC retry. The original failed batch
above is retained unchanged.

All three cache warmups and **all 12 slicing → backprojection workflows
(24 data operations) succeeded**. All 12 enabled operations returned usable
`partial` peaks. No by-name transfer failure or manifest-publication failure
recurred. Incomplete sweeps were caused by process exits or changing mappings /
process trees and excluded from peaks. These are sampled, partial measurements,
not proof of complete resident backing coverage.

The settings and alternating pair order are unchanged. Native Python 3.11.9 uses
12 workers and one BLAS/OpenMP thread, a shared existing chunk cache, and fresh
Windows-local outputs. Each backprojection consumes its own sliced intensity
volume. The isolated CloudFiles workaround above is unchanged (verified by
hash), identical for both treatments, and remains outside the repository.
Production and benchmark digests match the repository; production stayed fixed
throughout. No failed trials were retried or selected out. Linux benchmarks did
not run concurrently; the shared development host was not otherwise isolated.

| Path | Operation | Off median (s) | On median (s) | Difference (s) | Observed difference | MAD off / on (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Posterior Salivary Duct 2 | slice | 23.073508 | 23.576249 | +0.502741 | +2.179% | 0.684345 / 0.099444 |
| Posterior Salivary Duct 2 | backproject | 18.122470 | 18.202131 | +0.079661 | +0.440% | 0.086043 / 0.261774 |
| Posterior Salivary Duct 3 | slice | 22.707309 | 23.814665 | +1.107356 | +4.877% | 0.270171 / 1.342440 |
| Posterior Salivary Duct 3 | backproject | 18.782255 | 18.589627 | -0.192628 | -1.026% | 0.216100 / 0.406218 |
| Posterior Salivary Duct | slice | 35.535106 | 37.844008 | +2.308902 | +6.498% | 1.305884 / 1.203964 |
| Posterior Salivary Duct | backproject | 72.287309 | 75.070813 | +2.783504 | +3.851% | 0.455517 / 0.957966 |

The long path is an **unresolved performance finding** under the existing
in-operation target. Slicing was slower by 5.991% and 7.043% in the two pairs;
backprojection by 5.843% and 1.883%. Their median increases (2.309 s and 2.784 s)
exceed either treatment's MAD. These results do not support a negligible-overhead
claim or completion of the 1% engineering target. Two pairs on a shared host do
not establish a strict bound or isolate the cause; no extra trials were run to
seek a better result. On the short paths, positive median differences are smaller
than at least one treatment's MAD. Duct 3 slicing was slower in both pairs, but
one difference was only 0.156%; its median difference is below the enabled MAD.

Slicing had 16–21 complete and 6–21 incomplete sweeps per enabled operation;
backprojection had 16–68 complete and 3–10 incomplete sweeps. The requested
interval was one second. Long-path maximum sweep times reached 1.337 s for
slicing and 2.144 s for backprojection; observer CPU was 4.17–4.23 s and
6.58–7.16 s respectively. These are diagnostic costs, not additive wall-time
overhead. Every peak, interval, count, error and timing is retained in the raw
artifact.

The draft still has the Linux/default-Docker accounting blocker: resident shared
backing can be absent from PSS when no current mapping has touched it. The native
Windows working-set metric has the corresponding exclusion. Keeping an untouched
manager mapping alive fixes lifetime, not residency attribution. A revised metric
needs defined semantics and validation, and remaining native macOS / Docker
host-delivery requirements are still open.

## Recorded pipeline-only measurement

The [pipeline-only result](results/issue-138-pipeline-only.json) was refreshed after
removing the per-clock switches and worker reporting arguments. Both treatments
retain the pre-existing worker clocks; the existing `PipelineStep` methods
control accumulation. The run reused the same 384³ local source and segmentation
with four workers and one BLAS/OpenMP thread, using three alternating pairs per
operation after one disabled warmup. No data was regenerated or pixel-hashed.
Production code remained unchanged throughout the test, and the recorded source
digest matches the simplified implementation.

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

## RAM collector measurement (#139)

[All Linux pipeline samples and memory summaries](results/issue-139-pipeline-only.json)
use the same prepared 384³ source, four workers and one BLAS/OpenMP thread,
three alternating pairs per operation, and one disabled warmup. No pixels were
hashed. Production Python sources remained unchanged throughout the measurement.
The sampler requested a one-second interval.

| Operation | Disabled median | Enabled median | Difference | Observed overhead | MAD, disabled / enabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Slicing | 20.279846 s | 17.251620 s | -3028.226 ms | -14.932% | 415.349 / 594.883 ms |
| Backprojection | 9.115685 s | 8.975244 s | -140.441 ms | -1.541% | 45.338 / 108.476 ms |

Neither operation showed positive median overhead in this run. The large negative
slicing difference must not be interpreted as a speedup; this shared development
machine was not otherwise controlled, and three pairs do not establish a strict
upper bound. All samples are retained, without rerunning to select favorable
results. These observations meet the numerical engineering target on this Linux
workload; they do not validate native Windows/macOS overhead.

Slicing observed up to seven processes, 17–19 complete sweeps per run, 113–152 ms
maximum sweep durations and 1.493–1.900 s of observer CPU. Backprojection observed
up to five processes, 8–10 complete sweeps, 75–85 ms maximum sweep durations and
0.551–0.629 s of observer CPU. One backprojection sweep was incomplete because a
worker vanished during the read; the peak uses only complete sweeps. Observed
start-to-start intervals were approximately 1.000–1.010 s. Observer CPU is not
wall-time overhead: it can run concurrently on another core and includes polling
and completion of discarded end-of-window work.

### Native Windows diagnostic and unresolved performance limit

The [native Windows diagnostic](results/issue-139-windows-diagnostic.json) is a
separate component stress test, **not the full pipeline acceptance benchmark**.
Eight spawned workers each perform 81 in-place float32 additions on a disjoint
128 MiB slice of one 1 GiB `SharedNPArray`; the parent and workers map the full
allocation. It measures dispatch through worker completion, excluding collector
startup/shutdown, with a one-second requested sampling interval.

| Pair | Disabled | Enabled | Change |
| --- | ---: | ---: | ---: |
| 1 | 7.130173 s | 7.506819 s | +5.282% |
| 2 | 7.301253 s | 7.422279 s | +1.658% |
| 3 | 6.978654 s | 7.272467 s | +4.210% |

The ratio of medians is **+4.097%**: 7.130173 s versus 7.422279 s, a 292.105 ms
increase. MADs are 151.519 ms disabled and 84.540 ms enabled. The third pair used
a fresh identical fixture after the diagnostic's evidence write to a WSL UNC
path failed; the artifact records this interruption. All three enabled trials
produced complete samples, without collection errors. Actual sweeps reached
4.38–4.82 s during active writes, despite the requested one-second interval.
Shorter (~2 s) stress operations could end without any complete sample and
correctly reported unavailable memory.

The idle native reader cost was 50.66 ms for a 256 MiB allocation across three
processes and 511.09 ms for 1 GiB across nine. It counted the shared allocation
once in every sweep, beyond Windows' share-count cap. The diagnostic preserves
all paired times, sample/CPU details and idle measurements. This component test does not determine real pipeline acceptance. The original real-data Windows trials exposed transfer-lifetime and mapping-publication
failures. The post-rebase rerun above completed but retains performance and
accounting findings; native Windows acceptance remains open. The default Docker backend uses the Linux collector.

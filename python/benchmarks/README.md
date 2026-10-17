# Run-statistics overhead benchmark

The overhead budget applies to instrumentation **during `Pipeline.process()`**.
Use the existing prepared local fixture for a short focused comparison:

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
the same 1% target. Native Windows/macOS also include mapping-observation work in
the measured pipeline interval; the separate sampler excludes its own memory.

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

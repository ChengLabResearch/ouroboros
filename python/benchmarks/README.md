# Run-statistics overhead benchmark

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

The acceptance target is at most **1% median elapsed-time overhead**, calculated
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

RAM sampling is excluded from this benchmark until issue #139 supplies its
collector. The combined instrumentation must be measured against the same 1%
target when that collector is introduced.

## Recorded measurements

The [recorded lifecycle implementation run](results/issue-138-local-384.json)
used the default fixture and seven pairs on an Intel Xeon W-10855M, 12 logical
CPUs, Linux under WSL2, Python 3.11.13, and four workers. It was measured before
the final report-schema simplification; its source digest identifies the exact
measured Python code. The earlier
[pre-lifecycle-fix measurements](results/issue-138-local-384-before-lifecycle-fixes.json)
are retained separately and are not pooled with this run.

| Operation | Disabled median | Enabled median | Observed overhead | MAD, disabled / enabled |
| --- | ---: | ---: | ---: | ---: |
| Slicing | 8.463 s | 8.621 s | +1.87% | 0.149 / 0.146 s |
| Backprojection | 5.536 s | 5.565 s | +0.51% | 0.071 / 0.245 s |

**Slicing exceeded the agreed 1% observed-median target.** Backprojection met
that numerical criterion, but variation in both operations exceeded 1%, so
these measurements do not establish an instrumentation-cost bound at that
precision. The full ranges and all individual samples are in the artifact.
Later schema changes have not been measured by this run. No live Docker daemon
was available, so these numbers make no claim about Docker transfer overhead.

All measured TIFF pixels matched across both treatments and matched one
scientific run of each operation at baseline commit `1cf19b6`. The sliced image
contained 576 × 192 × 192 `uint16` pixels (42,467,328 pixel bytes), and the
backprojection contained 388 × 314 × 310 `uint8` pixels (37,767,920 pixel bytes;
552,589 compressed file bytes). Every verbose run delivered statistics identical
to its stdout JSON without collection errors; no run emitted a warning.

A separate diagnostic slice, excluded from the acceptance samples, timed the
existing reporting methods with wrappers: volume collection took 6.54 ms,
publication took 10.96 ms including record construction, formatting, and file
writing, and parent timing aggregation accumulated 1.30 ms. Those approximately
18.8 ms did not identify a reporting cost at the scale of the paired median
difference. They exclude worker-side timing and serialization costs; nested and
concurrent spans must not be summed indiscriminately. The diagnostic does not
replace the failed slicing criterion. The external handler-minus-operation
interval was about 11 ms for slicing and 12 ms for backprojection, also recorded
per sample in the artifact.

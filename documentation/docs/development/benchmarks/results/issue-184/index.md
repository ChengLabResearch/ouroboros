# Per-slice execution with unchanged partitions (#184)

This comparison uses the box-level implementation at `2226b31eadaf636b2122b6b9833b5d090a515091`
and per-slice execution at `081943057fb1f50100a25535b1afa5272049418d`.
The payload estimator and partition selection are unchanged, as explicitly scoped
by the owner. The estimator retains its conservative box-level scoring formula;
its output is not a measurement of the streamed worker's allocations or RAM.

`coordinate_grid()` still owns coordinate calculations. The slicing worker now
reuses one coordinate plane and passes its existing slice indices and TIFF mapping
to `slice_volume_from_grids()`. That helper owns one float64 coefficient workspace,
filters each native source channel once, and writes interpolated planes directly
into caller-owned views. Channels execute sequentially, regenerating the grid for
each channel to retain one coefficient workspace. The old full-grid allocation,
noncontiguous output stack, and per-channel interpolation-result array are removed.

## Fixed-box correctness and allocation evidence

Four baseline/candidate pairs use real source voxels and recorded annotation
geometry: the largest estimated-payload box in each current workload, plus the
recorded gapped nerve assignment. Every float pixel, generated coordinate grid and
extremum matches exactly. Instrumentation verifies one reused grid, one prefilter
per channel, and genuine TIFF plane destinations. Gap and adjacent sentinel
planes remain untouched. All sources in these probes have one channel; existing
dtype/channel tests also cover three channels and coefficient-workspace reuse.

The gapped case retains the historical assignment and box, but both executions
use current-main download extents and integer origins. Its 523 assigned indices
are 9856–10281 and 10289–10385; all seven intervening planes are checked.
Current production partitions remain contiguous for all three recorded workloads.

| Fixed source box | Assigned slices | Grid allocation before, bytes | Grid allocation after, bytes |
| --- | ---: | ---: | ---: |
| Daphnia, current box 5 | 146 | 17,520,000 | 120,000 |
| PSD, current box 43 | 71 | 34,080,000 | 480,000 |
| Nerve, current box 0 | 91 | 174,720,000 | 1,920,000 |
| Nerve, recorded gapped box 160 | 523 | 1,004,160,000 | 1,920,000 |

These are observed per-box allocation payloads, not process RAM. The gapped case
also eliminates a 334,720,000-byte output stack and a same-sized per-channel
interpolation result. Source storage and one float64 coefficient field remain.
The [fixed-box results](fixed-boxes.json) retain exact-output hashes and allocation
observations. Their instrumented execution times are not performance estimates.

## Complete-path measurements

The campaign uses the existing run-statistics benchmark and collector on native
Linux/WSL, with 12 logical CPUs and the unchanged 12-slot production budget:
3 download processes, 9 slicing processes, and 12 rewrite threads. Both values of
`alternate_slice_count_order` are exercised. Observed submissions and canonical
execution statistics verify `partition` and `alternating_slice_count` ordering.

Each of the three complete paths runs once per implementation and order: 12 runs,
without extra warmups. Previously cached source chunks are reused; OS page-cache
warmth is uncontrolled. Run order reverses baseline/candidate across order pairs.
Exact source boxes, assignments and geometry hashes must match the saved main
partitions in every run. Final normalized pixels and scientific TIFF metadata
are compared across all four runs for each specimen.

Full-pipeline wall time is the primary timing comparison. Worker-stage totals are
accumulated elapsed time and can overlap across workers; they must not be summed
as pipeline wall time. Existing worker timing boundaries are retained, including
extrema calculation after the worker timers. Grid callback time is attributed to
`generate_grid`, excluding it from `slice_volume`.

The existing collector requests samples every second and reports process-tree
non-shared-memory PSS plus managed shared-buffer residency once. Coverage and
errors accompany every peak; a partial result is not a complete peak guarantee.
The benchmark does not change the collector or substitute allocated bytes for RAM.

All 12 runs succeeded. The final normalized pixel hashes and scientific TIFF
metadata are identical across all four configurations for each specimen; all
30/82/158 source boxes and assignments remain unchanged. These are single-run
observations, with no estimate of timing variance.

| Specimen | Box order | Pipeline seconds, before → after | Change | Sampled peak GiB, before → after |
| --- | --- | ---: | ---: | ---: |
| Daphnia | Default | 22.95 → 22.46 | -2.14% | 0.747 → 0.741 |
| Daphnia | Alternating | 22.89 → 23.58 | +3.02% | 0.733 → 0.617 |
| PSD | Default | 55.33 → 55.87 | +0.97% | 1.882 → 1.455 |
| PSD | Alternating | 54.10 → 51.48 | -4.85% | 1.692 → 1.497 |
| Nerve | Default | 395.34 → 348.12 | -11.94% | 9.125 → 8.546 |
| Nerve | Alternating | 440.68 → 365.91 | -16.97% | 9.522 → 8.694 |

| Specimen | Box order | Slicing-step seconds, before → after | Complete/incomplete RAM sweeps, before → after | Collector CPU seconds, before → after |
| --- | --- | ---: | --- | ---: |
| Daphnia | Default | 4.87 → 4.34 | 21/2 → 23/0 | 1.14 → 0.87 |
| Daphnia | Alternating | 4.70 → 4.53 | 22/1 → 21/3 | 0.96 → 0.94 |
| PSD | Default | 33.52 → 35.03 | 38/18 → 47/9 | 5.89 → 6.23 |
| PSD | Alternating | 33.19 → 30.69 | 49/6 → 48/4 | 6.54 → 5.62 |
| Nerve | Default | 370.59 → 321.63 | 320/74 → 309/39 | 178.15 → 87.95 |
| Nerve | Alternating | 416.17 → 339.59 | 365/73 → 324/42 | 232.45 → 97.20 |

All RAM results are `partial` except Daphnia's candidate/default run (`sampled`).
The table reports peaks from complete sweeps; incomplete sweeps cannot establish
an upper bound on actual peak RAM. Errors reflect transient process/segment
availability and manager observations changing during a sweep. Across the 12
runs, 1,587 sweeps completed and 271 were incomplete. The requested interval was
1 second; observed nerve intervals/sweeps reached 1.477 seconds. Exact status,
coverage, errors and intervals are retained in [run measurements](campaign-runs.csv).

The nerve collector consumed 88–232 CPU seconds per run. These measurements
therefore describe the verbose pipeline with its existing sampler active, not
isolated interpolation performance. Collector CPU time is not an independently
measured elapsed-time penalty and cannot be subtracted from wall time. This
campaign does not separate algorithm gains from changed sampler contention.
The collector was not modified, and no additional timing-only campaign was run.

The nerve pipeline improved by 11.94%/16.97% in the default/alternating comparisons.
The smaller Daphnia/PSD timing changes do not establish a repeatable gain or
regression from one observation. Sampled peaks fell in all six comparisons, but
partial coverage limits the RAM conclusion. The fixed-box allocation removals
above are independent of those sampling limitations.

[Paired measurements](campaign-comparisons.csv) and [provenance](campaign-provenance.json)
record revisions, package versions, thread environment and run sequence. The run
CSV distinguishes canonical operation time, handler time, full pipeline time,
slicing-step time and accumulated worker-stage timings. `data_operation_elapsed_seconds`
in the original benchmark harness means the slicing step, not the whole operation.

## Comparison with the earlier 22.32 / 74.26 / 356.5 seconds

Those values are the medians of three warm-cache operation timings on October 18,
at `90b782941df87910037ba0a8f7926def3a5f415a`. The matching current boundary is
canonical `operation_elapsed_seconds` from the run-statistics JSON, rather than
the slightly longer CLI handler span or the shorter pipeline/slicing-step spans.

| Specimen | Historical operation median (range), seconds | Current default operation, seconds | Change from median |
| --- | ---: | ---: | ---: |
| Daphnia | 22.319 (22.238–22.620) | 22.547 | +1.02% |
| PSD | 74.259 (73.539–76.354) | 56.001 | -24.59% |
| Nerve | 356.492 (352.955–366.417) | 348.238 | -2.32% |

The candidate is not generally slower: Daphnia falls inside the old range, while
PSD and nerve are below it. The 395–441-second nerve results in this campaign
belong to the current-main box-level baseline, before this PR's per-slice change.
They are slower than the historical median, but these different campaigns do not
isolate which intervening change or runtime condition accounts for that slowdown.

| Specimen | Setup/planning seconds, historical → current | Slicing-step seconds, historical → current |
| --- | ---: | ---: |
| Daphnia | 15.124 → 17.862 | 6.651 → 4.341 |
| PSD | 15.216 → 20.375 | 57.965 → 35.029 |
| Nerve | 15.195 → 25.615 | 340.237 → 321.632 |

Setup/planning now takes longer while the slicing stage takes less time.
`VolumeCachePipelineStep` combines source metadata initialization and partition
planning. Prior isolated measurements of the identical current partitioner source
show 2.355/4.378/6.801 seconds for the three real geometries. That establishes a
measurable planning cost, but does not attribute the entire setup-stage increase
to partitioning.

Both campaigns used native WSL2, 12 logical CPUs, 3 download/9 slicing processes,
12 rewrite threads, and the same Python/NumPy/SciPy/CloudVolume versions. Raster
sizes/counts also match: 2656×100×100, 4901×200×200 and 10386×400×400. The old
partitions were 32/80/161 boxes versus 30/82/158 now; intervening production
changes include integer cutout geometry. The old BLAS thread variables were set
to 1; this campaign inherited them unset. psutil was 7.2.2 versus 7.1.3 now.
Three old warm-cache repeats versus one current run/configuration, uncontrolled
OS cache warmth/scheduling, and these configuration differences preclude a
controlled causal claim across revisions. Historical nerve collector CPU time
was 145–157 seconds; it was already non-negligible CPU consumption in that cohort.

[Historical comparison evidence](historical-comparison.json) retains the exact
operation times, stage medians, source identity and prior partition-only results.

## Validation and retained evidence

The 125 focused helper, slicing/statistics and TIFF metadata tests pass. These
include native/high-precision dtype interpolation, three-channel destinations,
both box orders, normalization/zero guarding, and mapping/worker lifetimes.
The final timing-boundary adjustment was followed by the 72 pipeline tests passing
again. All nine CI checks passed for implementation commit `0819430`, including
the server image, coverage checks, and memory-collector jobs on Linux, Windows
and macOS with Python 3.11/3.12. Those CI jobs do not constitute cross-platform
real-data performance validation. The only test warning is the pre-existing Google API Core Python 3.11
support warning. Native Windows, macOS and Windows-host/Docker workflows were not
measured in this campaign.

Raw source inputs stay outside the repository. The local campaign and pinned
source snapshots are under `/mnt/wsl/PHYSICALDRIVE2p1/ouroboros-184-20261027`.
The publication checkout retains the campaign/probe scripts and detailed audit
reports under `pub/bundles/issue184-implementation-20261027/`. Generated TIFFs are
removed only after output checks and evidence capture.

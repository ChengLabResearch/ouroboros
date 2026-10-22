# RAM collector cost and coverage after #162

This campaign measures merged production revision
[`e81c4d4`](https://github.com/ChengLabResearch/ouroboros/commit/e81c4d4743777d80b5b6fa70f8354d9c285ecd0e)
for [#163](https://github.com/ChengLabResearch/ouroboros/issues/163). Production
code, the one-second sampling interval, metric semantics, and report reduction
are unchanged. This is engineering evidence; it does not replace publication
Table 1 measurements tracked by
[ouroboros_pub#121](https://github.com/den-sq/ouroboros_pub/issues/121).

## Findings and remaining work

**The identity-growth regression is absent in the controlled probe; blanket
negligible-overhead and adequate-peak-coverage claims are not established.**
All 64 controlled-clock sweeps retain exactly 51 unique processes, with no failures
and early/late median sweep costs of 16.500 / 16.808 ms. New successful timed
reports have maximum counts of 7/6 for the four-worker fixture and at most 15/14
for 12-worker real-data slicing/backprojection. See the separate identity evidence
and its limits below.

The owner-approved **1% engineering target remains limited to #139's named longer
Salivary workload**, interpreted with variability. In the valid Docker cohort,
Long slicing changes by **+0.157%** (+0.020 s), below both MADs and with opposite
pair signs. Long backprojection changes by **+6.476%** (+0.957 s), above both MADs
(0.364 / 0.565 s), with positive changes in both pairs (+0.183% / +13.087%). The
latter is a material observed difference and does **not** establish the scoped
1% acceptance. Two shared-host pairs do not isolate the collector as the sole
cause or establish a statistical upper bound. Historical native Linux/Windows
acceptance remains specifically at the older recorded revision/deployments.

Nerve changes by **−15.345% slicing / −8.219% backprojection**. Disabled-treatment
MADs are 5.577 / 24.131 s, and the final disabled trial is substantially slower
than the first. These measurements illustrate inter-run variation, not a reporting
speedup, and do not override the Salivary finding. Fixed-local slicing (+5.198%)
and Short 3 slicing (+3.053%) are slower in both pairs with median differences
above both treatment MADs. Preserve those short-run costs outside the scoped
longer-run target; do not call them negligible. Short 2's failed warmup further
limits its timing interpretation.

Long Salivary slicing accepts only **3/13 and 7/14** attempted sweeps; Nerve
slicing accepts **8/29 and 16/28**. Across these four runs, the 50 incomplete
sweeps comprise 29 `FileNotFoundError`, 19 `OSError`, and two `NoSuchProcess`
records. Their retained examples name missing `/psm_*` segments, changed manager
observations, and process exits, respectively; none names process-tree churn.
Errors are counted by exception class, with only the first message retained per
class in each report. These examples point to **managed shared-memory allocation
and release racing the sweep** as the main remaining long-slicing coverage
limitation, between the manager manifest read and segment-open or residency
checks; individual lifecycle sequences were not traced. The campaign's only two
retained `Process tree changed during memory sweep` examples are both in Short 2
backprojection, consistent with removal of the clock-induced false churn in #162.

Short 3 slicing accepts only one sample, with a maximum of two processes, in each
enabled run: its non-null peaks do not establish worker-phase coverage. Nerve
backprojection accepts **150/150 and 177/177** sweeps, with maximum sweeps of 437 / 503 ms and
collector CPU of 15.522 / 24.066 s across 149.802 / 176.684 s windows. These CPU
figures describe observer work, not workload slowdown.

The original failing Nerve report had 1,551 reported processes, a maximum sweep
of 18.951 s, 1,132.140 collector CPU-seconds over 1,129.921 s, and 173 complete /
117 incomplete samples. The current counts and sweep costs corroborate removal
of duplicate accounting, but these are **12-worker versus 48-worker** slicing
runs with different durations/hardware. Real slicing's incomplete fraction did
not uniformly improve. No historical values are numerically repaired or replaced
by this comparison, and the controlled-clock result does not imply that real
allocation churn can no longer invalidate a sweep.

**#163 remains open** for follow-up on the observed Long backprojection cost and
an explicit assessment of managed-segment churn and the reported coverage limits
before broad overhead or peak-coverage claims. That follow-up must distinguish
treatment cost from host variation and retain the same timing boundary; this
campaign makes no sampler, interval, metric or scientific-workload changes to
obtain acceptance. The cache
contents were repaired for testing; the code path that created the bad entries
has not been identified or fixed. Native
macOS performance and macOS-host Docker delivery remain deferred under #139.

## Deployment and timing protocol

The CLI harness ran in an isolated Linux Docker container on Windows 11 Enterprise
22631 / WSL2, with an Intel Xeon W-10855M (6 cores, 12 logical CPUs). Docker
Desktop's Linux VM exposed 12 CPUs and 70.67 GiB RAM; the container had no extra
CPU/memory cap and a 64 GiB `/dev/shm`. Python was 3.11.12, psutil 7.1.3, NumPy
1.26.4, SciPy 1.11.2, tifffile 2024.12.12, CloudVolume 12.7.0 and CloudFiles 5.8.2.
All installed versions represented in `poetry.lock` match it. CloudVolume's
metadata says 12.7.0, but its recorded wheel origin and hash match the declared
12.7.0a0 fork. BLAS/OpenMP/MKL threads were limited to one in both treatments.
Existing application containers remained running on this shared development host.

The local image ID is
`sha256:3719d0a81c31dd078d78fed6398c6d2441a2e918fb7db4d6ca7771c3b5ce1ea2`.
It retains the installed dependency environment from local image
`sha256:44f155af7e0ff1971ccce2a70760da78472238984643a8cc442ba561bce8d37d`
and copies the exact fixed production source over it. Neither image has a registry
`RepoDigest`. This was not a fresh release-image certification. See
[provenance](provenance.json), the [campaign Dockerfile](Dockerfile), and the
[container provenance reader](container_provenance.py).

Reporting off omits `--verbose`; reporting on enables statistics and RAM
collection together. Each workload has two pairs in order off → on, then on → off.
Summaries use the existing benchmark `summary()` function, including per-treatment
MAD. **Overhead uses the existing data-step duration**: downloads, computation,
writes, worker/manager lifecycle and cleanup inside the step all count. Earlier
parsing/geometry/source setup, sampler startup/shutdown outside the step, and
final formatting/persistence/delivery do not. The external `Pipeline.process()`
timer is reported separately. Observer CPU time spans the broader sampling window
and is not additive elapsed-time overhead.

Workloads retain the existing scientific configurations:

- **Fixed local:** seed 138, 384³ uint16 source, 192 × 192 slices, four workers,
  one disabled warmup per operation, the existing pipeline benchmark mode.
- **Salivary:** the original three `PosteriorSalivaryDuct_cut_ng.json` paths,
  120 × 120 uint16 slices, adaptive distance 1 / ratio 0.5, 128 target slices
  per box, 12 workers. Short 2 has 10 points / 875 slices; Short 3 has 16 / 887;
  Long has 50 / 4,901. Each treatment backprojects its own intensity TIFF with
  160³ chunks, zlib, minimum bounding box and separate output planes.
- **Nerve:** `Long Nerve Trace` from `NerveCords_ng.json`, 59 points / 10,386 uint16
  slices of 120 × 120,
  the saved source/scientific options and the same 12-worker treatment protocol.
  Each treatment backprojects its own sliced intensity TIFF, as for Salivary.
  Explicit annotation selection leaves the original annotation file unchanged.

Real-data paths use one disabled slicing warmup, a shared CloudVolume cache and
uncleared OS caches; remote metadata requests may remain. No timed trial hashes
output pixels. Every attempt, including failed warmups and dependent skips,
remains in the raw results. The original failure used 48 workers; this 12-CPU,
12-worker campaign is corroborative, not a matched reproduction of that deployment.
No new native Windows/macOS or GUI/server delivery run was performed. Historical
native Linux/Windows comparisons and macOS workflow deferrals remain in the
[parent benchmark report](../../index.md).

## Source-cache failure and separate recovery cohort

The original Salivary Short 2 warmup failed with HTTP 416, although all 16
measured operations for the two short paths completed. That
failed warmup means Short 2 does not establish the intended warm-cache condition.
The original Long warmup and all four measured slices failed gzip decoding; their
four backprojections were skipped. They have no overhead estimate.

**The benchmark cache was repaired; the code path that created the bad entries
has not been identified or fixed.**

An offset is a byte position inside a file. This dataset is stored in large shard
files, each with index blocks locating its contents.
[Forensic evidence](source-cache-diagnostic.json) identifies two bad cached
`11b.shard-*.msi` files. Their bytes exactly match remote shard `11b` read at
positions specified by shard `152`'s index. Reading `11b` at its correct positions
returns valid indexes, and a fresh metadata-only CloudVolume reader succeeds.
All six other cached MSI entries pass gzip and expected-length checks. This
establishes a file/offset mismatch in the cached data; the originating code path
remains unproven. No race or other specific originating cause is established.

CloudVolume saved the incorrect bytes before checking whether they could be
decompressed. Subsequent attempts reused them and repeatedly failed with the same
gzip error, before the malformed indexes could supply image-chunk offsets. This
specific failure does not establish a general proof of image-data integrity.

The original cache and failed cohort were preserved. A separate cache copy
replaced exactly those two entries with the validated remote bytes, recorded in
[cache recovery provenance](cache-recovery.json). The separately named
`salivary-long-validated` cohort gets a full new disabled warmup and both complete
pairs. Its results are not pooled with the failed original cohort. Nerve follows
using that cache. No production code, dependency, source volume, or annotation
was patched to obtain these results.

"Cold-cache" describes the circumstances in which the bad entries were encountered;
it does not establish that an empty cache alone triggers the failure. Because the
originating code path remains unresolved, the problem could recur when uncached
indexes are downloaded. This is a data-loading/cache finding discovered during
the RAM campaign; there is no evidence that RAM collection caused it.

## Canonical data-step summaries

| Cohort / path | Operation | Off median s | On median s | Delta s | Change % | Off / on MAD s | Pair deltas s | Pair changes % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-local / Fixed local fixture | slice | 8.781005 | 9.237414 | 0.456409 | 5.197681 | 0.140387 / 0.114794 | 0.201227, 0.711590 | 2.255561, 8.235404 |
| fixed-local / Fixed local fixture | backproject | 5.854895 | 5.730546 | -0.124349 | -2.123853 | 0.041594 / 0.140005 | 0.057250, -0.305949 | 0.984816, -5.188664 |
| nerve / Long Nerve Trace | slice | 31.719848 | 26.852296 | -4.867552 | -15.345446 | 5.577276 / 0.348343 | 1.058067, -10.793171 | 4.047293, -28.938347 |
| nerve / Long Nerve Trace | backproject | 176.922093 | 162.380396 | -14.541696 | -8.219265 | 24.131323 / 13.446140 | -3.856513, -25.226880 | -2.524049, -12.547352 |
| salivary / Posterior Salivary Duct 2 | slice | 3.604959 | 3.249789 | -0.355170 | -9.852271 | 0.160511 / 0.316726 | -0.511385, -0.198956 | -14.846650, -5.283684 |
| salivary / Posterior Salivary Duct 2 | backproject | 2.537026 | 2.508138 | -0.028889 | -1.138686 | 0.153913 / 0.147190 | 0.272214, -0.329991 | 11.422606, -12.263049 |
| salivary / Posterior Salivary Duct 3 | slice | 2.748314 | 2.832232 | 0.083919 | 3.053460 | 0.029722 / 0.041032 | 0.013165, 0.154672 | 0.473898, 5.689426 |
| salivary / Posterior Salivary Duct 3 | backproject | 2.315126 | 2.438078 | 0.122952 | 5.310830 | 0.047437 / 0.095033 | -0.019517, 0.265422 | -0.826091, 11.704500 |
| salivary / Posterior Salivary Duct | slice | incomplete pairs | — | — | — | — | — | — |
| salivary / Posterior Salivary Duct | backproject | incomplete pairs | — | — | — | — | — | — |
| salivary-long-validated / Posterior Salivary Duct | slice | 12.457339 | 12.476935 | 0.019596 | 0.157303 | 0.073576 / 0.172273 | -0.226254, 0.265445 | -1.805563, 2.143494 |
| salivary-long-validated / Posterior Salivary Duct | backproject | 14.772077 | 15.728734 | 0.956657 | 6.476120 | 0.363892 / 0.565084 | 0.027681, 1.885634 | 0.182883, 13.087241 |

## External pipeline summaries (separate boundary)

| Cohort / path | Operation | Off median s | On median s | Delta s | Change % | Off / on MAD s | Pair deltas s | Pair changes % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-local / Fixed local fixture | slice | 8.831604 | 9.290229 | 0.458625 | 5.192992 | 0.137813 / 0.116874 | 0.203937, 0.713312 | 2.273689, 8.204849 |
| fixed-local / Fixed local fixture | backproject | 5.906702 | 5.790074 | -0.116627 | -1.974493 | 0.039232 / 0.147920 | 0.070524, -0.303779 | 1.201957, -5.109025 |
| nerve / Long Nerve Trace | slice | 32.981054 | 27.758425 | -5.222629 | -15.835239 | 5.635513 / 0.356757 | 0.769641, -11.214899 | 2.814503, -29.041677 |
| nerve / Long Nerve Trace | backproject | 177.838101 | 163.242437 | -14.595664 | -8.207276 | 24.212963 / 13.440854 | -3.823556, -25.367773 | -2.488887, -12.555130 |
| salivary / Posterior Salivary Duct 2 | slice | 3.789457 | 3.428439 | -0.361018 | -9.526895 | 0.192303 / 0.304959 | -0.473674, -0.248361 | -13.168032, -6.237462 |
| salivary / Posterior Salivary Duct 2 | backproject | 2.744761 | 2.680135 | -0.064627 | -2.354544 | 0.164924 / 0.152617 | 0.252914, -0.382167 | 9.803488, -13.134318 |
| salivary / Posterior Salivary Duct 3 | slice | 2.920185 | 3.022392 | 0.102207 | 3.500006 | 0.027161 / 0.043776 | 0.031269, 0.173144 | 1.060937, 5.984874 |
| salivary / Posterior Salivary Duct 3 | backproject | 2.548784 | 2.612204 | 0.063419 | 2.488224 | 0.092584 / 0.089462 | -0.118627, 0.245466 | -4.491128, 9.993739 |
| salivary / Posterior Salivary Duct | slice | incomplete pairs | — | — | — | — | — | — |
| salivary / Posterior Salivary Duct | backproject | incomplete pairs | — | — | — | — | — | — |
| salivary-long-validated / Posterior Salivary Duct | slice | 12.912314 | 12.944759 | 0.032445 | 0.251269 | 0.084283 / 0.175344 | -0.227183, 0.292072 | -1.748016, 2.276825 |
| salivary-long-validated / Posterior Salivary Duct | backproject | 15.242203 | 16.318097 | 1.075893 | 7.058648 | 0.347804 / 0.719750 | 0.008339, 2.143447 | 0.053492, 14.390963 |

## Every operation attempt

Warmups are excluded from summaries. Missing durations stay missing, including failed and dependent skipped trials.

| Cohort | Label | Phase | Reporting | Status | Data step s | Pipeline s |
|---|---|---|---|---|---:|---:|
| fixed-local | slice-warmup-0 | warmup | off | succeeded | 9.872730 | 9.940157 |
| fixed-local | backproject-warmup-0 | warmup | off | succeeded | 6.770284 | 6.819431 |
| fixed-local | slice-00-0 | measured | off | succeeded | 8.921393 | 8.969418 |
| fixed-local | slice-00-1 | measured | on | succeeded | 9.122620 | 9.173354 |
| fixed-local | slice-01-1 | measured | on | succeeded | 9.352207 | 9.407103 |
| fixed-local | slice-01-0 | measured | off | succeeded | 8.640618 | 8.693791 |
| fixed-local | backproject-00-0 | measured | off | succeeded | 5.813301 | 5.867470 |
| fixed-local | backproject-00-1 | measured | on | succeeded | 5.870551 | 5.937995 |
| fixed-local | backproject-01-1 | measured | on | succeeded | 5.590540 | 5.642154 |
| fixed-local | backproject-01-0 | measured | off | succeeded | 5.896489 | 5.945934 |
| nerve | path-0-warmup | warmup | off | succeeded | 105.372422 | 106.224567 |
| nerve | path-0-pair-1-0-slice | measured | off | succeeded | 26.142573 | 27.345541 |
| nerve | path-0-pair-1-0-backproject | measured | off | succeeded | 152.790769 | 153.625138 |
| nerve | path-0-pair-1-1-slice | measured | on | succeeded | 27.200639 | 28.115182 |
| nerve | path-0-pair-1-1-backproject | measured | on | succeeded | 148.934256 | 149.801583 |
| nerve | path-0-pair-2-1-slice | measured | on | succeeded | 26.503953 | 27.401668 |
| nerve | path-0-pair-2-1-backproject | measured | on | succeeded | 175.826537 | 176.683291 |
| nerve | path-0-pair-2-0-slice | measured | off | succeeded | 37.297124 | 38.616567 |
| nerve | path-0-pair-2-0-backproject | measured | off | succeeded | 201.053416 | 202.051064 |
| salivary | path-0-warmup | warmup | off | failed_trial | — | — |
| salivary | path-0-pair-1-0-slice | measured | off | succeeded | 3.444448 | 3.597154 |
| salivary | path-0-pair-1-0-backproject | measured | off | succeeded | 2.383114 | 2.579838 |
| salivary | path-0-pair-1-1-slice | measured | on | succeeded | 2.933063 | 3.123480 |
| salivary | path-0-pair-1-1-backproject | measured | on | succeeded | 2.655327 | 2.832752 |
| salivary | path-0-pair-2-1-slice | measured | on | succeeded | 3.566515 | 3.733399 |
| salivary | path-0-pair-2-1-backproject | measured | on | succeeded | 2.360948 | 2.527518 |
| salivary | path-0-pair-2-0-slice | measured | off | succeeded | 3.765470 | 3.981760 |
| salivary | path-0-pair-2-0-backproject | measured | off | succeeded | 2.690939 | 2.909685 |
| salivary | path-1-warmup | warmup | off | succeeded | 6.692638 | 6.851906 |
| salivary | path-1-pair-1-0-slice | measured | off | succeeded | 2.778035 | 2.947346 |
| salivary | path-1-pair-1-0-backproject | measured | off | succeeded | 2.362562 | 2.641369 |
| salivary | path-1-pair-1-1-slice | measured | on | succeeded | 2.791200 | 2.978615 |
| salivary | path-1-pair-1-1-backproject | measured | on | succeeded | 2.343045 | 2.522742 |
| salivary | path-1-pair-2-1-slice | measured | on | succeeded | 2.873264 | 3.066168 |
| salivary | path-1-pair-2-1-backproject | measured | on | succeeded | 2.533111 | 2.701666 |
| salivary | path-1-pair-2-0-slice | measured | off | succeeded | 2.718592 | 2.893024 |
| salivary | path-1-pair-2-0-backproject | measured | off | succeeded | 2.267689 | 2.456200 |
| salivary | path-2-warmup | warmup | off | failed_trial | — | — |
| salivary | path-2-pair-1-0-slice | measured | off | failed_trial | — | — |
| salivary | path-2-pair-1-0-backproject | measured | off | skipped | — | — |
| salivary | path-2-pair-1-1-slice | measured | on | failed_trial | — | — |
| salivary | path-2-pair-1-1-backproject | measured | on | skipped | — | — |
| salivary | path-2-pair-2-1-slice | measured | on | failed_trial | — | — |
| salivary | path-2-pair-2-1-backproject | measured | on | skipped | — | — |
| salivary | path-2-pair-2-0-slice | measured | off | failed_trial | — | — |
| salivary | path-2-pair-2-0-backproject | measured | off | skipped | — | — |
| salivary-long-validated | path-0-warmup | warmup | off | succeeded | 46.939174 | 47.400840 |
| salivary-long-validated | path-0-pair-1-0-slice | measured | off | succeeded | 12.530916 | 12.996597 |
| salivary-long-validated | path-0-pair-1-0-backproject | measured | off | succeeded | 15.135969 | 15.590007 |
| salivary-long-validated | path-0-pair-1-1-slice | measured | on | succeeded | 12.304662 | 12.769414 |
| salivary-long-validated | path-0-pair-1-1-backproject | measured | on | succeeded | 15.163650 | 15.598347 |
| salivary-long-validated | path-0-pair-2-1-slice | measured | on | succeeded | 12.649208 | 13.120103 |
| salivary-long-validated | path-0-pair-2-1-backproject | measured | on | succeeded | 16.293818 | 17.037847 |
| salivary-long-validated | path-0-pair-2-0-slice | measured | off | succeeded | 12.383763 | 12.828031 |
| salivary-long-validated | path-0-pair-2-0-backproject | measured | off | succeeded | 14.408184 | 14.894400 |

## Every enabled report

C/I/D = complete/incomplete/discarded sweeps. Counts and peaks derive from accepted sweeps; low counts can reflect missed worker-phase coverage. CPU is observer CPU time, not workload slowdown.

| Cohort | Label | Outcome | RAM status | Peak MiB | Max processes | C/I/D | Requested s | Observed min / max s | Max sweep ms | CPU s | Window s | Errors |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| fixed-local | slice-00-1 | succeeded | partial | 790.874 | 7 | 9/1/0 | 1.000 | 1.000271 / 1.005842 | 66.154 | 0.524864 | 9.173741 | FileNotFoundError ×1: [Errno 2] No such file or directory: '/psm_05f5b9ae' |
| fixed-local | slice-01-1 | succeeded | partial | 825.531 | 7 | 9/1/0 | 1.000 | 1.000264 / 1.000469 | 61.902 | 0.512859 | 9.407411 | OSError ×1: Manager observations changed during memory sweep |
| fixed-local | backproject-00-1 | succeeded | sampled | 487.446 | 6 | 6/0/0 | 1.000 | 1.000242 / 1.000365 | 44.654 | 0.254811 | 5.938279 | none |
| fixed-local | backproject-01-1 | succeeded | sampled | 491.402 | 6 | 6/0/0 | 1.000 | 1.000205 / 1.000355 | 41.202 | 0.244516 | 5.642384 | none |
| nerve | path-0-pair-1-1-slice | succeeded | partial | 1893.071 | 15 | 8/21/0 | 1.000 | 1.000242 / 1.002008 | 188.221 | 3.901276 | 28.115472 | FileNotFoundError ×16: [Errno 2] No such file or directory: '/psm_44763101'; NoSuchProcess ×1: process no longer exists (pid=110326); OSError ×4: Manager observations changed during memory sweep |
| nerve | path-0-pair-1-1-backproject | succeeded | sampled | 2424.739 | 14 | 150/0/0 | 1.000 | 1.000108 / 1.004761 | 437.096 | 15.521690 | 149.801894 | none |
| nerve | path-0-pair-2-1-slice | succeeded | partial | 2068.054 | 15 | 16/12/0 | 1.000 | 1.000023 / 1.003422 | 229.096 | 4.026687 | 27.401936 | FileNotFoundError ×3: [Errno 2] No such file or directory: '/psm_300e16d7'; OSError ×9: Manager observations changed during memory sweep |
| nerve | path-0-pair-2-1-backproject | succeeded | sampled | 2246.466 | 14 | 177/0/0 | 1.000 | 1.000067 / 1.015984 | 502.712 | 24.066265 | 176.683612 | none |
| salivary | path-0-pair-1-1-slice | succeeded | partial | 1343.079 | 15 | 3/1/0 | 1.000 | 1.000288 / 1.000417 | 134.800 | 0.384899 | 3.123695 | OSError ×1: Manager observations changed during memory sweep |
| salivary | path-0-pair-1-1-backproject | succeeded | partial | 635.469 | 14 | 2/1/0 | 1.000 | 1.000359 / 1.000462 | 123.339 | 0.252351 | 2.832994 | OSError ×1: Process tree changed during memory sweep |
| salivary | path-0-pair-2-1-slice | succeeded | partial | 1008.281 | 15 | 2/2/0 | 1.000 | 1.000335 / 1.006804 | 292.263 | 0.641104 | 3.733755 | FileNotFoundError ×1: [Errno 2] No such file or directory: '/psm_9f91b7b8'; OSError ×1: Manager observations changed during memory sweep |
| salivary | path-0-pair-2-1-backproject | succeeded | partial | 636.170 | 14 | 2/1/0 | 1.000 | 1.000271 / 1.000360 | 128.886 | 0.253488 | 2.527777 | OSError ×1: Process tree changed during memory sweep |
| salivary | path-1-pair-1-1-slice | succeeded | partial | 183.979 | 2 | 1/2/0 | 1.000 | 1.000369 / 1.000370 | 143.874 | 0.294494 | 2.978897 | FileNotFoundError ×1: [Errno 2] No such file or directory: '/psm_69e2df92'; OSError ×1: Manager observations changed during memory sweep |
| salivary | path-1-pair-1-1-backproject | succeeded | sampled | 1234.995 | 14 | 3/0/0 | 1.000 | 1.000273 / 1.000406 | 115.816 | 0.244369 | 2.523001 | none |
| salivary | path-1-pair-2-1-slice | succeeded | partial | 185.445 | 2 | 1/3/0 | 1.000 | 1.000277 / 1.000433 | 192.380 | 0.408568 | 3.066620 | OSError ×2: Manager observations changed during memory sweep; ProcessLookupError ×1: [Errno 3] No such process: '/proc/13030/smaps_rollup' |
| salivary | path-1-pair-2-1-backproject | succeeded | sampled | 1240.196 | 14 | 3/0/0 | 1.000 | 1.000277 / 1.000372 | 114.713 | 0.242687 | 2.701982 | none |
| salivary | path-2-pair-1-1-slice | failed | partial | 185.062 | 2 | 1/1/0 | 1.000 | 1.000261 / 1.000261 | 22.791 | 0.047736 | 1.017157 | ProcessLookupError ×1: [Errno 3] No such process: '/proc/13515/smaps_rollup' |
| salivary | path-2-pair-1-1-backproject | skipped | no report | — | — | —/—/— | — | — / — | — | — | — | none |
| salivary | path-2-pair-2-1-slice | failed | partial | 184.752 | 2 | 1/1/0 | 1.000 | 1.000316 / 1.000316 | 21.096 | 0.033275 | 1.012036 | ProcessLookupError ×1: [Errno 3] No such process: '/proc/13552/smaps_rollup' |
| salivary | path-2-pair-2-1-backproject | skipped | no report | — | — | —/—/— | — | — / — | — | — | — | none |
| salivary-long-validated | path-0-pair-1-1-slice | succeeded | partial | 1298.645 | 15 | 3/10/0 | 1.000 | 1.000299 / 1.000472 | 155.163 | 1.663375 | 12.769848 | FileNotFoundError ×6: [Errno 2] No such file or directory: '/psm_a11e470d'; OSError ×4: Manager observations changed during memory sweep |
| salivary-long-validated | path-0-pair-1-1-backproject | succeeded | sampled | 2075.643 | 14 | 16/0/0 | 1.000 | 1.000284 / 1.001533 | 487.291 | 2.275208 | 15.598886 | none |
| salivary-long-validated | path-0-pair-2-1-slice | succeeded | partial | 1207.062 | 15 | 7/7/0 | 1.000 | 1.000214 / 1.002094 | 235.375 | 1.931725 | 13.120475 | FileNotFoundError ×4: [Errno 2] No such file or directory: '/psm_c63e7192'; NoSuchProcess ×1: process no longer exists (pid=46503); OSError ×2: Manager observations changed during memory sweep |
| salivary-long-validated | path-0-pair-2-1-backproject | succeeded | partial | 2006.921 | 14 | 17/1/0 | 1.000 | 1.000277 / 1.001246 | 437.776 | 2.507177 | 17.038165 | NoSuchProcess ×1: process no longer exists (pid=46832) |

## Failed or skipped attempts

| Cohort | Label | Status | Error / dependency |
|---|---|---|---|
| salivary | path-0-warmup | failed_trial | Pipeline Error: Error during downloading and slicing: 416 Client Error: Requested Range Not Satisfiable for url: https://neuroglancer.hpc.psu.edu/assets/precomputed_repository/10mm/2BM/A3/Octo9_Merge_Pass4_Pass2_cleaned_16bit_precomputed_sharded_2/700_700_700/0e0.shard |
| salivary | path-2-warmup | failed_trial | Pipeline Error: Error during downloading and slicing: File is not in gzip format. Magic numbers 0x95, 0x4b did not match 0x1f, 0x8b. |
| salivary | path-2-pair-1-0-slice | failed_trial | Pipeline Error: Error during downloading and slicing: File is not in gzip format. Magic numbers 0x95, 0x4b did not match 0x1f, 0x8b. |
| salivary | path-2-pair-1-0-backproject | skipped | Matching slicing trial did not complete |
| salivary | path-2-pair-1-1-slice | failed_trial | Error during downloading and slicing: File is not in gzip format. Magic numbers 0x95, 0x4b did not match 0x1f, 0x8b. |
| salivary | path-2-pair-1-1-backproject | skipped | Matching slicing trial did not complete |
| salivary | path-2-pair-2-1-slice | failed_trial | Error during downloading and slicing: File is not in gzip format. Magic numbers 0x95, 0x4b did not match 0x1f, 0x8b. |
| salivary | path-2-pair-2-1-backproject | skipped | Matching slicing trial did not complete |
| salivary | path-2-pair-2-0-slice | failed_trial | Pipeline Error: Error during downloading and slicing: File is not in gzip format. Magic numbers 0x95, 0x4b did not match 0x1f, 0x8b. |
| salivary | path-2-pair-2-0-backproject | skipped | Matching slicing trial did not complete |

## Controlled-clock identity evidence

The separate probe completed 64 of 64 sweeps with 0 failures; `passed: true`. All reported, retained, unique-PID and unique-identity counts remained 51 (one root plus 50 real children). Fresh wrappers matched retained identities while public creation timestamps changed forward/backward and within sweeps.

Early / late median sweep time: 16.499633 / 16.808487 ms. Early / late median CPU: 16.495450 / 16.804750 ms. This is diagnostic evidence against duplicate-identity growth, not an operation-overhead acceptance measurement.


## Reproduction and validation

Run the existing harness in a matching dependency environment; use new output
folders so failed attempts cannot be overwritten. The recorded container commands
were (the `/campaign/inputs` files are copies of the saved options/annotations,
with path adapters only; hashes and complete options are in the raw evidence):

```sh
docker exec ouroboros-163 python /app/benchmarks/benchmark_run_statistics.py \
  --scope pipeline --size 384 --workers 4 --pairs 2 --timeout 3600 \
  --work-dir /campaign/results/fixed-local

docker exec ouroboros-163 python /app/benchmarks/benchmark_run_statistics.py \
  --real-slice-options /campaign/inputs/salivary-slice-options.json \
  --real-backproject-options /campaign/inputs/backproject-options.json \
  --cache-dir /campaign/cache --workers 12 --pairs 2 --timeout 3600 \
  --work-dir /campaign/results/salivary

docker exec ouroboros-163 python /app/benchmarks/benchmark_run_statistics.py \
  --real-slice-options /campaign/inputs/salivary-slice-options.json \
  --real-backproject-options /campaign/inputs/backproject-options.json \
  --annotation-layer 'Posterior Salivary Duct' \
  --cache-dir /campaign/cache-validated --workers 12 --pairs 2 --timeout 3600 \
  --work-dir /campaign/results/salivary-long-validated

docker exec ouroboros-163 python /app/benchmarks/benchmark_run_statistics.py \
  --real-slice-options /campaign/inputs/nerve-slice-options.json \
  --real-backproject-options /campaign/inputs/backproject-options.json \
  --annotation-layer 'Long Nerve Trace' \
  --cache-dir /campaign/cache-validated --workers 12 --pairs 2 --timeout 3600 \
  --work-dir /campaign/results/nerve

docker exec -e PYTHONPATH=/app:/app/benchmarks ouroboros-163 \
  python /campaign/probe_memory_identity.py \
  --output /campaign/identity-clock.json --mid-sweep-shift

docker exec ouroboros-163 python -m pytest \
  /app/test/common/test_memory_identity.py -q -p no:cacheprovider
```

The native identity checks passed **26 tests** inside the measured image. The
first test invocation emitted a pytest cache-permission warning because UID 1000
could not write `/app/.pytest_cache`; rerunning with `-p no:cacheprovider` passed
without warnings. Both invocation outcomes are retained in provenance. This is
not another implementation of #162's platform-contract tests.

The controlled-clock probe uses production `TreeSampler` to observe a separate
root and 50 waiting children. It changes only psutil's boot-time conversion,
including a shift after the first native memory read, while the OS clock remains
untouched. Canonical `psutil.Process` identities and `export_process_identity`
supply the evidence; no PID/timestamp identity algorithm is reimplemented. The
observer and its growing JSON ledger live outside the measured tree, and evidence
serialization is outside the timed sweep. The probe demonstrates repeated
rediscovery under injected clock movement, not long-workload overhead or managed
buffer coverage. Timed operation reports retain aggregate counts, not per-sweep
PID traces; the merged production unique-PID guard and separate identity probe
are the corresponding uniqueness evidence.

The new `--annotation-layer` option only selects which existing annotation layers
the benchmark schedules. It preserves input bytes and records/checks selection
on resume. The existing pipeline timers own operation durations, `RunStatistics`
and `MemoryCollector` own lifecycle/reporting, `TreeSampler` owns process discovery
and RAM reads, and the existing benchmark `summary()` owns the timing reduction.
No duplicate production writer, carrier, calculation or sampler was added or
removed. Exported identities, JSON reports and tables are read-only evidence
adapters, not another runtime authority.

Postflight verified unchanged production source, dependency resolution and harness
fingerprints, and an empty `/dev/shm`. Across all cohorts, 40 measured operations
succeeded, four measured slices failed and four dependent backprojections were
skipped. Five of seven excluded warmups succeeded; both failed warmups remain.

## Raw evidence

The following are byte-for-byte copies of the completed raw campaign outputs.
Provenance combines the recorded preflight with postflight checks and records the
other artifact hashes. Data-step and pipeline table summaries both use the existing
benchmark `summary()` function on each raw operation group; no failed group is
assigned an overhead estimate. `measured_median_at_most_one_percent` in raw JSON
is an arithmetic comparison, not an engineering acceptance verdict.

| Artifact | SHA-256 |
| --- | --- |
| [fixed-local.json](fixed-local.json) | `8d061202926d726d5111206805bbdf446a18ffc744a6e3afe61fe1e43e676f92` |
| [salivary-original.json](salivary-original.json) | `a0a6f28ce7db17bbba60a0aab995eea39d836fe6d39799a1ac6c2ab8d4376b88` |
| [salivary-long-validated.json](salivary-long-validated.json) | `563b2beb04e2b0e30085c14a42bbb4504b985b956f0a9ba7e845dae1930c5a37` |
| [nerve.json](nerve.json) | `c9b105960b8a6a97ee9870281507804bc8f05e983e42906afa1d4fb2f00462ce` |
| [identity-clock.json](identity-clock.json) | `b115280588b9222b5db0dc3397f768692f0905cb9023246c229dc6527eb5806e` |
| [source-cache-diagnostic.json](source-cache-diagnostic.json) | `65e9695f88caad1797698f3f7ea2815cc3141d631716bb71785a4f71d2077f57` |
| [cache-recovery.json](cache-recovery.json) | `7e921de0c6a34a5cab0bbb198b31ebe96f269184a361a33b3a9fd76e9976b4f3` |
| [provenance.json](provenance.json) | `45f15afe2d386a6946ff1dd3a0941b674ea770d1a8d3dd16095fc5fdde2fea8a` |

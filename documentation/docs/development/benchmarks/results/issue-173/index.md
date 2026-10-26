# Native download dtype and fresh shared buffers (#173)

This campaign compares baseline `fbf4297` with implementation
`138e79db923d819bc55520f3127fcc67e06699e3`. Downloads now retain the source
dtype, fresh shared segments receive no explicit zero-fill, and interpolation
explicitly returns float32. Allocation ownership, interpolation settings,
normalization, output writing, and RAM accounting are unchanged.

## Workloads and method

Both workloads use uint16 source pixels and four workers. The synthetic fixture
reuses the existing seed-138 generator at 256³, producing 387 slices of 128².
The representative fixture is a 480³ Octo9 crop, bounds
`[5760,6240,2688]`–`[6240,6720,3168]`, covering the first six annotation points
of `Posterior Salivary Duct 2`. Its original adaptive spacing and normalization
settings produce 392 slices of 120². Source pixels, annotation provenance, options,
dependency versions, and production-file hashes are retained in the raw results.
The synthetic output disables normalization; the crop retains normalization and
zero guarding, covering both output paths. Only `slice.py` and `volume_cache.py`
differ between the measured production trees.
Synthetic download geometry extends beyond the source's high-Z boundary, so its
matching output also exercises zero padding; the real crop contains its requests.

Each revision receives one excluded warmup, then three measured runs per workload,
with order baseline → modified, modified → baseline, baseline → modified.
Reporting and RAM collection are enabled throughout; BLAS/OpenMP use one thread.
The campaign calls the existing `benchmark_run_statistics.run_once` with pipeline
scope and reads the existing `SliceParallelPipelineStep` duration. That timer
includes download, slicing, writing, and worker/manager startup and shutdown in
the step. The broader pipeline duration and collector window remain in the raw
results. Pixel hashing uses the existing `tiff_evidence` after timing; semantic
metadata hashing excludes only TIFF strip/tile offsets and byte counts.

Docker ran Linux through Docker Desktop on Windows/WSL2, Python 3.11.12,
with local HTTP sources and WSL-backed caches/output. Native Windows ran Python
3.11.13 with identical raw chunks copied to NTFS, a `file://` source, and NTFS
caches/output. Both used NumPy 1.26.4, SciPy 1.11.2, tifffile 2024.12.12, and
psutil 7.1.3. Compare revisions within each platform: these storage and process
startup differences prevent treating the platforms as matched timing treatments.

These are warm, bounded local-source measurements. The real crop was downloaded
and repackaged into raw chunks before timing; these are not measurements of remote
shard downloads, the complete historical annotation paths, GUI/server delivery,
or replacements for manuscript or historical campaign values.

## Timing and sampled RAM

Times are data-step medians, with median absolute deviation (MAD) in parentheses.
RAM is the median of each run's sampled peak, followed by the observed range.
Three runs per treatment on a shared development host do not establish a speedup
or a statistical bound on performance.

| Platform / workload | Baseline s (MAD) | Modified s (MAD) | Change | Baseline peak MiB (range) | Modified peak MiB (range) |
| --- | --- | --- | --- | --- | --- |
| Docker / synthetic | 3.638 (0.089) | 3.678 (0.081) | +1.09% | 400.15 (367.83–412.10) | 328.14 (143.52–346.00) |
| Docker / Octo9 crop | 2.484 (0.157) | 2.286 (0.012) | −7.97% | 490.45 (486.47–491.24) | 449.86 (446.90–450.28) |
| Windows / synthetic | 16.244 (0.639) | 15.564 (0.018) | −4.18% | 647.59 (628.36–659.80) | 632.37 (627.26–632.47) |
| Windows / Octo9 crop | 13.483 (0.258) | 13.200 (0.042) | −2.10% | 772.48 (684.78–778.81) | 649.36 (624.01–738.26) |

Linux reports `process_tree_nonshmem_pss_plus_managed_resident_bytes`;
Windows reports `process_tree_private_resident_plus_managed_allocated_bytes`.
The former counts observed managed residency; the latter counts managed allocation
size, including untouched pages. Cross-platform peaks therefore have different
semantics. No historical RAM values were carried forward.

Every Docker crop run has three complete samples and no incomplete/discarded
samples. Synthetic baseline runs have complete/incomplete counts `3/2`, `3/1`,
`4/0`; modified runs have `3/2`, `2/2`, `1/3`, all with zero discarded samples.
The last modified synthetic run observed at most two processes in complete sweeps,
missing its worker peak; its 143.52 MiB result is retained, not evidence of that
workload's true peak. One-second sampling and short operations can miss peaks even
when every sweep is complete. Partial reports retain process-exit and changing
manager-observation errors in the raw evidence.

Windows synthetic complete/incomplete counts are `16/1`, `16/1`, `15/1` before
and `15/1`, `15/1`, `14/1` after; crop counts are `14/0`, `13/1`, `13/2` before
and `12/2`, `13/1`, `13/1` after. No samples were discarded. All but one report
are partial because the process tree changed during a sweep.

## Allocation and numerical evidence

The allocation probe requests a 256³ single-channel volume with a clipped 192³
write. It reads residency through the existing managed-memory facility before
inspecting background pixels. Two successive shared allocations have distinct
names and retain zero background in both revisions.

| Docker buffer | Baseline dtype / size | Modified dtype / size |
| --- | --- | --- |
| Local padded buffer | float64 / 128 MiB | uint16 / 32 MiB |
| Shared padded buffer | float32 / 64 MiB | uint16 / 32 MiB |
| Shared residency before inspection | 64 MiB | 18 MiB |

Native Windows confirms the same buffer dtypes/sizes, distinct fresh names, and
zero backgrounds; no Windows managed-residency value is claimed.
These byte reductions apply to uint16; native 64-bit sources require eight bytes
per voxel instead of the former shared float32 buffer's four.
Untouched padding pages explain why resident bytes differ from allocated bytes;
the probe is an observation of these particular buffers, not a whole-pipeline
peak estimate. Focused tests observe the actual manager-owned mappings, verify
exact allocation size, reject any pre-download/full-buffer write, and verify
source values, clipped zero background, attachment release, and failure cleanup.

Every measured uint16 output has exactly matching pixel and semantic TIFF-metadata
hashes across revisions and platforms for its workload. Both channel
and non-channel interpolation return float32 without first copying the input to
float32. Tests cover all precomputed source types (uint8/16/32/64 and float32),
plus signed integers and float64 interpolation. Wide integers and float64 may
differ from the former shared path because source precision now survives until
interpolation; bit-for-bit baseline equivalence is not claimed for those types.

Focused fixtures passed on Linux, native Windows, and Linux Docker (63 tests
each). The full Python suite passed with 469 tests, two skips, and 99% helper
coverage. No new production warning was introduced.

## Native Windows setup findings

Preflight attempts exposed UNC cache-locking failures, an illegal colon in an
HTTP-port cache path, denied port-80 binding, and CloudFiles' zero-width transfer
interval failure. The completed route uses NTFS-local inputs/output and the same
isolated [CloudFiles interval workaround](../issue-139-cloudfiles-zero-interval.patch)
previously recorded for #139, applied equally to both revisions. Campaign-local
psutil 7.1.3 and the CloudFiles copy leave shipped and global dependencies unchanged.
These setup failures are retained separately from the measured trials.

## Evidence

- [Docker runs, warmups, options, hashes, and memory reports](docker-results.json)
- [Native Windows runs, warmups, options, hashes, and memory reports](windows-results.json)
- Docker allocation probes: [baseline](docker-allocation-baseline.json), [modified](docker-allocation-modified.json)
- Native Windows allocation probes: [baseline](windows-allocation-baseline.json), [modified](windows-allocation-modified.json)
- [Windows workaround provenance](windows-workaround.json)
- [Environment, revisions, and hashes](provenance.json), [commands](commands.json)
- [Failed preflight reports and stderr](preflights.json)

Reproduction drivers preserve the existing production timers and memory collector:
[fixture preparation](prepare.py), [paired campaign](compare.py), and
[allocation probe](allocation_probe.py). With both revisions and the prior #163
annotation inputs available, use a fresh fixture/output directory:

```sh
python prepare.py --root FIXTURES --baseline BASELINE_REPO --prior-campaign ISSUE163_DIR
python compare.py --baseline BASELINE_REPO --modified MODIFIED_REPO --root FIXTURES --output-root RESULTS --source-root SOURCE_URL --platform-label PLATFORM
python allocation_probe.py MODIFIED_REPO SYNTHETIC_SOURCE_URL allocation.json
```

Use the platform's interpreter and source/cache arrangement described above;
Windows also requires the disclosed campaign-local dependency setup.

# Recorded partitioning geometry

These fixtures contain actual annotation points from the Daphnia antennae,
octopus posterior salivary duct, and octopus nerve-cord runs recorded in
`table1-48cpu-20261026`. Each JSON retains only the selected image layer and
annotation points, in their original order and precision. No specimen voxels
are included.

`manifest.json` records source-file and saved-geometry SHA256 hashes, the
original generation revision and dependency versions, all geometry settings,
and the selected source scale's metadata. The original saved archives contained
2,656, 4,901, and 10,386 rectangles respectively. Keeping the annotations uses
about 37 KB instead of storing 1.5 MB of derived geometry archives.

The benchmark regenerates all rectangles through the production
`geometry_pipeline`. Annotation and output mip are both zero, so this path
requires no metadata requests or voxel downloads. It reports whether each
regenerated rectangle array exactly matches the saved archive's array hash;
the hash is provenance, not a tolerance for changing scientific output.

From `python/`, with project dependencies installed:

```sh
python benchmarks/benchmark_partitioning.py --repeats 2 --output /tmp/partitioning.json
pytest test/helpers/test_partitioning_geometry.py
```

A recorded comparison, including a separately scoped native run on a cached
specimen crop, is in
[`benchmarks/results/bounding-box-partitioning.json`](../../../benchmarks/results/bounding-box-partitioning.json).
The report identifies its source revision, implementation hashes, and measurement
limits; rerun the benchmark when either the partitioner or worker allocations change.

Use `--specimen octopus_nerve_cords` to select one case. The report includes the
unchanged count reference, the production payload partition, allocation
components, requested source bytes, slice-count distributions, index runs, and
source-clipped chunk coverage. Requested-byte comparisons use the full
inclusive allocated extent, including out-of-source zero-fill and overlap.
No additional interpolation halo is added. Chunk references
describe geometry before cache reuse; they are not observed network traffic.
Partition wall time excludes geometry generation and report construction and
does not measure slicing runtime or worker memory.

The nerve test gates the requested payload improvement and the declared zero
increase in requested bytes, while checking that the original rectangles are
unchanged and every assignment has its exact corner-derived bounds. The other
two specimens are additional observations, not additional improvement gates.

For output validation, use a bounded set of slices from a local real-data crop:
include reassigned slices and points at different distances from a box face,
and check world coordinates independently of each partition. Compare samples
against a common local source crop at the actual production coordinates, and
report errors by distance from the cutout faces because changing a crop changes
cubic prefilter boundary conditions. A slice's own tight-box result is a
boundary-conditioned diagnostic, not a correctness oracle. Do not adopt an
intensity tolerance from an unrelated synthetic volume. Full-pipeline runtime and process-tree RAM require
separate matched runs with the same real input, cache state, worker count, and
the existing run-statistics collector; this geometry benchmark makes neither
claim.

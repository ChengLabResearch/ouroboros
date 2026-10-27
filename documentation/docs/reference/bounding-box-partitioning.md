# Bounding box partitioning

The slicing pipeline groups slice rectangles into source-volume requests. Its
partitioner seeks to reduce the largest estimated slicing-task payload without
increasing the cumulative requested source volume relative to the reference
partition for the same options. It does not inspect available RAM, schedule
workers by memory usage, or impose a hard task-memory limit.

## Geometry and assignment

Every slice is assigned to exactly one box. A box's bounds enclose all four
corners of every assigned rectangle; centers are used to propose splits, not to
decide how much source data to retain. The requested integer extent runs from
`floor(min)` through `ceil(max)`, inclusive on each axis. Downloads, slice grids,
and backprojection use the same integer origin. Requested regions outside the
source remain zero-filled.

`BoundingBox` owns these integer extents. `download_volume_geometry` converts
them to the half-open requests expected by CloudVolume and computes source
clipping. The cost model uses the full allocated extent, including any
zero-filled region, because the slicing worker receives that full array.

## Slicing-task payload model

For a box, define:

| Symbol | Meaning |
| --- | --- |
| `V` | Number of spatial voxels in the box's inclusive integer extent |
| `n` | Number of slices assigned to the box |
| `H`, `W` | Output height and width from `SliceOptions` |
| `C` | Source channel count from `CloudVolumeInterface` |
| `s` | Source dtype item size in bytes |
| `Q = nHW` | Number of output positions, excluding channels |
| `B = sVC` | Bytes in the allocated source array |

The production download array has a channel axis, including when `C = 1`.
The dominant array payload during slicing is estimated as:

```text
one consecutive run of slice indices: P = B + 12Q + 8V
multiple runs of slice indices:       P = B + 12Q + 8V + 4QC + 4Q
```

The `12Q` term is the float32 XYZ coordinate grid. Cubic interpolation processes
one channel at a time and creates a float64 prefilter array, contributing `8V`.
A consecutive run writes directly into the temporary TIFF mapping. A box whose
slices occupy multiple runs needs a float32 output array (`4QC`) and one
temporary channel result (`4Q`) before copying to that mapping.

The model describes dominant arrays for one slicing task. Process overhead,
small temporary arrays, download buffers, concurrently active tasks, and
resident TIFF or filesystem-cache pages also affect observed memory use. A
payload estimate must not be interpreted as process RSS or total run memory.

The source interface is initialized before partitioning and reused by the
resulting `VolumeCache`. Its channel count and dtype supply the cost calculation;
the output dimensions come directly from `SliceOptions`, not from inferred
rectangle edge lengths.

## Partition selection

The reference partition uses the existing slice-count policy with the requested
`target_slices_per_box` and `max_depth`. Its cumulative allocated source bytes,
`sum(B)`, establish the work allowance. The production policy permits **0%
increase** over that reference. Overlapping requests count repeatedly, and
zero-filled parts of requested boxes count toward the allowance. This quantity
differs from source-clipped bytes, compressed storage size, and network traffic.

The candidate tree is built once. At each node, proposed splits include center
medians and the largest gaps between centers on each spatial axis, a gap in the
assigned slice indices, and a path-order cut minimizing the larger child
payload among valid cuts. Both children must be nonempty, and each child's
integer volume must be at most 95% of the parent's. Child bounds always come
from the full rectangle corners.

The path-cut search first computes prefix and suffix corner bounds. As a cut
moves forward, the prefix box can only grow and the suffix box can only shrink.
Two binary searches locate the interval where both children meet the spatial
rule, using `BoundingBox`'s inclusive volume calculation. Only cuts in that
interval are scored. Every admissible cut is evaluated in its original order,
preserving tie-breaking and payload-callback inputs. This avoids constructing
boxes for every rejected cut without introducing a separate vectorized volume
formula or assuming that task payload is monotone along the path.

Candidates are compared by their largest child payload, with deterministic
tie-breaking. The tree is limited by `max_depth` and by the availability of a
valid split. Different payload thresholds select different leaves from that
same tree. Selection examines thresholds at actual node payloads and the fully
expanded frontier, then chooses the smallest maximum task payload within the
reference work allowance. If the candidates cannot improve that maximum within
the allowance, the reference partition is retained.

The requested-volume allowance is checked for a complete partition. It is not
charged independently at each split: an intermediate pair of boxes may contain
overlap that later subdivisions remove.

Planning evaluates more cuts than the count-based reference and can take
longer, especially for long paths. The tree's geometry and task costs are
reused across the threshold search. Reducing the estimated largest task does
not guarantee faster slicing or a proportional decrease in process-tree RAM.

Planning runs in `VolumeCachePipelineStep`. The later
`SliceParallelPipelineStep` downloads and interpolates the selected boxes,
then normalizes and writes the output. Changing the partition can therefore
change that later stage's elapsed time without moving planning into it.
The payload model does not optimize task execution order or completion time.
With a fixed worker count, a large task dispatched late can extend the final
period when other workers are idle, even if accumulated worker time decreases.
Worker durations overlap and must not be added to, or equated with, the
coordinator's elapsed time.

## Internal box ordering

The slicing step normally submits downloads in partition order. Python callers
can opt into alternating slice-count order when constructing the step:

```python
from ouroboros.pipeline import SliceParallelPipelineStep

step = SliceParallelPipelineStep(alternate_slice_count_order=True)
```

The keyword defaults to `False` and is internal to the Python pipeline; it is
not a CLI, UI, or serialized `SliceOptions` setting. The enabled order sorts
boxes by descending assigned slice count, breaking ties by original box index,
then takes the first, last, next first, next last, and so on. With equal counts,
the largest end takes the lower box index and the smallest end takes the higher
box index. An odd middle box is submitted once.

This changes download submission order. Slicing still starts as downloads
complete, so concurrent tasks need not start or finish in submission order.
`VolumeCache` remains the owner of boxes and slice assignments; the flag changes
neither the partition nor the download/slicing pool sizes, in-flight limit, or
box-release lifecycle. Run-statistics `execution.workers` records `box_order`
as `partition` or `alternating_slice_count` for the slicing step.

Alternating large and small slice counts can change task overlap, resource
contention, and the final period with idle workers. It does not guarantee faster
execution or lower peak memory, and slice count is not a measure of source-box
volume. Keep the default unless a comparison on the intended workload supports
changing it.

## Options, compatibility, and limits

`target_slices_per_box` controls the count-based reference and therefore the
production work allowance. It is not a maximum number of slices in a selected
box. Changing it can change both the reference and the selected partition;
lowering it does not guarantee lower task payload or lower observed RAM.

`max_depth` remains the maximum subdivision depth. Raising it allows additional
candidate splits but cannot overcome the spatial-reduction requirement or the
size of an indivisible slice. A single oblique rectangle can require a large
axis-aligned source volume. Identical rectangles and other nodes without a
valid spatial reduction can also remain together with a large payload.

The public `calculate_bounding_boxes_bsp_link_rects` helper retains its existing
positional arguments and `(bounding_boxes, rect_to_box_map)` return shape.
Geometry-only callers that omit the optional `task_payload` callback retain the
slice-count policy. The slicing pipeline supplies that callback to enable the
payload policy. The serialized `BoundingBoxParams` fields remain
`target_slices_per_box` and `max_depth`; their continued presence does not make
the production slice-count target a hard cap.

## Interpolation and comparison

Partition changes preserve the slice rectangles, raster dimensions, and
interpolation settings. They can still change sampled values. Cubic
prefiltering sees each cutout's boundaries, and float32 grid arithmetic can
round differently for different integer origins. Prefilter effects decay with
distance but have no universal eight-voxel, or other fixed-distance, cutoff.
Their magnitude depends on the source values and local variation.

A reference sampled from an individual slice's tight box has its own boundary
effects. It is not automatically an exact reference for a larger box. Output
comparisons should check world-coordinate agreement separately from filtering
residuals and use the same source data and interpolation coordinates when
isolating the filtering effect. Compare floating-point samples before final
normalization as well as written outputs: normalization uses global extrema
and can propagate a local sampling difference across the output.

The [run-statistics reference](run-statistics.md) describes cumulative allocated
and source-clipped download bytes and measured memory statistics. Those describe
different costs and should be reported separately when evaluating a partition.

## Reproducing geometry comparisons

Keep real specimen annotation states outside the repository. Automated tests
extend the existing generated-geometry and simulated-volume fixtures; real
specimen comparisons are separate benchmark observations. To regenerate
recorded paths through the normal geometry pipeline without downloading
specimen voxels, supply an external input directory. From `python/` in the
project environment, run:

```sh
python benchmarks/benchmark_partitioning.py --fixtures /path/to/local-inputs --repeats 2 --output /tmp/partitioning.json
```

The directory contains local annotation states and `manifest.json`. Its
`specimens` object maps each chosen name to `annotations_file` (a path relative
to the directory), `slice_options` (the existing `SliceOptions` geometry and
bounding-box settings), `source_dtype`, `channels`, and `source_scale` (the
source metadata's `size`, `voxel_offset`, and `chunk_sizes`). Each entry also
has `provenance.reduced_state_sha256` for the annotation file and
`provenance.saved_rectangles_sha256` for the original rectangle array's bytes.
These hashes detect changed inputs and report exact geometry reproduction.
Annotation and output mip levels must match for offline regeneration.

The report includes allocation components, slice-count distributions, index
runs, repeated chunk coverage, deterministic repeated output, and planning
time. Its geometry estimates are separate from the elapsed time and measured
memory of a real slicing run.

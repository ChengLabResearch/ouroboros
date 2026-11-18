# Round-trip accuracy

`ouroboros.metrics.round_trip_accuracy` measures how closely a source volume
survives slicing and backprojection. It is an offline measurement; normal runs
do not retain an extra weight volume or calculate these metrics automatically.

```python
from ouroboros.metrics import round_trip_accuracy

scores = round_trip_accuracy(
    source_xyz, backprojected_xyz, accumulated_weights_xyz,
    source_mask=source_mask_xyz,
    backprojected_mask=weighted_mask_xyz,
    slice_rects=rects_xyz - source_box.get_integer_origin(),
    slice_shape=(slice_height, slice_width),
)
```

All volumes must have the same nonempty **XYZ** shape, source voxel spacing,
origin, and bounding box: crop the source to the backprojected output's box.
Transpose TIFF **ZYX** data with `.transpose(2, 1, 0)` at input. Alignment and
origin are the caller's responsibility; the function does not register images.
Evaluate scalar channels separately. Supply the already averaged reconstruction
and the **raw accumulated weights**, before any output-MIP upsampling. A binary
coverage mask loses the information needed to distinguish interior and boundary.

The existing `backproject_box` returns sparse indices, weighted sums, and raw
weights in flattened ZYX order. Production `process_chunk` stores these in its
intermediate records; `volume_from_intermediates(..., accumulator=...)` merges
and divides the values, leaving the accumulator's last row as the raw weights.
An offline benchmark can capture those weights before the writer resets its
buffers. The pipeline regression test demonstrates this without introducing a
second weighted-merge implementation.

Masks are optional and must be supplied together. Use a separate 0/1 source mask
and its weighted round trip before thresholding. Preserve the mask's 0–1 units
so the 0.5 threshold remains meaningful. For this diagnostic, retain the floating
samples from `slice_volume_from_grids` and pass them directly to `backproject_box`,
then merge its records with `volume_from_intermediates`. This measures the
interpolation/weighted path without normalizing or quantizing the mask. It is
an in-memory diagnostic: the application TIFF backprojection loader accepts
only bool/uint8/uint16, not floating-point TIFFs. The intensity round trip can
use the normal default normalization and supported TIFF path. Do not pass a
normalized integer segmentation (such as 0–254) as a probability mask.

## Returned measurements

| Result | Definition |
| --- | --- |
| `coverage` | Counts and fractions of all bounding-box voxels with weight > 0 and weight ≥ 1. |
| `intensity.covered` | Intensity scores where weight > 0. |
| `intensity.interior` | Scores where weight ≥ 1. This is the requested weight criterion, not a guarantee of geometric completeness. |
| `intensity.boundary` | Scores where 0 < weight < 1. |
| `masks.weighted` | Full-box mask scores after thresholding the weighted mask at ≥ 0.5. |
| `masks.binary` | Full-box mask scores using the same nonzero conversion as `make_backprojection_binary`. |
| `geometry` | Supplied rectangle corners and raster shape, so coverage can be interpreted with the swept geometry. |

Each intensity region reports a count, Pearson correlation, raw RMSE and PSNR,
and a least-squares fit `source = fit_scale * backprojected + fit_offset`.
`nrmse` is the fitted RMSE divided by that region's source peak-to-peak range.
`fitted_psnr` uses fitted RMSE; `psnr` uses raw RMSE. Both use the same source
range. A separate fit per region accounts for rescaling while the raw scores
retain evidence of absolute intensity changes. Compare regions with their
different ranges and fits in mind.

Each mask result reports Dice, IoU, precision, recall, HD95, the XYZ centroid
offset (reconstruction minus source), and its Euclidean length in voxels.
Scores include uncovered source foreground as false negatives. HD95 is the
larger of the two directed 95th-percentile nearest-surface distances, measured
between six-connected foreground boundary voxel centers; outside the box is
background. Distances are in voxel units, not physical units.

With rectangle geometry, each mask result also reports centroid offsets and
lengths for every slice, in trace order. These compare the source and final
reconstructed masks re-sliced on the supplied rectangles using Ouroboros's
cubic sampler, then thresholded at 0.5. This additional diagnostic interpolation
can affect per-slice results. Rectangles must be local to the supplied box, and
the raster shape must match the original sampling; no trace is inferred from
array axes.

Empty intensity regions return `None` for scores. Constant regions have no
Pearson correlation or range-normalized scores. Zero error with nonzero range
gives infinite PSNR. Two empty masks have Dice/IoU 1 and HD95 0; one empty mask
has infinite HD95. Precision/recall with a zero denominator and centroid offsets
with an empty mask/plane are `None`. The result contains Python dictionaries,
lists, and scalars; callers exporting strict JSON must encode infinities.

## Expected losses and scope

- Cubic forward interpolation and trilinear backprojection blur boundaries.
- Coverage is limited to voxels reached by the swept rectangles, within the
  reported bounding box. Changing rectangle size, raster resolution, slice
  spacing, or curvature changes coverage.
- Overlapping rectangles average multiple samples, especially inside bends.
- Default `normalize_output` rescales intensities and dtype conversion can
  quantize them. Affine fitting compensates for rescaling, not clipping or blur.
- Binary output includes any nonzero weighted value. Trilinear splatting can
  spread a nonnegative segmented sample to adjacent voxels, producing roughly
  one-voxel dilation. This is **not a strict bound for a cubic round trip**:
  ringing, normalization, and the production writer's inclusion of negative
  nonzero values can enlarge the mask further. Precision exposes that loss.
- Upsampling to another MIP introduces another interpolation. Compare arrays
  on one common grid; this metric does not convert MIPs or weights for you.

The synthetic sinusoidal-tube regression runs in the existing pipeline test
suite and uses production interpolation, TIFF conversion, accumulation, and
merge code. It is a regression check, not evidence for real-data accuracy.
The Octo9/pass-7/Regions manual-label benchmark, its figure and table, belong
with the paper, importing this function from a pinned package version. This
function works in memory and is intended for bounded regions of interest.

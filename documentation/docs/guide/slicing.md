# Slicing

![Long Sliced Volume](../assets/slicing/long-slice.png)

Slicing is one of the primary features of Ouroboros, available in the CLI and the desktop app.

### Using the Slice Page

**Basic Usage Demo**

![Basic Usage Demo](../assets/slicing/Slice%20Page%20Demo.gif)

**Reusing Options from a Previous Run**

![Reusing Options from a Previous Run](../assets/slicing/Slice%20Page%20Options%20Demo.gif)


### Slicing Options

📁 - Drag and drop files from File Explorer panel into this option.

- 📁 `Neuroglancer JSON` - Path to the Neuroglancer configuration JSON file (exported from Neuroglancer with `{}` icon, or through the Neuroglancer Plugin)
- `Neuroglancer Image Layer` - Select Neuroglancer image layer to slice from.
- `Neuroglancer Annotation Layer` - Select Neuroglancer annotation layer to slice from.
- `Slice Width` - The output width of each slice image.
- `Slice Height` - The output height of each slice image.
- 📁 `Output File Folder` - The folder to save all the resulting files into.
- `Output File Name` - Base name for all output files.
- `Annotation MIP Level` - The source MIP used for annotation coordinates when the annotation source has no explicit coordinate metadata. 0 is the highest resolution. Explicit annotation source scales take precedence over this setting.
- `Output MIP Level` - The MIP level to output slices in (essentially a downsample option). 1 is a good starting point.
- `Slicing Parameters`
    - `Distance Between Slices` - The distance between each slice along the annotation path.
    - `Use Adaptive Slicing` - Rather than just using equidistant slices, add more slices in more curved areas.
    - `Adaptive Slicing Ratio` - 1 indicates to consider distance and curvature equally, 0.5 is biased towards distance, and 2 is biased towards curvature.
- `Output Single File` - Whether to output one tiff stack file or a folder of files.
- `Normalize Output` - Whether to normalize the output data.
- `Zero-Guard Output` - Whether to fix sub-zero slice values that may cause overflow artifacts.
- `Performance Logging` - Choose **Off** (the default) for no performance report, **Standard** to save performance reports and per-step statistics without monitoring RAM, or **Detailed (with RAM)** to save the full report including RAM measurements. RAM monitoring adds overhead. Progress bars and basic step timings remain available in all three modes. The choice is saved as `logging_mode` with your options. Older `reporting_enabled` values load as Off for `false` and Detailed for `true`; an explicit `logging_mode` takes precedence, and missing settings reset to Off. The saved preference applies to desktop runs; CLI runs select their mode with `--logging-mode off|standard|detailed`, defaulting to Off independently of the saved preference.
- `Bounding Box Parameters`
    - `Max Depth` - Limits how many times a bounding box can be subdivided. Reaching this limit can leave a large task.
    - `Target Slices Per Box` - Sets the slice-count target for the reference partition used to limit cumulative requested volume. The slicing partitioner seeks smaller task payloads within that allowance; this value is not a hard slice-count or RAM limit. Lowering it does not guarantee lower peak memory. See [Bounding box partitioning](../reference/bounding-box-partitioning.md).

### How Does Slicing Work?

**Spline Fitting**

Ouroboros first converts the annotation path from the Neuroglancer JSON file into image-source voxel coordinates at `Output MIP Level`. For local annotations with explicit `source.transform.outputDimensions`, the supported coordinate space has dimensions ordered `x`, `y`, `z`, with scales expressed in meters (`m`) and an identity annotation transform. For an untransformed image, each coordinate is multiplied by its annotation scale and divided by the image source's resolution at the output MIP.

Image placement in the selected image layer's `source.transform` is supported for ordered XYZ meter dimensions and finite, invertible 3 × 4 affine matrices, including translations, rotations, and scaling. Ouroboros applies the inverse placement before fitting the path. Image `inputDimensions`, when supplied, can use different scales from `outputDimensions`; when omitted, they use the image source's native MIP-0 resolution. Unsupported dimensions, singular matrices, and legacy layer-level transforms report an error.

Explicit annotation scales require reading image-source resolution metadata even when the annotation and output MIP settings match. This step does not download image voxels. Without explicit annotation coordinate metadata, Ouroboros retains the existing `Annotation MIP Level` conversion for untransformed images. A nontrivial image placement requires explicit annotation scales and otherwise reports an error, because the annotation's global coordinate units are unknown. The viewer-wide `dimensions` setting does not supply missing annotation metadata.

Ouroboros fits a spline curve to the converted path. This produces a smooth, differentiable curve along the length of the annotations.

If the line segments between annotation points were used, then there could be highly discontinuous slice transitions around sharp corners.

**Slice Coordinate Frames**

Ouroboros samples the spline in output-MIP voxel coordinates. `Distance Between Slices` sets the spacing for equidistant sampling; adaptive sampling also accounts for curvature. Slice width and height use the same output-MIP voxel coordinates.

To follow any possible path, each point must have its own coordinate frame normal to the spline. For easier segmentation, it is also important that the axes of the point coordinate frames don't flip between points. 

To achieve this, Ouroboros [calculates rotation-minimizing frames](https://github.com/ChengLabResearch/ouroboros/blob/05bac7b1a0a45d6c93e70cc3220397ebc7e11f8b/python/ouroboros/helpers/spline.py#L96) based on the change in orientation of the tangent vectors of adjacent points.

**Rectangles and Grids**

Ouroboros calculates the corners of the slice rectangle using each point and its associated coordinate axes (as vectors).

_Pseudocode_

```python
width_vector # 'u' unit vector scaled to slice width / 2 
height_vector # 'v' unit vector scaled to slice height / 2

top_left = point - width_vector + height_vector
top_right = point + width_vector + height_vector
bottom_right = point + width_vector - height_vector
bottom_left = point - width_vector - height_vector
```

Then, for each point, a 2D grid is generated with given dimensions (slice width and height). Each point in the 2D grid is a 3D point which is calculated by interpolating between the four corner points.

**Volumes and Cubic Interpolation**

At this point, Ouroboros has a full grid of 3D points associated with each slice, but has yet to download any data from the cloud-hosted volume.

Ouroboros groups slices into bounding boxes, using the full rectangle corners to keep every slice inside its assigned volume. It evaluates several spatial and path-order splits to reduce the largest estimated slicing-task payload. The selected partition requests no more cumulative volume than the reference partition made with the same bounding-box options. Geometry and subdivision limits can prevent further improvement; the partitioner does not measure available RAM or guarantee that a run will fit in memory. [Bounding box partitioning](../reference/bounding-box-partitioning.md) describes the cost model and stopping rules.

Ouroboros downloads each bounding box from the integer voxel at `floor(min)` through `ceil(max)`, inclusive on every axis. Slice coordinates and backprojection use this same integer origin. Regions outside the source volume remain zero-filled.

For each volume, `scipy.ndimage.map_coordinates` extracts values at the slice grid coordinates using cubic interpolation with its default prefilter. Changing bounding boxes can change the prefilter's boundary effects and float32 coordinate rounding. The filtering effect decays with distance but has no fixed voxel-distance cutoff; no extra padding is downloaded to reduce it.

Each slice is written into a temporary TIFF as it is completed. After all slices finish, Ouroboros normalizes them and writes the final output, either a single tiff stack or a folder of tiffs.

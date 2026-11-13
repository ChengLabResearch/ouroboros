# Backprojection 

After straightened volume produced by the [slicing step](./slicing.md) is segmented, Ouroboros's Backproject Page can be used to project the segmentation back into the coordinate space of the original volume.

### Using the Backproject Page

**Basic Usage Demo**

![Basic Usage Demo](../assets/backproject/Backproject%20Page%20Demo.gif)

**Reusing Options from a Previous Run**

See the same section under [Slice Page](./slicing.md).

**Compression Options**

One of the options on the Backproject page allows the user to configure the output compression format.

Common options

- `zlib`
- `zstd`
- `none` (no compression, not recommended)

For more information, see [Tifffile](https://github.com/cgohlke/tifffile/blob/166092f3e7b38cd1af430846157711f916ed5200/tifffile/tifffile.py#L13068C9-L13068C20), the Python package responsible for the compression.

**Output Position**

Backprojection writes only the minimum bounding box of the slices, rather than the space of the entire source scan. To allow placing outputs - such as segmentations - in the correct place, the minimum corner at the output MIP level is recorded in the output name, `<Output File Name>_<x>_<y>_<z>`. These values can be negative when slices extend past the edge of the source volume; that part of the output contains no data.

**Plugin Segmentation Inputs**

The `Straightened Volume File` can be the segmentation TIFF produced by the autoseg plugin, as long as it was generated from the same slicing configuration selected in `Slice Options File`.

Ouroboros checks the plugin output before backprojection:

- Shape must match the slicing config as `D/V/U`: number of slices, slice height, slice width. Single-stack TIFFs use page count for `D`; directory outputs use the number of TIFF files.
- Dtype must be a mask-friendly integer or boolean type: `bool`, `uint8`, or `uint16`.
- Compressed plugin TIFFs are accepted when tifffile can read them; Ouroboros rewrites compressed or non-memmappable inputs to an uncompressed temporary stack before processing.

Shape mismatches, dtype mismatches, and missing slice configuration report separate errors so plugin output problems do not require manual reshaping to diagnose.

### Backprojection Options

📁 - Drag and drop files from File Explorer panel into this option.

- 📁 `Straightened Volume File` - Path to the volume of slices to backproject (e.g. the output tif of the slicing step).
- 📁 `Slice Options File` - Path to the `-slice-options.json` file saved by the Slice page, which includes the information needed for backprojection.
- 📁 `Output File Folder` - The folder to save all the resulting files into.
- `Output File Name` - Base name for all output files.
- `Output MIP Level` - The MIP level to write the backprojection at. Choose a level finer than the slicing MIP level to upsample, for example if you downsampled in the slicing step. A level coarser than the slicing MIP level is rejected with an error.
- `Upsample Order` - The OpenCV interpolation used to resize each page when upsampling: 0 nearest, 1 linear, 2 cubic (the default), 3 area, 4 Lanczos. If you check the binary option, feel free to set this to 0.
- `Backprojection Compression` - The compression to use for the backprojected tiff(s); the default is `zlib`. See [Compression Options](#using-the-backproject-page).
- `Output Single File` - Whether to output one tiff stack file or a folder of files.
- `Performance Logging` - Choose **Off** (the default) for no performance report, **Standard** to save performance reports and per-step statistics without monitoring RAM, or **Detailed (with RAM)** to save the full report including RAM measurements. RAM monitoring adds overhead. Progress bars and basic step timings remain available in all three modes. The choice is saved as `logging_mode` with your options. Older `reporting_enabled` values load as Off for `false` and Detailed for `true`; an explicit `logging_mode` takes precedence, and missing settings reset to Off. The saved preference applies to desktop runs; CLI runs select their mode with `--logging-mode off|standard|detailed`, defaulting to Off independently of the saved preference.
- `Binary Backprojection` - Whether or not to binarize all the values of the backprojection. Enable this to backproject a single-part segmentation.
- `Discrete Backprojection` - Whether to limit backprojected values to discrete integers. Enable this to backproject a multi-part segmentation.
- `Process Count` - Number of parallel processes to use during the backprojection process. Reducing the count trades parallel throughput for lower memory usage. Inspect memory use on the intended workload when choosing a count.
- `Chunk Size` - Size of each dimension (x/y/z) of the processing chunk.  Increasing will increase memory usage; performance is a more complex tradeoff and making it too large will negatively affect performance.

#### Currently Inactive Options

- `Output Min Bounding Box` - Not currently applied: the output always covers the minimum bounding box. See **Output Position** above.
- `Offset in Filename` - Not currently applied: the output name always includes the position.

### How Does Backprojection Work?

Ouroboros reloads the Neuroglancer JSON named by the slice options and regenerates the slice rectangles with the slicing geometry, so that JSON and those options must still match the straightened volume. From the rectangles, it derives one bounding box for the output and recalculates the 2D coordinate grids of 3D points for each slice.

Older slicing runs could ignore explicit annotation source scales or image-source placement. If either affected an existing straightened volume, regenerate that volume and its derived segmentations before backprojection. Backprojection uses the same inverse image placement and coordinate conversion described in [Spline Fitting](./slicing.md#how-does-slicing-work), which can change slice positions and count. Transformed images require explicit annotation scales when regenerating geometry.

**Trilinear Interpolation**

Backprojection iterates through 3D chunks of the straightened volume, in the following order:

1. The chunks are loaded from a memory map of the straightened volume.
2. A custom rapid trilinear interpolation implementation is used to map the slice data into a sparse representation of the backprojected volume, limited to a bounding box containing only points associated with the chunk.  
   1. These sparse representations are written to disk intermediates, with individual files for each slice of the backprojected volume they are associated with.
3. As chunks are completed, it evaluates which slices of the backprojected volume can be fully written by completed chunks.
   1. The data for these slices are loaded from the intermediates, then written to disk in the target folder as they are completed.
   2. When upsampling, each page is resized to the output size in X and Y and written once per output Z position. X and Y follow the ratio between the two MIP levels' sizes; Z uses the whole-number ratio of their resolutions.
4. Once all slices are completed, they are converted to a single file if this configuration option is enabled.

**Why Is Interpolation Needed?**

The slices in the straightened volume were originally produced through interpolation. Each slice has a 2D grid of 3D points, and the values at these points were approximated through interpolation.

In order to put the data back, we have to interpolate in the opposite direction. Each point contributes to the value of its neighbors based on how close it is to them. 

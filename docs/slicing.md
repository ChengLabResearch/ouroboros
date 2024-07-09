# Slicing

![Long Sliced Volume](./assets/slicing/long-slice.png)

Slicing is one of the primary features of Ouroboros, available in the CLI and the desktop app.

### What Does Slicing Mean Here?

A user of Ouroboros may have a multi-terabyte (or more) volumetric scan, hosted with the Neuroglancer of tools (e.g. [cloud-volume](https://github.com/seung-lab/cloud-volume)). 

Perhaps there is a long, relatively sparse structure (ROI), like a nerve or a blood vessel that crosses the entire scan. Even with a well-equipped computer, it would be difficult to segment the entire stucture in one pass due to RAM limitations.

Ouroboros provides a solution. A user first traces the structure in Neuroglancer with sequential annotation points, and then saves the JSON configuration to a file.

Ouroboros opens this configuration file and cuts rectangular slices along the annotation path, producing a straightened volume with the ROI at the center of each slice (usually much smaller than the original scan).

![Circle of Slices](./assets/slicing/circle-slices.png)
_Every tenth slice in a circular annotation path, rendered in Ouroboros's Slicing Page._

From there, the user segments the much smaller straightened volume with their choice of segmentation system. Then, Ouroboros [backprojects](./backproject.md) the segmented slices into the original volume space (unstraightens it), producing a full segmentation.

### How Does Slicing Work?

**Spline Fitting**

Ouroboros fits a spline curve to the annotation path from the Neuroglancer JSON file. This produces a smooth, differentiable curve along the length of the annotations. 

If the line segments between annotation points were used, then there would could be highly discontinuous slice transitions around sharp corners.

**Slice Coordinate Frames**

Ouroboros calculates equidistant points along the spline curve. Each point is "1" apart from the previous to match the appearance of isotropic slices taken from one of the axes of the coordinate space of the full volume (the coordinate space of the annotation points).

To follow any possible path, each point must have its own coordinate frame normal to the spline. For easier segmentation, it is also important that the axes of the point coordinate frames don't flip between points. 

To achieve this, Ouroboros [calculates rotation-minimizing frames](https://github.com/We-Gold/ouroboros/blob/05bac7b1a0a45d6c93e70cc3220397ebc7e11f8b/python/ouroboros/helpers/spline.py#L96) based on the change in orientation of the tangent vectors of adjacent points.

**Rectangles and Grids**

Ouroboros calculates the corners of the slice rectangle at using each point and its associated coordinate axes (as vectors).

_Pseudocode_

```python
width_vector # The 'u' unit vector scaled to slice width / 2 
height_vector # The 'v' unit vector scaled to slice height / 2

top_left = point - width_vector + height_vector
top_right = point + width_vector + height_vector
bottom_right = point + width_vector - height_vector
bottom_left = point - width_vector - height_vector
```

Then, for each point, a 2D grid is generated with given dimensions (slice width and height). Each point in the 2D grid is a 3D point which is calculated by interpolating between the four corner points.

**Volumes and Trilinear Interpolation**

At this point, Ouroboros has a full grid of 3D points associated with each slice, but has yet to download any data from the cloud-hosted volume.

To avoid downloading unnecessary data, Ouroboros employs binary space partitioning to divide the minimum bounding box of the ROI recursively until each volume contains few enough slices to fit into available RAM.

Ouroboros then downloads the data for each of these bounding boxes from the cloud-hosted volume. As each download completes, a new process is spawned to handle calculations. 

For each volume, `scipy.ndimage.map_coordinates` is used to extract the values at the coordinates of the slice grids from the 3D volume using trilinear interpolation. 

Each of these slices are saved as local tiff images and are later combined into a single output tiff.
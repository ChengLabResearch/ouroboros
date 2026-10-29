from dataclasses import dataclass, asdict, fields, astuple
from functools import partial
from collections.abc import Callable, Sequence

from cloudvolume import VolumeCutout
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

from .bounding_boxes import BoundingBox
from .spline import Spline
from .shapes import DataShape, TFIter, DataRange

INDEXING = "xy"

NO_COLOR_CHANNELS_DIMENSIONS = 3
COLOR_CHANNELS_DIMENSIONS = NO_COLOR_CHANNELS_DIMENSIONS + 1


@dataclass(frozen=True)
class SliceTaskPayload:
    """Major simultaneously live arrays in the 4-D slicing worker, in bytes.

    This is a planning estimate, not resident memory. Mapped output pages,
    downloads/decompression, other workers and small temporaries are excluded.
    Keep this model aligned with the worker's reused coordinate plane and
    ``slice_volume_from_grids``' reused coefficient workspace/direct output.
    """

    source_bytes: int
    grid_bytes: int
    coefficient_bytes: int
    output_bytes: int
    channel_result_bytes: int

    @property
    def peak_bytes(self) -> int:
        # The interpolation phase retains the grid and dominates grid creation.
        return (self.source_bytes + self.grid_bytes + self.coefficient_bytes
                + self.output_bytes + self.channel_result_bytes)


def estimate_slice_task_payload(
    box: BoundingBox, slice_count: int, contiguous: bool, *,
    slice_shape: tuple[int, int], source_dtype, channels: int,
) -> SliceTaskPayload:
    """Estimate a nonempty production task from its owning geometry/metadata.

    ``slice_shape`` is the configured raster shape, independent of rectangle
    edge lengths. The worker reuses one float32 XYZ plane and one float64
    coefficient workspace across sequential channels, writing into TIFF views.
    ``slice_count`` and ``contiguous`` remain callback-compatible inputs but do
    not affect retained array sizes, including for gapped assignments. Heap
    output components are always zero.
    """
    voxels = int(box.calculate_volume())
    samples = int(slice_shape[0]) * int(slice_shape[1])
    return SliceTaskPayload(
        source_bytes=voxels * int(channels) * np.dtype(source_dtype).itemsize,
        grid_bytes=3 * np.dtype(np.float32).itemsize * samples,
        coefficient_bytes=np.dtype(np.float64).itemsize * voxels,
        output_bytes=0,
        channel_result_bytes=0,
    )


@dataclass(frozen=True)
class SliceTaskPayloadEstimator:
    """Pickleable planning adapter containing only source/raster metadata.

    The canonical estimator owns every component; this adapter transfers no
    live source interface and computes or caches no independent cost model.
    """

    slice_shape: tuple[int, int]
    source_dtype: np.dtype
    channels: int

    def __call__(self, box: BoundingBox, slice_count: int, contiguous: bool) -> int:
        return estimate_slice_task_payload(
            box, slice_count, contiguous, slice_shape=self.slice_shape,
            source_dtype=self.source_dtype, channels=self.channels,
        ).peak_bytes


@dataclass
class FrontProj(DataShape): V: int; U: int      # noqa: E701,E702
@dataclass
class FrontProjStack(DataShape): D: int; V: int; U: int      # noqa: E701,E702


def calculate_slice_rects(
    times: np.ndarray, spline: Spline, width, height, spline_points=None
) -> np.ndarray:
    """
    Calculate the slice rectangles for a spline at a set of time points.

    Parameters:
    ----------
        times (numpy.ndarray): The time points at which to calculate the slice rectangles.
        spline (Spline): The spline object.
        width (float): The width of the slice rectangles.
        height (float): The height of the slice rectangles.
        spline_points (numpy.ndarray): The points on the spline at the given time points (3, n).

    Returns:
    -------
        numpy.ndarray: The slice rectangles at the given time points (n, 4, 3).
    """

    # Calculate the tangent, normal, and binormal vectors
    tangent_vectors, normal_vectors, binormal_vectors = (
        spline.calculate_rotation_minimizing_vectors(times)
    )

    # Transpose the vectors for vectpr-by-vector indexing (3, n) -> (n, 3)
    tangent_vectors = tangent_vectors.T
    normal_vectors = normal_vectors.T
    binormal_vectors = binormal_vectors.T

    if spline_points is None:
        spline_points = spline(times)

    # (3, n) -> (n, 3)
    spline_points = spline_points.T

    rects = []

    _width, w_remainder = divmod(width, 2)
    _height, h_remainder = divmod(height, 2)

    width_left = _width
    width_right = _width + w_remainder

    height_top = _height
    height_bottom = _height + h_remainder

    for i in range(len(times)):
        point = spline_points[i]

        localx = normal_vectors[i]
        localy = binormal_vectors[i]

        width_left_vec = localx * width_left
        width_right_vec = localx * width_right
        height_top_vec = localy * height_top
        height_bottom_vec = localy * height_bottom

        top_left = point - width_left_vec + height_top_vec
        top_right = point + width_right_vec + height_top_vec
        bottom_right = point + width_right_vec - height_bottom_vec
        bottom_left = point - width_left_vec - height_bottom_vec

        rects.append(np.array([top_left, top_right, bottom_right, bottom_left]))

    # Output the rects in the form (n, 4, 3)
    return np.array(rects)


def coordinate_grid(rect: np.ndarray, shape: tuple[int, int] | FrontProj,
                    floor: np.ndarray = None, flip: bool = False, *,
                    output: np.ndarray | None = None) -> np.ndarray:
    """
    Generate a coordinate grid for a rectangle, relative to space of rectangle.

    Parameters:
    ----------
        rect (numpy.ndarray): The corners of the rectangle as a list of 3D coordinates.
        shape (tuple[int, int] | FrontProj): The height and width of the grid.
        floor (ndarray): Extra minimum value to use as baseline, instead of rect[0]
        flip (bool): Whether to flip the slice rects coordiante order (e.g. X/Y/Z to Z/Y/X)
        output (ndarray): Optional writable float32 destination of shape (height, width, 3).

    Returns:
    -------
        numpy.ndarray: The supplied destination, or a newly allocated float32 grid.
    """
    # Addition adds an extra rect[0] so we extend floor by it.
    floor = (rect[0] if floor is None else rect[0] + floor).astype(np.float32)
    if isinstance(shape, tuple): shape = FrontProj(*shape)     # noqa: E701
    if flip:
        floor = np.flip(floor)
        l_rect = np.flip(rect, axis=1)
    else:
        l_rect = rect

    u = np.linspace(l_rect[0], l_rect[1], shape.U, dtype=np.float32)
    v = np.linspace(l_rect[0], l_rect[3], shape.V, dtype=np.float32)

    target_shape = (shape.V, shape.U, 3)
    if output is None:
        output = np.empty(target_shape, dtype=np.float32)
    elif not isinstance(output, np.ndarray) or output.dtype != np.float32:
        raise TypeError("output must be a float32 NumPy array")
    elif output.shape != target_shape:
        raise ValueError(f"output must have shape {target_shape}")
    elif not output.flags.writeable:
        raise ValueError("output must be writable")

    np.add(u.reshape(1, shape.U, 3), v.reshape(shape.V, 1, 3), out=output)
    np.subtract(output, floor, out=output)
    return output


def slice_volume_from_grids(
    volume: VolumeCutout, grids: Callable[[int], np.ndarray],
    *, output: np.ndarray, slice_indices: Sequence[int]
) -> np.ndarray:
    """Interpolate planes with cubic splines, reusing one coefficient buffer.

    Each source channel is filtered once, then sampled at the supplied grids.

    Parameters
    ----------
    volume : ndarray
        Source voxels with shape (x, y, z) or (x, y, z, channels).
    grids : callable
        Takes a slice index and returns one (height, width, 3) grid of
        box-local coordinates. May reuse its array; called for each channel.
    output : ndarray
        Writable C-contiguous float32 stack with shape (n, height, width),
        plus a trailing channel axis for a 4-D source. The destination stack
        length n includes any unassigned planes.
    slice_indices : sequence of int
        Destination plane indices passed to the grid callable.

    Returns
    -------
    ndarray
        The supplied output object. Only selected planes are modified.
    """
    has_color_channels, num_channels = detect_color_channels(volume)
    if output.dtype != np.float32 or not output.flags.c_contiguous or not output.flags.writeable:
        raise ValueError(f"output must be a writable C-contiguous float32 array.")
    if not len(slice_indices):
        return output

    # Reuse the workspace across channels: allocating on each iteration would
    # briefly retain both the previous and next channel's coefficient arrays.
    coefficients = np.empty(volume.shape[:3], dtype=np.float64)
    for channel in range(num_channels):
        source = volume[..., channel] if has_color_channels else volume
        spline_filter(source, order=3, output=coefficients, mode="constant")
        for index in slice_indices:
            grid = grids(index)
            plane = output[index]
            destination = (plane.reshape(-1, num_channels)[:, channel]
                           if has_color_channels else plane.reshape(-1))
            map_coordinates(coefficients, grid.reshape(-1, 3).T,
                            output=destination, order=3, mode="constant", cval=0,
                            prefilter=False)
    return output


def _apply_weights(values, weights, corner):
    # This looks stupid but is twice as fast as fancy indexing with prod
    c_weights = weights[corner[0], 0, :] * weights[corner[1], 1, :] * weights[corner[2], 2, :]
    w_values = values * c_weights[..., np.newaxis]
    return w_values, c_weights


def _points_and_weights(points: np.ndarray, bounding_box_shape: tuple[int], squish_type: type):
    int_points = np.empty(points.shape, dtype=squish_type)
    weights = np.empty(points.shape, np.float32)
    np.modf(points, out=(weights, int_points), casting="unsafe")

    # Weight values are 1-remainder for floor, remainder for 'ceiling'.
    weights = np.stack([1 - weights, weights], axis=0)

    return np.ravel_multi_index(int_points, bounding_box_shape).astype(squish_type), weights


def backproject_box(bounding_box: BoundingBox, slice_rects: np.ndarray, slices: np.ndarray):
    if slices.shape[0] == 0:
        # No slices, just return
        return np.empty((0), dtype=np.uint32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    channels = 1 if len(slices.shape) < 4 else slices.shape[-1]

    # Plot the backprojected X/Y/Z coordinates for each of the slice rects in this chunk.
    grid_call = partial(coordinate_grid, shape=slices[0].shape[:2], floor=bounding_box.get_integer_origin(), flip=True)
    precise_points = np.concatenate(list(map(grid_call, slice_rects)))

    # Flatten values to 1-Dimension Array for Efficient Allocation.
    # Use ZYX domain rather than XYZ as the former is what is written to disk.
    values = slices.flatten().reshape(-1, channels)
    zyx_shape = np.flip(bounding_box.get_shape())
    flat_shape = np.prod(zyx_shape)

    # Include the largest corner increment for zero-weight boundary writes.
    guard = np.ravel_multi_index((1, 1, 1), zyx_shape)
    capacity = flat_shape + guard
    squish_type = np.min_scalar_type(capacity - 1)

    # Allocate the flattened data volume.
    volume = np.zeros((1 + channels, capacity), dtype=np.float32)

    # Get the top corner (integer) points and full weight matrix.
    points, weights = _points_and_weights(precise_points.reshape(-1, 3).T, zyx_shape, squish_type)

    # Sequentially allocate the points and weights for each corner to the flattened array.
    for corner in np.array(list(np.ndindex(2, 2, 2))):
        w_values, c_weights = _apply_weights(values, weights, corner)
        point_inc = np.ravel_multi_index(corner, zyx_shape).astype(squish_type)

        for i in range(0, channels):
            np.add.at(volume[i], points + point_inc, w_values[..., i])
        np.add.at(volume[-1], points + point_inc, c_weights)

    # Get indicies of the flattened Z-Y-X backprojected domain that have values.
    nz_vol = np.flatnonzero(volume[-1, :flat_shape]).astype(squish_type)

    # Return indicies and only the volume region with values.
    return nz_vol, volume[:-1, nz_vol].squeeze(), volume[-1, nz_vol].squeeze()


def make_volume_binary(volume: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a volume to binary format.

    Note: To view a binary volume, use the threshold feature in ImageJ.

    Parameters:
    ----------
        volume (numpy.ndarray): The volume to convert to binary format.

    Returns:
    -------
        numpy.ndarray: The binary volume.
    """

    return (volume > 0).astype(dtype)


def detect_color_channels(data: np.ndarray, none_value=1) -> tuple[bool, int]:
    """
    Detect the number of color channels in a volume.

    Parameters:
    ----------
        data (numpy.ndarray): The volume data.
        none_value (int): The value to return if the volume has no color channels.

    Returns:
    -------
        tuple: A tuple containing the following:
            - has_color_channels (bool): Whether the volume has color channels.
            - num_color_channels (int): The number of color channels in the volume.
    """

    has_color_channels, num_color_channels = detect_color_channels_shape(
        data.shape, none_value
    )

    return has_color_channels, num_color_channels


def detect_color_channels_shape(shape: tuple, none_value=1) -> tuple[bool, int]:
    """
    Detect the number of color channels in a volume.

    Parameters:
    ----------
        shape (tuple): The shape of the volume data.
        none_value (int): The value to return if the volume has no color channels.

    Returns:
    -------
        tuple: A tuple containing the following:
            - has_color_channels (bool): Whether the volume has color channels.
            - num_color_channels (int): The number of color channels in the volume.
    """

    has_color_channels = len(shape) == COLOR_CHANNELS_DIMENSIONS
    num_color_channels = shape[-1] if has_color_channels else none_value

    return has_color_channels, num_color_channels


class BackProjectIter(TFIter):
    def __init__(self, dr: DataRange, shape: FrontProjStack, slice_rects: np.ndarray):
        self.__shape = asdict(shape)
        self.__step_f = {f.name: i for i, f in enumerate(fields(dr.step))}
        self.__step_v = asdict(dr.step)
        self.__u_gap = (slice_rects[:, 1, :] - slice_rects[:, 0, :]) / (shape.U - 1)
        self.__v_gap = (slice_rects[:, 3, :] - slice_rects[:, 0, :]) / (shape.V - 1)
        self.__top_r = slice_rects[:, 0, :]
        self.__step = dr.step

    def __call__(self, pos) -> tuple:
        # D/V/U (Z/Y/X in straightend volume) position.
        pos_step = {field: np.s_[pos[index]: min(pos[index] + self.__step_v[field], self.__shape[field])]
                    for field, index in self.__step_f.items()}

        # D/V/U Dimensions (Z/Y/X in the staightened volume.)
        start = FrontProjStack(D=pos_step["D"].start if "D" in pos_step else 0,
                               V=pos_step["V"].start if "V" in pos_step else 0,
                               U=pos_step["U"].start if "U" in pos_step else 0)
        stop = FrontProjStack(D=pos_step["D"].stop if "D" in pos_step else self.shape.D,
                              V=pos_step["V"].stop if "V" in pos_step else self.shape.V,
                              U=pos_step["U"].stop if "U" in pos_step else self.shape.U)
        shape = FrontProjStack(D=stop.D - start.D, V=stop.V - start.V, U=stop.U - start.U)

        # Slice index for chunk in the straightened volume.
        chunk = tuple([np.s_[:] if key not in self.__step_f else pos_step[key] for key in self.__shape])

        # Slice rects for the chunk
        chunk_rects = (np.array(
                        [self.__u_gap[chunk[0]] * start.U + self.__v_gap[chunk[0]] * start.V,
                         self.__u_gap[chunk[0]] * (stop.U - 1) + self.__v_gap[chunk[0]] * start.V,
                         self.__u_gap[chunk[0]] * (stop.U - 1) + self.__v_gap[chunk[0]] * (stop.V - 1),
                         self.__u_gap[chunk[0]] * start.U + self.__v_gap[chunk[0]] * (stop.V - 1)])
                       + self.__top_r[chunk[0]]).transpose(1, 0, 2)

        # Bounding box for the specific chunk
        bbox = BoundingBox.from_rects(chunk_rects)

        # N-dimension index of this chunk in the straightened volume.
        index = astuple(type(self.__step)(*pos) // self.__step)

        return chunk, shape, chunk_rects, bbox, index

from functools import partial
import pytest

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers import slice as slice_module
from ouroboros.helpers.slice import (
    calculate_slice_rects,
    detect_color_channels,
    make_volume_binary,
    slice_volume_from_grids,
    coordinate_grid,
    FrontProj,
    backproject_box,
    FrontProjStack,
    BackProjectIter
)
from ouroboros.helpers.spline import Spline
from test.sample_data import generate_sample_curve_helix


def generate_rects(width, height, z):
    # Sample points arranged in a simple curve
    sample_points = generate_sample_curve_helix(start_z=z.start, end_z=z.stop)

    # Initialize Spline object
    spline = Spline(sample_points, degree=3)

    # Generate a range of t values
    equidistant_times = spline.calculate_equidistant_parameters(
        distance_between_points=1
    )

    # Calculate slice rects
    return calculate_slice_rects(equidistant_times, spline, width, height)


def generate_bounded_rects(width, height, count):
    return np.array([BoundingBox.bounds_to_rect(width.start, width.stop - 1, height.start, height.stop - 1, i, i)
                     for i in range(count)])


def test_calculate_slice_rects():
    # Sample points arranged in a simple curve
    sample_points = generate_sample_curve_helix()

    # Initialize Spline object
    spline = Spline(sample_points, degree=3)

    # Generate a range of t values
    equidistant_times = spline.calculate_equidistant_parameters(
        distance_between_points=1
    )

    WIDTH = 100
    HEIGHT = 100

    # Calculate slice rects
    slice_rects = calculate_slice_rects(equidistant_times, spline, WIDTH, HEIGHT)

    # Assert that the method returns a list of numpy arrays
    assert isinstance(slice_rects, np.ndarray), "Slice rects should be a numpy array"
    for rect in slice_rects:
        assert isinstance(rect, np.ndarray), "Each slice rect should be a numpy array"
        assert rect.shape == (4, 3), "Each slice rect should have shape (4, 3)"

        # Check if the slice rects are valid rectangles
        top_left = rect[0]
        top_right = rect[1]
        bottom_right = rect[2]
        bottom_left = rect[3]

        assert np.allclose(
            top_left + bottom_right, top_right + bottom_left
        ), "The diagonals of the rectangle should intersect at the center"

        # Make sure rectangles have expected width and height
        assert np.allclose(
            np.linalg.norm(top_left - top_right), WIDTH
        ), "Width should be 100"
        assert np.allclose(
            np.linalg.norm(top_left - bottom_left), HEIGHT
        ), "Height should be 100"


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("floor", [None, np.array([-50.0, -60.0, -40.0])])
@pytest.mark.parametrize("provided", [False, True])
def test_coordinate_grid_preserves_formula_in_its_destination(flip, floor, provided, monkeypatch):
    rect = np.array([[-42.64727347, -54.72166585, -4.78695662],
                     [-47.22139466, 43.8212048, -21.16926598],
                     [40.81561612, 55.54768491, 24.78695662],
                     [45.38973732, -42.99518573, 41.16926598]])
    height, width = 5, 7
    adjusted_floor = (rect[0] if floor is None else rect[0] + floor).astype(np.float32)
    local_rect = np.flip(rect, axis=1) if flip else rect
    if flip:
        adjusted_floor = np.flip(adjusted_floor)
    u = np.linspace(local_rect[0], local_rect[1], width, dtype=np.float32)
    v = np.linspace(local_rect[0], local_rect[3], height, dtype=np.float32)
    # Preserve the original two-operation formula, including float32 rounding.
    expected = np.add(u.reshape(1, width, 3), v.reshape(height, 1, 3)) - adjusted_floor
    backing = np.full((2, height, width * 2, 3), -999, dtype=np.float32)
    destination = backing[1, :, ::2] if provided else None
    allocations, writes = [], []
    empty, add, subtract = np.empty, np.add, np.subtract

    def observe_empty(*args, **kwargs):
        result = empty(*args, **kwargs)
        if result.shape == expected.shape:
            allocations.append(result)
        return result

    def observe_add(left, right, *, out):
        writes.append(out)
        return add(left, right, out=out)

    def observe_subtract(left, right, *, out):
        assert left is out
        writes.append(out)
        return subtract(left, right, out=out)

    with monkeypatch.context() as patch:
        patch.setattr(slice_module.np, "empty", observe_empty)
        patch.setattr(slice_module.np, "add", observe_add)
        patch.setattr(slice_module.np, "subtract", observe_subtract)
        actual = coordinate_grid(rect, FrontProj(height, width) if provided else (height, width),
                                 floor=floor, flip=flip, output=destination)

    assert actual.shape == (height, width, 3)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert len(writes) == 2 and all(target is actual for target in writes)
    if provided:
        assert actual is destination
        assert np.shares_memory(actual, backing) and not actual.flags.c_contiguous
        assert allocations == []
        assert np.all(backing[0] == -999) and np.all(backing[1, :, 1::2] == -999)
    else:
        assert len(allocations) == 1 and allocations[0] is actual


@pytest.mark.parametrize("invalid", ["dtype", "shape", "readonly", "not_array"])
def test_coordinate_grid_rejects_invalid_destination(invalid):
    output = np.zeros((5, 7, 3), dtype=np.float32)
    error, message = ValueError, "shape"
    if invalid == "dtype":
        output = output.astype(np.float64)
        error, message = TypeError, "float32"
    elif invalid == "shape":
        output = output.transpose(1, 0, 2)
    elif invalid == "readonly":
        output.flags.writeable = False
        message = "writable"
    else:
        output = output.tolist()
        error, message = TypeError, "float32"
    with pytest.raises(error, match=message):
        coordinate_grid(np.arange(12).reshape(4, 3), (5, 7), output=output)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("channels", [None, 1, 3])
def test_native_dtype_interpolation_preserves_float32_baseline(dtype, channels, monkeypatch, tmp_path):
    amplitude = 255 if dtype == np.float32 else np.iinfo(dtype).max
    step = np.array([0, 0, 0, amplitude, amplitude, amplitude, 0, 0], dtype=dtype)
    volume = np.broadcast_to(step[:, None, None], (8, 4, 4)).copy()
    if channels is not None:
        volume = np.stack([np.roll(volume, channel, axis=0) for channel in range(channels)], axis=-1)
    grids = np.array([[1.5, 1.25, 1.75], [2.25, 1.25, 1.75], [3.5, 1.25, 1.75],
                      [4.5, 1.25, 1.75], [5.5, 1.25, 1.75], [6.5, 1.25, 1.75]],
                     dtype=np.float32).reshape(1, 2, 3, 3)
    grids = np.concatenate((grids, grids[:, :, ::-1]))
    coordinates = grids.reshape(-1, 3).T
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 8, 0, 4, 0, 4))
    old_volume = volume.astype(np.float32)
    if channels is None:
        expected = map_coordinates(old_volume, coordinates).reshape(2, 2, 3)
    else:
        expected = np.stack([
            map_coordinates(old_volume[..., channel], coordinates).reshape(2, 2, 3)
            for channel in range(channels)
        ], axis=-1)

    interpolated_inputs = []

    def observe_input(data, *args, **kwargs):
        # Only the small interpolation output may be converted to float32.
        assert data.dtype == volume.dtype
        assert np.shares_memory(data, volume)
        interpolated_inputs.append(data.dtype)
        return map_coordinates(data, *args, **kwargs)

    monkeypatch.setattr(slice_module, "map_coordinates", observe_input)
    actual = slice_volume_from_grids(volume, box, grids, width=3, height=2)
    assert actual.shape == (2, 2, 3) + (() if channels is None else (channels,))
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    # Cubic overshoot and fractional values must survive integer source storage.
    assert expected.min() < 0
    assert expected.max() > amplitude
    assert np.any(expected != np.rint(expected))

    path = tmp_path / "native-dtype.tif"
    destination = tifffile.memmap(path, shape=expected.shape, dtype=np.float32,
                                 photometric="minisblack")
    try:
        mapped = slice_volume_from_grids(volume, box, grids, width=3, height=2, output=destination)
        assert mapped is destination
        np.testing.assert_array_equal(mapped, actual)
        destination.flush()
    finally:
        destination._mmap.close()
    np.testing.assert_array_equal(tifffile.imread(path), expected)
    assert len(interpolated_inputs) == 2 * (channels or 1)


@pytest.mark.parametrize("dtype", [np.uint32, np.uint64, np.int32, np.int64, np.float64])
@pytest.mark.parametrize("channels", [None, 3])
def test_high_precision_source_interpolates_before_float32_output(dtype, channels, tmp_path):
    values = np.array([2**24, 2**24 + 1, 2**24 + 5, 2**24 + 9,
                       2**24 + 7, 2**24 + 1, 2**24, 2**24], dtype=dtype)
    volume = np.broadcast_to(values[:, None, None], (8, 4, 4)).copy()
    if channels is not None:
        volume = np.stack([np.roll(volume, channel, axis=0) for channel in range(channels)], axis=-1)
    grids = np.array([[1.25, 1.25, 1.75], [2.25, 1.25, 1.75], [3.5, 1.25, 1.75],
                      [4.5, 1.25, 1.75], [5.5, 1.25, 1.75], [6.5, 1.25, 1.75]],
                     dtype=np.float32).reshape(1, 2, 3, 3)
    coordinates = grids.reshape(-1, 3).T
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 8, 0, 4, 0, 4))
    if channels is None:
        expected = map_coordinates(volume, coordinates, output=np.float32).reshape(1, 2, 3)
    else:
        expected = np.stack([
            map_coordinates(volume[..., channel], coordinates, output=np.float32).reshape(1, 2, 3)
            for channel in range(channels)
        ], axis=-1)

    actual = slice_volume_from_grids(volume, box, grids, width=3, height=2)

    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    # The previous eager conversion discarded source bits before interpolation.
    old = slice_volume_from_grids(volume.astype(np.float32), box, grids, width=3, height=2)
    assert np.any(actual != old)

    path = tmp_path / "high-precision-dtype.tif"
    destination = tifffile.memmap(path, shape=expected.shape, dtype=np.float32,
                                 photometric="minisblack")
    try:
        mapped = slice_volume_from_grids(volume, box, grids, width=3, height=2, output=destination)
        assert mapped is destination
        np.testing.assert_array_equal(mapped, actual)
        destination.flush()
    finally:
        destination._mmap.close()
    np.testing.assert_array_equal(tifffile.imread(path), expected)


@pytest.mark.parametrize("channels", [None, 1, 3])
def test_slice_volume_writes_flat_interpolation_into_tiff_destination(channels, tmp_path, monkeypatch):
    volume = np.arange(80, dtype=np.uint16).reshape(5, 4, 4)
    if channels is not None:
        volume = np.stack([np.roll(volume, channel, axis=0) for channel in range(channels)], axis=-1)
    grids = np.array([[0.25, 1.25, 1.75], [1.25, 1.25, 1.75], [2.5, 1.25, 1.75],
                      [3.5, 1.25, 1.75], [4.5, 1.25, 1.75], [-0.5, 1.25, 1.75]],
                     dtype=np.float32).reshape(1, 2, 3, 3)
    grids = np.concatenate((grids, grids[:, :, ::-1]))
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 5, 0, 4, 0, 4))
    expected = slice_volume_from_grids(volume, box, grids, width=3, height=2)
    path = tmp_path / "destination.tif"
    mapping = tifffile.memmap(path, shape=(4, *expected.shape[1:]), dtype=np.float32,
                             photometric="minisblack")
    mapping[:] = -999
    destination = mapping[1:3]
    writes = []

    def observe_map(data, coordinates, **kwargs):
        assert set(kwargs) == {"output"}, "Preserve SciPy's interpolation defaults"
        target = kwargs["output"]
        expected_target = (destination.reshape(-1) if channels is None
                           else destination.reshape(-1, channels)[:, len(writes)])
        assert np.shares_memory(data, volume)
        assert np.shares_memory(coordinates, grids)
        np.testing.assert_array_equal(coordinates, grids.reshape(-1, 3).T)
        assert target.shape == expected_target.shape
        assert target.strides == expected_target.strides
        assert target.ctypes.data == expected_target.ctypes.data
        writes.append(target)
        return map_coordinates(data, coordinates, **kwargs)

    try:
        monkeypatch.setattr(slice_module, "map_coordinates", observe_map)
        actual = slice_volume_from_grids(volume, box, grids, width=3, height=2, output=destination)
        assert actual is destination
        assert len(writes) == (channels or 1)
        mapping.flush()
    finally:
        mapping._mmap.close()
    expected_file = np.full((4, *expected.shape[1:]), -999, dtype=np.float32)
    expected_file[1:3] = expected
    np.testing.assert_array_equal(tifffile.imread(path), expected_file)


def test_slice_volume_rejects_destinations_that_cannot_be_filled_in_place():
    volume = np.ones((3, 3, 3), dtype=np.uint16)
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 3, 0, 3, 0, 3))
    grids = np.ones((1, 2, 3, 3), dtype=np.float32)
    readonly = np.empty((1, 2, 3), dtype=np.float32)
    readonly.flags.writeable = False
    for output in (readonly, np.empty((1, 2, 3), dtype=np.float64),
                   np.empty((1, 3, 2), dtype=np.float32),
                   np.empty((1, 2, 6), dtype=np.float32)[:, :, ::2]):
        with pytest.raises(ValueError, match="writable C-contiguous float32 array"):
            slice_volume_from_grids(volume, box, grids, width=3, height=2, output=output)


def test_slice_volume_from_grids_empty_grids():
    volume = np.random.rand(10, 10, 10).astype(np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 10, 0, 10, 0, 10))
    grids = np.empty((0, 10, 10, 3)).astype(np.float32)
    width, height = 10, 10

    result = slice_volume_from_grids(volume, bounding_box, grids, width, height)

    assert result.shape == (0, height, width)
    assert result.dtype == np.float32


def test_slice_volume_from_grids_invalid_dimensions():
    volume = np.random.rand(10, 10, 10).astype(np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 10, 0, 10, 0, 10))
    grids = np.random.rand(5, 10, 10, 2).astype(np.float32)  # Invalid grid dimensions
    width, height = 10, 10

    with pytest.raises(ValueError):
        slice_volume_from_grids(volume, bounding_box, grids, width, height)


def test_backproject_basic():
    volume = np.zeros(np.prod((10, 10, 10)), dtype=np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 10, 0, 10, 0, 10))
    rects = generate_bounded_rects(range(0, 10), range(0, 10), 5)
    slices = np.random.rand(5, 10, 10).astype(np.float32)

    lookup, totals, weights = backproject_box(bounding_box, rects, slices)
    volume[lookup] = (totals / weights)

    assert volume.dtype == np.float32
    assert np.any(volume > 0)


def test_backproject_empty_slices():
    volume = np.zeros(np.prod((10, 10, 10)), dtype=np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 10, 0, 10, 0, 10))
    rects = generate_bounded_rects(range(0, 10), range(0, 10), 5)
    slices = np.random.rand(0, 10, 10).astype(np.float32)

    lookup, totals, weights = backproject_box(bounding_box, rects, slices)
    volume[lookup] = (totals / weights)

    assert volume.dtype == np.float32
    assert np.all(volume == 0)


def test_backproject_slices_large_volume():
    volume = np.zeros(np.prod((100, 100, 100)), dtype=np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 100, 0, 100, 0, 100))
    rects = generate_bounded_rects(range(0, 100), range(0, 100), 50)
    slices = np.random.rand(50, 100, 100).astype(np.float32)

    lookup, totals, weights = backproject_box(bounding_box, rects, slices)
    volume[lookup] = (totals / weights)

    assert volume.dtype == np.float32
    assert np.any(volume > 0)


def test_make_volume_binary():
    volume = np.array(
        [
            [[0.1, 0.5, 0.9], [0.2, 0.6, 0.8], [0.3, 0.7, 0.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
        ],
        dtype=np.float32,
    )

    binary_volume = make_volume_binary(volume)

    expected_binary_volume = np.array(
        [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ],
        dtype=np.uint8,
    )

    assert binary_volume.shape == volume.shape
    assert binary_volume.dtype == np.uint8
    assert np.array_equal(binary_volume, expected_binary_volume)


def test_detect_color_channels_no_channels():
    data = np.random.rand(10, 10, 10)  # 3D array
    has_color_channels, num_color_channels = detect_color_channels(data)
    assert not has_color_channels
    assert num_color_channels == 1


def test_detect_color_channels_with_channels():
    data = np.random.rand(10, 10, 10, 3)  # 4D array with 3 color channels
    has_color_channels, num_color_channels = detect_color_channels(data)
    assert has_color_channels
    assert num_color_channels == 3


def test_detect_color_channels_with_different_channels():
    data = np.random.rand(10, 10, 10, 4)  # 4D array with 4 color channels
    has_color_channels, num_color_channels = detect_color_channels(data)
    assert has_color_channels
    assert num_color_channels == 4


def test_detect_color_channels_custom_none_value():
    data = np.random.rand(10, 10, 10)  # 3D array
    none_value = 0
    has_color_channels, num_color_channels = detect_color_channels(
        data, none_value=none_value
    )
    assert not has_color_channels
    assert num_color_channels == none_value


def make_rect(point, u_vec, v_vec):
    return np.array([np.array([point[x], point[x] + u_vec[x], point[x] + u_vec[x] + v_vec[x], point[x] + v_vec[x]])
                     for x in range(point.shape[0])])


def test_backproject_values():
    bounds = np.random.rand(27).reshape(3, 3, 3) * 20
    rects = make_rect(bounds[0], bounds[1], bounds[2])
    slices = (np.random.rand(36) * 20).reshape(3, 4, 3)
    bounding_box = BoundingBox.from_rects(rects)

    zyx_shape = np.flip(bounding_box.get_shape())

    box_lookup, box_totals, box_weights = backproject_box(bounding_box, rects, slices)

    # Total weighted values assigned should be the same as sum of the slices.
    assert np.allclose([np.sum(box_totals)], [np.sum(slices)])

    # Total weights should be same as the # of points (1.0 per point)
    assert int(np.round(np.sum(box_weights))) == np.prod(slices.shape)

    # Bounds should be the shape of the box.
    # Would crash if beyond these bounds.
    _ = np.flip(np.unravel_index(box_lookup, (zyx_shape)))


def test_backproject_iter_2D():
    FPStackRange = FrontProjStack.drange((0, 0, 0), (3, 4, 3), (2, 2, 2))
    bounds = np.random.randint(0, 20, 27).reshape(3, 3, 3)
    rects = make_rect(bounds[0], bounds[1], bounds[2])
    func_call = partial(BackProjectIter, shape=FrontProjStack(3, 4, 3), slice_rects=rects)

    chunks = []
    shapes = []
    chunk_rects_set = []
    bboxes = []
    indexes = []

    print(rects.shape)

    for chunk, shape, chunk_rects, bbox, index in FPStackRange.get_iter(func_call):
        chunks.append(chunk)
        shapes.append(shape)
        chunk_rects_set.append(chunk_rects)
        bboxes.append(bbox)
        indexes.append(index)

    chunks_match = [np.s_[0:2, 0:2, 0:2], np.s_[0:2, 0:2, 2:3], np.s_[0:2, 2:4, 0:2], np.s_[0:2, 2:4, 2:3],
                    np.s_[2:3, 0:2, 0:2], np.s_[2:3, 0:2, 2:3], np.s_[2:3, 2:4, 0:2], np.s_[2:3, 2:4, 2:3]]
    shapes_match = [FrontProjStack(*x) for x in
                    [(2, 2, 2), (2, 2, 1), (2, 2, 2), (2, 2, 1), (1, 2, 2), (1, 2, 1), (1, 2, 2), (1, 2, 1)]]
    indexes_match = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    assert chunks == chunks_match
    assert shapes == shapes_match
    assert indexes == indexes_match
    assert np.all(chunk_rects_set[0][0][0].astype(int) == rects[0][0])
    assert np.all(chunk_rects_set[-1][-1][2].astype(int) == rects[-1][2])
    assert bboxes[2].approx_bounds() == BoundingBox.from_rects(chunk_rects_set[2]).approx_bounds()

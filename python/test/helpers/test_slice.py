from dataclasses import astuple
from functools import partial
from pathlib import Path
import pytest

import numpy as np
import tifffile as tf

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.slice import (
    calculate_slice_rects,
    detect_color_channels,
    make_volume_binary,
    slice_volume_from_grids,
    coordinate_grid,
    backproject_box,
    FrontProjStack,
    BackProjectIter
)
from ouroboros.helpers.shapes import ImageStack
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


def test_generate_coordinate_grid_for_rect():
    rect = np.array([[-42.64727347, -54.72166585,  -4.78695662],
                     [-47.22139466,  43.8212048,  -21.16926598],
                     [40.81561612,  55.54768491,  24.78695662],
                     [45.38973732, -42.99518573,  41.16926598]])
#     rect = np.array([[0, 0, 0], [2, 2, 0], [2, 3, 2], [0, 1, 2]])
#     rect = np.array([[ 57.35272653,  45.27833415,  95.21304338],
#        [ 52.77860534, 143.8212048 ,  78.83073402],
#        [140.81561612, 155.54768491, 124.78695662],
#        [145.38973732,  57.00481427, 141.16926598]])

    WIDTH = 100
    HEIGHT = 60

    # Generate a coordinate grid for the rectangle
    cg = coordinate_grid(rect, (HEIGHT, WIDTH))

    # Assert that the method returns a numpy array
    assert isinstance(
        cg, np.ndarray
    ), "Coordinate grid should be a numpy array"
    assert cg.shape == (
        HEIGHT,
        WIDTH,
        3,
    ), "Coordinate grid should have shape (60, 100, 3)"
    assert np.allclose(
        cg[0][0], rect[0]
    ), "The first coordinate should be the top left corner of the rectangle"


def test_generate_coordinate_grid_flipped():
    rect = np.array([[-42.64727347, -54.72166585,  -4.78695662],
                     [-47.22139466,  43.8212048,  -21.16926598],
                     [40.81561612,  55.54768491,  24.78695662],
                     [45.38973732, -42.99518573,  41.16926598]])
    floor = np.array([-50.0, -60.0, -40.0])
#     rect = np.array([[0, 0, 0], [2, 2, 0], [2, 3, 2], [0, 1, 2]])
#     rect = np.array([[ 57.35272653,  45.27833415,  95.21304338],
#        [ 52.77860534, 143.8212048 ,  78.83073402],
#        [140.81561612, 155.54768491, 124.78695662],
#        [145.38973732,  57.00481427, 141.16926598]])

    WIDTH = 100
    HEIGHT = 60

    # Generate a coordinate grid for the rectangle
    cg = coordinate_grid(rect, (HEIGHT, WIDTH), flip=True, floor=floor)

    # Assert that the method returns a numpy array
    assert isinstance(cg, np.ndarray), "Coordinate grid should be a numpy array"
    assert cg.shape == (HEIGHT, WIDTH, 3, ), "Coordinate grid should have shape (60, 100, 3)"
    assert np.allclose(cg[0][0], np.flip(rect[0] - floor)
                       ), "The first coordinate should be the top left corner of the rectangle"


def test_slice_volume_from_grids_single_channel():
    volume = np.random.rand(10, 10, 10).astype(np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 10, 0, 10, 0, 10))
    grids = np.random.rand(5, 10, 10, 3).astype(np.float32)
    width, height = 10, 10

    result = slice_volume_from_grids(volume, bounding_box, grids, width, height)

    assert result.shape == (5, height, width)
    assert result.dtype == np.float32


def test_slice_volume_from_grids_multi_channel():
    volume = np.random.rand(10, 10, 10, 3).astype(np.float32)
    bounding_box = BoundingBox(BoundingBox.bounds_to_rect(0, 10, 0, 10, 0, 10))
    grids = np.random.rand(5, 10, 10, 3).astype(np.float32)
    width, height = 10, 10

    result = slice_volume_from_grids(volume, bounding_box, grids, width, height)

    assert result.shape == (5, height, width, 3)
    assert result.dtype == np.float32


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
    box_points = np.flip(np.unravel_index(box_lookup, (zyx_shape)))
    

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

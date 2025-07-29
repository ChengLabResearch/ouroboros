import numpy as np
from ouroboros.helpers.bounding_boxes import (
    BoundingBox,
    BoundingBoxParams,
    calculate_bounding_boxes_bsp_link_rects,
    boxes_dim_range
)
from ouroboros.helpers.slice import calculate_slice_rects
from ouroboros.helpers.spline import Spline
from test.sample_data import generate_sample_curve_helix


def test_bounding_box_initialization():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    assert bbox.x_min == 0
    assert bbox.x_max == 1
    assert bbox.y_min == 0
    assert bbox.y_max == 1
    assert bbox.z_min == 0
    assert bbox.z_max == 1


def test_bounding_box_get_shape():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    shape = bbox.get_shape()
    assert shape == (2, 2, 2)


def test_bounding_box_approx_bounds():
    rect = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    bbox = BoundingBox(rect)
    approx_bounds = bbox.approx_bounds()
    assert approx_bounds == (0, 2, 0, 2, 0, 2)


def test_bounding_box_intersects():
    rect1 = np.array([[0, 0, 0], [1, 1, 1]])
    rect2 = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    bbox1 = BoundingBox(rect1)
    bbox2 = BoundingBox(rect2)
    assert bbox1.intersects(bbox2) is True

    rect3 = np.array([[2, 2, 2], [3, 3, 3]])
    bbox3 = BoundingBox(rect3)
    assert bbox1.intersects(bbox3) is False


def test_bounding_box_intersection():
    # Note: The intersection is calculated based on approximate bounds
    rect1 = np.array([[0, 0, 0], [2, 2, 2]])
    rect2 = np.array([[1, 1, 1], [3, 3, 3]])
    bbox1 = BoundingBox(rect1)
    bbox2 = BoundingBox(rect2)
    intersection = bbox1.intersection(bbox2)
    assert intersection.x_min == 1
    assert intersection.x_max == 2
    assert intersection.y_min == 1
    assert intersection.y_max == 2
    assert intersection.z_min == 1
    assert intersection.z_max == 2


def test_bounding_box_bound_boxes():
    rect1 = np.array([[0, 0, 0], [1, 1, 1]])
    rect2 = np.array([[1, 1, 1], [1.5, 1.5, 1.5]])
    bbox1 = BoundingBox(rect1)
    bbox2 = BoundingBox(rect2)
    combined_bbox = BoundingBox.bound_boxes([bbox1, bbox2])
    assert combined_bbox.x_min == 0
    assert combined_bbox.x_max == 2
    assert combined_bbox.y_min == 0
    assert combined_bbox.y_max == 2
    assert combined_bbox.z_min == 0
    assert combined_bbox.z_max == 2


def test_bounding_box_bound_boxes_not_approx():
    rect1 = np.array([[0, 0, 0], [1, 1, 1]])
    rect2 = np.array([[1, 1, 1], [1.5, 1.5, 1.5]])
    bbox1 = BoundingBox(rect1)
    bbox2 = BoundingBox(rect2)
    combined_bbox = BoundingBox.bound_boxes([bbox1, bbox2], use_approx_bounds=False)
    assert combined_bbox.x_min == 0
    assert combined_bbox.x_max == 1.5
    assert combined_bbox.y_min == 0
    assert combined_bbox.y_max == 1.5
    assert combined_bbox.z_min == 0
    assert combined_bbox.z_max == 1.5


def test_bounding_box_to_prism():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    prism = bbox.to_prism()

    expected_prism = np.array(
        [
            [[0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]],
            [[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
            [[1, 0, 1], [0, 0, 1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 1, 0]],
            [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]],
        ]
    )
    np.testing.assert_array_equal(prism, expected_prism)


def test_bounding_box_calculate_volume():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    volume = bbox.calculate_volume()
    assert volume == 8


def test_to_empty_volume():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    empty_volume = bbox.to_empty_volume()
    assert empty_volume.shape == (2, 2, 2)
    assert empty_volume[0, 0, 0] == 0
    assert empty_volume.dtype == np.float32
    assert empty_volume.ndim == 3
    assert empty_volume.size == 8
    assert np.all(empty_volume == 0)


def test_to_dict():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    bbox_dict = bbox.to_dict()

    assert bbox_dict["x_min"] == 0
    assert bbox_dict["x_max"] == 1
    assert bbox_dict["y_min"] == 0
    assert bbox_dict["y_max"] == 1
    assert bbox_dict["z_min"] == 0
    assert bbox_dict["z_max"] == 1


def test_from_dict():
    bbox_dict = {
        "x_min": 0,
        "x_max": 1,
        "y_min": 0,
        "y_max": 1,
        "z_min": 0,
        "z_max": 1,
    }
    bbox = BoundingBox.from_dict(bbox_dict)
    assert bbox.x_min == 0
    assert bbox.x_max == 1
    assert bbox.y_min == 0
    assert bbox.y_max == 1
    assert bbox.z_min == 0
    assert bbox.z_max == 1


def test_from_rects():
    # Create a sample array of rectangles (n, 4, 3) where n is the number of rectangles,
    # 4 is the number of points in each rectangle, and 3 is the number of dimensions
    # Note: These rectangles are not actual rectangle, but just a collection of points
    # for the sake of testing
    rects = np.array(
        [
            [[0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]],
            [[0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 1]],
            [[1, 1, 1], [2, 2, 2], [2, 1, 2], [1, 2, 2]],
            [[2, 2, 2], [3, 3, 3], [3, 2, 3], [2, 3, 3]],
        ]
    )

    combined_bbox = BoundingBox.from_rects(rects)
    assert combined_bbox.x_min == 0
    assert combined_bbox.x_max == 3
    assert combined_bbox.y_min == 0
    assert combined_bbox.y_max == 3
    assert combined_bbox.z_min == 0
    assert combined_bbox.z_max == 3


def test_bounding_box_params():
    params = BoundingBoxParams()

    dict_params = params.to_dict()

    assert dict_params["max_depth"] == 10
    assert dict_params["target_slices_per_box"] == 128

    new_params = BoundingBoxParams.from_dict(dict_params)

    assert new_params.max_depth == 10
    assert new_params.target_slices_per_box == 128


def test_to_cloudvolume_bbox():
    rect = np.array([[0, 0, 0], [1, 1, 1]])
    bbox = BoundingBox(rect)
    cloudvolume_bbox = bbox.to_cloudvolume_bbox()
    assert cloudvolume_bbox.dx == 1
    assert cloudvolume_bbox.dy == 1
    assert cloudvolume_bbox.dz == 1
    assert cloudvolume_bbox.ndim == 3


def test_longest_dimension():
    rect = np.array([[0, 0, 0], [2, 1, 1]])
    bbox = BoundingBox(rect)
    assert bbox.longest_dimension() == 0

    rect = np.array([[0, 0, 0], [1, 2, 1]])
    bbox = BoundingBox(rect)
    assert bbox.longest_dimension() == 1

    rect = np.array([[0, 0, 0], [1, 1, 2]])
    bbox = BoundingBox(rect)
    assert bbox.longest_dimension() == 2


def test_calculate_bounding_boxes_bsp_link_rects_empty():
    rects = np.array([])
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects)
    assert bounding_boxes == []
    assert rect_to_box_map == []


def test_calculate_bounding_boxes_bsp_link_rects_single_rect():
    rects = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects)
    assert len(bounding_boxes) == 1
    assert rect_to_box_map == [0]


def test_calculate_bounding_boxes_bsp_link_rects_multiple_non_overlapping_rects():
    rects = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[2, 2, 2], [3, 2, 2], [3, 3, 2], [2, 3, 2]],
        ]
    )
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects)
    assert len(bounding_boxes) == 1
    assert rect_to_box_map == [0, 0]


def test_calculate_bounding_boxes_bsp_link_rects_multiple_overlapping_rects():
    rects = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0.5, 0.5, 0], [1.5, 0.5, 0], [1.5, 1.5, 0], [0.5, 1.5, 0]],
        ]
    )
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects)
    assert len(bounding_boxes) == 1
    assert rect_to_box_map == [0, 0]


def test_calculate_bounding_boxes_bsp_link_rects_full_curve():
    # Sample points arranged in a simple curve
    sample_points = generate_sample_curve_helix(
        start_z=2000, end_z=2500, num_points=1000, radius=30
    )

    # Initialize Spline object
    spline = Spline(sample_points, degree=3)

    # Generate a range of t values
    equidistant_times = spline.calculate_equidistant_parameters(
        distance_between_points=1
    )

    WIDTH = 10
    HEIGHT = 10

    # Calculate slice rects
    slice_rects = calculate_slice_rects(equidistant_times, spline, WIDTH, HEIGHT)

    bounding_boxes, link_rects = calculate_bounding_boxes_bsp_link_rects(slice_rects)

    # Make sure that all slices are contained in their respective bounding boxes
    for i, rect in enumerate(slice_rects):
        bbox = bounding_boxes[link_rects[i]]

        x_min, x_max, y_min, y_max, z_min, z_max = bbox.approx_bounds()

        assert x_min <= rect[:, 0].min()
        assert x_max >= rect[:, 0].max()
        assert y_min <= rect[:, 1].min()
        assert y_max >= rect[:, 1].max()
        assert z_min <= rect[:, 2].min()
        assert z_max >= rect[:, 2].max()


def test_boxes_dim_range():
    rects = [np.array([[1, 1, 1], [1.5, 3.5, 1.5]]), np.array([[0, 1, 1], [4.5, 2.5, 3.5]])]
    bboxes = [BoundingBox(rect) for rect in rects]

    # Test default (z)
    assert np.all(boxes_dim_range(bboxes) == np.arange(1, 5, dtype=int))

    # Test parameter
    assert np.all(boxes_dim_range(bboxes, 'x') == np.arange(0, 6, dtype=int))


def test_shoud_be_divided():
    rects = [np.array([[1, 1, 1], [1.5, 3.5, 1.5]]), np.array([[0, 1, 1], [4.5, 2.5, 3.5]])]
    bboxes = [BoundingBox(rect) for rect in rects]

    # Test where utilized volume is very small (<10%) and box should be divided.
    assert bboxes[0].should_be_divided(1)

    # Test where utilized volume is reasonable (>10^) and box should not be divided.
    assert not bboxes[0].should_be_divided(10)
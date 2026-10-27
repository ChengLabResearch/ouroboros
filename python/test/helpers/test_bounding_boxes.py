import numpy as np
import pytest
from ouroboros.helpers.bounding_boxes import (
    BoundingBox,
    BoundingBoxParams,
    calculate_bounding_boxes_bsp_link_rects,
    boxes_dim_range,
    _PartitionNode,
    _path_partition,
    _select_partition_threshold,
    slice_indices_are_contiguous,
)
from ouroboros.helpers.slice import calculate_slice_rects, estimate_slice_task_payload
from ouroboros.helpers.spline import Spline
from test.sample_data import generate_sample_curve_helix


def payload(box, count, contiguous):
    return int(box.calculate_volume()) + count * (12 if contiguous else 20)


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


def test_integer_origin_preserves_the_geometric_bounds():
    bbox = BoundingBox(np.array([[-1.9, 2.3, 4], [3.1, 5.8, 4]]))

    origin = bbox.get_integer_origin()

    assert origin.dtype == np.int64
    np.testing.assert_array_equal(origin, [-2, 2, 4])
    assert bbox.get_shape() == (7, 5, 1)
    np.testing.assert_array_equal(bbox.get_min(), [-1.9, 2.3, 4])
    np.testing.assert_allclose(bbox.to_cloudvolume_bbox().minpt, [-1.9, 2.3, 4])
    np.testing.assert_allclose(bbox.to_cloudvolume_bbox().maxpt, [3.1, 5.8, 4])


def test_bounding_box_volume_slice_and_extrema():
    rect = np.array([[1, 2, 3], [4, 6, 8]])
    bbox = BoundingBox(rect)

    vol_slice = bbox.vol_slice()
    assert vol_slice[0] == slice(1, 4)
    assert vol_slice[1] == slice(2, 6)
    assert vol_slice[2] == slice(3, 8)
    np.testing.assert_array_equal(bbox.get_min(dtype=np.int16), np.array([1, 2, 3], dtype=np.int16))
    np.testing.assert_array_equal(bbox.get_max(dtype=np.float32), np.array([4, 6, 8], dtype=np.float32))


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


@pytest.mark.parametrize("task_payload", [None, payload], ids=["count", "payload"])
def test_calculate_bounding_boxes_bsp_link_rects_empty(task_payload):
    rects = np.array([])
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=task_payload)
    assert bounding_boxes == []
    assert rect_to_box_map == []


@pytest.mark.parametrize("task_payload", [None, payload], ids=["count", "payload"])
def test_calculate_bounding_boxes_bsp_link_rects_single_rect(task_payload):
    rects = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=task_payload)
    assert len(bounding_boxes) == 1
    assert rect_to_box_map == [0]


@pytest.mark.parametrize("task_payload", [None, payload], ids=["count", "payload"])
def test_calculate_bounding_boxes_bsp_link_rects_multiple_non_overlapping_rects(task_payload):
    rects = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[2, 2, 2], [3, 2, 2], [3, 3, 2], [2, 3, 2]],
        ]
    )
    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=task_payload)
    assert len(bounding_boxes) == (1 if task_payload is None else 2)
    assert_assignment(rects, bounding_boxes, rect_to_box_map)
    if task_payload is not None:
        reference = calculate_bounding_boxes_bsp_link_rects(rects)
        peak, volume = partition_metrics(bounding_boxes, rect_to_box_map)
        reference_peak, reference_volume = partition_metrics(*reference)
        assert peak < reference_peak
        assert volume <= reference_volume


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


@pytest.mark.parametrize("task_payload", [None, payload], ids=["count", "payload"])
def test_calculate_bounding_boxes_bsp_link_rects_unsplittable_partition(task_payload):
    rect = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    rects = np.repeat(rect[None, :, :], 3, axis=0)

    bounding_boxes, rect_to_box_map = calculate_bounding_boxes_bsp_link_rects(
        rects,
        target_slices_per_box=1,
        max_depth=5,
        task_payload=task_payload,
    )

    assert len(bounding_boxes) == 1
    assert rect_to_box_map == [0, 0, 0]


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

    original = slice_rects.copy()

    def model(box, count, contiguous):
        return estimate_slice_task_payload(
            box, count, contiguous, slice_shape=(HEIGHT, WIDTH),
            source_dtype=np.uint16, channels=1,
        ).peak_bytes

    reference = calculate_bounding_boxes_bsp_link_rects(slice_rects)
    selected = calculate_bounding_boxes_bsp_link_rects(slice_rects, task_payload=model)
    for result in (reference, selected):
        assert_assignment(slice_rects, *result)
    peak, volume = partition_metrics(*selected, model=model)
    reference_peak, reference_volume = partition_metrics(*reference, model=model)
    assert peak <= reference_peak
    assert volume <= reference_volume
    np.testing.assert_array_equal(slice_rects, original)


def test_boxes_dim_range():
    rects = [np.array([[1, 1, 1], [1.5, 3.5, 1.5]]), np.array([[0, 1, 1], [4.5, 2.5, 3.5]])]
    bboxes = [BoundingBox(rect) for rect in rects]

    # Test default (z)
    assert np.all(boxes_dim_range(bboxes) == np.arange(1, 5, dtype=int))

    # Test parameter
    assert np.all(boxes_dim_range(bboxes, 'x') == np.arange(0, 6, dtype=int))


@pytest.mark.parametrize("dim", ["x", "y", "z"])
@pytest.mark.parametrize("maximum, expected", [(3, [1, 2, 3]), (3.5, [1, 2, 3, 4])])
def test_boxes_dim_range_stops_at_the_last_allocated_voxel(dim, maximum, expected):
    boxes = [BoundingBox(np.full((2, 3), 1.5)), BoundingBox(np.full((2, 3), maximum))]
    np.testing.assert_array_equal(boxes_dim_range(boxes, dim), expected)
    np.testing.assert_array_equal(boxes_dim_range(boxes[1:], dim), [3] if maximum == 3 else [3, 4])


def test_shoud_be_divided():
    rects = [np.array([[1, 1, 1], [1.5, 3.5, 1.5]]), np.array([[0, 1, 1], [4.5, 2.5, 3.5]])]
    bboxes = [BoundingBox(rect) for rect in rects]

    # Test where utilized volume is very small (<10%) and box should be divided.
    assert bboxes[0].should_be_divided(1)

    # Test where utilized volume is reasonable (>10^) and box should not be divided.
    assert not bboxes[0].should_be_divided(10)


def yz_planes(positions, width=1):
    return np.array([
        [[x, 0, 0], [x, width, 0], [x, width, width], [x, 0, width]]
        for x in positions
    ], dtype=float)


def xy_planes(positions, width=9):
    return np.array([
        [[x, 0, 0], [x + width, 0, 0], [x + width, width, 0], [x, width, 0]]
        for x in positions
    ], dtype=float)


def partition_metrics(boxes, mapping, model=payload):
    groups = [np.flatnonzero(np.asarray(mapping) == i) for i in range(len(boxes))]
    return (
        max(model(box, len(indices), slice_indices_are_contiguous(indices))
            for box, indices in zip(boxes, groups)),
        sum(int(box.calculate_volume()) for box in boxes),
    )


def assert_assignment(rects, boxes, mapping):
    assert len(mapping) == len(rects)
    assert set(mapping) == set(range(len(boxes)))
    for index, box_index in enumerate(mapping):
        box = boxes[box_index]
        assert np.all(rects[index] >= box.get_min())
        assert np.all(rects[index] <= box.get_max())
    for box_index, box in enumerate(boxes):
        expected = BoundingBox.from_rects(rects[np.asarray(mapping) == box_index])
        assert box.to_dict() == expected.to_dict()


def test_search_can_cross_an_infeasible_intermediate_volume():
    # The preferred first path cut has source volume 700 + 700, above the
    # reference's 1000. Its descendants total only 400. Rejecting a first split
    # against the global cap would miss this valid final partition.
    rects = xy_planes([0, 60, 30, 90])

    def model(box, count, contiguous):
        return int(box.calculate_volume()) + 10 * count + (0 if contiguous else 10000)

    result = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=model)
    assert len(result[0]) == 4
    assert partition_metrics(*result, model=model) == (110, 400)
    assert_assignment(rects, *result)


def test_search_expands_the_lowest_payload_node_when_children_cost_more():
    # Payload is permitted to increase after a split (for example when the
    # assignment gains an output buffer). The lowest node score must not hide
    # a final frontier that meets both comparison constraints.
    root = BoundingBox(BoundingBox.bounds_to_rect(0, 99, 0, 0, 0, 0))
    left = BoundingBox(BoundingBox.bounds_to_rect(0, 29, 0, 0, 0, 0))
    right = BoundingBox(BoundingBox.bounds_to_rect(70, 99, 0, 0, 0, 0))
    nodes = [
        _PartitionNode(root, np.array([0, 1]), 10, (1, 2)),
        _PartitionNode(left, np.array([0]), 20),
        _PartitionNode(right, np.array([1]), 20),
    ]
    threshold = _select_partition_threshold(nodes, reference_peak=30, reference_volume=70)
    assert threshold is not None
    assert threshold < nodes[0].payload
    assert max(node.payload for node in nodes[1:]) == 20
    assert sum(node.box.calculate_volume() for node in nodes[1:]) == 60


def test_spatial_overlap_cannot_exceed_reference_source_volume():
    rects = xy_planes([0, 5, 10, 15], width=10)
    reference = calculate_bounding_boxes_bsp_link_rects(rects)
    # Each nontrivial partition duplicates source voxels, although it reduces
    # this count-only model. The requested-volume constraint wins.
    result = calculate_bounding_boxes_bsp_link_rects(
        rects, task_payload=lambda box, count, contiguous: count
    )
    assert [box.to_dict() for box in result[0]] == [box.to_dict() for box in reference[0]]
    assert result[1] == reference[1]


def test_coincident_centers_with_distinct_orientations_can_split():
    xy = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]])
    yz = xy[:, [2, 0, 1]]
    rects = np.stack([xy, yz])
    assert np.array_equal(rects.mean(axis=1), np.zeros((2, 3)))
    result = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=payload)
    assert len(result[0]) == 2
    assert_assignment(rects, *result)


@pytest.mark.parametrize("positions, indices", [
    ([15, 17, 14, 14, 24, 11], [0, 1, 2, 3, 4, 5]),
    ([0, 18, 19], [0, 1, 2]),  # Child at exactly 95% of the inclusive volume.
    ([0, 19, 18], [0, 1, 2]),  # Invalid cuts at the end of the path.
    ([0, 0, 19, 19], [0, 1, 2, 3]),  # One valid cut between invalid ends.
    ([-3.8, -2.1, -0.4, 1.6], [0, 2, 3, 7]),  # Fractional bounds and index gaps.
    ([-0.1, -0.1, 17.0, 18], [0, 1, 2, 3]),
    ([-0.1, -0.1, 17.1, 18], [0, 1, 2, 3]),  # Ceil removes a boundary-valid cut.
    ([0, 2, 4], [0, 1, 2]),  # Equal scores retain the first cut.
    ([0, 19, 0], [0, 1, 2]),  # No cut satisfies both child constraints.
    ([0, 0, 0], [0, 1, 2]),
    ([0, 19], [0, 1]),
])
def test_valid_path_cut_survives_an_invalid_unrestricted_minimum(positions, indices):
    rects = yz_planes(positions)
    indices = np.asarray(indices)
    box = BoundingBox.from_rects(rects)
    model = lambda child, count, contiguous: int(child.calculate_volume()) + 20 * count
    # Exhaustive candidates derive extents from actual corners, independently of
    # the prefix/suffix search. Observe callback arguments as well as the winner.
    expected_calls, candidates = [], []
    for cut in range(1, len(rects)):
        left, right = BoundingBox.from_rects(rects[:cut]), BoundingBox.from_rects(rects[cut:])
        volumes = [int(child.calculate_volume()) for child in (left, right)]
        if 20 * max(volumes) > 19 * int(box.calculate_volume()):
            continue
        costs = []
        for child, assigned in ((left, indices[:cut]), (right, indices[cut:])):
            contiguous = slice_indices_are_contiguous(assigned)
            expected_calls.append((child.to_dict(), len(assigned), contiguous))
            costs.append(model(child, len(assigned), contiguous))
        candidates.append(((max(costs), sum(volumes)), cut, left, right))
    calls = []

    def observed(child, count, contiguous):
        calls.append((child.to_dict(), count, contiguous))
        return model(child, count, contiguous)

    children = _path_partition(indices, rects, int(box.calculate_volume()), observed)
    assert calls == expected_calls
    if not candidates:
        assert children is None
        return
    _, cut, left, right = min(candidates, key=lambda candidate: candidate[0])
    assert [child.box.to_dict() for child in children] == [left.to_dict(), right.to_dict()]
    np.testing.assert_array_equal(children[0].indices, indices[:cut])
    np.testing.assert_array_equal(children[1].indices, indices[cut:])
    # The unrestricted minimax cut after slice 4 leaves a child equal to the
    # parent extent. The valid cut after slice 5 must still be considered.
    if positions == [15, 17, 14, 14, 24, 11]:
        assert [len(child.indices) for child in children] == [5, 1]
    assert max(child.box.calculate_volume() for child in children) <= 0.95 * box.calculate_volume()


def test_returning_path_is_deterministic_and_assigned_once():
    rects = yz_planes([0, 2, 4, 6, 8, 6, 4, 2, 0])
    first = calculate_bounding_boxes_bsp_link_rects(rects, 3, 6, task_payload=payload)
    second = calculate_bounding_boxes_bsp_link_rects(rects, 3, 6, task_payload=payload)
    assert first[1] == second[1]
    assert np.array_equal(
        np.array([box.approx_bounds() for box in first[0]]),
        np.array([box.approx_bounds() for box in second[0]]),
    )
    reference = calculate_bounding_boxes_bsp_link_rects(rects, 3, 6)
    peak, volume = partition_metrics(*first)
    reference_peak, reference_volume = partition_metrics(*reference)
    assert peak <= reference_peak
    assert volume <= reference_volume
    assert_assignment(rects, *first)


@pytest.mark.parametrize("max_depth", [0, 1])
def test_depth_limit_is_respected(max_depth):
    rects = yz_planes([0, 5, 10, 15, 20, 25])
    result = calculate_bounding_boxes_bsp_link_rects(rects, 128, max_depth, task_payload=payload)
    assert len(result[0]) <= 2 ** max_depth
    assert_assignment(rects, *result)


def test_reference_is_retained_without_payload_improvement():
    rects = yz_planes([0, 5, 10, 15])
    reference = calculate_bounding_boxes_bsp_link_rects(rects, 2, 6)
    result = calculate_bounding_boxes_bsp_link_rects(
        rects, 2, 6, task_payload=lambda box, count, contiguous: 1
    )
    assert result[1] == reference[1]
    assert [box.to_dict() for box in result[0]] == [box.to_dict() for box in reference[0]]


@pytest.mark.parametrize("indices, expected", [
    ([], False), ([-1], False), ([0], True), ([2, 3, 4], True),
    ([0, 2], False), ([0, 2, 1, 3], False), ([1, 1], False),
])
def test_contiguity_matches_output_interval_contract(indices, expected):
    assert slice_indices_are_contiguous(np.asarray(indices)) is expected

"""Behavioral checks for payload selection within the bounding-box owner."""

import numpy as np
import pytest

from ouroboros.helpers.bounding_boxes import (
    BoundingBox,
    _PartitionNode,
    _calculate_bounding_boxes_count,
    _path_partition,
    _select_partition_threshold,
    calculate_bounding_boxes_bsp_link_rects,
    slice_indices_are_contiguous,
)


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


def payload(box, count, contiguous):
    return int(box.calculate_volume()) + count * (12 if contiguous else 20)


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


def test_payload_splits_separated_slices_below_count_target():
    rects = yz_planes([0, 1, 100, 101])
    reference = _calculate_bounding_boxes_count(rects, 128, 10)
    result = calculate_bounding_boxes_bsp_link_rects(rects, 128, 10, task_payload=payload)
    peak, volume = partition_metrics(*result)
    reference_peak, reference_volume = partition_metrics(*reference)
    assert len(reference[0]) == 1
    assert len(result[0]) > 1
    assert peak < reference_peak
    assert volume <= reference_volume
    assert_assignment(rects, *result)


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
    reference = _calculate_bounding_boxes_count(rects)
    # Each nontrivial partition duplicates source voxels, although it reduces
    # this count-only model. The requested-volume constraint wins.
    result = calculate_bounding_boxes_bsp_link_rects(
        rects, task_payload=lambda box, count, contiguous: count
    )
    assert [box.to_dict() for box in result[0]] == [box.to_dict() for box in reference[0]]
    assert result[1] == reference[1]


def test_identical_rectangles_remain_unsplit():
    rects = np.repeat(yz_planes([0]), 200, axis=0)
    boxes, mapping = calculate_bounding_boxes_bsp_link_rects(
        rects, 1, 10, task_payload=payload
    )
    assert len(boxes) == 1
    assert mapping == [0] * len(rects)


def test_coincident_centers_with_distinct_orientations_can_split():
    xy = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]])
    yz = xy[:, [2, 0, 1]]
    rects = np.stack([xy, yz])
    assert np.array_equal(rects.mean(axis=1), np.zeros((2, 3)))
    result = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=payload)
    assert len(result[0]) == 2
    assert_assignment(rects, *result)


def test_valid_path_cut_survives_an_invalid_unrestricted_minimum():
    rects = yz_planes([15, 17, 14, 14, 24, 11])
    box = BoundingBox.from_rects(rects)
    model = lambda child, count, contiguous: int(child.calculate_volume()) + 20 * count
    children = _path_partition(np.arange(len(rects)), rects, int(box.calculate_volume()), model)
    # The unrestricted minimax cut after slice 4 leaves a child equal to the
    # parent extent. The valid cut after slice 5 must still be considered.
    assert children is not None
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
    reference = _calculate_bounding_boxes_count(rects, 3, 6)
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


def test_reference_is_unchanged_without_a_model_or_without_improvement():
    rects = yz_planes([0, 5, 10, 15])
    reference = _calculate_bounding_boxes_count(rects, 2, 6)
    for result in (
        calculate_bounding_boxes_bsp_link_rects(rects, 2, 6),
        calculate_bounding_boxes_bsp_link_rects(
            rects, 2, 6, task_payload=lambda box, count, contiguous: 1
        ),
    ):
        assert result[1] == reference[1]
        assert [box.to_dict() for box in result[0]] == [box.to_dict() for box in reference[0]]


@pytest.mark.parametrize("rects", [np.empty((0, 4, 3)), yz_planes([0])])
def test_empty_and_single_slice_inputs(rects):
    boxes, mapping = calculate_bounding_boxes_bsp_link_rects(rects, task_payload=payload)
    assert len(boxes) == len(rects)
    assert len(mapping) == len(rects)


@pytest.mark.parametrize("indices, expected", [
    ([], False), ([-1], False), ([0], True), ([2, 3, 4], True),
    ([0, 2], False), ([0, 2, 1, 3], False), ([1, 1], False),
])
def test_contiguity_matches_output_interval_contract(indices, expected):
    assert slice_indices_are_contiguous(np.asarray(indices)) is expected

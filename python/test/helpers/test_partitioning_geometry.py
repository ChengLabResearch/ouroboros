"""Acceptance evidence from the recorded 10,386-slice nerve annotation path."""

import numpy as np

from benchmarks.benchmark_partitioning import load_specimen, partition_report
from ouroboros.helpers.bounding_boxes import (
    _calculate_bounding_boxes_count,
    calculate_bounding_boxes_bsp_link_rects,
)
from ouroboros.helpers.slice import estimate_slice_task_payload


def test_real_nerve_reduces_task_payload_without_more_requested_source_bytes():
    rects, specimen = load_specimen("octopus_nerve_cords")
    original = rects.copy()
    config = specimen["slice_options"]
    params = config["bounding_box_params"]

    def task_payload(box, count, contiguous):
        return estimate_slice_task_payload(
            box, count, contiguous,
            slice_shape=(config["slice_height"], config["slice_width"]),
            source_dtype=specimen["source_dtype"], channels=specimen["channels"],
        ).peak_bytes

    reference_boxes, reference_mapping = _calculate_bounding_boxes_count(rects, **params)
    boxes, mapping = calculate_bounding_boxes_bsp_link_rects(rects, **params, task_payload=task_payload)
    reference = partition_report(rects, reference_boxes, reference_mapping, specimen, include_chunks=False)
    selected = partition_report(rects, boxes, mapping, specimen, include_chunks=False)

    # The recorded input reproduces the issue's real failure case, not a substitute curve.
    assert rects.shape == (10386, 4, 3)
    assert reference["boxes"] == 161
    assert reference["slice_count_distribution"]["max"] == 523
    assert reference["noncontiguous_boxes"] == 14
    # These are the requested change's payload reduction and declared 0% byte bound.
    assert selected["max_payload_bytes"] < reference["max_payload_bytes"]
    assert selected["cumulative_requested_source_bytes"] <= reference["cumulative_requested_source_bytes"]
    # partition_report validates exactly-once links and tight assigned-corner bounds.
    assert selected["exactly_once_assignment_and_tight_corner_bounds"]
    np.testing.assert_array_equal(rects, original)

"""Subtree execution preserves the serial planner's scientific contract."""

from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing
import os
import pickle

import numpy as np
import pytest

from ouroboros.helpers import bounding_boxes as partitioning
from ouroboros.helpers import slice as slicing
from ouroboros.helpers.slice import calculate_slice_rects, estimate_slice_task_payload
from ouroboros.helpers.spline import Spline
from test.sample_data import generate_sample_curve_helix


START_METHODS = [
    method for method in ("spawn", "fork")
    if method in multiprocessing.get_all_start_methods()
]


def _production_payload(box, count, contiguous):
    return estimate_slice_task_payload(
        box, count, contiguous, slice_shape=(4, 4),
        source_dtype=np.uint16, channels=2,
    ).peak_bytes


def _nonmonotone_payload(box, count, contiguous):
    # Deliberately violates monotonicity and distinguishes gapped assignments.
    return (int(box.calculate_volume()) % 29 + count % 7
            + (11 if contiguous else 37))


@pytest.fixture
def curve_rects():
    points = generate_sample_curve_helix(
        start_z=-12, end_z=12, radius=8, num_points=100,
    )
    return calculate_slice_rects(
        np.linspace(0, 1, 96), Spline(points, degree=3), 4, 4,
    )


class _ObservedPool:
    """Observe native process handles without owning their shutdown policy."""

    def __init__(self, start_method, on_enter=None):
        self.context = multiprocessing.get_context(start_method)
        self.on_enter = on_enter
        self.processes = []
        self.exited = False

    @contextmanager
    def __call__(self):
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=self.context) as pool:
                if self.on_enter is not None:
                    self.on_enter()
                try:
                    yield pool
                finally:
                    # Retain the process objects before executor shutdown clears
                    # its registry, so joining is checked after any outcome.
                    self.processes.extend(pool._processes.values())
        finally:
            self.exited = True

    def assert_joined(self):
        assert self.exited
        assert len(self.processes) == 1
        assert all(process.exitcode is not None for process in self.processes)
        assert all(not process.is_alive() for process in self.processes)


def _assert_same_tree(actual, expected):
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert type(result.box) is partitioning.BoundingBox
        assert result.box.to_dict() == reference.box.to_dict()
        np.testing.assert_array_equal(result.indices, reference.indices)
        assert result.payload == reference.payload
        assert result.children == reference.children


def _assert_same_partition(actual, expected):
    assert [box.to_dict() for box in actual[0]] == [
        box.to_dict() for box in expected[0]
    ]
    assert actual[1] == expected[1]


@pytest.mark.parametrize("start_method", START_METHODS)
@pytest.mark.parametrize("payload", [_production_payload, _nonmonotone_payload])
def test_native_subtrees_preserve_exact_tree_and_selected_partition(
    curve_rects, start_method, payload,
):
    original = curve_rects.copy()
    expected_tree = partitioning._partition_tree(curve_rects, 5, payload)
    assert len(expected_tree) > 3
    if payload is _nonmonotone_payload:
        assert any(
            expected_tree[child].payload > node.payload
            for node in expected_tree for child in node.children or ()
        )
    observed = _ObservedPool(start_method)
    actual_tree = partitioning._partition_tree(
        curve_rects, 5, payload, executor_factory=observed,
    )
    observed.assert_joined()
    _assert_same_tree(actual_tree, expected_tree)

    expected = partitioning.calculate_bounding_boxes_bsp_link_rects(
        curve_rects, target_slices_per_box=16, max_depth=5,
        task_payload=payload,
    )
    observed = _ObservedPool(start_method)
    actual = partitioning.calculate_bounding_boxes_bsp_link_rects(
        curve_rects, target_slices_per_box=16, max_depth=5,
        task_payload=payload, executor_factory=observed,
    )
    observed.assert_joined()
    _assert_same_partition(actual, expected)
    np.testing.assert_array_equal(curve_rects, original)


@pytest.mark.parametrize("kind, depth", [
    ("curve", 0), ("curve", 1), ("single", 5), ("coincident", 5),
])
def test_no_pool_without_independent_subtrees(curve_rects, kind, depth):
    rects = curve_rects
    if kind == "single":
        rects = rects[:1]
    elif kind == "coincident":
        rects = np.repeat(rects[:1], 8, axis=0)

    def forbidden_pool():
        pytest.fail("Planner allocated a pool without independent subtree work")

    expected = partitioning._partition_tree(rects, depth, _production_payload)
    actual = partitioning._partition_tree(
        rects, depth, _production_payload, executor_factory=forbidden_pool,
    )
    _assert_same_tree(actual, expected)


@pytest.mark.parametrize("empty, payload", [(False, None), (True, _production_payload)])
def test_reference_only_calls_do_not_allocate_a_pool(curve_rects, empty, payload):
    rects = curve_rects[:0] if empty else curve_rects

    def forbidden_pool():
        pytest.fail("Reference-only planning must not allocate a pool")

    expected = partitioning.calculate_bounding_boxes_bsp_link_rects(rects)
    actual = partitioning.calculate_bounding_boxes_bsp_link_rects(
        rects, task_payload=payload, executor_factory=forbidden_pool,
    )
    _assert_same_partition(actual, expected)


def test_default_serial_path_retains_nested_pure_callbacks(curve_rects):
    offset = 13

    def nested_payload(box, count, contiguous):
        return _nonmonotone_payload(box, count, contiguous) + offset

    # A uniform cost offset preserves every candidate and threshold comparison.
    expected = partitioning.calculate_bounding_boxes_bsp_link_rects(
        curve_rects, task_payload=_nonmonotone_payload,
    )
    actual = partitioning.calculate_bounding_boxes_bsp_link_rects(
        curve_rects, task_payload=nested_payload,
    )
    _assert_same_partition(actual, expected)


def test_serializable_payload_adapter_calls_the_canonical_estimator(monkeypatch):
    adapter = slicing.SliceTaskPayloadEstimator((7, 11), np.dtype("uint16"), 3)
    restored = pickle.loads(pickle.dumps(adapter))
    box = partitioning.BoundingBox(np.array([[0, 0, 0], [2, 3, 4]]))
    estimate = slicing.SliceTaskPayload(17, 23, 31, 0, 0)
    calls = []

    def canonical_estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return estimate

    monkeypatch.setattr(slicing, "estimate_slice_task_payload", canonical_estimate)

    assert restored(box, 5, False) == estimate.peak_bytes
    assert calls == [((box, 5, False), {
        "slice_shape": (7, 11), "source_dtype": np.dtype("uint16"), "channels": 3,
    })]


def test_both_branches_reuse_the_canonical_builder_and_nodes(curve_rects, monkeypatch):
    build = partitioning._partition_subtree
    built = []
    submitted = []

    def observed_builder(*args):
        result = build(*args)
        built.append(result)
        return result

    class ImmediateExecutor:
        def submit(self, function, *args):
            submitted.append(function)
            future = Future()
            future.set_result(function(*args))
            return future

    @contextmanager
    def factory():
        yield ImmediateExecutor()

    monkeypatch.setattr(partitioning, "_partition_subtree", observed_builder)
    nodes = partitioning._partition_tree(
        curve_rects, 5, _nonmonotone_payload, executor_factory=factory,
    )

    assert submitted == [observed_builder]
    assert len(built) == 2
    # Assembly must reuse the canonical nodes, rather than reconstructing their
    # bounds, assignments, or payload from a competing representation.
    assert {id(node) for node in nodes[1:]} == {
        id(node) for subtree in built for node in subtree
    }


@dataclass
class _FailingPayload:
    parent_pid: int
    failure: str
    armed: bool = False

    def __call__(self, box, count, contiguous):
        worker = os.getpid() != self.parent_pid
        if worker and self.failure == "worker_exception":
            raise ValueError("worker payload failed")
        if worker and self.failure == "worker_exit":
            os._exit(23)
        if not worker and self.failure == "parent_exception" and self.armed:
            raise ValueError("parent payload failed")
        return _production_payload(box, count, contiguous)


@pytest.mark.parametrize("failure, error, message", [
    ("worker_exception", ValueError, "worker payload failed"),
    ("worker_exit", BrokenProcessPool, "terminated abruptly"),
    ("parent_exception", ValueError, "parent payload failed"),
])
def test_spawn_pool_joins_after_worker_or_parent_failure(
    curve_rects, failure, error, message,
):
    payload = _FailingPayload(os.getpid(), failure)
    observed = _ObservedPool(
        "spawn", on_enter=lambda: setattr(payload, "armed", True),
    )
    with pytest.raises(error, match=message):
        partitioning._partition_tree(
            curve_rects, 5, payload, executor_factory=observed,
        )
    observed.assert_joined()

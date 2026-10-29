"""Planner allocation belongs to the existing pipeline resource owner."""

from concurrent.futures import Future
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.common.run_statistics import RunStatistics
from ouroboros.common.server_types import SliceTask
from ouroboros.helpers.bounding_boxes import BoundingBox, BoundingBoxParams
from ouroboros.helpers.options import SliceOptions
from ouroboros.helpers.slice import SliceTaskPayloadEstimator
from ouroboros.helpers.volume_cache import CloudVolumeInterface, VolumeCache
from ouroboros.pipeline import volume_cache_pipeline as planning
from ouroboros.pipeline.pipeline import Pipeline, PipelineStep
from ouroboros.pipeline.pipeline_input import PipelineInput


def _input(tmp_path, count, *, depth=5, budget=4):
    options = SliceOptions.model_construct(
        output_file_folder=str(tmp_path), output_file_name="planned",
        neuroglancer_json="annotations.json", slice_width=7, slice_height=11,
        process_count=budget,
        bounding_box_params=BoundingBoxParams(max_depth=depth, target_slices_per_box=16),
    )
    rectangle = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float)
    return PipelineInput(
        slice_options=options, slice_rects=np.broadcast_to(rectangle, (count, 4, 3)),
        source_url="memory://source",
    )


def _source():
    source = object.__new__(CloudVolumeInterface)
    source.dtype = np.dtype("uint16")
    source.available_mips = [0]
    source.source_url = "memory://source"
    source.cv = SimpleNamespace(shape=(2000, 10, 10, 3))
    return source


def _context(monkeypatch, platform="linux"):
    context = SimpleNamespace(get_start_method=lambda: "fork", requests=[])
    # Replace only this module's sys reference; changing sys.platform itself
    # would also change dependency behavior on the actual test host.
    monkeypatch.setattr(planning, "sys", SimpleNamespace(platform=platform))

    def get_context(method):
        assert method == "fork"
        context.requests.append(method)
        return context

    monkeypatch.setattr(planning.multiprocessing, "get_context", get_context)
    return context


class _InlineExecutor:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("pool_entered")
        return self

    def submit(self, function, *args):
        self.events.append("submitted")
        future = Future()
        future.set_result(function(*args))
        return future

    def __exit__(self, *_):
        self.events.append("pool_closed")


@pytest.mark.parametrize("platform, count, depth, budget, workers", [
    ("linux", 511, 5, 4, 0),
    ("linux", 512, 5, 4, 1),
    ("darwin", 511, 5, 4, 0),
    ("darwin", 512, 5, 4, 1),
    ("win32", 512, 5, 4, 0),
    ("win32", 8192, 5, 4, 0),
    ("win32", 20000, 5, 4, 0),
    ("unmeasured", 20000, 5, 4, 0),
    ("linux", 512, 1, 4, 0),
    ("darwin", 512, 1, 4, 0),
    ("linux", 512, 5, 0, 0),
    ("darwin", 512, 5, 0, 0),
    ("linux", 512, 5, 1, 1),
    ("darwin", 512, 5, 1, 1),
    ("linux", 512, 5, 48, 1),
    ("darwin", 512, 5, 48, 1),
])
def test_planner_policy_honors_platform_cutoff_and_worker_budget(
    tmp_path, monkeypatch, platform, count, depth, budget, workers,
):
    data = _input(tmp_path, count, depth=depth, budget=budget)
    context = _context(monkeypatch, platform)
    source = _source()
    step = planning.VolumeCachePipelineStep()
    events, allocations = [], []
    monkeypatch.setattr(planning, "CloudVolumeInterface", lambda _: source)

    def executor(*, max_workers, mp_context):
        assert mp_context is context
        assert max_workers == step.execution_resources["planner_processes"]
        allocations.append(max_workers)
        return _InlineExecutor(events)

    def partition(rects, *, target_slices_per_box, max_depth, task_payload,
                  executor_factory):
        assert rects is data.slice_rects
        assert max_depth == depth
        assert target_slices_per_box == 16
        assert type(task_payload) is SliceTaskPayloadEstimator
        assert asdict(task_payload) == {
            "slice_shape": (11, 7), "source_dtype": source.dtype,
            "channels": source.num_channels,
        }
        assert (executor_factory is not None) is bool(workers)
        if executor_factory is not None:
            with executor_factory():
                pass
        return [BoundingBox.from_rects(rects)], [0] * len(rects)

    monkeypatch.setattr(planning, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(planning, "calculate_bounding_boxes_bsp_link_rects", partition)
    result, error = Pipeline([step]).process(data)

    assert error is None and result is data
    assert allocations == ([1] if workers else [])
    assert context.requests == (["fork"] if workers else [])
    assert step.execution_resources == {
        "planner_processes": workers,
        **({"planner_start_method": "fork"} if workers else {}),
    }
    assert data.volume_cache.cv is source
    assert events == (["pool_entered", "pool_closed"] if workers else [])


def test_real_planning_closes_pool_before_cache_and_next_step(tmp_path, monkeypatch):
    data = _input(tmp_path, 512, budget=48)
    rects = data.slice_rects.copy()
    rects[:, :, 0] += 3 * np.arange(len(rects))[:, None]
    data.slice_rects = rects
    original = rects.copy()
    source = _source()
    context = _context(monkeypatch)
    estimator = SliceTaskPayloadEstimator((11, 7), source.dtype, source.num_channels)
    expected_boxes, expected_mapping = planning.calculate_bounding_boxes_bsp_link_rects(
        rects, target_slices_per_box=16, max_depth=5, task_payload=estimator,
    )
    step = planning.VolumeCachePipelineStep()
    events, allocations = [], []

    def executor(*, max_workers, mp_context):
        assert mp_context is context
        allocations.append(max_workers)
        assert max_workers == step.execution_resources["planner_processes"]
        return _InlineExecutor(events)

    def cache(*args, **kwargs):
        assert events[-1] == "pool_closed"
        events.append("cache_created")
        return VolumeCache(*args, **kwargs)

    class NextStep(PipelineStep):
        def __init__(self):
            super().__init__(())

        def _process(self, _):
            assert events[-1] == "cache_created"
            events.append("next_step")

    monkeypatch.setattr(planning, "CloudVolumeInterface", lambda _: source)
    monkeypatch.setattr(planning, "ProcessPoolExecutor", executor)
    monkeypatch.setattr(planning, "VolumeCache", cache)
    pipeline = Pipeline([step, NextStep()])
    result, error = pipeline.process(data)

    assert error is None and result is data
    assert allocations == [1]
    assert context.requests == ["fork"]
    assert events == [
        "pool_entered", "submitted", "pool_closed", "cache_created", "next_step",
    ]
    assert [box.to_dict() for box in data.volume_cache.bounding_boxes] == [
        box.to_dict() for box in expected_boxes
    ]
    assert data.volume_cache.link_rects == expected_mapping
    np.testing.assert_array_equal(rects, original)

    task = SliceTask(
        task_id="planner-allocation", options="unused", pipeline=pipeline,
        pipeline_input=data,
    )
    statistics = RunStatistics(task)
    assert statistics.to_record()["execution"]["workers"] == [
        {"step": step.step_name, **step.execution_resources},
    ]
    # Reporting observes the owner's current record, rather than keeping an
    # independent requested budget or stale allocation snapshot.
    step.reset_statistics()
    assert statistics.to_record()["execution"]["workers"] == []


def test_unsplittable_root_reports_zero_actual_workers(tmp_path, monkeypatch):
    data = _input(tmp_path, 512)
    context = _context(monkeypatch)
    monkeypatch.setattr(planning, "CloudVolumeInterface", lambda _: _source())

    def forbidden_pool(**_):
        pytest.fail("Unsplittable geometry must not allocate a planner process")

    monkeypatch.setattr(planning, "ProcessPoolExecutor", forbidden_pool)
    step = planning.VolumeCachePipelineStep()
    pipeline = Pipeline([step])
    _, error = pipeline.process(data)

    assert error is None
    assert context.requests == []
    assert step.execution_resources == {"planner_processes": 0}
    assert len(data.volume_cache.bounding_boxes) == 1
    task = SliceTask(
        task_id="serial-allocation", options="unused", pipeline=pipeline,
        pipeline_input=data,
    )
    assert RunStatistics(task).to_record()["execution"]["workers"] == [
        {"step": step.step_name, "planner_processes": 0},
    ]

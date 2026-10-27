import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import threading
import weakref
from contextlib import contextmanager
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import psutil
import pytest
import tifffile
from cloudvolume import Bbox
from scipy.ndimage import map_coordinates

from ouroboros.common.pipelines import geometry_pipeline
from ouroboros.common.run_statistics import RunStatistics
from ouroboros.common.server_types import SliceTask
from ouroboros.helpers import slice as interpolation
from ouroboros.helpers import files as files_module
from ouroboros.helpers import volume_cache as cache_module
from ouroboros.helpers.bounding_boxes import (
    BoundingBox, calculate_bounding_boxes_bsp_link_rects, slice_indices_are_contiguous,
)
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers.shapes import ImgSliceC
from ouroboros.pipeline import backproject_pipeline as backproject_module
from ouroboros.pipeline import pipeline as pipeline_module
from ouroboros.pipeline import slice_parallel_pipeline as slice_module
from ouroboros.pipeline.pipeline import Pipeline, PipelineStep
from ouroboros.pipeline.pipeline_input import BasePipelineInput, PipelineInput
from test.sample_data import generate_sample_curve_helix


class FunctionStep(PipelineStep):
    def __init__(self, action=lambda step: None):
        super().__init__(())
        self.action = action

    def _process(self, inputs):
        return self.action(self)


def clock(values):
    return SimpleNamespace(perf_counter=iter(values).__next__)


def test_overlapping_worker_samples_have_explicit_nested_semantics_and_repeatable_summaries(monkeypatch):
    barrier = threading.Barrier(2)

    def work(step):
        def worker():
            barrier.wait(timeout=2)
            step.add_timing("compute", np.float64(0.75))
            step.add_timing("total_worker", 1.0, includes=("compute",))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: worker(), range(2)))

    step = FunctionStep(work)
    pipeline = Pipeline([step]).set_statistics_enabled(True)
    monkeypatch.setattr(pipeline_module, "time", clock([0, 0, 1, 1]))
    pipeline.process(BasePipelineInput())
    original = pipeline.get_step_statistics()
    assert original == pipeline.get_step_statistics()
    assert json.loads(json.dumps(original)) == original
    detail = original[0]["custom_times"]
    assert detail["compute"]["accumulated_elapsed_seconds"] == 1.5
    assert detail["compute"]["accumulated_elapsed_seconds"] > pipeline.elapsed_seconds
    assert detail["compute"]["scope"] == "worker"
    assert detail["total_worker"]["includes"] == ["compute"]
    assert detail["total_worker"]["loops"] == 2
    assert "total" not in detail["compute"]
    original[0]["custom_times"]["compute"]["mean"] = -1
    original[0].pop("pipeline")
    assert pipeline.get_step_statistics()[0]["custom_times"]["compute"]["mean"] == 0.75
    assert step.timing["custom_times"]["compute"] == [0.75, 0.75]
    assert pipeline.get_steps_progress_and_durations()[0][2] == step.get_duration() == 1


@pytest.mark.parametrize("failure, status", [("bad input", "failed"), (ValueError("worker failure"), "failed"),
                                             (KeyboardInterrupt(), "interrupted")])
def test_failed_or_interrupted_steps_finalize_owned_durations_without_success_progress(monkeypatch, failure, status):
    def work(step):
        step.update_progress(0.25)
        if isinstance(failure, BaseException):
            raise failure
        return failure
    step, skipped = FunctionStep(work), FunctionStep()
    pipeline = Pipeline([step, skipped])
    observed = []
    step.listen_for_progress(observed.append)
    monkeypatch.setattr(pipeline_module, "time", clock([10, 11, 14, 15]))
    if isinstance(failure, BaseException):
        with pytest.raises(type(failure)):
            pipeline.process(BasePipelineInput())
    else:
        assert pipeline.process(BasePipelineInput()) == (None, failure)
    assert pipeline.elapsed_seconds == 5
    assert step.get_duration() == 3
    assert pipeline.status == step.timing["status"] == status
    assert skipped.timing["status"] == "pending"
    assert observed == [0, 0.25]
    assert pipeline.get_steps_progress_and_durations()[0] == [step.step_name, 0.25, 3]


def test_disabled_statistics_keep_progress_and_step_duration_but_no_detailed_samples():
    def work(step):
        step.add_timing("worker", 12)
        step.add_timing_list("other", [3, 5])
        step.add_byte_count("intermediate_payload_bytes", 100)
    step = FunctionStep(work)
    pipeline = Pipeline([step])
    pipeline.process(BasePipelineInput())
    assert pipeline.get_step_statistics()[0]["custom_times"] == {}
    assert pipeline.get_step_statistics()[0]["byte_counts"] == {}
    assert step.timing["custom_times"] == {} and step.byte_counts == {}
    assert step.get_progress() == 1
    assert pipeline.elapsed_seconds >= step.get_duration() > 0
    pipeline.set_statistics_enabled(True).process(BasePipelineInput())
    assert step.get_time_statistics()["custom_times"]["worker"]["loops"] == 1
    assert step.byte_counts == {"intermediate_payload_bytes": 100}
    pipeline.process(BasePipelineInput())
    assert step.get_time_statistics()["custom_times"]["worker"]["loops"] == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_status_snapshot_derives_lifecycle_from_step_owner_and_keeps_unknown_work_indeterminate(enabled):
    observed = []

    def work(step):
        step.set_stage("downloading_and_slicing", determinate=True)
        step.update_progress(0.75)
        observed.append(step.get_status())
        step.set_stage("normalizing_and_writing")
        observed.append(step.get_status())
        return "rewrite failed"

    step, skipped = FunctionStep(work), FunctionStep()
    pipeline = Pipeline([step, skipped]).set_statistics_enabled(enabled)
    assert pipeline.get_steps_status()[0]["status"] == "pending"
    pipeline.process(BasePipelineInput())
    assert observed[0]["status"] == observed[1]["status"] == "running"
    assert observed[0]["progress"] == 0.75 and observed[1]["progress"] is None
    status, pending = pipeline.get_steps_status()
    assert status["status"] == step.timing["status"] == "failed"
    assert status["stage"] == "normalizing_and_writing" and status["progress"] is None
    assert status["duration_seconds"] == step.get_duration() > 0
    assert pending["status"] == skipped.timing["status"] == "pending"
    assert pipeline.get_steps_progress_and_durations()[0][1] == step.get_progress() == 0.75
    status["status"] = "succeeded"
    assert step.timing["status"] == "failed", "Serialized status must never become a runtime authority"


def make_rects():
    return np.array([[[0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1]]], dtype=float)


def slice_options(tmp_path):
    return SliceOptions(output_file_folder=str(tmp_path), output_file_name="sample", neuroglancer_json="local.json",
                        slice_width=2, slice_height=2, process_count=4)


@pytest.mark.parametrize("slice_indices", [[1, 2], [2, 0], [-2, -1]],
                         ids=["contiguous", "noncontiguous", "negative_indices"])
@pytest.mark.parametrize("channels", [1, 3])
def test_slice_worker_returns_nested_timing_samples_for_the_owning_step(
    tmp_path, monkeypatch, slice_indices, channels,
):
    rects = np.repeat(make_rects(), 3, axis=0)
    rects[:, :, 2] = np.array([0.5, 1.0, 1.5])[:, None]
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 3, 0, 3, 0, 3))
    options = slice_options(tmp_path)
    path = tmp_path / "slices.tif"
    file_shape = (3, 2, 2, channels)
    tifffile.imwrite(path, np.full(file_shape, -999, dtype=np.float32),
                     photometric="rgb" if channels == 3 else "minisblack")
    volume = np.arange(np.prod(box.get_shape()) * channels, dtype=np.uint16).reshape(
        *box.get_shape(), channels)
    grids = np.stack([slice_module.coordinate_grid(rects[index], (2, 2), floor=box.get_min())
                     for index in slice_indices])
    interpolate, open_mapping = slice_module.slice_volume_from_grids, slice_module.memmap
    expected = interpolate(volume, box, grids, 2, 2)
    mappings, outputs = [], []
    observed = {"grid": [], "coefficients": [], "writes": []}
    real_filter = interpolation.spline_filter
    real_map = interpolation.map_coordinates

    def observe_filter(*args, **kwargs):
        coefficients = real_filter(*args, **kwargs)
        assert coefficients.dtype == np.float64
        assert coefficients is kwargs["output"]
        assert np.shares_memory(args[0], volume) and args[0].dtype == volume.dtype
        observed["coefficients"].append(coefficients)
        return coefficients

    def observe_map(*args, **kwargs):
        channel, local_index = divmod(len(observed["writes"]), len(slice_indices))
        target = mappings[0][slice_indices[local_index]].reshape(-1, channels)[:, channel]
        assert args[0] is observed["coefficients"][-1]
        assert kwargs["prefilter"] is False
        assert kwargs["output"].ctypes.data == target.ctypes.data
        assert kwargs["output"].strides == target.strides
        result = real_map(*args, **kwargs)
        assert result is kwargs["output"]
        observed["writes"].append(result)
        return result

    def observe_mapping(*args, **kwargs):
        result = open_mapping(*args, **kwargs)
        mappings.append(result)
        return result

    def observe_interpolation(source, bounding_box, grids, width, height, *, output=None, slice_indices=None):
        assert source is volume and bounding_box is box
        assert callable(grids)
        outputs.append(output)
        assert output is mappings[0]

        def grid_at(index):
            grid = grids(index)
            assert grid.flags.owndata and grid.shape == (height, width, 3)
            observed["grid"].append(grid)
            return grid

        result = interpolate(source, bounding_box, grid_at, width, height,
                             output=output, slice_indices=slice_indices)
        assert result is output
        return result

    monkeypatch.setattr(slice_module, "memmap", observe_mapping)
    monkeypatch.setattr(slice_module, "slice_volume_from_grids", observe_interpolation)
    monkeypatch.setattr(interpolation, "spline_filter", observe_filter)
    monkeypatch.setattr(interpolation, "map_coordinates", observe_map)
    monkeypatch.setattr(slice_module, "time", clock(count()))
    try:
        index, durations, minimum, maximum = slice_module.process_worker_save_parallel(
            options, volume, box, slice_indices, 0, rects, str(path),
        )
    finally:
        for mapping in mappings:
            mapping._mmap.close()
    assert len(mappings) == len(outputs) == 1
    expected_file = np.full(file_shape, -999, dtype=np.float32)
    expected_file[slice_indices] = expected
    np.testing.assert_array_equal(tifffile.imread(path).reshape(file_shape), expected_file)
    assert (minimum, maximum) == (expected.min(), expected.max())
    assert index == 0
    assert set(durations) == {"initial_load", "generate_grid", "slice_volume", "memmap_write", "total_process"}
    assert all(len(samples) == 1 and samples[0] > 0 for samples in durations.values())
    assert sum(samples[0] for key, samples in durations.items() if key != "total_process") <= durations["total_process"][0]
    assert len(observed["writes"]) == len(observed["grid"]) == len(slice_indices) * channels
    assert all(grid is observed["grid"][0] for grid in observed["grid"])
    assert len(observed["coefficients"]) == channels
    assert all(workspace is observed["coefficients"][0] for workspace in observed["coefficients"])
    estimate = interpolation.estimate_slice_task_payload(
        box, len(slice_indices), slice_indices_are_contiguous(slice_indices),
        slice_shape=(options.slice_height, options.slice_width),
        source_dtype=volume.dtype, channels=channels,
    )
    assert estimate.source_bytes == volume.nbytes
    assert observed["grid"][0].nbytes == estimate.grid_bytes // len(slice_indices)
    assert observed["coefficients"][0].nbytes == estimate.coefficient_bytes
    # Partition scoring intentionally retains its pre-streaming formula.
    samples = len(slice_indices) * options.slice_height * options.slice_width
    buffered = not slice_indices_are_contiguous(slice_indices)
    assert estimate.grid_bytes == samples * 3 * np.dtype(np.float32).itemsize
    assert estimate.output_bytes == (samples * channels * 4 if buffered else 0)
    assert estimate.channel_result_bytes == (samples * 4 if buffered else 0)
    live_payload = volume.nbytes + observed["grid"][0].nbytes + observed["coefficients"][0].nbytes
    assert live_payload < estimate.peak_bytes


@pytest.mark.parametrize("clipped", [False, True], ids=["inside_source", "clipped_source"])
@pytest.mark.parametrize("comparison", ["count_extents", "payload_partition"])
def test_downloaded_slice_agrees_across_fractional_boxes_and_full_volume(
    tmp_path, monkeypatch, clipped, comparison,
):
    # A generated coordinate-encoding field exposes shifted samples. Exercise
    # both the original tight/whole-path bounds and changed payload assignments.
    world_origin = np.full(3, 960)
    x, y, z = np.ogrid[:192, :192, :192]
    world = (128 + 0.5 * x + 0.25 * y + 0.125 * z).astype(np.float32)
    source_origin = np.array([1030, 960, 960]) if clipped else world_origin
    world[:source_origin[0] - world_origin[0]] = 0
    source_bounds = Bbox(source_origin, world_origin + world.shape)

    def download(bbox, **kwargs):
        assert source_bounds.contains_bbox(bbox)
        start, stop = bbox.minpt - world_origin, bbox.maxpt - world_origin
        return world[tuple(slice(int(low), int(high)) for low, high in zip(start, stop))][..., None]

    cv = SimpleNamespace(cv=SimpleNamespace(bounds=source_bounds, num_channels=1, download=download),
                         dtype=world.dtype)
    points = generate_sample_curve_helix(start_z=0, end_z=12, radius=3, num_points=30) + 1040
    annotations = tmp_path / "path.json"
    annotations.write_text(json.dumps({"layers": [
        {"type": "annotation", "name": "path", "annotations": [
            {"type": "point", "point": point.tolist()} for point in points]},
        {"type": "image", "name": "source", "source": "precomputed://unused"},
    ]}))
    options = slice_options(tmp_path).model_copy(update={
        "slice_height": 61, "slice_width": 63, "neuroglancer_json": str(annotations),
        "neuroglancer_annotation_layer": "path", "neuroglancer_image_layer": "source",
    })
    pipeline, data = geometry_pipeline(options)
    _, error = pipeline.process(data)
    assert error is None
    rects = data.slice_rects
    original = rects.copy()
    selected = len(rects) // 2
    rect = rects[selected]

    def payload(box, count, contiguous):
        return interpolation.estimate_slice_task_payload(
            box, count, contiguous, slice_shape=(61, 63),
            source_dtype=world.dtype, channels=1,
        ).peak_bytes

    choices = ((1, None), (len(rects), None)) if comparison == "count_extents" else ((4, None), (4, payload))
    partitions = [calculate_bounding_boxes_bsp_link_rects(
        rects, target_slices_per_box=target, task_payload=model,
    ) for target, model in choices]
    groups = [sorted(tuple(np.flatnonzero(np.asarray(links) == i)) for i in range(len(boxes)))
              for boxes, links in partitions]
    assert groups[0] != groups[1], "Compare actual memberships, not just renumbered boxes"
    np.testing.assert_array_equal(rects, original)
    boxes = [partition[links[selected]] for partition, links in partitions]
    if comparison == "count_extents":
        assert np.any(boxes[0].get_min() % 1 != boxes[1].get_min() % 1)

    # Independent float64 coordinates for every original output plane.
    horizontal = np.linspace(0, 1, 63)[None, None, :, None]
    vertical = np.linspace(0, 1, 61)[None, :, None, None]
    world_grids = (rects[:, 0, None, None] + (rects[:, 1] - rects[:, 0])[:, None, None] * horizontal
                   + (rects[:, 3] - rects[:, 0])[:, None, None] * vertical - world_origin)
    world_grid = world_grids[selected]
    reference = map_coordinates(world, world_grid.reshape(-1, 3).T,
                                order=3, mode="constant", output=np.float32).reshape(61, 63)
    if not clipped:
        stack_reference = map_coordinates(
            world, world_grids.reshape(-1, 3).T, output=np.float32,
        ).reshape(len(rects), 61, 63)
    interpolate = slice_module.slice_volume_from_grids
    results, coordinate_references, observed_grids = [], [], []

    for index, (partition, links) in enumerate(partitions):
        path = tmp_path / f"partition-{index}.tif"
        tifffile.imwrite(path, np.full((len(rects), 61, 63, 1), -999, dtype=np.float32),
                         photometric="minisblack")
        actual_world_grids = np.full(world_grids.shape, np.nan)
        stack_interior = np.zeros((len(rects), 61, 63), dtype=bool)
        selected_grid = None

        def interpolate_observed(volume, box, grids, width, height, *, output=None, slice_indices=None):
            assert isinstance(output, np.memmap)
            assert slice_indices == indices

            def grid_at(index):
                nonlocal selected_grid
                grid = grids(index)
                assert np.all(grid >= 0)
                assert np.all(grid <= np.array(volume.shape[:3]) - 1)
                actual_world_grids[index] = grid + box.get_integer_origin() - world_origin
                distance = np.minimum(grid, np.array(box.get_shape()) - 1 - grid).min(axis=-1)
                stack_interior[index] = distance > 8
                if index == selected:
                    selected_grid = grid.copy()
                return grid

            return interpolate(volume, box, grid_at, width, height,
                               output=output, slice_indices=slice_indices)

        monkeypatch.setattr(slice_module, "slice_volume_from_grids", interpolate_observed)
        for box_index, box in enumerate(partition):
            indices = np.flatnonzero(np.asarray(links) == box_index).tolist()
            volume, returned_box, _ = cache_module.download_volume(cv, box, mip=0)
            assert returned_box is box
            assert volume.shape[:3] == box.get_shape()
            if clipped:
                missing = max(0, source_origin[0] - box.get_integer_origin()[0])
                assert np.all(volume[:missing] == 0)
                assert np.all(volume[missing:] > 0)
            slice_module.process_worker_save_parallel(options, volume, box, indices, box_index, rects, str(path))
        stack = tifffile.imread(path).reshape(len(rects), 61, 63)
        if not clipped:
            assert np.all(stack > 0), "Every assigned slice must be written without constant-fill zero edges"
        assert np.all(np.isfinite(actual_world_grids))
        # Bound the existing float32 grid arithmetic separately from filtering.
        np.testing.assert_allclose(actual_world_grids, world_grids, rtol=0,
                                   atol=3*np.spacing(np.float32(rects.max())))
        if not clipped:
            # Retain the removed partition test's stack-wide numerical and
            # TIFF assignment checks using this existing smooth-field fixture.
            at_actual_coordinates = map_coordinates(
                world, actual_world_grids.reshape(-1, 3).T, output=np.float32,
            ).reshape(len(rects), 61, 63)
            assert stack_interior.sum() > 100
            np.testing.assert_array_max_ulp(
                stack[stack_interior], at_actual_coordinates[stack_interior], maxulp=1,
            )
            np.testing.assert_array_max_ulp(
                stack[stack_interior], stack_reference[stack_interior], maxulp=32,
            )
        sampled = stack[selected]
        results.append(sampled)
        observed_grids.append(selected_grid)
        coordinate_references.append(map_coordinates(
            world, actual_world_grids[selected].reshape(-1, 3).T,
            order=3, mode="constant", output=np.float32,
        ).reshape(61, 63))
        if not clipped:
            box = boxes[index]
            lookup, totals, weights = backproject_module.backproject_box(box, rect[None], sampled[None])
            positions = np.array(np.unravel_index(lookup, box.get_shape()[::-1])).T[:, ::-1]
            positions += box.get_integer_origin()
            np.testing.assert_allclose(weights.sum(), sampled.size, rtol=1e-6)
            np.testing.assert_allclose(totals.sum(), sampled.sum(), rtol=1e-6)
            np.testing.assert_allclose(weights @ positions / weights.sum(),
                                       (world_grid + world_origin).mean(axis=(0, 1)), rtol=0,
                                       atol=3*np.spacing(np.float32(rect.max())))

    # Preserve the established middle-plane bounds for this smooth field.
    # >8 voxels is a fixture-specific residual check, not a universal cutoff.
    interior = np.ones((61, 63), dtype=bool)
    for box, grid in zip(boxes, observed_grids):
        distance = np.minimum(grid, np.array(box.get_shape()) - 1 - grid).min(axis=-1)
        interior &= distance > 8
    assert interior.sum() > 100
    coordinate_tolerance = 3 * np.spacing(np.float32(rect.max()))
    edge_tolerance = coordinate_tolerance * 2 * world.max()
    for sampled, coordinate_reference in zip(results, coordinate_references):
        np.testing.assert_array_max_ulp(sampled[interior], coordinate_reference[interior], maxulp=1)
        if clipped:
            np.testing.assert_allclose(sampled[interior], reference[interior], rtol=0, atol=edge_tolerance)
        else:
            np.testing.assert_array_max_ulp(sampled[interior], reference[interior], maxulp=32)
    if clipped:
        np.testing.assert_allclose(results[0][interior], results[1][interior], rtol=0, atol=2*edge_tolerance)
    else:
        np.testing.assert_array_max_ulp(results[0][interior], results[1][interior], maxulp=16)


def _fail_mapped_slice_worker(failure_stage, **worker_args):
    from ouroboros.helpers import slice as interpolation_module

    # Only weak references survive the failed task. The real executor owns
    # exception transport and traceback disposal; this probe does neither.
    global _failed_slice_mapping_probe
    record = _failed_slice_mapping_probe = {
        "pid": os.getpid(), "mappings": [], "backings": [], "calls": 0, "flushes": 0,
    }
    open_mapping = slice_module.memmap
    interpolate = interpolation_module.map_coordinates

    def observe_mapping(*args, **kwargs):
        result = open_mapping(*args, **kwargs)
        record["mappings"].append(weakref.ref(result))
        record["backings"].append(weakref.ref(result._mmap))
        return result

    def interpolate_or_fail(*args, **kwargs):
        record["calls"] += 1
        assert np.shares_memory(kwargs["output"], record["mappings"][0]())
        if failure_stage == "interpolation" and record["calls"] == 2:
            raise OSError("injected interpolation failure")
        return interpolate(*args, **kwargs)

    def fail_flush(mapping):
        record["flushes"] += 1
        assert np.shares_memory(mapping, record["mappings"][0]())
        raise OSError("injected flush failure")

    with patch.object(slice_module, "memmap", observe_mapping), \
            patch.object(interpolation_module, "map_coordinates", interpolate_or_fail), \
            patch.object(np.memmap, "flush", fail_flush):
        return slice_module.process_worker_save_parallel(**worker_args)


def _inspect_failed_slice_mapping():
    record = _failed_slice_mapping_probe
    return {
        "pid": os.getpid(), "failed_pid": record["pid"],
        "mappings_released": [ref() is None for ref in record["mappings"]],
        "backings_released": [ref() is None for ref in record["backings"]],
        "calls": record["calls"], "flushes": record["flushes"],
    }


@pytest.mark.parametrize("start_method", multiprocessing.get_all_start_methods())
@pytest.mark.parametrize("failure_stage", ["interpolation", "flush"])
def test_failed_mapped_slice_releases_mapping_before_pool_shutdown(tmp_path, start_method, failure_stage):
    options = slice_options(tmp_path).model_copy(update={"slice_width": 3})
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 5, 0, 4, 0, 4))
    volume = np.arange(80, dtype=np.uint16).reshape(5, 4, 4)
    volume = np.stack([np.roll(volume, channel, axis=0) for channel in range(3)], axis=-1)
    rects = np.array([[[0.25, 0.5, z], [3.75, 0.5, z],
                       [3.75, 2.5, z], [0.25, 2.5, z]] for z in [0.5, 1.25, 2.5]])
    grids = slice_module.coordinate_grid(rects[1], (2, 3), floor=box.get_min())[None]
    expected = np.full((3, 2, 3, 3), -999, dtype=np.float32)
    interpolated = slice_module.slice_volume_from_grids(volume, box, grids, 3, 2)
    # The interpolation failure is injected after the first real channel write.
    # A flush failure occurs after every channel has been written.
    if failure_stage == "interpolation":
        expected[1:2, ..., 0] = interpolated[..., 0]
    else:
        expected[1:2] = interpolated
    path = tmp_path / "failed-slices.tif"
    tifffile.imwrite(path, np.full(expected.shape, -999, dtype=np.float32), photometric="rgb")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context(start_method),
    ) as pool:
        pid = pool.submit(os.getpid).result(timeout=30)
        failed = pool.submit(
            _fail_mapped_slice_worker, failure_stage,
            config=options, volume=volume, bounding_box=box, slice_indices=[1],
            volume_index=0, slice_rects=rects, temporary_path=str(path),
        )
        with pytest.raises(OSError, match=f"injected {failure_stage} failure"):
            failed.result(timeout=30)
        # Retain the failed Future while checking the same live worker. There
        # is no manual close, traceback clearing, GC, or worker restart here.
        observed = pool.submit(_inspect_failed_slice_mapping).result(timeout=30)
        assert observed == {
            "pid": pid, "failed_pid": pid,
            "mappings_released": [True], "backings_released": [True],
            "calls": 2 if failure_stage == "interpolation" else 3,
            "flushes": 0 if failure_stage == "interpolation" else 1,
        }
        np.testing.assert_array_equal(tifffile.imread(path), expected)
        renamed = path.with_name("released-slices.tif")
        path.rename(renamed)
        renamed.unlink()  # Also proves the mapping no longer blocks Windows deletion.
        assert not renamed.exists()


def test_download_worker_returns_elapsed_time_with_the_owned_geometry(monkeypatch):
    source = SimpleNamespace(
        bounds=Bbox((0, 0, 0), (4, 4, 4)), num_channels=1,
        download=lambda bbox, **kwargs: np.full((*bbox.size3(), 1), 5),
    )
    cv = SimpleNamespace(cv=source, dtype=np.uint16)
    box = BoundingBox(BoundingBox.bounds_to_rect(-1, 3, 0, 3, 0, 3))
    monkeypatch.setattr(cache_module, "time", clock([10, 12]))
    marker = object()
    actual, returned_box, elapsed, returned_marker = cache_module.download_volume(
        cv, box, mip=0, marker=marker,
    )
    assert returned_box is box
    assert returned_marker is marker
    assert elapsed == 2
    assert np.all(actual[0] == 0)
    assert np.all(actual[1:] == 5)


def test_backproject_worker_timers_are_disjoint_and_bytes_are_separate(tmp_path, monkeypatch):
    rects = make_rects()
    box = BoundingBox.from_rects(rects)
    monkeypatch.setattr(backproject_module.tifffile, "memmap", lambda *args, **kwargs: np.ones((1, 2, 2)))
    monkeypatch.setattr(backproject_module, "backproject_box", lambda *args: (
        np.array([0, 1], dtype=np.uint32), np.array([3, 4], dtype=np.float32), np.array([1, 1], dtype=np.float32)))
    monkeypatch.setattr(backproject_module, "time", clock(range(13)))
    options = BackprojectOptions(output_file_folder=str(tmp_path), output_file_name="sample",
                                 straightened_volume_path="unused", slice_options_path="unused")
    durations, byte_count, index, stack = backproject_module.process_chunk(
        options, "unused", rects, (slice(None),) * 3, (0, 0, 0), box,
        intermediate_threads=1,
    )
    assert durations["split"] == durations["stack"] == [1]
    assert durations["total_chunk_process"] == [12]
    assert byte_count == 24
    assert "total_bytes" not in durations
    assert index == (0, 0, 0)
    assert stack.tolist() == [0]


def test_writer_returns_nested_timings_and_uses_dispatched_merge_threads(tmp_path, monkeypatch):
    merge = Mock(return_value=np.arange(4, dtype=np.float32))
    writer = Mock()
    shape = ImgSliceC(2, 2, 1)
    monkeypatch.setattr(files_module, "volume_from_intermediates", merge)
    monkeypatch.setattr(files_module, "time", clock([0, 2, 3, 7, 8]))
    timings = files_module.write_conv_vol(
        writer, tmp_path, shape, np.uint8, None, tmp_path, 0, 0, False, merge_threads=3,
    )
    assert timings == {"Merge Volume": 2, "Write Merged": 4, "Total Chunk Merge": 8}
    merge.assert_called_once_with(tmp_path, shape, False, thread_count=3)
    assert writer.call_count == 1


@pytest.mark.parametrize("workers", [4, 8, 12])
@pytest.mark.parametrize("failed_stage", ["startup", "download", "slice"])
@pytest.mark.parametrize("enabled", [False, True])
def test_slice_callback_failure_releases_waiter_and_records_failure(tmp_path, monkeypatch, failed_stage, enabled, workers):
    options = slice_options(tmp_path)
    options.process_count = workers
    rects = make_rects()
    cache = Mock(spec=cache_module.VolumeCache)
    cache.has_color_channels.return_value = False
    cache.get_resolution_um.return_value = np.ones(3)
    cache.volumes = [None]
    cache.bounding_boxes = [BoundingBox.from_rects(rects)]
    cache.cv = object()
    cache.mip = 0
    cache.use_shared = False
    cache.cache_volume = False
    cache.get_slice_indices.return_value = [0]

    dispatched = []

    class InlineExecutor:
        def __init__(self, *args, **kwargs):
            dispatched.append(kwargs["max_workers"])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, func, **kwargs):
            future = concurrent.futures.Future()
            if func is slice_module.os.getpid:
                if failed_stage == "startup":
                    future.set_exception(RuntimeError("startup failed"))
                else:
                    future.set_result(func())
            elif func.func is slice_module.download_volume and failed_stage != "download":
                future.set_result((np.ones((1, 1, 1)), cache.bounding_boxes[0], 0.25, 0))
            else:
                future.set_exception(RuntimeError(f"{failed_stage} failed"))
            return future

    @contextmanager
    def manager(**kwargs):
        yield (SimpleNamespace(address="unused"),)

    monkeypatch.setattr(slice_module.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(slice_module, "SharedNPManager", manager)
    # TIFF metadata creation predates the pinned tifffile API; keep this failure
    # fixture focused on dispatch and callback propagation.
    monkeypatch.setattr(slice_module, "build_straightened_tiff_metadata", lambda **kwargs: {"photometric": "minisblack"})
    original_event = threading.Event
    wait_results = []

    class ObservedEvent:
        def __init__(self):
            self.event = original_event()
        def set(self):
            self.event.set()
        def wait(self):
            result = self.event.wait(timeout=1)
            wait_results.append(result)
            assert result, "The callback failed without releasing the owning thread"

    monkeypatch.setattr(slice_module.threading, "Event", ObservedEvent)
    step = slice_module.SliceParallelPipelineStep()
    data = PipelineInput(slice_options=options, slice_rects=rects, volume_cache=cache)
    output, error = Pipeline([step]).set_statistics_enabled(enabled).process(data)
    assert output is None and f"{failed_stage} failed" in error
    assert wait_results == ([] if failed_stage == "startup" else [True])
    assert step.timing["status"] == "failed"
    assert step.get_progress() < 1
    assert step.get_duration() > 0
    assert dispatched == [workers // 4, workers * 3 // 4]
    assert step.execution_resources["rewrite_threads"] == workers
    assert step.execution_resources["download_processes"] == workers // 4
    assert step.execution_resources["slice_processes"] == workers * 3 // 4
    assert step.timing["custom_times"] == ({"Download Time": [0.25]} if enabled and failed_stage == "slice" else {})


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("alternate", [False, True])
@pytest.mark.parametrize("fail, normalize, zero_guard", [
    (False, False, False), (False, False, True),
    (False, True, False), (False, True, True), (True, True, True),
])
def test_slice_rewrite_reports_running_unknown_length_work_after_workers_finish(
    tmp_path, monkeypatch, single, alternate, fail, normalize, zero_guard
):
    options = slice_options(tmp_path).model_copy(update={
        "make_single_file": single, "slice_height": 3, "slice_width": 5,
        "normalize_output": normalize, "zeroguard_output": zero_guard,
    })
    rects = np.repeat(np.array([[[0.4, 0.25, 0.6], [7.6, 0.25, 0.6],
                                [7.6, 6.75, 0.6], [0.4, 6.75, 0.6]]]), 12, axis=0)
    rects[:, :, 2] = np.tile([0.6, 2.4, 5.2], 4)[:, None]
    x, y, z = np.indices((9, 8, 7))
    volume = np.where((x >= 2) & (x <= 3), (y + 1) * 100 + z * 31, 0).astype(np.uint16)
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 9, 0, 8, 0, 7))

    # Preserve the prior box-level SciPy reference independently of streaming.
    # Compare each real final writer against its float32 result and the owning converter.
    grids = np.stack([slice_module.coordinate_grid(rect, (3, 5), floor=box.get_min()) for rect in rects])
    interpolate = slice_module.slice_volume_from_grids
    reference = map_coordinates(volume, grids.reshape(-1, 3).T,
                                output=np.float32).reshape(len(rects), 3, 5)
    assert reference.min() < 0, "Cubic overshoot must exercise zero-guard when normalization is disabled"
    expected = np.stack([files_module.np_convert(
        target_dtype=volume.dtype, source=plane, normalize=normalize, zero_guard=zero_guard,
        preset_min=reference.min(), preset_max=reference.max(),
    ) for plane in reference])
    # Five boxes exceed the initial four-download window; tied counts also
    # exercise deterministic ordering. All use the same controlled source field.
    counts = [2, 4, 1, 3, 2]
    mapping = np.repeat(np.arange(len(counts)), counts).tolist()
    cv = SimpleNamespace(dtype=volume.dtype, has_color_channels=False,
                         get_resolution_um=lambda mip: np.ones(3))
    cache = cache_module.VolumeCache([box] * len(counts), mapping, cv, mip=0)
    monkeypatch.setattr(cache, "connect_shm", lambda *args: None)
    step = (slice_module.SliceParallelPipelineStep(alternate_slice_count_order=True)
            if alternate else slice_module.SliceParallelPipelineStep())
    expected_order = [1, 2, 3, 4, 0] if alternate else [0, 1, 2, 3, 4]
    observed, mapped_shapes, mapped_indices, downloaded, sliced = [], [], [], [], []

    def interpolate_observed(*args, output=None, **kwargs):
        assert isinstance(output, np.memmap), "Final pixels must come from the mapped worker path"
        mapped_shapes.append(output.shape)
        mapped_indices.append(kwargs["slice_indices"])
        return interpolate(*args, output=output, **kwargs)

    class InlineExecutor:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, func, **kwargs):
            future = concurrent.futures.Future()
            if getattr(func, "func", None) is slice_module.download_volume:
                assert step.get_status()["stage"] == "downloading_and_slicing"
                index = kwargs["volume_index"]
                downloaded.append(index)
                assert kwargs["bounding_box"] is cache.bounding_boxes[index]
                future.set_result((volume, box, 0.25, index))
            else:
                if getattr(func, "func", None) is slice_module.process_worker_save_parallel:
                    index = kwargs["volume_index"]
                    sliced.append(index)
                    assert kwargs["slice_indices"] == cache.get_slice_indices(index)
                future.set_result(func(**kwargs))
            return future

    @contextmanager
    def manager(**kwargs):
        yield (SimpleNamespace(address="unused"),)

    rewrite_name = "memmap_normalized" if single else "write_normalized"
    rewrite = getattr(slice_module, rewrite_name)

    def rewrite_observed(*args):
        observed.append(step.get_status())
        assert step.get_progress() == 1, "Worker fraction remains a compatibility output, not completion"
        if fail:
            raise OSError("cannot rewrite output")
        return rewrite(*args)

    monkeypatch.setattr(slice_module.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(slice_module, "SharedNPManager", manager)
    monkeypatch.setattr(slice_module, "build_straightened_tiff_metadata", lambda **kwargs: {"photometric": "minisblack"})
    monkeypatch.setattr(slice_module, "slice_volume_from_grids", interpolate_observed)
    monkeypatch.setattr(slice_module, rewrite_name, rewrite_observed)
    data = PipelineInput(slice_options=options, slice_rects=rects, volume_cache=cache)
    pipeline = Pipeline([step])
    _, error = pipeline.process(data)
    assert len(observed) == len(rects)
    assert downloaded == sliced == expected_order
    assert mapped_shapes == [expected.shape] * len(counts)
    assert mapped_indices == [cache.get_slice_indices(index) for index in expected_order]
    assert cache.link_rects == mapping and cache.bounding_boxes == [box] * len(counts)
    assert step.execution_resources["box_order"] == ("alternating_slice_count" if alternate else "partition")
    task = SliceTask(task_id="ordering", options="unused", pipeline=pipeline, pipeline_input=data)
    record = RunStatistics(task).to_record()
    assert record["execution"]["workers"] == [{"step": step.step_name, **step.execution_resources}]
    assert observed[0]["stage"] == "normalizing_and_writing"
    assert observed[0]["status"] == "running" and observed[0]["progress"] is None
    if fail:
        assert "normalizing and writing" in error and "cannot rewrite output" in error
        assert step.get_status()["status"] == "failed" and step.get_status()["progress"] is None
    else:
        assert error is None
        assert step.get_status()["status"] == "succeeded" and step.get_status()["progress"] == 1
        output_path = Path(data.output_file_path)
        if single:
            actual = tifffile.imread(output_path)
            assert actual.dtype == volume.dtype
            np.testing.assert_array_equal(actual, expected)
        else:
            filenames = [files_module.format_tiff_name(i, files_module.num_digits_for_n_files(len(rects)))
                         for i in range(len(rects))]
            assert sorted(path.name for path in output_path.iterdir()) == filenames
            for index, filename in enumerate(filenames):
                actual = tifffile.imread(output_path / filename)
                assert actual.dtype == volume.dtype
                np.testing.assert_array_equal(actual, expected[index])


def test_statistics_filename_keeps_operation_and_run_identity():
    assert files_module.format_statistics_file("scan", "slice", "abc") == "scan-slice-abc-statistics.json"
    assert files_module.format_statistics_file("scan", "backproject", "abc") != \
        files_module.format_statistics_file("scan", "slice", "abc")


@pytest.mark.parametrize("failure_source", ["writer", "progress_callback"])
@pytest.mark.parametrize("enabled", [False, True])
def test_backprojection_writer_failure_reaches_owner_without_callback_warning(
    tmp_path, monkeypatch, caplog, failure_source, enabled
):
    source = tmp_path / "straightened.tif"
    tifffile.imwrite(source, np.ones((1, 2, 2), dtype=np.uint8), photometric="minisblack")
    rects = make_rects()
    options = BackprojectOptions(output_file_folder=str(tmp_path), output_file_name="sample",
                                 straightened_volume_path=str(source), slice_options_path="unused", process_count=4)
    cache = Mock(spec=cache_module.VolumeCache)
    cache.bounding_boxes = [BoundingBox.from_rects(rects)]
    cache.get_resolution_um.return_value = np.ones(3)
    submitted_resources = []

    class InlineExecutor:
        def __init__(self, count):
            submitted_resources.append(count)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, func, **kwargs):
            future = concurrent.futures.Future()
            if func is backproject_module.process_chunk:
                assert kwargs["intermediate_threads"] == 12
                future.set_result(({"back_project": [1], "total_chunk_process": [2]}, 24, (0, 0, 0), [0]))
            else:
                assert kwargs["merge_threads"] == 4
                if failure_source == "writer":
                    future.set_exception(OSError("cannot write output"))
                else:
                    future.set_result({})
            return future

    monkeypatch.setattr(backproject_module.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    step = backproject_module.BackprojectPipelineStep()
    step.add_byte_count = Mock(wraps=step.add_byte_count)
    if failure_source == "progress_callback":
        def fail_completed_progress(progress):
            if progress == 1:
                raise RuntimeError("cannot publish progress")
        step.listen_for_progress(fail_completed_progress)
    data = PipelineInput(backproject_options=options, slice_options=slice_options(tmp_path),
                         volume_cache=cache, slice_rects=rects)
    output, error = Pipeline([step]).set_statistics_enabled(enabled).process(data)
    message = "cannot write output" if failure_source == "writer" else "cannot publish progress"
    assert output is None and message in error
    assert step.timing["status"] == "failed"
    if failure_source == "writer":
        assert step.get_progress() < 1
    assert step.get_duration() > 0
    assert submitted_resources == [step.execution_resources["backproject_processes"],
                                   step.execution_resources["writer_processes"]]
    assert "exception calling callback" not in caplog.text
    step.add_byte_count.assert_called_once_with("intermediate_payload_bytes", 24)
    assert step.byte_counts == ({"intermediate_payload_bytes": 24} if enabled else {})
    if enabled:
        assert step.timing["custom_times"]["back_project"] == [1]
        assert step.get_time_statistics()["custom_times"]["total_chunk_process"]["includes"] == ["back_project"]
    else:
        assert step.timing["custom_times"] == {}


@pytest.mark.parametrize("fail_export", [False, True])
def test_backproject_single_tiff_export_and_cleanup_keep_step_running(tmp_path, monkeypatch, fail_export):
    source = tmp_path / "straightened.tif"
    tifffile.imwrite(source, np.ones((1, 2, 2), dtype=np.uint8), photometric="minisblack")
    rects = make_rects()
    options = BackprojectOptions(output_file_folder=str(tmp_path), output_file_name="sample",
                                 straightened_volume_path=str(source), slice_options_path="unused",
                                 process_count=4, make_single_file=True)
    cache = Mock(spec=cache_module.VolumeCache)
    cache.bounding_boxes = [BoundingBox.from_rects(rects)]
    cache.get_resolution_um.return_value = np.ones(3)
    step = backproject_module.BackprojectPipelineStep()
    observed = []

    class InlineExecutor:
        def __init__(self, count):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, func, **kwargs):
            assert step.get_status()["stage"] == "backprojecting_and_writing"
            future = concurrent.futures.Future()
            if func is backproject_module.process_chunk:
                future.set_result(({}, 0, (0, 0, 0), [0]))
            else:
                path = kwargs["target_folder"] / "0.tif"
                tifffile.imwrite(path, np.ones((2, 2), dtype=np.uint16), photometric="minisblack")
                kwargs["source_path"].parent.mkdir(exist_ok=True)
                future.set_result({})
            return future

    read = backproject_module.tifffile.imread
    remove = backproject_module.shutil.rmtree

    def read_observed(path, *args, **kwargs):
        observed.append(step.get_status())
        assert step.get_progress() == 1
        if fail_export:
            raise OSError("cannot export TIFF")
        return read(path, *args, **kwargs)

    def cleanup_observed(*args, **kwargs):
        observed.append(step.get_status())
        return remove(*args, **kwargs)

    monkeypatch.setattr(backproject_module.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(backproject_module.tifffile, "imread", read_observed)
    monkeypatch.setattr(backproject_module.shutil, "rmtree", cleanup_observed)
    data = PipelineInput(backproject_options=options, slice_options=slice_options(tmp_path),
                         volume_cache=cache, slice_rects=rects)
    pipeline = Pipeline([step])
    if fail_export:
        with pytest.raises(OSError, match="cannot export TIFF"):
            pipeline.process(data)
        assert step.get_status()["status"] == "failed"
        assert step.get_status()["stage"] == "exporting_tiff"
    else:
        _, error = pipeline.process(data)
        assert error is None and step.get_status()["status"] == "succeeded"
        assert [item["stage"] for item in observed] == ["exporting_tiff", "cleaning_intermediates",
                                                       "cleaning_intermediates"]
        assert (tmp_path / data.output_file_path).is_file()
    assert observed[0]["stage"] == "exporting_tiff"
    assert all(item["status"] == "running" and item["progress"] is None for item in observed)


class LocalVolume:
    bounds = Bbox((0, 0, 0), (4, 4, 4))
    num_channels, shape = 1, (4, 4, 4, 1)

    def download(self, bbox, **kwargs):
        return np.full((*bbox.size3(), 1), 37, dtype=np.uint16)

    def mip_resolution(self, mip):
        return np.full(3, 1000)


@pytest.mark.skipif(sys.platform != 'linux', reason='Inspects Linux mappings and descriptors')
def test_released_slice_volume_has_no_worker_attachments_before_pool_shutdown(tmp_path, monkeypatch):
    source = object.__new__(cache_module.CloudVolumeInterface)
    source.cv, source.available_mips, source.dtype = LocalVolume(), [0], np.dtype(np.uint16)
    rects = make_rects()
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 3, 0, 3, 0, 3))
    cache = cache_module.VolumeCache([box], [0], source, use_shared=True)
    cache.cache_volume = [False]
    options = slice_options(tmp_path).model_copy(update={'make_single_file': False, 'normalize_output': False})
    real_executor = concurrent.futures.ProcessPoolExecutor
    real_thread_pool = slice_module.multiprocessing.pool.ThreadPool
    pools, released, inspected = [], [], []

    def executor(*args, **kwargs):
        pool = real_executor(*args, mp_context=multiprocessing.get_context('fork'), **kwargs)
        pools.append(pool)
        return pool

    remove_volume = cache.remove_volume

    def release(index, destroy_shared=False):
        released.append('/dev/shm/' + cache.volumes[index].name)
        remove_volume(index, destroy_shared=destroy_shared)

    @contextmanager
    def inspect_before_rewrite(*args, **kwargs):
        # All slice callbacks have finished, but the manager and pools are live.
        assert len(released) == 1 and cache.volumes == [None]
        assert len(pools[1]._processes) == 3
        deadline = time.monotonic() + 5
        while True:
            retained = []
            for pool in pools:
                for pid in pool._processes:
                    paths = [r.path for r in psutil.Process(pid).memory_maps(grouped=False)]
                    for fd in Path(f'/proc/{pid}/fd').iterdir():
                        try:
                            paths.append(os.readlink(fd))
                        except FileNotFoundError:
                            pass
                    retained.extend((pid, p) for p in paths if p.removesuffix(' (deleted)') in released)
            if not retained:
                break
            assert time.monotonic() < deadline, f'Workers retain released backing: {retained}'
            time.sleep(0.01)  # Allow the worker to discard its completed task argument.
        inspected.append(True)
        with real_thread_pool(*args, **kwargs) as pool:
            yield pool

    monkeypatch.setattr(slice_module.concurrent.futures, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(cache, 'remove_volume', release)
    monkeypatch.setattr(slice_module.multiprocessing.pool, 'ThreadPool', inspect_before_rewrite)
    # Existing TIFF metadata compatibility is unrelated to allocation lifetime.
    monkeypatch.setattr(slice_module, 'build_straightened_tiff_metadata',
                        lambda **kwargs: {'photometric': 'minisblack'})
    data = PipelineInput(slice_options=options, slice_rects=rects, volume_cache=cache)
    _, error = Pipeline([slice_module.SliceParallelPipelineStep()]).process(data)
    assert error is None, error
    assert inspected == [True]

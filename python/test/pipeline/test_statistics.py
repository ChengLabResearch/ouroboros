import concurrent.futures
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import psutil
import pytest
import tifffile
from cloudvolume import Bbox

from ouroboros.helpers import files as files_module
from ouroboros.helpers import volume_cache as cache_module
from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers.shapes import ImgSliceC
from ouroboros.pipeline import backproject_pipeline as backproject_module
from ouroboros.pipeline import pipeline as pipeline_module
from ouroboros.pipeline import slice_parallel_pipeline as slice_module
from ouroboros.pipeline.pipeline import Pipeline, PipelineStep
from ouroboros.pipeline.pipeline_input import BasePipelineInput, PipelineInput


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


def test_slice_worker_returns_nested_timing_samples_for_the_owning_step(tmp_path, monkeypatch):
    rects = make_rects()
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 3, 0, 3, 0, 3))
    options = slice_options(tmp_path)
    path = tmp_path / "slices.tif"
    tifffile.memmap(path, shape=(1, 2, 2), dtype=np.float32, photometric="minisblack")
    monkeypatch.setattr(slice_module, "time", clock(range(9)))
    index, durations, _, _ = slice_module.process_worker_save_parallel(
        options, np.arange(27).reshape(3, 3, 3), box, [0], 0, rects, str(path),
    )
    assert index == 0
    assert durations == {
        "initial_load": [1], "generate_grid": [1], "slice_volume": [1],
        "memmap_write": [1], "total_process": [8],
    }


def test_download_worker_returns_elapsed_time_with_the_owned_geometry(monkeypatch):
    source = SimpleNamespace(
        bounds=Bbox((0, 0, 0), (4, 4, 4)), num_channels=1,
        download=lambda bbox, **kwargs: np.full((*bbox.size3(), 1), 5),
    )
    cv = SimpleNamespace(cv=source)
    box = BoundingBox(BoundingBox.bounds_to_rect(-1, 3, 0, 3, 0, 3))
    monkeypatch.setattr(cache_module, "time", clock([10, 12]))
    actual, returned_box, elapsed = cache_module.download_volume(cv, box, mip=0)
    assert returned_box is box
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
@pytest.mark.parametrize("fail", [False, True])
def test_slice_rewrite_reports_running_unknown_length_work_after_workers_finish(tmp_path, monkeypatch, single, fail):
    options = slice_options(tmp_path).model_copy(update={"make_single_file": single})
    rects = make_rects()
    cache = Mock(spec=cache_module.VolumeCache)
    cache.has_color_channels.return_value = False
    cache.get_resolution_um.return_value = np.ones(3)
    cache.get_volume_dtype.return_value = np.uint16
    cache.volumes = [None]
    cache.bounding_boxes = [BoundingBox(BoundingBox.bounds_to_rect(0, 3, 0, 3, 0, 3))]
    cache.cv, cache.mip = object(), 0
    cache.use_shared = cache.cache_volume = False
    cache.get_slice_indices.return_value = [0]
    step = slice_module.SliceParallelPipelineStep()
    observed = []

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
                future.set_result((np.arange(27).reshape(3, 3, 3), cache.bounding_boxes[0], 0.25, 0))
            else:
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
    monkeypatch.setattr(slice_module, rewrite_name, rewrite_observed)
    data = PipelineInput(slice_options=options, slice_rects=rects, volume_cache=cache)
    _, error = Pipeline([step]).process(data)
    assert len(observed) == 1
    assert observed[0]["stage"] == "normalizing_and_writing"
    assert observed[0]["status"] == "running" and observed[0]["progress"] is None
    if fail:
        assert "normalizing and writing" in error and "cannot rewrite output" in error
        assert step.get_status()["status"] == "failed" and step.get_status()["progress"] is None
    else:
        assert error is None
        assert step.get_status()["status"] == "succeeded" and step.get_status()["progress"] == 1
        assert Path(data.output_file_path).exists()


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

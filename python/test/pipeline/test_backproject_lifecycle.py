"""Backprojection worker timing, failure propagation and export cleanup."""

import concurrent.futures
from pathlib import Path
import sys
from unittest.mock import Mock
import numpy as np
import pytest
import tifffile
from ouroboros.common.logging_mode import LoggingMode
from ouroboros.helpers import files as files_module
from ouroboros.helpers import volume_cache as cache_module
from ouroboros.helpers.bounding_boxes import (
    BoundingBox,
)
from ouroboros.helpers.options import BackprojectOptions
from ouroboros.pipeline import backproject_pipeline as backproject_module
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput
from test.harness import ContextExecutor, clock, make_rects, slice_options


pytestmark = pytest.mark.needs("backprojection")


def test_backproject_worker_timers_are_disjoint_and_bytes_are_separate(tmp_path, monkeypatch):
    rects = make_rects()
    box = BoundingBox.from_rects(rects)
    monkeypatch.setattr(backproject_module.tifffile, "memmap", lambda *args, **kwargs: np.ones((1, 2, 2)))
    payload = (
        np.array([0, 1], dtype=np.uint32),
        np.array([3, -1, 4, -1], dtype=np.float32)[::2],
        np.array([1, -1, 1, -1], dtype=np.float32)[::2],
    )
    monkeypatch.setattr(backproject_module, "backproject_box", lambda *args: payload)
    append = Mock(wraps=backproject_module.write_raw_intermediate)
    monkeypatch.setattr(backproject_module, "write_raw_intermediate", append)
    monkeypatch.setattr(backproject_module, "time", clock(range(13)))
    intermediate_directory = tmp_path / "intermediates"
    info = files_module.StraightenedVolumeInfo(
        Path("unused"), Path("unused"), files_module.FrontProjectionShape(1, 2, 2),
        np.dtype(np.uint8), 1, False,
    )
    durations, byte_count, index, stack = backproject_module.process_chunk(
        intermediate_directory, info, rects, (slice(None),) * 3, (0, 0, 0), box,
        intermediate_threads=1,
    )
    assert durations["split"] == durations["stack"] == [1]
    assert durations["total_chunk_process"] == [12]
    assert byte_count == 24
    assert "total_bytes" not in durations
    assert index == (0, 0, 0)
    assert stack.tolist() == [0]
    append.assert_called_once()
    path, *segments = append.call_args.args
    assert path == intermediate_directory / "i_00000.dat"
    assert all(isinstance(part, np.ndarray) and part.flags.c_contiguous for part in segments[3:])
    expected = (
        np.array([box.get_shape()[0], box.get_shape()[0], 0, 0, 1, 2], dtype=np.uint32).tobytes()
        + np.array([array.dtype.str for array in payload], dtype="S8").tobytes()
        + b"".join(array.tobytes() for array in payload)
    )
    assert path.read_bytes() == expected
    if sys.platform != "win32":
        assert list(path.parent.glob("*.lock")) == []


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
    interface = Mock(spec=cache_module.CloudVolumeInterface)
    interface.get_resolution_um.return_value = np.ones(3)
    interface.get_resolution_nm.return_value = np.ones(3)
    interface.get_volume_shape.return_value = np.ones(3)
    monkeypatch.setattr(backproject_module, "CloudVolumeInterface", lambda url: interface)
    submitted_resources = []

    class InlineExecutor(ContextExecutor):
        def __init__(self, count):
            submitted_resources.append(count)

        def submit(self, func, **kwargs):
            future = concurrent.futures.Future()
            if func is backproject_module.process_chunk:
                assert kwargs["intermediate_threads"] == 12
                future.set_result(({"back_project": [1], "total_chunk_process": [2]}, 24, (0, 0, 0), [0]))
            else:
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
                         source_url="memory://source", slice_rects=rects)
    output, error = Pipeline([step]).set_logging_mode(
        LoggingMode.STANDARD if enabled else LoggingMode.OFF).process(data)
    message = "cannot write output" if failure_source == "writer" else "cannot publish progress"
    assert output is None and message in error
    assert step.timing["status"] == "failed"
    if failure_source == "writer":
        assert step.get_progress() < 1
    assert step.get_duration() > 0
    assert submitted_resources == [step.exec_config["backproject_processes"],
                                   step.exec_config["writer_processes"]]
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
    interface = Mock(spec=cache_module.CloudVolumeInterface)
    interface.get_resolution_um.return_value = np.ones(3)
    interface.get_resolution_nm.return_value = np.ones(3)
    interface.get_volume_shape.return_value = np.ones(3)
    monkeypatch.setattr(backproject_module, "CloudVolumeInterface", lambda url: interface)
    step = backproject_module.BackprojectPipelineStep()
    observed = []

    class InlineExecutor(ContextExecutor):
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
                         source_url="memory://source", slice_rects=rects)
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

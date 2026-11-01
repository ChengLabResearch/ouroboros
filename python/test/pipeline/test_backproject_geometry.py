"""Backprojection owns one envelope without allocating slicing partitions."""

from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from ouroboros.common.logging_mode import LoggingMode
from ouroboros.common.pipelines import (
    backproject_pipeline, slice_pipeline, visualization_pipeline,
)
from ouroboros.common.run_statistics import RunStatistics
from ouroboros.common.server_types import BackProjectTask
from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers import volume_cache as cache_module
from ouroboros.pipeline import backproject_pipeline as backproject_module
from ouroboros.pipeline import volume_cache_pipeline as planning
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput


def _options(tmp_path):
    slicing = SliceOptions(
        output_file_folder=str(tmp_path), output_file_name="sample",
        neuroglancer_json="annotations.json", slice_width=2, slice_height=2,
        output_mip_level=1, process_count=4,
    )
    backprojecting = BackprojectOptions(
        output_file_folder=str(tmp_path), output_file_name="sample",
        straightened_volume_path=str(tmp_path / "straightened.tif"),
        slice_options_path="slice.json", output_mip_level=1, process_count=4,
    )
    return slicing, backprojecting


@pytest.fixture
def source_metadata(monkeypatch):
    opened, resolutions = [], []

    def resolution(mip):
        resolutions.append(mip)
        return np.array((100, 200, 300)) * 2 ** mip

    native = SimpleNamespace(
        shape=(20, 30, 40, 1), dtype=np.dtype("uint16"),
        available_mips=[0, 1], info={"type": "image"},
        cache=SimpleNamespace(enabled=True),
        mip_volume_size=lambda mip: np.array((20, 30, 40)) // 2 ** mip,
        mip_resolution=resolution,
    )

    def open_source(url, **kwargs):
        opened.append(url)
        return native

    monkeypatch.setattr(cache_module, "CloudVolume", open_source)
    return SimpleNamespace(opened=opened, resolutions=resolutions)


def test_only_backprojection_omits_the_slicing_partition_step(tmp_path):
    slicing, backprojecting = _options(tmp_path)
    pipeline, _ = backproject_pipeline(backprojecting, slicing)
    assert [type(step).__name__ for step in pipeline.steps] == [
        "ParseJSONPipelineStep", "SlicesGeometryPipelineStep", "BackprojectPipelineStep",
    ]
    for build in (slice_pipeline, visualization_pipeline):
        pipeline, _ = build(slicing)
        assert any(isinstance(step, planning.VolumeCachePipelineStep) for step in pipeline.steps)


def test_backprojection_setup_and_reporting_share_the_direct_envelope(
    tmp_path, monkeypatch, source_metadata,
):
    slicing, backprojecting = _options(tmp_path)
    tifffile.imwrite(backprojecting.straightened_volume_path,
                     np.ones((1, 2, 2), dtype=np.uint8), photometric="minisblack")
    rects = np.array([[[-1.25, 2.5, 1.25], [0.75, 2.5, 1.25],
                       [0.75, 4.5, 1.25], [-1.25, 4.5, 1.25]]])
    data = PipelineInput(
        slice_options=slicing, backproject_options=backprojecting,
        source_url="memory://source", slice_rects=rects,
    )
    envelopes, submitted = [], []
    from_rects = BoundingBox.from_rects

    def envelope(values):
        box = from_rects(values)
        if values is data.slice_rects:
            envelopes.append(box)
        return box

    def forbidden(*args, **kwargs):
        pytest.fail("Backprojection and its reporting must not create slicing partitions or a cache")

    class ObserveExecutor:
        def __init__(self, count):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, **kwargs):
            assert function is backproject_module.process_chunk
            submitted.append(kwargs["full_bounding_box"])
            future = Future()
            # The ownership assertion needs setup and the worker boundary, not
            # image processing or another process pool in this regression test.
            future.set_exception(RuntimeError("stop after observing worker geometry"))
            return future

    monkeypatch.setattr(BoundingBox, "from_rects", staticmethod(envelope))
    monkeypatch.setattr(BoundingBox, "bound_boxes", staticmethod(forbidden))
    monkeypatch.setattr(planning, "calculate_bounding_boxes_bsp_link_rects", forbidden)
    monkeypatch.setattr(planning, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(cache_module.VolumeCache, "__init__", forbidden)
    monkeypatch.setattr(backproject_module.concurrent.futures, "ProcessPoolExecutor", ObserveExecutor)

    pipeline = Pipeline([backproject_module.BackprojectPipelineStep()]).set_logging_mode(LoggingMode.STANDARD)
    result, error = pipeline.process(data)
    assert result is None and "stop after observing worker geometry" in error
    assert len(envelopes) == 1 and submitted == [envelopes[0]]
    assert data.backprojection_box is envelopes[0]
    assert data.volume_cache is None
    assert source_metadata.opened == ["memory://source"]
    assert source_metadata.resolutions == [1]
    assert data.output_file_path == "sample_-2_2_1"

    # Reporting must serialize these owners, without another geometry scan or
    # metadata connection. It also preserves the former rounded envelope bounds.
    monkeypatch.setattr(BoundingBox, "from_rects", staticmethod(forbidden))
    owner = data.backprojection_source
    task = BackProjectTask(task_id="direct-envelope", options="unused",
                           pipeline=pipeline, pipeline_input=data)
    statistics = RunStatistics(task)
    statistics.configure(slicing)
    statistics.configure(backprojecting)
    statistics.capture_volumes()
    assert statistics.collection_errors == []
    assert data.backprojection_source is owner
    assert data.backprojection_box is envelopes[0]
    assert source_metadata.opened == ["memory://source"]
    geometry = statistics.volumes["bounding_box"]
    assert geometry["bounds"] == {"minimum": [-2, 2, 1], "maximum": [1, 5, 2]}
    assert geometry["dimensions"] == [4, 4, 2]
    assert geometry["mip"] == 1
    assert geometry["voxel_spacing_um"] == [0.2, 0.4, 0.6]
    assert "download_partitions" not in geometry
    source = statistics.volumes["source"]
    assert source["full_resolution"]["dimensions"] == [20, 30, 40]
    assert source["used_resolution"]["dimensions"] == [10, 15, 20]
    record = statistics.to_record()
    assert record["cache"]["flush_requested"] == slicing.flush_cache
    assert record["cache"]["cloudvolume_cache_enabled"] is True
    assert all(worker["step"] != "VolumeCachePipelineStep" for worker in record["execution"]["workers"])


def test_backprojection_owners_round_trip_through_pipeline_input_json(source_metadata):
    source = cache_module.CloudVolumeInterface("memory://source")
    box = BoundingBox(np.array([[-1.25, 2.5, 1.25], [0.75, 4.5, 1.25]]))
    data = PipelineInput(
        source_url=source.source_url, backprojection_source=source, backprojection_box=box,
    )
    restored = PipelineInput.from_json(data.to_json())
    assert isinstance(restored, PipelineInput)
    assert isinstance(restored.backprojection_source, cache_module.CloudVolumeInterface)
    assert restored.backprojection_source.to_dict() == source.to_dict()
    assert isinstance(restored.backprojection_box, BoundingBox)
    assert restored.backprojection_box.to_dict() == box.to_dict()
    assert restored.volume_cache is None

"""Backprojection owns one envelope without allocating slicing partitions."""

from concurrent.futures import Future

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
from test.harness import ContextExecutor


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
    envelopes, submitted, validated = [], [], []
    from_rects = BoundingBox.from_rects
    validate = backproject_module.validate_straightened_volume_for_backprojection

    def inspect_input(*args):
        info = validate(*args)
        validated.append(info)
        return info

    def envelope(values):
        box = from_rects(values)
        if values is data.slice_rects:
            envelopes.append(box)
        return box

    def forbidden(*args, **kwargs):
        pytest.fail("Backprojection and its reporting must not create slicing partitions or a cache")

    class ObserveExecutor(ContextExecutor):
        def submit(self, function, **kwargs):
            assert function is backproject_module.process_chunk
            submitted.append(kwargs["full_bounding_box"])
            assert kwargs["volume_info"] is validated[0]
            assert "straightened_volume_path" not in kwargs
            assert "config" not in kwargs
            assert kwargs["intermediate_directory"].parent == tmp_path
            future = Future()
            # The ownership assertion needs setup and the worker boundary, not
            # image processing or another process pool in this regression test.
            future.set_exception(RuntimeError("stop after observing worker geometry"))
            return future

    monkeypatch.setattr(BoundingBox, "from_rects", staticmethod(envelope))
    monkeypatch.setattr(backproject_module, "validate_straightened_volume_for_backprojection", inspect_input)
    monkeypatch.setattr(BoundingBox, "bound_boxes", staticmethod(forbidden))
    monkeypatch.setattr(planning, "calculate_bounding_boxes_bsp_link_rects", forbidden)
    monkeypatch.setattr(planning, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(cache_module.VolumeCache, "__init__", forbidden)
    monkeypatch.setattr(backproject_module.concurrent.futures, "ProcessPoolExecutor", ObserveExecutor)

    pipeline = Pipeline([backproject_module.BackprojectPipelineStep()]).set_logging_mode(LoggingMode.STANDARD)
    result, error = pipeline.process(data)
    assert result is None and "stop after observing worker geometry" in error
    assert len(envelopes) == 1 and submitted == [envelopes[0]]
    assert len(validated) == 1
    assert data.backprojection_box is envelopes[0]
    assert data.volume_cache is None
    assert source_metadata.opened == ["memory://source"]
    assert set(source_metadata.resolutions) == {1}
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


@pytest.mark.needs("backprojection")
@pytest.mark.parametrize(
    "xyz,channels,single_file,binary,expected_bigtiff,output_mip",
    [
        # The exact 617 x 2048 x 1630 output envelope reported in #218.
        ((1630, 2048, 617), 3, True, False, True, 1),
        ((1630, 2048, 617), 1, True, False, False, 1),
        ((1630, 2048, 617), 3, True, True, True, 1),
        ((1630, 2048, 617), 1, True, True, False, 1),
        ((32768, 32768, 1), 3, False, False, True, 1),
        ((32768, 32768, 1), 1, False, False, False, 1),
        # Under 4 GiB at the slicing MIP; over it once upsampled to the output MIP (#231).
        ((1024, 1024, 1024), 1, True, False, False, 1),
        ((1024, 1024, 1024), 1, True, False, True, 0),
        ((32768, 32768, 1), 1, False, False, True, 0),
    ],
)
def test_backprojection_bigtiff_reaches_the_writer(
    tmp_path, monkeypatch, source_metadata, xyz, channels, single_file, binary, expected_bigtiff, output_mip,
):
    slicing, options = _options(tmp_path)
    options.make_single_file = single_file
    options.make_backprojection_binary = binary
    options.backprojection_compression = "none"
    options.output_mip_level = output_mip
    pixels = np.ones((2, 2, channels), dtype=np.uint16)
    tifffile.imwrite(options.straightened_volume_path, pixels,
                     photometric="rgb" if channels == 3 else "minisblack")
    rects = np.array([[[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]])
    envelope = BoundingBox(BoundingBox.bounds_to_rect(
        0, xyz[0] - 1, 0, xyz[1] - 1, 0, xyz[2] - 1,
    ))
    from_rects = BoundingBox.from_rects
    monkeypatch.setattr(BoundingBox, "from_rects", staticmethod(
        lambda values: envelope if values is rects else from_rects(values)
    ))

    # Exercise real TIFF writers and headers using the reported large envelope,
    # but substitute a tiny plane for the expensive backprojection payload.
    def one_page_ready(processed, slice_rects, min_dim, writeable, chunk_size):
        writeable[0] = 1

    monkeypatch.setattr(backproject_module, "update_writable_rects", one_page_ready)

    class TinyPayloadExecutor(ContextExecutor):
        def submit(self, function, **kwargs):
            future = Future()
            if function is backproject_module.process_chunk:
                future.set_result(({}, 0, (0, 0, 0), [0]))
            else:
                assert kwargs["shape"].C == channels
                assert (kwargs["shape"].X, kwargs["shape"].Y) == xyz[:2]
                scale = 2 ** (slicing.output_mip_level - output_mip)
                assert (kwargs["output_shape"].X, kwargs["output_shape"].Y) == (xyz[0] * scale, xyz[1] * scale)
                assert kwargs["output_shape"].C == channels
                assert kwargs["z_steps"] == scale
                kwargs["source_path"].parent.mkdir(exist_ok=True)
                kwargs["writer"](kwargs["target_folder"] / "0.tif",
                                 data=pixels.astype(np.uint8 if binary else np.uint16))
                future.set_result({})
            return future

    monkeypatch.setattr(backproject_module.concurrent.futures, "ProcessPoolExecutor", TinyPayloadExecutor)
    data = PipelineInput(slice_options=slicing, backproject_options=options,
                         source_url="memory://source", slice_rects=rects)
    result, error = Pipeline([backproject_module.BackprojectPipelineStep()]).process(data)
    assert error is None, error
    path = tmp_path / result.output_file_path
    if not single_file:
        path /= "0.tif"
    with tifffile.TiffFile(path) as tif:
        assert tif.is_bigtiff is expected_bigtiff
        numerator, denominator = tif.pages[0].tags["XResolution"].value
        # 100 nm voxels at mip 0, written as voxels per centimeter.
        assert numerator / denominator == pytest.approx(1e5 / 2 ** output_mip, rel=1e-3)
        assert tif.pages[0].dtype == np.dtype(np.uint8 if binary else np.uint16)

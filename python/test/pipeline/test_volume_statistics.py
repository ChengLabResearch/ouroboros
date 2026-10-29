"""Check reporting against small independent arrays and native geometry calls."""

import json
from types import SimpleNamespace

from cloudvolume import Bbox
import numpy as np
import pytest
import tifffile

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.files import generate_tiff_write
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers.volume_cache import CloudVolumeInterface, VolumeCache
from ouroboros.pipeline.pipeline_input import PipelineInput
from ouroboros.pipeline import volume_statistics, volume_cache_pipeline
from ouroboros.helpers.slice import estimate_slice_task_payload


def _cache(channels=1, dtype=np.uint16, mip=1, kind="image", available_mips=(0, 1)):
    """Use real owning interfaces with metadata-only CloudVolume backing."""
    sizes = {0: (20, 30, 40), 1: (10, 15, 20)}
    spacing = {0: (100, 200, 300), 1: (200, 400, 600)}
    native = SimpleNamespace(
        shape=(*sizes[0], channels), num_channels=channels,
        mip=0, bounds=Bbox((0, 0, 0), sizes[0]), info={"type": kind},
        mip_volume_size=lambda level: np.array(sizes[level]),
        mip_resolution=lambda level: np.array(spacing[level]),
    )
    interface = object.__new__(CloudVolumeInterface)
    interface.cv = native
    interface.available_mips = list(available_mips)
    interface.dtype = np.dtype(dtype)
    interface.source_url = "memory://source"
    boxes = [BoundingBox(BoundingBox.bounds_to_rect(-1.9, 4.2, 2.3, 6.7, 0.1, 5.1))]
    return VolumeCache(boxes, [0], interface, mip=mip)


def _input(tmp_path, cache=None):
    return PipelineInput(
        slice_options=SliceOptions(
            output_file_folder=str(tmp_path), output_file_name="sample",
            output_mip_level=1, slice_width=4, slice_height=3,
            neuroglancer_json="annotations.json", neuroglancer_image_layer="scan",
            neuroglancer_annotation_layer="nerve",
        ),
        source_url="memory://source", json_path="annotations.json",
        volume_cache=cache if cache is not None else _cache(),
    )


def _write_stack(path, values, spacing=(0.2, 0.4, 0.6)):
    with tifffile.TiffWriter(path) as writer:
        write = generate_tiff_write(
            writer.write, compression=None, micron_resolution=np.array(spacing),
            backprojection_offset=None,
            photometric="rgb" if values.ndim == 4 and values.shape[-1] > 1 else "minisblack",
        )
        for plane in values:
            write(plane)


@pytest.mark.parametrize("channels", [1, 3])
def test_source_straightened_and_storage_use_actual_dtype_and_channels(tmp_path, channels, monkeypatch):
    cache = _cache(channels=channels)
    data = _input(tmp_path, cache)
    values = np.arange(2 * 3 * 4 * channels, dtype=np.uint8).reshape(2, 3, 4, channels)
    path = tmp_path / "actual-output.tif"
    _write_stack(path, values)
    data.output_file_path = str(path)

    def forbid_pixels(*args, **kwargs):
        raise AssertionError("Reporting must not read or download pixels")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbid_pixels)
    cache.cv.cv.download = forbid_pixels
    result = volume_statistics.collect_volume_statistics(data, "slice")
    json.dumps(result, allow_nan=False)

    assert result["source"]["full_resolution"]["dimensions"] == [20, 30, 40]
    expected_source_bytes = np.empty((20, 30, 40, channels), np.uint16).nbytes
    assert result["source"]["full_resolution"]["logical_uncompressed_bytes"] == expected_source_bytes
    used = result["source"]["used_resolution"]
    assert used["dimensions"] == [10, 15, 20]
    assert used["voxel_spacing_um"] == [0.2, 0.4, 0.6]
    assert used["logical_uncompressed_bytes"] == np.empty((10, 15, 20, channels), np.uint16).nbytes
    straightened = result["straightened"]
    assert straightened["dimensions"] == [2, 3, 4]
    assert straightened["axis_order"] == "DVU"
    assert straightened["channels"] == channels
    assert straightened["dtype"] == "uint8"
    assert straightened["logical_uncompressed_bytes"] == values.nbytes
    assert straightened["physical_file_bytes"] == path.stat().st_size
    assert straightened["data_kind"] == "image"
    reduction = result["storage_reduction"]["source_used_resolution"]
    assert reduction["numerator_bytes"] == values.nbytes
    assert reduction["denominator_bytes"] == 10 * 15 * 20 * channels * 2
    assert reduction["percentage"] == 100 * (1 - values.nbytes / (10 * 15 * 20 * channels * 2))
    assert reduction["reference"] == "source.used_resolution.logical_uncompressed_bytes"
    assert result["source"]["url"] == "memory://source"
    assert "backprojected_output" not in result
    downloads = result["bounding_box"]["download_partitions"]
    assert downloads["allocated_logical_bytes"] == np.empty((8, 6, 7, channels), np.uint16).nbytes
    assert downloads["source_clipped_logical_bytes"] == np.empty((6, 6, 7, channels), np.uint16).nbytes


def test_geometry_uses_native_owners_and_distinguishes_endpoints_and_clipping(tmp_path, monkeypatch):
    data = _input(tmp_path)
    original_bound = BoundingBox.bound_boxes
    original_download = volume_statistics.download_volume_geometry
    envelopes = []
    partitions = []

    def bound(boxes, *args, **kwargs):
        assert boxes is data.volume_cache.bounding_boxes
        envelopes.append(boxes)
        return original_bound(boxes, *args, **kwargs)

    def download(interface, box):
        assert interface is data.volume_cache.cv
        assert box is data.volume_cache.bounding_boxes[0]
        partitions.append(box)
        return original_download(interface, box)

    monkeypatch.setattr(BoundingBox, "bound_boxes", staticmethod(bound))
    monkeypatch.setattr(volume_statistics, "download_volume_geometry", download)
    result = volume_statistics.collect_volume_statistics(data, "slice")["bounding_box"]
    assert len(envelopes) == 1
    assert len(partitions) == 1
    assert result["bounds"] == {"minimum": [-2, 2, 0], "maximum": [5, 7, 6]}
    assert result["dimensions"] == [8, 6, 7]
    assert result["logical_uncompressed_bytes"] == np.empty((8, 6, 7), np.uint16).nbytes
    downloads = result["download_partitions"]
    assert downloads["count"] == 1
    assert downloads["allocated_logical_bytes"] == np.empty((8, 6, 7), np.uint16).nbytes
    assert downloads["source_clipped_logical_bytes"] == np.empty((6, 6, 7), np.uint16).nbytes
    # Preserve current production clipping semantics, including its default mip.
    assert result["mip"] == 1
    assert result["download_partitions"]["clipping_bounds_mip"] == 0


def test_download_geometry_uses_the_box_integer_origin_and_shape(monkeypatch):
    cache = _cache()
    box = cache.bounding_boxes[0]
    origin = box.get_integer_origin
    shape = box.get_shape
    calls = []

    def observe_origin():
        calls.append("origin")
        return origin()

    def observe_shape():
        calls.append("shape")
        return shape()

    def forbid_float_conversion():
        pytest.fail("Download geometry must not reinterpret the geometric bounds")

    monkeypatch.setattr(box, "get_integer_origin", observe_origin)
    monkeypatch.setattr(box, "get_shape", observe_shape)
    monkeypatch.setattr(box, "to_cloudvolume_bbox", forbid_float_conversion)
    requested, clipped, allocated = volume_statistics.download_volume_geometry(cache.cv, box)

    assert calls == ["origin", "shape"]
    assert requested == Bbox((-2, 2, 0), (6, 8, 7))
    assert clipped == Bbox((0, 2, 0), (6, 8, 7))
    assert (allocated.X, allocated.Y, allocated.Z, allocated.C) == (8, 6, 7, 1)


def test_integer_geometry_and_downloads_share_the_inclusive_extent(tmp_path):
    data = _input(tmp_path)
    data.volume_cache.bounding_boxes = [BoundingBox(BoundingBox.bounds_to_rect(1, 4, 2, 6, 3, 8))]
    result = volume_statistics.collect_volume_statistics(data, "slice")["bounding_box"]
    assert result["dimensions"] == [4, 5, 6]
    assert result["download_partitions"]["allocated_logical_bytes"] == np.empty((4, 5, 6), np.uint16).nbytes
    data.volume_cache.bounding_boxes.append(BoundingBox(BoundingBox.bounds_to_rect(2, 5, 2, 6, 3, 8)))
    result = volume_statistics.collect_volume_statistics(data, "slice")["bounding_box"]
    assert result["dimensions"] == [5, 5, 6]
    assert result["download_partitions"]["count"] == 2
    # Overlapping requests each contribute their own logical allocation, not a union or RAM peak.
    expected_request_bytes = 2 * np.empty((4, 5, 6), np.uint16).nbytes
    assert result["download_partitions"]["allocated_logical_bytes"] == expected_request_bytes
    assert result["download_partitions"]["source_clipped_logical_bytes"] == expected_request_bytes


def test_backprojection_uses_input_and_actual_rescaled_tiled_output_headers(tmp_path):
    data = _input(tmp_path)
    input_path = tmp_path / "plugin-mask.tif"
    input_values = np.zeros((2, 3, 4), dtype=np.uint8)
    _write_stack(input_path, input_values)
    output_dir = tmp_path / "actual-output"
    output_dir.mkdir()
    output_plane = np.zeros((7, 9), dtype=np.uint16)
    for index in (10, 2):
        _write_stack(output_dir / f"{index}.tif", output_plane[np.newaxis], spacing=(0.1, 0.2, 0.3))
    data.backproject_options = BackprojectOptions(
        output_file_folder=str(tmp_path), output_file_name="requested-name",
        straightened_volume_path=str(input_path), slice_options_path="slice.json",
        output_mip_level=0, make_single_file=False,
    )
    data.output_file_path = "actual-output"
    result = volume_statistics.collect_volume_statistics(data, "backproject")
    assert result["straightened"]["logical_uncompressed_bytes"] == input_values.nbytes
    assert result["straightened"]["data_kind"] == "unknown"
    output = result["backprojected_output"]
    assert output["dimensions"] == [2, 7, 9]
    assert output["axis_order"] == "ZYX"
    assert output["logical_uncompressed_bytes"] == 2 * output_plane.nbytes
    assert output["requested_mip"] == 0
    assert output["header_spacing"]["values"] == pytest.approx([0.1, 0.2, 0.3], rel=1e-4)
    assert output["plane_index_extent"]["minimum"] == 2
    assert output["plane_index_extent"]["maximum"] == 10
    assert output["plane_index_extent"]["maximum"] - output["plane_index_extent"]["minimum"] + 1 == 9
    assert output["dimensions"][0] == 2  # Seven missing planes are not counted as stored data.
    assert "download_partitions" not in result["bounding_box"]
    assert output["final_nonzero_extent"] is None


def test_backprojection_single_file_rgb_output_and_missing_resolution(tmp_path):
    data = _input(tmp_path, _cache(channels=3, kind="segmentation"))
    path = tmp_path / "bp.tif"
    values = np.zeros((5, 6, 7, 3), dtype=np.uint16)
    with tifffile.TiffWriter(path) as tif:
        for plane in values:
            tif.write(plane, photometric="rgb")
    data.backproject_options = BackprojectOptions(
        output_file_folder=str(tmp_path), output_file_name="bp",
        straightened_volume_path="missing.tif", slice_options_path="slice.json",
    )
    data.output_file_path = str(path)
    result = volume_statistics.collect_volume_statistics(data, "backproject")
    assert result["source"]["data_kind"] == "segmentation"
    assert result["straightened"]["status"] == "unavailable"
    output = result["backprojected_output"]
    assert output["dimensions"] == [5, 6, 7]
    assert output["channels"] == 3
    assert output["logical_uncompressed_bytes"] == values.nbytes
    assert output["header_spacing"]["values"] == [None, None, None]
    assert result["storage_reduction"]["source_full_resolution"]["status"] == "unavailable"


def test_incomplete_runs_and_missing_full_resolution_are_explicit(tmp_path):
    data = _input(tmp_path, _cache(available_mips=(1,)))
    result = volume_statistics.collect_volume_statistics(data, "slice")
    assert result["source"]["full_resolution"]["status"] == "unavailable"
    assert result["source"]["used_resolution"]["mip"] == 1
    assert result["straightened"]["status"] == "unavailable"
    data.volume_cache = None
    result = volume_statistics.collect_volume_statistics(data, "slice")
    assert result["bounding_box"]["status"] == "unavailable"
    assert volume_statistics.collect_volume_statistics(None, "slice")["status"] == "unavailable"
    with pytest.raises(ValueError, match="Unknown operation"):
        volume_statistics.collect_volume_statistics(data, "typo")


def test_invalid_output_header_is_unavailable_without_hiding_programming_errors(tmp_path, monkeypatch):
    data = _input(tmp_path)
    path = tmp_path / "bad.tif"
    path.write_bytes(b"not a tiff")
    data.output_file_path = str(path)
    assert volume_statistics.collect_volume_statistics(data, "slice")["straightened"]["status"] == "unavailable"

    def bug(path):
        raise RuntimeError("unexpected defect")

    monkeypatch.setattr(volume_statistics, "inspect_straightened_volume", bug)
    with pytest.raises(RuntimeError, match="unexpected defect"):
        volume_statistics.collect_volume_statistics(data, "slice")


@pytest.mark.parametrize("channels", [1, 3])
def test_partition_planning_uses_configured_raster_and_reuses_source_interface(tmp_path, monkeypatch, channels):
    source = _cache(channels=channels).cv
    data = _input(tmp_path)
    config = data.slice_options
    rects = np.array([[[0, 0, 0], [100, 0, 0], [100, 200, 0], [0, 200, 0]]])
    box = BoundingBox.from_rects(rects)
    events = []

    def interface(url):
        assert url == data.source_url
        events.append("metadata")
        return source

    def partition(received, *, target_slices_per_box, max_depth, task_payload,
                  executor_factory=None):
        assert executor_factory is None
        events.append("partition")
        assert received is rects
        assert target_slices_per_box == config.bounding_box_params.target_slices_per_box
        assert max_depth == config.bounding_box_params.max_depth
        # Rectangle edge lengths differ from the configured 3x4 raster.
        expected = estimate_slice_task_payload(
            box, 1, True, slice_shape=(3, 4), source_dtype=source.dtype,
            channels=source.num_channels,
        ).peak_bytes
        for count, contiguous in ((1, True), (7, True), (7, False)):
            assert task_payload(box, count, contiguous) == expected
        return [box], [0]

    monkeypatch.setattr(volume_cache_pipeline, "CloudVolumeInterface", interface)
    monkeypatch.setattr(volume_cache_pipeline, "calculate_bounding_boxes_bsp_link_rects", partition)
    assert volume_cache_pipeline.VolumeCachePipelineStep()._process(
        (config, rects, data.source_url, data)
    ) is None
    assert events == ["metadata", "partition"]
    assert data.volume_cache.cv is source
    assert data.volume_cache.bounding_boxes[0] is box
    assert data.volume_cache.link_rects == [0]

"""Check planning costs against the actual worker's allocation/dispatch seams."""
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.ndimage._interpolation as scipy_interpolation
import tifffile

from ouroboros.helpers import slice as interpolation
from ouroboros.helpers.bounding_boxes import BoundingBox, slice_indices_are_contiguous
from ouroboros.helpers.options import SliceOptions
from ouroboros.pipeline import slice_parallel_pipeline as worker
from ouroboros.pipeline import volume_cache_pipeline as cache_pipeline


@pytest.mark.parametrize("indices", [[1, 2], [0, 2]])
@pytest.mark.parametrize("channels", [1, 3])
def test_estimated_payload_matches_observed_worker_arrays(tmp_path, monkeypatch, indices, channels):
    box = BoundingBox(BoundingBox.bounds_to_rect(0, 11, 0, 12, 0, 13))
    volume = np.arange(np.prod(box.get_shape()) * channels, dtype=np.uint16).reshape(
        *box.get_shape(), channels)
    rects = np.array([[[2, 2, z], [8, 2, z], [8, 9, z], [2, 9, z]] for z in (3, 5, 7)])
    config = SliceOptions(slice_height=5, slice_width=7, neuroglancer_json="unused",
                          output_file_folder=str(tmp_path), output_file_name="sample", process_count=4)
    path = tmp_path / "sample.tif"
    target = tifffile.memmap(path, shape=(3, 5, 7, channels), dtype=np.float32,
                            photometric="rgb" if channels == 3 else "minisblack")
    del target
    observed = {"grid": [], "coefficients": [], "results": [], "buffer": []}
    real_filter = scipy_interpolation.spline_filter
    real_map = interpolation.map_coordinates
    real_slice = worker.slice_volume_from_grids

    def filtered(*args, **kwargs):
        coefficients = real_filter(*args, **kwargs)
        observed["coefficients"].append(coefficients.nbytes)
        assert coefficients.dtype == np.float64
        return coefficients

    def mapped(source, coordinates, **kwargs):
        result = real_map(source, coordinates, **kwargs)
        destination = kwargs.get("output")
        if isinstance(destination, np.ndarray):
            assert result is destination
            assert isinstance(destination.base, np.memmap)
        else:
            observed["results"].append(result.nbytes)
        return result

    def sliced(source, bounding_box, grids, width, height, *, output=None):
        assert source is volume and bounding_box is box
        assert grids.flags.owndata
        observed["grid"].append(grids.nbytes)
        result = real_slice(source, bounding_box, grids, width, height, output=output)
        if output is None:
            assert result.flags.owndata
            observed["buffer"].append(result.nbytes)
        else:
            assert result is output and isinstance(output, np.memmap)
        return result

    monkeypatch.setattr(scipy_interpolation, "spline_filter", filtered)
    monkeypatch.setattr(interpolation, "map_coordinates", mapped)
    monkeypatch.setattr(worker, "slice_volume_from_grids", sliced)
    worker.process_worker_save_parallel(config, volume, box, indices, 0, rects, str(path))
    estimate = interpolation.estimate_slice_task_payload(
        box, len(indices), slice_indices_are_contiguous(indices),
        slice_shape=(config.slice_height, config.slice_width), source_dtype=volume.dtype, channels=channels)
    assert estimate.source_bytes == volume.nbytes
    assert observed["grid"] == [estimate.grid_bytes]
    assert observed["coefficients"] == [estimate.coefficient_bytes] * channels
    assert observed["buffer"] == ([] if estimate.output_bytes == 0 else [estimate.output_bytes])
    assert observed["results"] == ([] if estimate.channel_result_bytes == 0
                                   else [estimate.channel_result_bytes] * channels)
    # Channels are sequential; their workspaces are not added together.
    assert estimate.peak_bytes == (volume.nbytes + max(observed["grid"])
                                   + max(observed["coefficients"])
                                   + max(observed["buffer"], default=0)
                                   + max(observed["results"], default=0))


def test_pipeline_uses_configured_raster_and_same_source_metadata_owner(tmp_path, monkeypatch):
    config = SliceOptions(slice_height=5, slice_width=7, neuroglancer_json="unused",
                          output_file_folder=str(tmp_path), output_file_name="sample", process_count=4)
    rects = np.array([[[0, 0, 0], [100, 0, 0], [100, 200, 0], [0, 200, 0]]])
    box = BoundingBox.from_rects(rects)
    source = SimpleNamespace(dtype=np.dtype("uint16"), num_channels=3)
    data = SimpleNamespace()
    events = []

    def interface(url):
        events.append("metadata")
        assert url == "file://source"
        return source

    def partition(received, *, target_slices_per_box, max_depth, task_payload):
        events.append("partition")
        assert received is rects
        assert target_slices_per_box == config.bounding_box_params.target_slices_per_box
        assert max_depth == config.bounding_box_params.max_depth
        assert task_payload(box, 1, True) == interpolation.estimate_slice_task_payload(
            box, 1, True, slice_shape=(5, 7), source_dtype=source.dtype, channels=source.num_channels).peak_bytes
        return [box], [0]

    def cache(boxes, links, received_source, **kwargs):
        events.append("cache")
        assert received_source is source
        assert boxes[0] is box and links == [0]
        return source

    monkeypatch.setattr(cache_pipeline, "CloudVolumeInterface", interface)
    monkeypatch.setattr(cache_pipeline, "calculate_bounding_boxes_bsp_link_rects", partition)
    monkeypatch.setattr(cache_pipeline, "VolumeCache", cache)
    assert cache_pipeline.VolumeCachePipelineStep()._process((config, rects, "file://source", data)) is None
    assert events == ["metadata", "partition", "cache"]
    assert data.volume_cache is source

"""Upsampled backprojection derives its output shape and origin from the source pyramid."""

import json

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from ouroboros.helpers import files as files_module
from ouroboros.helpers.files import write_conv_vol
from ouroboros.helpers.shapes import ImgSliceC, NGOrder
from ouroboros.helpers.volume_cache import CloudVolumeInterface
from ouroboros.pipeline.backproject_pipeline import get_output_origin, get_output_shape, is_big_tiff
from test.helpers.test_files import _write_intermediate_file

PYRAMID = [(2594, 3190, 7788), (1297, 1595, 3894), (649, 798, 1947)]
ROUNDED_UP = [(2595, 3191, 7788), (1298, 1596, 3894)]
PAGE = ImgSliceC(Y=2, X=3, C=1)
DOUBLED_PAGE = ImgSliceC(Y=4, X=6, C=1)


def _source(tmp_path, sizes):
    """A real local precomputed source, so sizes arrive as cloud-volume reports them."""
    scales = [{"key": str(mip), "size": list(size), "resolution": [2 ** mip] * 3, "voxel_offset": [0, 0, 0],
               "chunk_sizes": [[64, 64, 64]], "encoding": "raw"} for mip, size in enumerate(sizes)]
    info = {"data_type": "uint8", "num_channels": 1, "type": "image", "scales": scales}
    (tmp_path / "info").write_text(json.dumps(info), encoding="utf-8")
    return CloudVolumeInterface(f"precomputed://file://{tmp_path}")


def _write_page(path):
    """A 2 x 3 page whose pixel values are their row-major indexes."""
    _write_intermediate_file(path, np.arange(6), np.arange(6), np.ones(6), source_rows=3, target_rows=3)


@pytest.mark.parametrize("sizes,mip,write,expected", [
    pytest.param(PYRAMID, 1, (10, 15, 20, 1), (20, 30, 40, 1), id="one-level"),
    pytest.param(PYRAMID, 2, (649, 798, 1947, 1), (2594, 3190, 7788, 1), id="two-levels-full-volume"),
    pytest.param(ROUNDED_UP, 1, (1298, 1596, 3894, 1), (2595, 3191, 7788, 1), id="rounded-up-mip-full-volume"),
    pytest.param(PYRAMID, 2, (57, 101, 10, 1), (228, 404, 40, 1), id="region-rounds-up"),
    pytest.param(PYRAMID, 1, (10, 15, 20, 3), (20, 30, 40, 3), id="channels-unscaled"),
])
def test_output_shape_maps_the_written_region_to_the_target_mip(tmp_path, sizes, mip, write, expected):
    assert get_output_shape(_source(tmp_path, sizes), mip, 0, NGOrder(*write)) == NGOrder(*expected)


def test_output_z_uses_the_whole_resolution_multiplier(tmp_path):
    sizes = [(40, 80, 2595), (20, 40, 1298)]
    assert get_output_shape(_source(tmp_path, sizes), 1, 0, NGOrder(20, 40, 1298, 1)).Z == 2596


def test_same_mip_output_shape_is_unchanged(tmp_path):
    write_shape = NGOrder(10, 15, 20, 2)
    assert get_output_shape(_source(tmp_path, PYRAMID), 1, 1, write_shape) == write_shape


@pytest.mark.parametrize("mip,origin,expected", [
    pytest.param(1, (3, 4, 5), (6, 8, 10), id="one-level"),
    pytest.param(2, (10, 20, 30), (40, 80, 120), id="rounds-up"),
    pytest.param(1, (-2, 2, 1), (-4, 4, 2), id="negative-origin"),
])
def test_output_origin_maps_to_target_mip_voxels(tmp_path, mip, origin, expected):
    output_origin = get_output_origin(_source(tmp_path, PYRAMID), mip, 0, np.array(origin))
    assert_array_equal(output_origin, expected)
    assert np.issubdtype(output_origin.dtype, np.integer)


def test_same_mip_output_origin_is_unchanged(tmp_path):
    assert_array_equal(get_output_origin(_source(tmp_path, PYRAMID), 1, 1, np.array((-2, 2, 1))), (-2, 2, 1))


@pytest.mark.parametrize("resolve", [
    pytest.param(lambda source: get_output_shape(source, 0, 1, NGOrder(10, 15, 20, 1)), id="shape"),
    pytest.param(lambda source: get_output_origin(source, 0, 1, np.array((1, 2, 3))), id="origin"),
])
def test_coarser_target_is_rejected(tmp_path, resolve):
    with pytest.raises(NotImplementedError, match="from 0 to 1"):
        resolve(_source(tmp_path, PYRAMID))


@pytest.mark.parametrize("shape,dtype,expected", [
    pytest.param(NGOrder(2 ** 12, 2 ** 10, 2 ** 10, 1), np.uint8, False, id="exactly-at-limit"),
    pytest.param(NGOrder(2 ** 12 + 1, 2 ** 10, 2 ** 10, 1), np.uint8, True, id="just-over-limit"),
    pytest.param(NGOrder(2 ** 12, 2 ** 10, 2 ** 10, 2), np.uint8, True, id="channels-count"),
    pytest.param(NGOrder(2 ** 12, 2 ** 10, 2 ** 10, 1), np.uint16, True, id="sample-size-counts"),
    pytest.param(NGOrder(2 ** 12, 2 ** 10, 2 ** 10, 1), np.dtype("uint16"), True, id="dtype-instance"),
    pytest.param(ImgSliceC(2 ** 16, 2 ** 16, 1), np.uint8, False, id="page-at-limit"),
    pytest.param(ImgSliceC(2 ** 16, 2 ** 16, 3), np.uint8, True, id="page-channels-over-limit"),
    pytest.param(ImgSliceC(70000, 70000, 1), np.uint8, True, id="axis-over-uint16"),
])
def test_big_tiff_decision_counts_every_axis_and_sample_size(shape, dtype, expected):
    assert bool(is_big_tiff(shape, dtype)) is expected


@pytest.fixture
def writes(monkeypatch):
    monkeypatch.setattr(files_module, "_writer_page_buffers", None)
    return []


@pytest.mark.parametrize("output_shape", [
    pytest.param(DOUBLED_PAGE, id="whole-factor"),
    pytest.param(ImgSliceC(Y=5, X=7, C=1), id="non-integer-factor"),
])
def test_writer_resizes_pages_to_the_output_shape(tmp_path, writes, output_shape):
    source = tmp_path / "page.dat"
    _write_page(source)
    perf = write_conv_vol(lambda path, data: writes.append((path.name, data.copy())), source, tmp_path,
                          PAGE, output_shape, 1, np.uint16, 0, 0, False)
    assert [name for name, _ in writes] == ["00000.tif"]
    assert writes[0][1].shape[:2] == (output_shape.Y, output_shape.X)
    assert writes[0][1].dtype == np.uint16
    assert "Zoom" in perf


def test_writer_keeps_orientation_when_resizing(tmp_path, writes):
    source = tmp_path / "page.dat"
    _write_page(source)
    write_conv_vol(lambda path, data: writes.append(data.copy()), source, tmp_path,
                   PAGE, DOUBLED_PAGE, 1, np.uint16, 0, 0, False)
    assert_array_equal(writes[0], [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5], [3, 3, 4, 4, 5, 5]])


def test_writer_repeats_each_page_for_the_output_z_positions(tmp_path, writes):
    source = tmp_path / "page.dat"
    _write_page(source)
    write_conv_vol(lambda path, data: writes.append(path.name), source, tmp_path,
                   PAGE, DOUBLED_PAGE, 2, np.uint16, 3, 0, False)
    assert writes == ["00006.tif", "00007.tif"]


def test_writer_resizes_binary_pages_as_bytes(tmp_path, writes):
    source = tmp_path / "page.dat"
    _write_page(source)
    write_conv_vol(lambda path, data: writes.append(data.copy()), source, tmp_path,
                   PAGE, DOUBLED_PAGE, 1, bool, 0, 0, False)
    assert writes[0].dtype == np.uint8
    assert writes[0].shape == (4, 6)


def test_writer_reuses_source_storage_across_resized_pages(tmp_path, writes):
    source = tmp_path / "page.dat"
    outputs = []
    for index in range(2):
        _write_page(source)
        write_conv_vol(lambda path, data: writes.append(data.copy()), source, tmp_path,
                       PAGE, DOUBLED_PAGE, 1, np.uint16, index, 0, False)
        outputs.append(files_module._writer_page_buffers[1])
    assert outputs[0] is outputs[1]
    assert outputs[0].shape == (2, 3, 1)
    assert_array_equal(writes[0], writes[1])

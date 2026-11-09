"""Upsampled backprojection derives its output shape and origin from the source pyramid."""

import json

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from ouroboros.helpers.shapes import ImgSliceC, NGOrder
from ouroboros.helpers.volume_cache import CloudVolumeInterface
from ouroboros.pipeline.backproject_pipeline import get_output_origin, get_output_shape, is_big_tiff

PYRAMID = [(2594, 3190, 7788), (1297, 1595, 3894), (649, 798, 1947)]
ROUNDED_UP = [(2595, 3191, 7788), (1298, 1596, 3894)]


def _source(tmp_path, sizes):
    """A real local precomputed source, so sizes arrive as cloud-volume reports them."""
    scales = [{"key": str(mip), "size": list(size), "resolution": [2 ** mip] * 3, "voxel_offset": [0, 0, 0],
               "chunk_sizes": [[64, 64, 64]], "encoding": "raw"} for mip, size in enumerate(sizes)]
    info = {"data_type": "uint8", "num_channels": 1, "type": "image", "scales": scales}
    (tmp_path / "info").write_text(json.dumps(info), encoding="utf-8")
    return CloudVolumeInterface(f"precomputed://file://{tmp_path}")


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

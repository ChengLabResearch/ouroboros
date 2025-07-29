import pytest
from unittest.mock import MagicMock, patch
from ouroboros.helpers.volume_cache import (
    VolumeCache,
    CloudVolumeInterface,
    get_mip_volume_sizes,
    update_writable_boxes,
    update_writable_rects
)
from ouroboros.helpers.bounding_boxes import BoundingBox, boxes_dim_range


@pytest.fixture
def mock_cloud_volume():
    with patch("ouroboros.helpers.volume_cache.CloudVolume") as MockCloudVolume:
        mock_cv = MockCloudVolume.return_value
        mock_cv.available_mips = [0, 1, 2]
        mock_cv.dtype = "uint8"
        mock_cv.shape = (100, 100, 100, 3)
        mock_cv.cache.flush = MagicMock()
        mock_cv.mip_volume_size = lambda mip: (100, 100, 100)
        yield mock_cv


@pytest.fixture
def bounding_boxes():
    return [
        BoundingBox(BoundingBox.bounds_to_rect(0, 0, 0, 10, 10, 10)),
        BoundingBox(BoundingBox.bounds_to_rect(10, 10, 10, 20, 20, 20)),
    ]


@pytest.fixture
def cloud_volume_interface(mock_cloud_volume):
    return CloudVolumeInterface("test_source_url")


@pytest.fixture
def volume_cache(bounding_boxes, cloud_volume_interface):
    return VolumeCache(
        bounding_boxes=bounding_boxes,
        link_rects=[0, 1],
        cloud_volume_interface=cloud_volume_interface,
        mip=None,
        flush_cache=True,
    )


def test_volume_cache_init(volume_cache, bounding_boxes):
    assert volume_cache.bounding_boxes == bounding_boxes
    assert volume_cache.link_rects == [0, 1]
    assert volume_cache.cv.source_url == "test_source_url"
    assert volume_cache.mip == 0
    assert volume_cache.flush_cache is True
    assert volume_cache.volumes == [None, None]
    assert volume_cache.cache_volume == [False, False]
    assert volume_cache.volume_index(1) == 1
    assert volume_cache.get_num_channels() == 3

    volume_cache.set_volume_mip(3)
    assert volume_cache.mip == 3


def test_volume_cache_to_dict(volume_cache):
    volume_cache_dict = volume_cache.to_dict()
    assert volume_cache_dict["bounding_boxes"] == [
        bb.to_dict() for bb in volume_cache.bounding_boxes
    ]
    assert volume_cache_dict["link_rects"] == volume_cache.link_rects
    assert volume_cache_dict["cv"]["source_url"] == "test_source_url"
    assert volume_cache_dict["mip"] == volume_cache.mip
    assert volume_cache_dict["flush_cache"] == volume_cache.flush_cache


def test_volume_cache_from_dict(volume_cache):
    with patch(
        "ouroboros.helpers.volume_cache.CloudVolumeInterface.from_dict"
    ) as mock_from_dict:
        volume_cache_dict = {
            "bounding_boxes": [bb.to_dict() for bb in volume_cache.bounding_boxes],
            "link_rects": volume_cache.link_rects,
            "cv": {"source_url": "test_source_url"},
            "mip": 0,
            "flush_cache": True,
        }
        mock_from_dict.return_value = volume_cache.cv
        new_volume_cache = VolumeCache.from_dict(volume_cache_dict)
        assert len(new_volume_cache.bounding_boxes) == len(volume_cache.bounding_boxes)
        assert new_volume_cache.link_rects == volume_cache.link_rects
        assert new_volume_cache.cv.source_url == "test_source_url"
        assert new_volume_cache.mip == 0
        assert new_volume_cache.flush_cache is True
        mock_from_dict.assert_called_once_with(volume_cache_dict["cv"])


def test_volume_cache_get_volume_gigabytes(volume_cache):
    with patch(
        "ouroboros.helpers.volume_cache.calculate_gigabytes_from_dimensions"
    ) as mock_calculate:
        mock_calculate.return_value = 1.0
        gigabytes = volume_cache.get_volume_gigabytes()
        assert gigabytes == 1.0
        mock_calculate.assert_called_once()


def test_volume_cache_flush(volume_cache):
    with patch.object(volume_cache.cv, "flush_cache") as mock_flush_cache:
        volume_cache.flush_local_cache()
        mock_flush_cache.assert_called_once()


def test_cloud_volume_interface_init(mock_cloud_volume):
    cvi = CloudVolumeInterface("test_source_url")
    assert cvi.source_url == "test_source_url"
    assert cvi.available_mips == [0, 1, 2]
    assert cvi.dtype == "uint8"


def test_cloud_volume_interface_to_dict(cloud_volume_interface):
    cvi_dict = cloud_volume_interface.to_dict()
    assert cvi_dict == {"source_url": "test_source_url"}


def test_cloud_volume_interface_get_volume_shape(cloud_volume_interface):
    shape = cloud_volume_interface.get_volume_shape(0)
    assert shape == (100, 100, 100)


def test_cloud_volume_interface_get_resolution_nm(cloud_volume_interface):
    with patch.object(cloud_volume_interface.cv, "mip_resolution", return_value=1000):
        resolution = cloud_volume_interface.get_resolution_nm(0)
        assert resolution == 1000


def test_cloud_volume_interface_get_resolution_um(cloud_volume_interface):
    with patch.object(cloud_volume_interface.cv, "mip_resolution", return_value=1000):
        resolution = cloud_volume_interface.get_resolution_um(0)
        assert resolution == 1.0


def test_cloud_volume_interface_flush_cache(cloud_volume_interface):
    with patch.object(cloud_volume_interface.cv.cache, "flush") as mock_flush:
        cloud_volume_interface.flush_cache()
        mock_flush.assert_called_once()


def test_volume_cache_has_color_channels(volume_cache):
    assert volume_cache.has_color_channels() == volume_cache.cv.has_color_channels


def test_volume_cache_should_cache_last_volume(volume_cache):
    assert volume_cache.should_cache_last_volume([1, 2, 3, 1]) is True
    assert volume_cache.should_cache_last_volume([1, 2, 3, 4]) is False


def test_volume_cache_get_dtype(volume_cache):
    assert volume_cache.get_volume_dtype() == "uint8"


def test_volume_cache_get_available_mips(volume_cache):
    assert volume_cache.get_volume_mip() == 0


def test_volume_cache_get_shape(volume_cache):
    assert volume_cache.get_volume_shape() == (100, 100, 100)


def test_request_volume_for_slice(volume_cache):
    slice_index = 1
    with patch.object(
        volume_cache, "volume_index", return_value=1
    ) as mock_volume_index:
        volume_data, bounding_box = volume_cache.request_volume_for_slice(slice_index)

        mock_volume_index.assert_called_once_with(slice_index)
        assert bounding_box == volume_cache.bounding_boxes[1]
        assert volume_data == volume_cache.volumes[1]


def test_create_processing_data(volume_cache):
    # Patch volume_cache.download to set the volume data
    with patch.object(volume_cache, "download_volume") as mock_download:

        def mock_download_func(volume_index, bounding_box, parallel):
            volume_cache.volumes[volume_index] = bounding_box.to_empty_volume()

        mock_download.side_effect = mock_download_func

        # Call the method
        processing_data = volume_cache.create_processing_data(0)

        # Check the return values
        assert processing_data[0] is not None
        assert processing_data[1] == volume_cache.bounding_boxes[0]
        assert processing_data[2] == [0]
        assert processing_data[3] == 0


def test_get_mip_volume_sizes(mock_cloud_volume):
    with patch.object(mock_cloud_volume, "mip_volume_size") as mock_mip_volume_size:
        mock_mip_volume_size.return_value = (100, 100, 100)

        sizes = get_mip_volume_sizes("test_source_url")

        assert sizes == {0: (100, 100, 100), 1: (100, 100, 100), 2: (100, 100, 100)}


def test_get_mip_volume_sizes_error(mock_cloud_volume):
    with patch.object(mock_cloud_volume, "mip_volume_size") as mock_mip_volume_size:
        mock_mip_volume_size.return_value = (100, 100, 100)
        mock_mip_volume_size.side_effect = Exception

        sizes = get_mip_volume_sizes("test_source_url")

        assert sizes == {}


def test_cloud_volume_interface_from_dict(mock_cloud_volume):
    cvi = CloudVolumeInterface.from_dict({"source_url": "test_source_url"})
    assert cvi.source_url == "test_source_url"
    assert cvi.available_mips == [0, 1, 2]
    assert cvi.dtype == "uint8"


def test_cloud_volume_channels(cloud_volume_interface):
    assert cloud_volume_interface.num_channels == 3


def test_volume_cache_remove_volume(volume_cache):
    slice_index = 1
    with patch.object(
        volume_cache, "volume_index", return_value=1
    ) as mock_volume_index:
        volume_data, bounding_box = volume_cache.request_volume_for_slice(slice_index)

        mock_volume_index.assert_called_once_with(slice_index)
        assert volume_data == volume_cache.volumes[1]
        volume_cache.remove_volume(1)
        assert volume_cache.volumes[1] is None


def test_boxes_dim_range(volume_cache):
    import numpy as np

    remaining = volume_cache.bounding_boxes.copy()

    assert np.all(boxes_dim_range(remaining) == np.arange(10, 22, dtype=int))
    assert np.all(boxes_dim_range([BoundingBox(BoundingBox.bounds_to_rect(0, 0, 0, 10, 10, 10))]) == np.array([10, 11], dtype=int))
    assert np.all(boxes_dim_range(remaining, dim="x") == np.arange(0, 12, dtype=int))

    with pytest.raises(ValueError) as ve:
        boxes_dim_range(remaining, dim='q')
        
    assert np.all(boxes_dim_range([]) == np.array([], dtype=int))


def test_update_writeable_boxes(volume_cache):
    import numpy as np

    remaining = volume_cache.bounding_boxes.copy()

    writeable = np.empty(0, dtype=int)
    writeable = update_writable_boxes(volume_cache, writeable, remaining, 0)

    assert np.all(boxes_dim_range(remaining) == np.array([20, 21], dtype=int))
    assert np.all(writeable == np.array([10, 11], dtype=int))


def generate_chunked_rects(start, stop, step):
    import numpy as np
    return (np.array([BoundingBox.bounds_to_rect(start, stop, start, stop, z, z + 1)
                     for z in range(start, stop)]),
           np.array([BoundingBox.bounds_to_rect(x, x + step, y, y + step, z, z + step)
                     for z in range(start, stop, step)
                     for y in range(start, stop, step)
                     for x in range(start, stop, step)]))


def test_update_writeable_rects(volume_cache):
    import numpy as np
    chunk_size = 2
    min_dim = 4
    max_dim = 12
    # This is not the right way to setup the rects but it works for testing.
    full_rects, chunk_rects = generate_chunked_rects(min_dim, max_dim, chunk_size)

    writeable = np.zeros(max_dim - min_dim, dtype=int)
    processed = np.zeros(chunk_rects.shape, dtype=bool)

    update_writable_rects(processed, full_rects, min_dim, writeable, chunk_size)

    assert not np.any(writeable)

    processed[0, 0, 1] = True

    update_writable_rects(processed, full_rects, min_dim, writeable, chunk_size)

    assert not np.any(writeable)

    processed[0, :, :] = True
    
    update_writable_rects(processed, full_rects, min_dim, writeable, chunk_size)

    assert np.all(writeable[0:2])
    assert not np.any(writeable[2:])

    writeable[0:2] = 2
    processed[1, :, :] = True

    update_writable_rects(processed, full_rects, min_dim, writeable, chunk_size)

    assert np.all(writeable[0:4])
    assert not np.any(writeable[4:])
    assert np.all(writeable[0:2] == 2)
    
	# Properly handle all fields processed.
    processed[:] = True
    update_writable_rects(processed, full_rects, min_dim, writeable, chunk_size)

    assert np.all(writeable)
    assert np.all(writeable[0:2] == 2)

"""Native download storage, allocation ownership, and untouched background."""

import gc
import json
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from types import SimpleNamespace

from cloudvolume import Bbox
import numpy as np
import psutil
import pytest

from ouroboros.common.memory_ipc import process_from_identity
from ouroboros.common.memory_managed import ManagerSession
from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.mem import SharedNPArray, SharedNPManager
from ouroboros.helpers.volume_cache import CloudVolumeInterface, download_volume


SOURCE_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64, np.float32]


@pytest.fixture(scope="module")
def shared_manager():
    with ManagerSession() as session:
        with SharedNPManager(ctx=multiprocessing.get_context("spawn")) as (manager,):
            yield manager, session.context
            assert manager._request("list_segments") == []
        assert session.context.failure_flag.value == 0


def _snapshot(manager, context):
    return json.loads(Path(context.directory, f"{manager._process.pid}.json").read_text())


def _source(dtype, channels=1, download=None):
    values = np.arange(3 * 3 * 2 * channels, dtype=dtype).reshape(3, 3, 2, channels)
    if np.issubdtype(dtype, np.integer):
        values.flat[0] = np.iinfo(dtype).min
        values.flat[-1] = np.iinfo(dtype).max
    else:
        values /= 7
        values.flat[-1] = 1 + 2**-40

    def read(box, **kwargs):
        assert box == Bbox((0, 0, 0), (3, 3, 2))
        assert kwargs == {"mip": 2, "parallel": 3}
        return values

    source = object.__new__(CloudVolumeInterface)
    source.dtype = np.dtype(dtype)
    source.cv = SimpleNamespace(
        bounds=Bbox((0, 0, 0), (3, 3, 2)), num_channels=channels,
        download=read if download is None else download,
    )
    return source, values


@pytest.mark.parametrize("dtype", SOURCE_DTYPES)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("use_shared", [False, True])
def test_download_preserves_source_values_and_clipped_zero_background(
        dtype, channels, use_shared, shared_manager):
    manager, _ = shared_manager
    source, values = _source(dtype, channels)
    box = BoundingBox(BoundingBox.bounds_to_rect(-1, 4, 0, 3, 0, 2))
    result, *_ = download_volume(
        source, box, mip=2, parallel=3, use_shared=use_shared,
        shm_address=manager.address, shm_authkey=bytes(manager._authkey),
    )
    try:
        pixels = result.array() if use_shared else result
        assert pixels.shape == (5, 3, 2, channels)
        assert pixels.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(pixels[1:4], values)
        assert np.all(pixels[0] == 0)
        assert np.all(pixels[4] == 0)
    finally:
        if use_shared:
            manager.remove_termed(result)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_download_allocates_native_bytes_without_writing_the_full_buffer(
        dtype, shared_manager, monkeypatch):
    manager, context = shared_manager
    writes = []
    enter = SharedNPArray.__enter__

    class ObservedArray(np.ndarray):
        def __setitem__(self, key, value):
            writes.append(self[key].size)
            return super().__setitem__(key, value)

        def fill(self, value):
            pytest.fail("Download filled the complete shared buffer")

    def observe(array):
        # Observe the real manager-owned mapping; this view never owns memory.
        return enter(array).view(ObservedArray)

    source, values = _source(dtype)
    read = source.cv.download

    def download(box, **kwargs):
        assert writes == [], "Download wrote shared pages before receiving source data"
        return read(box, **kwargs)

    source.cv.download = download
    monkeypatch.setattr(SharedNPArray, "__enter__", observe)
    # Page-aligned byte counts make the manager's native size exact on Windows too.
    box = BoundingBox(BoundingBox.bounds_to_rect(-1, 63, 0, 64, 0, 16))
    result, *_ = download_volume(
        source, box, mip=2, parallel=3, use_shared=True,
        shm_address=manager.address, shm_authkey=bytes(manager._authkey),
    )
    name = result.name
    try:
        voxel_count = 64 * 64 * 16
        expected_bytes = voxel_count * np.dtype(dtype).itemsize
        assert result.size() == expected_bytes
        assert not result.owns_unlink
        snapshot = _snapshot(manager, context)
        assert process_from_identity(snapshot["owner"]) == psutil.Process(manager._process.pid)
        segment, = snapshot["segments"]
        assert segment["name"] == name
        assert segment["size"] == expected_bytes
        assert writes == [values.size]
        assert manager._request("list_segments") == [name]
    finally:
        manager.remove_termed(result)


@pytest.mark.parametrize("failure", ["download", "assignment"])
def test_failed_download_remains_owned_until_manager_shutdown(failure):
    with ManagerSession() as session:
        with SharedNPManager(ctx=multiprocessing.get_context("spawn")) as (manager,):
            def fail(box, **kwargs):
                if failure == "download":
                    raise OSError("deliberate download failure")
                return np.zeros((2, 2), dtype=np.uint16)

            source, _ = _source(np.uint16, download=fail)
            box = BoundingBox(BoundingBox.bounds_to_rect(-1, 4, 0, 3, 0, 2))
            with pytest.raises(OSError if failure == "download" else ValueError):
                download_volume(
                    source, box, mip=2, parallel=3, use_shared=True,
                    shm_address=manager.address, shm_authkey=bytes(manager._authkey),
                )
            gc.collect()
            snapshot = _snapshot(manager, session.context)
            segment, = snapshot["segments"]
            name = segment["name"]
            assert manager._request("list_segments") == [name]
        assert _snapshot(manager, session.context)["segments"] == []
        assert session.context.failure_flag.value == 0
        with pytest.raises(FileNotFoundError):
            attachment = SharedMemory(name)
            attachment.close()

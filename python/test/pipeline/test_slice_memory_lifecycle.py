"""Exercise actual slicing dispatch/release with tiny local data and live pools."""

import concurrent.futures
from contextlib import contextmanager
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import sys
import time

from cloudvolume import Bbox
import numpy as np
import psutil
import pytest

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.mem import SharedNPManager
from ouroboros.helpers.options import SliceOptions
from ouroboros.helpers.volume_cache import CloudVolumeInterface, VolumeCache
from ouroboros.pipeline import slice_parallel_pipeline as slicing
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput


class LocalVolume:
    """Picklable replacement for remote storage; production download code runs."""

    bounds = Bbox((0, 0, 0), (8, 8, 8))
    num_channels = 1
    shape = (8, 8, 8, 1)

    def download(self, bbox, **kwargs):
        return np.full((*bbox.size3(), 1), 37, dtype=np.uint16)

    def mip_resolution(self, mip):
        return np.full(3, 1000)


def linux_attachments(pids, names):
    """Inspect actual mappings AND descriptors, including unlinked backing."""
    targets = {'/dev/shm/' + name for name in names}
    retained = []
    for pid in pids:
        for region in psutil.Process(pid).memory_maps(grouped=False):
            if region.path.removesuffix(' (deleted)') in targets:
                retained.append((pid, 'mapping', region.path))
        for descriptor in Path(f'/proc/{pid}/fd').iterdir():
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            if target.removesuffix(' (deleted)') in targets:
                retained.append((pid, 'descriptor', target))
    return retained


@pytest.mark.parametrize('start_method', multiprocessing.get_all_start_methods())
def test_released_slice_volume_has_no_worker_attachments_before_pool_shutdown(
    tmp_path, monkeypatch, start_method,
):
    context = multiprocessing.get_context(start_method)
    source = object.__new__(CloudVolumeInterface)
    source.cv = LocalVolume()
    source.available_mips = [0]
    source.dtype = np.dtype(np.uint16)
    rects = np.array([[[0, 0, z], [2, 0, z], [2, 2, z], [0, 2, z]] for z in (1, 5)], dtype=float)
    boxes = [BoundingBox(BoundingBox.bounds_to_rect(0, 3, 0, 3, z, z + 3)) for z in (0, 4)]
    cache = VolumeCache(boxes, [0, 1], source, use_shared=True)
    # Release one completed volume while intentionally retaining another.
    cache.cache_volume = [False, True]
    options = SliceOptions(output_file_folder=str(tmp_path), output_file_name='local',
                           neuroglancer_json='unused', slice_width=2, slice_height=2,
                           make_single_file=False, normalize_output=False)
    real_executor = concurrent.futures.ProcessPoolExecutor
    pools, managers, released, inspected = [], [], [], []

    def executor(*args, **kwargs):
        pool = real_executor(*args, mp_context=context, **kwargs)
        pools.append(pool)
        return pool

    def manager(**kwargs):
        owner = SharedNPManager(ctx=context, **kwargs)
        managers.append(owner)
        return owner

    remove_volume = cache.remove_volume

    def release(index, destroy_shared=False):
        if not cache.cache_volume[index]:
            released.append(cache.volumes[index].name)
        remove_volume(index, destroy_shared=destroy_shared)

    @contextmanager
    def inspect_before_rewrite(*args, **kwargs):
        # This is the production boundary after all slice callbacks and before
        # manager/pool shutdown. Keep the real conversion and TIFF writes.
        assert len(released) == 1
        cached = cache.volumes[1]
        assert managers[0]._request('list_segments') == [cached.name]
        with cached as pixels:
            assert np.all(pixels == 37)
        pids = [pid for pool in pools for pid in pool._processes]
        if start_method == 'fork':
            assert len(pools[1]._processes) == 3
        assert all(psutil.pid_exists(pid) for pid in pids)
        deadline = time.monotonic() + 5
        while True:
            if sys.platform == 'linux':
                remaining = linux_attachments(pids, released)
            elif sys.platform == 'win32':
                # On Windows an openable name proves some handle still owns
                # the buffer; on POSIX unlink alone would not prove cleanup.
                remaining = []
                for name in released:
                    try:
                        handle = SharedMemory(name)
                    except FileNotFoundError:
                        continue
                    handle.close()
                    remaining.append(name)
            else:
                # macOS exercises real dispatch, manager release and cached
                # attachment validity; Linux provides the /proc retention proof.
                remaining = []
            if not remaining:
                break
            assert time.monotonic() < deadline, f'Workers retain released backing: {remaining}'
            time.sleep(0.01)
        inspected.append(True)
        from multiprocessing.pool import ThreadPool
        with ThreadPool(*args, **kwargs) as pool:
            yield pool

    monkeypatch.setattr(slicing.concurrent.futures, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(slicing, 'SharedNPManager', manager)
    monkeypatch.setattr(cache, 'remove_volume', release)
    # Patch the call site without replacing multiprocessing.pool.ThreadPool,
    # which inspect_before_rewrite uses to perform the real output conversion.
    from types import SimpleNamespace
    monkeypatch.setattr(slicing, 'multiprocessing', SimpleNamespace(
        pool=SimpleNamespace(ThreadPool=inspect_before_rewrite)))
    # Existing metadata keyword compatibility is unrelated to allocation lifetime.
    monkeypatch.setattr(slicing, 'build_straightened_tiff_metadata',
                        lambda **kwargs: {'photometric': 'minisblack'})
    data = PipelineInput(slice_options=options, slice_rects=rects, volume_cache=cache)
    try:
        output, error = Pipeline([slicing.SliceParallelPipelineStep(processes=4)]).process(data)
        assert error is None, error
        assert output is not None
        assert inspected == [True]
    finally:
        for volume in cache.volumes:
            if volume is not None:
                volume.shutdown()

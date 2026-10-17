from concurrent.futures import ProcessPoolExecutor
import ctypes
import gc
import json
import mmap
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import pickle
import shutil
import weakref

import numpy as np
import psutil
import pytest
import tifffile

from ouroboros.common import memory_mappings as mappings
from ouroboros.helpers.mem import SharedNPArray
from ouroboros.helpers.shapes import ImgSlice


@pytest.fixture(autouse=True)
def isolated_observations(monkeypatch):
    # Exercise the native mapping adapter on Linux, without changing Linux's
    # production process-metric path or pretending to test native residency.
    monkeypatch.setattr(mappings, "_SUPPORTED", True)
    for name, value in (("_context", None), ("_session_context", None),
                        ("_session_count", 0), ("_registry", {}), ("_errors", []),
                        ("_create_time", None)):
        monkeypatch.setattr(mappings, name, value)
    yield
    if mappings._context is not None:
        shutil.rmtree(mappings._context.directory, ignore_errors=True)
        mappings._context = None


def snapshot(context, pid=None):
    return json.loads(Path(context.directory, f"{pid or os.getpid()}.json").read_text())


def _configure_native_test_worker(context):
    # Spawn imports the module afresh, so enable its native-only adapter in the
    # test process before invoking the actual executor initializer.
    mappings._SUPPORTED = True
    mappings.configure_mappings(context)


def _inspect_unpickled_mapping(array):
    before = snapshot(mappings._context)
    array.shutdown()
    return before, snapshot(mappings._context)


def test_linux_does_not_register_or_create_an_observation_channel(monkeypatch):
    monkeypatch.setattr(mappings, "_SUPPORTED", False)

    def unexpected(*args, **kwargs):
        pytest.fail("disabled mapping observation performed work")

    monkeypatch.setattr(mappings, "_ensure_process", unexpected)
    monkeypatch.setattr(mappings.tempfile, "mkdtemp", unexpected)
    sentinel = object()
    monkeypatch.setattr(tifffile, "memmap", lambda *args, **kwargs: sentinel)
    mappings.observe_shared_memory(object())
    mappings.refresh_mappings()
    assert mappings.tracked_tiff_memmap("unchanged-argument") is sentinel
    session = mappings.MappingSession().acquire()
    assert session.context is None
    assert mappings.executor_mapping_options(None) == {}
    assert mappings._registry == {}


def test_existing_owner_is_observed_when_first_session_starts(monkeypatch):
    shared = SharedMemory(create=True, size=4096)
    array = SharedNPArray(shared.name, ImgSlice(2, 4), np.uint8)
    describe = mappings._describe
    calls = []

    def recorded(*args):
        calls.append(args)
        return describe(*args)

    monkeypatch.setattr(mappings, "_describe", recorded)
    try:
        assert mappings._context is None
        assert calls == []
        assert all(observation.record is None for observation in mappings._registry.values())
        with mappings.MappingSession() as session:
            observed = snapshot(session.context)
            assert observed["pid"] == os.getpid()
            assert observed["create_time"] == psutil.Process().create_time()
            assert observed["errors"] == []
            assert len(calls) == 1
            record, = observed["mappings"]
            assert record["identity"] == ["shm", shared.name]
            assert record["size"] == 4096  # Full mapping, not the eight-byte logical array.
            assert record["offset"] == 0
            with array as data:
                assert record["address"] == data.ctypes.data
        assert mappings._context is None
        # Reopening a session observes the same live allocation.
        with mappings.MappingSession() as later:
            assert snapshot(later.context)["mappings"] == [record]
    finally:
        array.shutdown()
        shared.close()
        shared.unlink()


def test_overlapping_sessions_share_channel_until_last_release():
    first = mappings.MappingSession().acquire()
    second = mappings.MappingSession().acquire()
    assert first.acquire() is first
    assert first.context is second.context
    directory = Path(first.context.directory)
    first.release()
    first.release()
    assert directory.is_dir()
    assert mappings._context is second.context
    second.release()
    assert not directory.exists()
    assert mappings._context is None


def test_unreported_pool_inherits_an_existing_observation_session():
    assert mappings.executor_mapping_options(None) == {}
    with mappings.MappingSession() as session:
        options = mappings.executor_mapping_options(None)
        assert options["initializer"] is mappings.configure_mappings
        assert options["initargs"] == (session.context,)
        assert mappings._session_count == 1
    assert mappings.executor_mapping_options(None) == {}


def test_shared_views_pickle_and_shutdown_follow_actual_handles():
    shared = SharedMemory(create=True, size=4096)
    arrays = []
    try:
        with mappings.MappingSession() as session:
            root = SharedNPArray(shared.name, ImgSlice(2, 4), np.uint8)
            view = root[1:]
            restored = pickle.loads(pickle.dumps(view))
            arrays.extend((root, view, restored))
            records = snapshot(session.context)["mappings"]
            assert len(records) == 3
            assert len({record["address"] for record in records}) == 3
            assert all(record["identity"] == ["shm", shared.name] for record in records)
            assert all(record["size"] == 4096 for record in records)
            for remaining, array in zip((2, 1, 0), arrays):
                array.shutdown()
                assert len(snapshot(session.context)["mappings"]) == remaining
    finally:
        for array in arrays:
            array.shutdown()
        shared.close()
        shared.unlink()


def test_executor_initializer_precedes_shared_array_unpickle():
    shared = SharedMemory(create=True, size=4096)
    array = SharedNPArray(shared.name, ImgSlice(2, 4), np.uint8)
    try:
        with mappings.MappingSession() as session:
            options = mappings.executor_mapping_options(session.context)
            assert options["initializer"] is mappings.configure_mappings
            assert options["initargs"] == (session.context,)
            options["initializer"] = _configure_native_test_worker
            with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"), **options) as pool:
                before, after = pool.submit(_inspect_unpickled_mapping, array).result(timeout=15)
            assert before["pid"] != os.getpid()
            record, = before["mappings"]
            assert record["identity"] == ["shm", shared.name]
            assert record["size"] == 4096
            assert before["errors"] == after["errors"] == []
            assert after["mappings"] == []
            assert session.context.failure_flag.value == 0
            assert len(snapshot(session.context)["mappings"]) == 1
    finally:
        array.shutdown()
        shared.close()
        shared.unlink()


def test_tiff_mapping_observes_full_backing_offset_identity_mode_and_view_lifetime(tmp_path):
    path = tmp_path / "data.tif"
    alias = tmp_path / "alias.tif"
    tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint8))
    os.link(path, alias)
    # Created before reporting starts; retain its original backing identity but
    # defer address capture until reporting begins.
    root = mappings.tracked_tiff_memmap(path, mode="r")
    assert all(observation.record is None for observation in mappings._registry.values())
    with mappings.MappingSession() as session:
        record, = snapshot(session.context)["mappings"]
        mapping_offset = root.offset - root.offset % mmap.ALLOCATIONGRANULARITY
        assert root.offset != mapping_offset  # Header lies inside the mapping.
        assert record["address"] == root.ctypes.data - (root.offset - mapping_offset)
        assert record["offset"] == mapping_offset
        assert record["size"] == len(root._mmap)
        assert record["mode"] == "r"
        stat = path.stat()
        assert record["identity"] == ["file", stat.st_dev, stat.st_ino]
        other = mappings.tracked_tiff_memmap(alias, mode="c")
        records = snapshot(session.context)["mappings"]
        assert len(records) == 2
        assert records[0]["identity"] == records[1]["identity"]
        assert records[1]["mode"] == "c"
        other._mmap.close()
        mappings.refresh_mappings()
        assert snapshot(session.context)["mappings"] == [record]
        view = np.asarray(root)[1:]
        reference = weakref.ref(root)
        del root
        gc.collect()
        assert reference() is not None
        assert snapshot(session.context)["mappings"] == [record]
        del view
        gc.collect()
        assert reference() is None
        assert snapshot(session.context)["mappings"] == []


def test_shared_buffer_export_retains_mapping_after_close_failure():
    shared = SharedMemory(create=True, size=4096)
    try:
        with mappings.MappingSession() as session:
            mappings.observe_shared_memory(shared)
            exported = ctypes.c_char.from_buffer(shared.buf)
            with pytest.raises(BufferError):
                shared.close()
            mappings.refresh_mappings()
            assert len(snapshot(session.context)["mappings"]) == 1
            del exported
            shared.close()
            mappings.refresh_mappings()
            assert snapshot(session.context)["mappings"] == []
    finally:
        shared.close()
        shared.unlink()


def test_tiff_path_replacement_does_not_reidentify_an_existing_mapping(tmp_path):
    path = tmp_path / "data.tif"
    moved = tmp_path / "original.tif"
    tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint8))
    array = mappings.tracked_tiff_memmap(path, mode="r")
    original = path.stat()
    # Renaming an open file is supported on POSIX; Windows file-sharing rules
    # may prevent this particular lifecycle.
    if os.name == "nt":
        array._mmap.close()
        pytest.skip("Windows may deny replacing an open memory-mapped file")
    path.rename(moved)
    tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint8))
    assert path.stat().st_ino != original.st_ino
    try:
        with mappings.MappingSession() as session:
            record, = snapshot(session.context)["mappings"]
            assert record["identity"] == ["file", original.st_dev, original.st_ino]
            assert record["name"] == str(path)
    finally:
        array._mmap.close()


def test_manifest_write_failure_is_visible_without_breaking_allocation(monkeypatch):
    shared = SharedMemory(create=True, size=4096)
    try:
        with mappings.MappingSession() as session:
            replace = mappings.os.replace

            def denied(*args, **kwargs):
                raise PermissionError("test observation destination unavailable")

            monkeypatch.setattr(mappings.os, "replace", denied)
            array = SharedNPArray(shared.name, ImgSlice(2, 4), np.uint8)
            try:
                assert session.context.failure_flag.value == 1
                assert list(Path(session.context.directory).glob("*.tmp")) == []
                with array as data:
                    data[0, 0] = 9
                    assert shared.buf[0] == 9
                monkeypatch.setattr(mappings.os, "replace", replace)
                mappings.refresh_mappings()
                observed = snapshot(session.context)
                assert len(observed["mappings"]) == 1
                assert observed["errors"] == ["PermissionError: test observation destination unavailable"]
            finally:
                array.shutdown()
    finally:
        shared.close()
        shared.unlink()


def test_metadata_failure_preserves_original_tiff_mapping_and_reports_gap(tmp_path, monkeypatch):
    path = tmp_path / "data.tif"
    tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint8))
    array = tifffile.memmap(path, mode="r")
    monkeypatch.setattr(tifffile, "memmap", lambda *args, **kwargs: array)

    def denied(*args):
        raise OSError("test metadata unavailable")

    monkeypatch.setattr(mappings, "_describe", denied)
    with mappings.MappingSession() as session:
        assert mappings.tracked_tiff_memmap(path, mode="r") is array
        observed = snapshot(session.context)
        assert observed["mappings"] == []
        assert observed["errors"] == ["OSError: test metadata unavailable"]
        assert session.context.failure_flag.value == 1
    array._mmap.close()


def test_inactive_file_identity_failure_is_reported_by_later_session(tmp_path, monkeypatch):
    path = tmp_path / "data.tif"
    tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint8))
    array = tifffile.memmap(path, mode="r")
    monkeypatch.setattr(tifffile, "memmap", lambda *args, **kwargs: array)

    def denied(*args, **kwargs):
        raise OSError("test original backing identity unavailable")

    with monkeypatch.context() as unavailable:
        unavailable.setattr(mappings.os, "stat", denied)
        assert mappings.tracked_tiff_memmap(path, mode="r") is array
    with mappings.MappingSession() as session:
        observed = snapshot(session.context)
        assert observed["mappings"] == []
        assert observed["errors"] == ["OSError: test original backing identity unavailable"]
        assert session.context.failure_flag.value == 1
    array._mmap.close()

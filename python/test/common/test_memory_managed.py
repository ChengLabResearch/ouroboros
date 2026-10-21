import json
import mmap
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import psutil
import pytest

from ouroboros.common import memory_managed as managed
from ouroboros.common.memory_ipc import process_from_identity
from ouroboros.helpers.mem import SharedNPArray, SharedNPManager, _OwnedSharedMemoryTracker
from ouroboros.helpers.shapes import ImgSlice


def snapshot(context, pid):
    return json.loads(Path(context.directory, f"{pid}.json").read_text())


def _check_initializer(context, marker):
    active = managed.current_context()
    assert active.directory == context.directory
    assert active.failure_flag.value == 0
    Path(marker).write_text("caller initializer ran")


def _check_fork_session(connection):
    inherited = managed.current_context()
    with managed.ManagerSession() as session:
        connection.send((inherited is None, session.context.directory))
    connection.close()


def test_inactive_observation_does_not_read_allocations_or_create_files(monkeypatch):
    assert managed.current_context() is None

    class UnreadableSegments:
        def values(self):
            pytest.fail("inactive observer inspected allocations")

    def unexpected(*args, **kwargs):
        pytest.fail("inactive observer created observation resources")

    monkeypatch.setattr(managed.tempfile, "mkdtemp", unexpected)
    monkeypatch.setattr(managed.psutil, "Process", unexpected)
    managed.publish_owned_segments(UnreadableSegments())


def test_overlapping_sessions_share_channel_until_last_release():
    first = managed.ManagerSession().acquire()
    second = managed.ManagerSession().acquire()
    try:
        assert first.acquire() is first
        assert first.context is second.context
        directory = Path(first.context.directory)
        first.release()
        first.release()
        assert directory.is_dir()
        assert managed.current_context() is second.context
    finally:
        first.release()
        second.release()
    assert not directory.exists()
    assert managed.current_context() is None


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="Requires fork")
def test_fork_drops_inherited_channel_without_removing_parent_session():
    context = multiprocessing.get_context("fork")
    with managed.ManagerSession() as session:
        reader, writer = context.Pipe(duplex=False)
        child = context.Process(target=_check_fork_session, args=(writer,))
        try:
            child.start()
            writer.close()
            assert reader.poll(10)
            reset, child_directory = reader.recv()
            child.join(10)
            assert child.exitcode == 0
            assert reset
            assert child_directory != session.context.directory
            assert not Path(child_directory).exists()
            assert Path(session.context.directory).is_dir()
            assert managed.current_context() is session.context
        finally:
            if child.is_alive():
                child.terminate()
                child.join(10)
            reader.close()
            writer.close()


def test_owner_snapshot_follows_allocations_not_attachments(tmp_path):
    with managed.ManagerSession() as session:
        manager = SharedNPManager(ctx=multiprocessing.get_context("spawn"))
        marker = tmp_path / "initializer.txt"
        manager.start(_check_initializer, (session.context, str(marker)))
        arrays = []
        shared_list = None
        try:
            pid = manager._process.pid
            assert marker.read_text() == "caller initializer ran"
            initial = snapshot(session.context, pid)
            assert initial["segments"] == []
            assert process_from_identity(initial["owner"]) == psutil.Process(pid)
            assert "pid" not in initial and "create_time" not in initial

            array = manager.TermedNPArray(ImgSlice(2, 4), np.uint8)
            arrays.append(array)
            name = array.name
            before = snapshot(session.context, pid)
            record, = before["segments"]
            assert record["name"] == name
            assert record["size"] >= 8
            assert set(record) == ({"name", "size", "address"} if sys.platform == "darwin"
                                   else {"name", "size"})
            assert manager._request("list_segments") == [name]
            # Creating views and serializing attachments do not become another
            # source of allocation state or update the owner's observation.
            arrays.extend([array[1:], pickle.loads(pickle.dumps(array))])
            with arrays[-1] as data:
                data[0, 0] = 17
            assert snapshot(session.context, pid) == before
            for attached in arrays:
                attached.shutdown()
            attachment = SharedMemory(name)
            try:
                assert attachment.buf[0] == 17
            finally:
                attachment.close()
            assert snapshot(session.context, pid) == before
            manager.remove_termed(name)
            after = snapshot(session.context, pid)
            assert after["segments"] == []
            assert after["generation"] > before["generation"]

            # Inherited ShareableList uses the same owner as scientific arrays.
            shared_list = manager.ShareableList([1, 2])
            retained_name = shared_list.shm.name
            shared_list.shm.close()
            listed, = snapshot(session.context, pid)["segments"]
            assert listed["name"] == retained_name
            assert manager._request("list_segments") == [retained_name]
        finally:
            for attached in arrays:
                attached.shutdown()
            if shared_list is not None:
                shared_list.shm.close()
            manager.shutdown()
        assert snapshot(session.context, pid)["segments"] == []
        assert session.context.failure_flag.value == 0
        assert {path.stem for path in Path(session.context.directory).glob("*.json")} == {str(pid)}
        with pytest.raises(FileNotFoundError):
            SharedMemory(retained_name)


@pytest.mark.parametrize("failure", ["observation_file_io", "export_process_identity"])
def test_observation_failure_does_not_break_owner_allocation(monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise PermissionError("test observation destination unavailable")

    with managed.ManagerSession() as session:
        monkeypatch.setattr(managed, failure, unavailable)
        owner = _OwnedSharedMemoryTracker()
        try:
            name = owner.allocate(64, termed=True)
            assert session.context.failure_flag.value == 1
            attachment = SharedMemory(name)
            try:
                attachment.buf[0] = 23
                assert attachment.buf[0] == 23
            finally:
                attachment.close()
            assert owner.segment_names == [name]
            owner.destroy_segment(name, termed_only=True)
            assert owner.segment_names == []
            assert list(Path(session.context.directory).iterdir()) == []
        finally:
            owner.unlink()
        with pytest.raises(FileNotFoundError):
            SharedMemory(name)


def test_allocate_release_cycle_changes_snapshot_even_with_a_coarse_clock(monkeypatch):
    monkeypatch.setattr(time, "monotonic_ns", lambda: 123)
    with managed.ManagerSession() as session:
        owner = _OwnedSharedMemoryTracker()
        try:
            before = snapshot(session.context, os.getpid())
            name = owner.allocate(64)
            owner.destroy_segment(name)
            after = snapshot(session.context, os.getpid())
            assert before["segments"] == after["segments"] == []
            assert before["generation"] != after["generation"]
        finally:
            owner.unlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX backing residency")
def test_residency_does_not_fault_pages_or_release_manager_owned_memory():
    with managed.ManagerSession() as session:
        manager = SharedNPManager(ctx=multiprocessing.get_context("spawn"))
        manager.start()
        producer = None
        try:
            producer = manager.SharedMemory(4 * mmap.PAGESIZE)
            segment, = snapshot(session.context, manager._process.pid)["segments"]
            # Two reads must leave newly allocated pages untouched.
            assert managed.resident_bytes(segment) == 0
            assert managed.resident_bytes(segment) == 0
            producer.buf[0] = 1
            producer.buf[2 * mmap.PAGESIZE] = 2
            assert managed.resident_bytes(segment) == 2 * mmap.PAGESIZE
            producer.close()
            producer = None
            assert managed.resident_bytes(segment) == 2 * mmap.PAGESIZE
            assert manager._request("list_segments") == [segment["name"]]
            assert manager._request("release_segment", segment["name"])
            with pytest.raises(FileNotFoundError):
                managed.resident_bytes(segment)
        finally:
            if producer is not None:
                producer.close()
            manager.shutdown()

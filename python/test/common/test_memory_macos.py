import ctypes
import errno
import multiprocessing
import os
import sys
from types import SimpleNamespace

import pytest

from ouroboros.common import memory_macos


PAGE = 4096


def region(address, pages=1, *, mode=2, private=0, shared=0, refs=1):
    return memory_macos._ProcRegionInfo(
        address=address, size=pages * PAGE, share_mode=mode,
        private_pages_resident=private, shared_pages_resident=shared,
        ref_count=refs,
    )


def region_reader(monkeypatch, regions, *, terminal_error=errno.EINVAL):
    pending = iter(regions)

    def proc_pidinfo(pid, flavor, address, output, size):
        assert pid == 123 and flavor == 7
        item = next(pending, None)
        if item is None:
            ctypes.set_errno(terminal_error)
            return 0
        ctypes.memmove(output, ctypes.byref(item), size)
        return size

    native = SimpleNamespace(
        page_size=PAGE, shared_region=(0, 0), proc_pidinfo=proc_pidinfo,
    )
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    return native


def test_region_abi_matches_darwin_header():
    assert ctypes.sizeof(memory_macos._ProcRegionInfo) == 96
    assert memory_macos._ProcRegionInfo.offset.offset == 16
    assert memory_macos._ProcRegionInfo.private_pages_resident.offset == 64
    assert memory_macos._ProcRegionInfo.address.offset == 80


def test_private_counts_exclude_shared_and_manager_owned_pages(monkeypatch):
    region_reader(monkeypatch, [
        region(PAGE, 4, private=2, shared=1),
        region(5 * PAGE, 4, mode=1, private=1, shared=2, refs=2),
        region(9 * PAGE, 4, mode=1, private=1, shared=2, refs=1),
        region(13 * PAGE, 2, mode=4, shared=2),
        region(15 * PAGE, 2, private=2),
        region(17 * PAGE, 2, mode=8, private=2),
    ])
    assert memory_macos.private_resident_bytes(123, [
        {"address": 15 * PAGE, "size": 2 * PAGE},
    ]) == 9 * PAGE


def test_adjacent_owner_ranges_cover_a_merged_region(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, 2, private=2)])
    assert memory_macos.private_resident_bytes(123, [
        {"address": PAGE, "size": PAGE},
        {"address": 2 * PAGE, "size": PAGE},
    ]) == 0


def test_partial_private_region_coverage_is_not_guessed(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, 2, private=1)])
    with pytest.raises(OSError, match="partially overlaps"):
        memory_macos.private_resident_bytes(123, [{"address": PAGE, "size": PAGE}])


@pytest.mark.parametrize("regions", [[], [region(PAGE, private=1)]])
def test_permission_errors_do_not_become_zero_or_partial_samples(monkeypatch, regions):
    region_reader(monkeypatch, regions, terminal_error=errno.EPERM)
    with pytest.raises(PermissionError):
        memory_macos.private_resident_bytes(123)


def test_region_walk_rejects_nonprogressing_results(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, private=1), region(PAGE, private=1)])
    with pytest.raises(OSError, match="changed during"):
        memory_macos.private_resident_bytes(123)


def test_truncated_region_reply_is_rejected(monkeypatch):
    native = region_reader(monkeypatch, [])
    native.proc_pidinfo = lambda *args: ctypes.sizeof(memory_macos._ProcRegionInfo) - 1
    with pytest.raises(OSError, match="Truncated"):
        memory_macos.private_resident_bytes(123)


def test_intel_shared_cache_exclusion_preserves_private_pages(monkeypatch):
    native = region_reader(monkeypatch, [
        region(PAGE, mode=1, private=1), region(2 * PAGE, private=1),
    ])
    native.shared_region = (PAGE, 3 * PAGE)
    assert memory_macos.private_resident_bytes(123) == PAGE


@pytest.mark.parametrize("mode", [3, 4, 5, 6, 7])
def test_shared_modes_are_excluded_without_worker_mapping_tracking(monkeypatch, mode):
    region_reader(monkeypatch, [region(PAGE, 8, mode=mode, private=2, shared=6)])
    assert memory_macos.private_resident_bytes(123) == 0


def test_sole_manager_mapping_is_excluded_when_libproc_calls_it_private(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, 8, private=8), region(9 * PAGE, private=1)])
    assert memory_macos.private_resident_bytes(123, [
        {"name": "owned-buffer", "address": PAGE, "size": 8 * PAGE - 17},
    ]) == PAGE


@pytest.mark.parametrize("address,size", [(-1, PAGE), (PAGE, -1)])
def test_invalid_owner_ranges_are_rejected(monkeypatch, address, size):
    region_reader(monkeypatch, [])
    with pytest.raises(ValueError, match="nonnegative"):
        memory_macos.private_resident_bytes(123, [{"address": address, "size": size}])


def _native_shared_child(connection, name):
    from multiprocessing.shared_memory import SharedMemory

    private = bytearray(8 * 1024 * 1024)
    private[::4096] = b"x" * (len(private) // 4096)
    connection.send("ready")
    if connection.recv() != "attach":
        connection.close()
        return
    shared = SharedMemory(name=name)
    try:
        for offset in range(0, shared.size, os.sysconf("SC_PAGE_SIZE")):
            shared.buf[offset] = 1
        connection.send("attached")
        connection.recv()
    finally:
        shared.close()
        connection.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Native macOS smoke test")
def test_native_same_user_child_private_and_manager_shared_memory():
    from multiprocessing.shared_memory import SharedMemory

    # The parent retains the owning view, as SharedNPManager does in production.
    # A child's second mapping must not add the shared allocation to its private
    # count; the collector accounts for that allocation through the owner.
    size = 16 * 1024 * 1024
    shm = SharedMemory(create=True, size=size)
    context = multiprocessing.get_context("spawn")
    connection, child_connection = context.Pipe()
    child = context.Process(target=_native_shared_child, args=(child_connection, shm.name))
    try:
        child.start()
        child_connection.close()
        assert connection.poll(10) and connection.recv() == "ready"
        before = memory_macos.private_resident_bytes(child.pid)
        assert before >= 8 * 1024 * 1024
        connection.send("attach")
        assert connection.poll(10) and connection.recv() == "attached"
        after = memory_macos.private_resident_bytes(child.pid)
        # Allow small interpreter bookkeeping growth, but not the 16 MiB
        # shared allocation. This is a correctness check, not a benchmark.
        assert abs(after - before) < size // 4
    finally:
        if child.is_alive():
            connection.send("exit")
            child.join(10)
            if child.is_alive():
                child.terminate()
                child.join(10)
        connection.close()
        child_connection.close()
        shm.close()
        shm.unlink()

import ctypes
from concurrent.futures import ThreadPoolExecutor
import json
import mmap
from multiprocessing.shared_memory import SharedMemory
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.common import memory_windows as windows


PAGE = windows.PAGE_SIZE


def page(address, *, shared=False, count=0):
    return (address // PAGE << 12) | (int(shared) << 8) | (count << 5)


def mapping(address, pages, *, offset=0, identity=("shm", "buffer")):
    return {"address": address, "size": pages * PAGE, "offset": offset,
            "identity": list(identity), "kind": identity[0], "name": str(identity[-1])}


class NativeAPI:
    """Exercise the production native-buffer boundary with scripted OS results."""

    def __init__(self, entries=(), *, failures=(), open_error=0, close_error=0,
                 working_set_bytes=0, memory_info_error=0):
        self.entries = np.asarray(entries, dtype=np.uint64)
        self.failures = iter(failures)
        self.open_error = open_error
        self.close_error = close_error
        self.working_set_bytes = working_set_bytes
        self.memory_info_error = memory_info_error
        self.memory_info_calls = []
        self.error = 0
        self.opened = []
        self.closed = []
        self.capacities = []

    def OpenProcess(self, access, inherit, pid):
        self.opened.append((access, inherit, pid))
        self.error = self.open_error
        return 0 if self.open_error else 123

    def QueryWorkingSet(self, handle, pointer, size):
        assert handle == 123
        raw = (ctypes.c_uint64 * (size // 8)).from_address(pointer.value)
        buffer = np.ctypeslib.as_array(raw)
        self.capacities.append(len(buffer) - 1)
        failure = next(self.failures, None)
        if failure is not None:
            self.error, required = failure
            buffer[0] = required
            return 0
        buffer[0] = self.entries.size
        buffer[1:self.entries.size + 1] = self.entries
        return 1

    def GetProcessMemoryInfo(self, handle, pointer, size):
        counters = ctypes.cast(pointer, ctypes.POINTER(windows._ProcessMemoryCounters)).contents
        assert counters.cb == size == ctypes.sizeof(counters)
        self.memory_info_calls.append(handle)
        self.error = self.memory_info_error
        counters.WorkingSetSize = self.working_set_bytes
        return not self.memory_info_error

    def CloseHandle(self, handle):
        self.closed.append(handle)
        self.error = self.close_error
        return not self.close_error

    def last_error(self):
        return self.error


def install(monkeypatch, api):
    monkeypatch.setattr(windows, "_windows_api", lambda: api)
    return api


def test_unique_and_registered_pages_are_partitioned_from_one_snapshot(monkeypatch):
    base = 0x10000
    api = install(monkeypatch, NativeAPI([
        page(base, shared=True, count=1),      # Single-user buffer page: union, not USS.
        page(base + PAGE, shared=True, count=7),
        page(base + 2 * PAGE),               # A private COW copy in the mapped range.
        page(base + 4 * PAGE),               # Private heap outside registered mappings.
        page(base + 5 * PAGE, shared=True, count=1),
        page(base + 6 * PAGE, shared=True, count=2),  # Untracked shared library omitted.
    ]))
    private, buffers = windows.read_process(42, [mapping(base, 4, offset=8 * PAGE)])
    assert private == 3 * PAGE
    np.testing.assert_array_equal(buffers[("shm", "buffer")], [8 * PAGE, 9 * PAGE])
    assert buffers[("shm", "buffer")].dtype == np.uint64
    assert api.opened == [(0x0410, False, 42)]
    assert len(api.capacities) == 1
    assert api.closed == [123]


def test_backing_offsets_deduplicate_views_and_more_than_seven_processes(monkeypatch):
    observed = []
    for worker in range(9):
        base = 0x10000 + worker * 0x10000
        alias = base + 8 * PAGE
        install(monkeypatch, NativeAPI([
            page(base + PAGE, shared=True, count=7),
            page(base, shared=True, count=7),
            page(alias, shared=True, count=7),
        ]))
        private, buffers = windows.read_process(worker + 1, [
            mapping(base, 2, offset=4 * PAGE),
            mapping(alias, 1, offset=4 * PAGE),
        ])
        assert private == 0
        np.testing.assert_array_equal(buffers[("shm", "buffer")], [4 * PAGE, 5 * PAGE])
        observed.append(buffers[("shm", "buffer")])
    assert np.unique(np.concatenate(observed)).size * PAGE == 2 * PAGE


def test_nonresident_registered_pages_do_not_become_allocated_size(monkeypatch):
    base = 0x10000
    install(monkeypatch, NativeAPI([page(base + PAGE, shared=True, count=2)]))
    private, buffers = windows.read_process(42, [mapping(base, 100), mapping(base + 200 * PAGE, 5,
                                                           identity=("file", 7, 19))])
    assert private == 0
    np.testing.assert_array_equal(buffers[("shm", "buffer")], [PAGE])
    assert buffers[("file", 7, 19)].size == 0


def test_overlapping_observations_of_same_mapping_do_not_duplicate_private_pages(monkeypatch):
    base = 0x10000
    install(monkeypatch, NativeAPI([page(base), page(base + 3 * PAGE, shared=True, count=1)]))
    private, buffers = windows.read_process(42, [
        mapping(base, 4), mapping(base, 2), mapping(base + PAGE, 3, offset=PAGE),
    ])
    assert private == PAGE
    np.testing.assert_array_equal(buffers[("shm", "buffer")], [3 * PAGE])


@pytest.mark.parametrize("observations", [
    [mapping(0x10000, 3), mapping(0x10000 + PAGE, 1, identity=("shm", "other"))],
    [mapping(0x10000, 3), mapping(0x10000 + PAGE, 1, offset=0)],
    [mapping(0x10001, 1)],
    [mapping(0x10000, 1, offset=1)],
    [mapping(0x10000, 0)],
    [mapping(0x10000, 1, offset=(1 << 64) - PAGE + PAGE)],
])
def test_invalid_mapping_provenance_fails_before_reading_process(monkeypatch, observations):
    api = install(monkeypatch, NativeAPI())
    with pytest.raises(ValueError):
        windows.read_process(42, observations)
    assert api.opened == []


@pytest.mark.parametrize("pid", [0, -1, 1 << 32, 1.5])
def test_invalid_pid_is_not_truncated_or_wrapped(monkeypatch, pid):
    api = install(monkeypatch, NativeAPI())
    with pytest.raises((ValueError, TypeError)):
        windows.read_process(pid, [])
    assert api.opened == []


def test_native_buffer_grows_to_reported_requirement_then_reads_complete_snapshot(monkeypatch):
    monkeypatch.setattr(windows, "_INITIAL_ENTRIES", 1)
    api = install(monkeypatch, NativeAPI([page(0x10000 + i * PAGE) for i in range(5)],
                                        failures=[(24, 5)]))
    assert windows.read_process(42, []) == (5 * PAGE, {})
    assert api.capacities == [1, 5]
    assert api.closed == [123]


def test_current_working_set_sizes_one_page_query_but_does_not_supply_metric(monkeypatch):
    entries = [page(0x10000 + i * PAGE) for i in range(5000)]
    api = install(monkeypatch, NativeAPI(entries, working_set_bytes=10000 * PAGE))
    assert windows.read_process(42, []) == (5000 * PAGE, {})
    assert api.memory_info_calls == [123]
    assert len(api.capacities) == 1
    assert api.capacities[0] > 10000
    assert api.closed == [123]


def test_capacity_hint_headroom_stays_within_native_buffer_bound(monkeypatch):
    monkeypatch.setattr(windows, "_MAX_BUFFER_BYTES", 64)
    api = install(monkeypatch, NativeAPI([page(0x10000)], working_set_bytes=1000 * PAGE))
    assert windows.read_process(42, []) == (PAGE, {})
    assert api.capacities == [7]


def test_capacity_hint_failure_is_explicit_and_releases_handle(monkeypatch):
    api = install(monkeypatch, NativeAPI(memory_info_error=5))
    with pytest.raises(OSError, match="GetProcessMemoryInfo") as caught:
        windows.read_process(42, [])
    assert caught.value.errno == 5
    assert api.capacities == []
    assert api.closed == [123]


@pytest.mark.parametrize("error", [5, 87, 299])
def test_permission_disappearance_and_partial_copy_failures_are_not_zero_samples(monkeypatch, error):
    api = install(monkeypatch, NativeAPI(failures=[(error, 0)]))
    with pytest.raises(OSError, match="QueryWorkingSet") as caught:
        windows.read_process(42, [])
    assert caught.value.errno == error
    assert api.closed == [123]


def test_open_failure_does_not_close_invalid_handle(monkeypatch):
    api = install(monkeypatch, NativeAPI(open_error=5))
    with pytest.raises(OSError, match="OpenProcess"):
        windows.read_process(42, [])
    assert api.closed == []


def test_collection_and_cleanup_errors_preserve_original_collection_failure(monkeypatch):
    api = install(monkeypatch, NativeAPI(failures=[(299, 0)], close_error=6))
    with pytest.raises(OSError, match="QueryWorkingSet") as caught:
        windows.read_process(42, [])
    assert caught.value.errno == 299
    assert api.closed == [123]


def test_close_failure_after_success_is_explicit(monkeypatch):
    install(monkeypatch, NativeAPI(close_error=6))
    with pytest.raises(OSError, match="CloseHandle"):
        windows.read_process(42, [])


def test_growth_and_churn_have_explicit_bounds_and_release_handle(monkeypatch):
    monkeypatch.setattr(windows, "_INITIAL_ENTRIES", 1)
    monkeypatch.setattr(windows, "_MAX_BUFFER_BYTES", 64)
    api = install(monkeypatch, NativeAPI(failures=[(24, 100)]))
    with pytest.raises(OSError, match="buffer limit"):
        windows.read_process(42, [])
    assert api.closed == [123]

    monkeypatch.setattr(windows, "_MAX_QUERY_ATTEMPTS", 2)
    api = install(monkeypatch, NativeAPI(failures=[(24, 2), (24, 3)]))
    with pytest.raises(OSError, match="bounded collection retries"):
        windows.read_process(42, [])
    assert len(api.capacities) == 2
    assert api.closed == [123]


def test_success_with_truncated_native_payload_is_rejected(monkeypatch):
    api = NativeAPI()

    def incomplete(handle, pointer, size):
        ctypes.c_uint64.from_address(pointer.value).value = size // 8
        return 1

    api.QueryWorkingSet = incomplete
    install(monkeypatch, api)
    with pytest.raises(OSError, match="incomplete working-set"):
        windows.read_process(42, [])
    assert api.closed == [123]


def test_import_is_portable_but_unsupported_native_execution_is_explicit(monkeypatch):
    windows._windows_api.cache_clear()
    monkeypatch.setattr(windows, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(NotImplementedError, match="requires Windows"):
        windows._windows_api()
    monkeypatch.setattr(windows, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(windows, "_POINTER_BYTES", 4)
    with pytest.raises(NotImplementedError, match="64-bit"):
        windows._windows_api()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows API smoke test")
def test_native_windows_shared_buffer_union_uses_actual_worker_mapping():
    child = None
    shared = SharedMemory(create=True, size=4 * PAGE)
    with ThreadPoolExecutor(max_workers=1) as reader:
        try:
            shared.buf[::PAGE] = b"x" * 4
            address = ctypes.addressof(ctypes.c_char.from_buffer(shared.buf))
            child = subprocess.Popen([
                sys.executable, "-B", "-c",
                "import ctypes,json,sys; from multiprocessing.shared_memory import SharedMemory; "
                "s=SharedMemory(name=sys.argv[1]); s.buf[::int(sys.argv[2])]=b'y'*4; "
                "print(json.dumps({'address':ctypes.addressof(ctypes.c_char.from_buffer(s.buf)),"
                "'size':len(s.buf)}),flush=True); sys.stdin.readline(); s.close()",
                shared.name, str(PAGE),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            observed = json.loads(reader.submit(child.stdout.readline).result(timeout=10))
            registered = mapping(address, 4, identity=("shm", shared.name))
            _, parent_pages = windows.read_process(os.getpid(), [registered])
            registered.update(observed)
            _, child_pages = windows.read_process(child.pid, [registered])
            identity = ("shm", shared.name)
            expected = np.arange(4, dtype=np.uint64) * PAGE
            np.testing.assert_array_equal(parent_pages[identity], expected)
            np.testing.assert_array_equal(child_pages[identity], expected)
            np.testing.assert_array_equal(np.union1d(parent_pages[identity], child_pages[identity]), expected)
        finally:
            if child is not None:
                try:
                    child.communicate("exit\n", timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.communicate(timeout=5)
            shared.close()
            shared.unlink()


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows API smoke test")
def test_native_windows_copy_on_write_is_private_not_shared(tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(b"x" * (4 * PAGE))
    with path.open("r+b") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY) as view:
        assert sum(view[offset] for offset in range(0, 4 * PAGE, PAGE)) == 4 * ord("x")
        view[0] = ord("y")
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        registered = mapping(address, 4, identity=("file", path.stat().st_dev, path.stat().st_ino))
        private, buffers = windows.read_process(os.getpid(), [registered])
        assert private >= PAGE
        np.testing.assert_array_equal(buffers[tuple(registered["identity"])],
                                      np.arange(1, 4, dtype=np.uint64) * PAGE)

import ctypes
import mmap
from multiprocessing.shared_memory import SharedMemory
import os
import sys
from types import SimpleNamespace

import pytest

from ouroboros.common import memory_windows as windows


class NativeAPI:
    """Exercise the production EX2 boundary with scripted OS results."""

    def __init__(self, private_bytes=123456, *, omitted=(), open_error=0,
                 close_error=0, memory_info_error=0):
        self.private_bytes = private_bytes
        self.omitted = omitted
        self.open_error = open_error
        self.close_error = close_error
        self.memory_info_error = memory_info_error
        self.memory_info_calls = []
        self.error = 0
        self.opened = []
        self.closed = []

    def OpenProcess(self, access, inherit, pid):
        self.opened.append((access, inherit, pid))
        self.error = self.open_error
        return 0 if self.open_error else 123

    def GetProcessMemoryInfo(self, handle, pointer, size):
        counters = ctypes.cast(pointer, ctypes.POINTER(windows._ProcessMemoryCountersEX2)).contents
        assert counters.cb == size == ctypes.sizeof(counters)
        self.memory_info_calls.append(handle)
        self.error = self.memory_info_error
        # Give neighboring fields distinct values: none may supply this metric.
        counters.WorkingSetSize = 100000000
        counters.PrivateUsage = 200000000
        if "PrivateWorkingSetSize" not in self.omitted:
            counters.PrivateWorkingSetSize = self.private_bytes
        if "SharedCommitUsage" not in self.omitted:
            counters.SharedCommitUsage = 300000000
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


def test_ex2_abi_uses_windows_size_t_and_ulong64_fields():
    if ctypes.sizeof(ctypes.c_void_p) == 8:
        assert ctypes.sizeof(windows._ProcessMemoryCountersEX2) == 96
        assert windows._ProcessMemoryCountersEX2.PrivateWorkingSetSize.offset == 80
        assert windows._ProcessMemoryCountersEX2.SharedCommitUsage.offset == 88


@pytest.mark.parametrize("private_bytes", [0, 123456, 1 << 34])
def test_private_residency_comes_from_one_ex2_call(monkeypatch, private_bytes):
    api = install(monkeypatch, NativeAPI(private_bytes))
    assert windows.private_resident_bytes(42) == private_bytes
    assert api.opened == [(0x0410, False, 42)]
    assert api.memory_info_calls == [123]
    assert api.closed == [123]


@pytest.mark.parametrize("omitted", [
    ("PrivateWorkingSetSize",), ("SharedCommitUsage",),
    ("PrivateWorkingSetSize", "SharedCommitUsage"),
])
def test_older_api_cannot_silently_produce_zero_or_another_metric(monkeypatch, omitted):
    api = install(monkeypatch, NativeAPI(omitted=omitted))
    with pytest.raises(NotImplementedError, match="EX2 fields"):
        windows.private_resident_bytes(42)
    assert api.closed == [123]


@pytest.mark.parametrize("pid", [0, -1, 1 << 32, 1.5])
def test_invalid_pid_is_not_truncated_or_wrapped(monkeypatch, pid):
    api = install(monkeypatch, NativeAPI())
    with pytest.raises((ValueError, TypeError)):
        windows.private_resident_bytes(pid)
    assert api.opened == []


@pytest.mark.parametrize("error", [5, 87, 299])
def test_permission_disappearance_and_partial_copy_failures_are_not_zero_samples(monkeypatch, error):
    api = install(monkeypatch, NativeAPI(memory_info_error=error))
    with pytest.raises(OSError, match="GetProcessMemoryInfo") as caught:
        windows.private_resident_bytes(42)
    assert caught.value.errno == error
    assert api.closed == [123]


def test_open_failure_does_not_close_invalid_handle(monkeypatch):
    api = install(monkeypatch, NativeAPI(open_error=5))
    with pytest.raises(OSError, match="OpenProcess"):
        windows.private_resident_bytes(42)
    assert api.memory_info_calls == []
    assert api.closed == []


def test_collection_and_cleanup_errors_preserve_original_collection_failure(monkeypatch):
    api = install(monkeypatch, NativeAPI(memory_info_error=299, close_error=6))
    with pytest.raises(OSError, match="GetProcessMemoryInfo") as caught:
        windows.private_resident_bytes(42)
    assert caught.value.errno == 299
    assert api.closed == [123]


def test_close_failure_after_success_is_explicit(monkeypatch):
    install(monkeypatch, NativeAPI(close_error=6))
    with pytest.raises(OSError, match="CloseHandle"):
        windows.private_resident_bytes(42)


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
def test_native_private_counter_counts_touched_heap_but_excludes_shared_buffer():
    size = 8 * 1024 * 1024
    # Allocate the handle first so initialization does not contaminate the delta.
    shared = SharedMemory(create=True, size=size)
    try:
        baseline = windows.private_resident_bytes(os.getpid())
        private = bytearray(size)
        private[::mmap.PAGESIZE] = b"p" * (size // mmap.PAGESIZE)
        private_delta = windows.private_resident_bytes(os.getpid()) - baseline
        assert private_delta >= size - 1024 * 1024

        baseline = windows.private_resident_bytes(os.getpid())
        shared.buf[::mmap.PAGESIZE] = b"s" * (size // mmap.PAGESIZE)
        shared_delta = windows.private_resident_bytes(os.getpid()) - baseline
        assert shared_delta < 1024 * 1024
    finally:
        shared.close()
        shared.unlink()

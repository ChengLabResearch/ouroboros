"""Windows page-accounting adapter for the operation memory collector.

The caller owns process and mapping lifetimes. This module only reads a process's
current working set and translates registered mapping addresses into backing
allocation offsets. The collector unions those offsets across processes.
"""

import ctypes
from functools import lru_cache
import mmap
from operator import index
import sys

import numpy as np


PAGE_SIZE = mmap.PAGESIZE
_POINTER_BYTES = ctypes.sizeof(ctypes.c_void_p)
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_ERROR_BAD_LENGTH = 24
_INITIAL_ENTRIES = 4096
_MAX_QUERY_ATTEMPTS = 8
# Match psutil's existing Windows working-set scan ceiling. Exceeding the bound
# is an explicit collection failure; a partial buffer must never become a sample.
_MAX_BUFFER_BYTES = 256 * 1024 * 1024
_SHARED = np.uint64(1 << 8)


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _WindowsAPI:
    def __init__(self):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.OpenProcess = kernel32.OpenProcess
        self.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        self.OpenProcess.restype = ctypes.c_void_p
        self.GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        self.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters),
                                             ctypes.c_uint32]
        self.GetProcessMemoryInfo.restype = ctypes.c_int32
        self.QueryWorkingSet = psapi.QueryWorkingSet
        self.QueryWorkingSet.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
        self.QueryWorkingSet.restype = ctypes.c_int32
        self.CloseHandle = kernel32.CloseHandle
        self.CloseHandle.argtypes = [ctypes.c_void_p]
        self.CloseHandle.restype = ctypes.c_int32
        self.last_error = ctypes.get_last_error


@lru_cache(maxsize=1)
def _windows_api():
    if sys.platform != "win32":
        raise NotImplementedError("Windows working-set collection requires Windows")
    if _POINTER_BYTES != 8:
        raise NotImplementedError("Windows working-set collection requires 64-bit Python")
    return _WindowsAPI()


def _api_error(api, operation):
    code = api.last_error()
    return OSError(code, f"{operation} failed (Windows error {code})")


def _working_set(pid):
    api = _windows_api()
    handle = api.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid)
    if not handle:
        raise _api_error(api, "OpenProcess")
    completed = False
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not api.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise _api_error(api, "GetProcessMemoryInfo")
        pages = (counters.WorkingSetSize + PAGE_SIZE - 1) // PAGE_SIZE
        # RSS is only a disposable capacity hint. Page descriptors from the
        # successful QueryWorkingSet call remain the sole accounting source.
        # Headroom avoids a sizing scan on every sweep, without caching state.
        capacity = min(_MAX_BUFFER_BYTES // 8 - 1,
                       max(_INITIAL_ENTRIES, pages + max(_INITIAL_ENTRIES, pages // 8)))
        for _ in range(_MAX_QUERY_ATTEMPTS):
            if (capacity + 1) * 8 > _MAX_BUFFER_BYTES:
                raise OSError("Windows working-set enumeration exceeds the buffer limit")
            # The native structure starts with ULONG_PTR NumberOfEntries,
            # followed by that many ULONG_PTR page descriptors. NumPy supplies
            # aligned contiguous storage and avoids copying the native snapshot.
            buffer = np.empty(capacity + 1, dtype=np.uint64)
            buffer[0] = 0
            if api.QueryWorkingSet(handle, ctypes.c_void_p(buffer.ctypes.data), buffer.nbytes):
                count = int(buffer[0])
                if count > capacity:
                    raise OSError("Windows returned an incomplete working-set enumeration")
                completed = True
                return buffer[1:count + 1]
            error = _api_error(api, "QueryWorkingSet")
            if error.errno != _ERROR_BAD_LENGTH:
                raise error
            # ERROR_BAD_LENGTH reports the required NumberOfEntries. Growing
            # geometrically also handles a working set that changes during reads.
            capacity = max(capacity * 2, int(buffer[0]))
            del buffer
        raise OSError("Windows working set kept changing during bounded collection retries")
    finally:
        if not api.CloseHandle(handle) and completed:
            raise _api_error(api, "CloseHandle")


def _mapping_ranges(mappings):
    ranges = []
    for mapping in mappings:
        address, size, offset = (index(mapping[key]) for key in ("address", "size", "offset"))
        if (address <= 0 or size <= 0 or offset < 0
                or address % PAGE_SIZE or offset % PAGE_SIZE
                or address + size > 1 << 64 or offset + size > 1 << 64):
            raise ValueError("Windows mapping observations require valid page-aligned addresses and offsets")
        identity = tuple(mapping["identity"])
        hash(identity)
        ranges.append((address, address + size, offset, identity))
    ranges.sort(key=lambda value: value[0])
    merged = []
    for address, end, offset, identity in ranges:
        if merged and address < merged[-1][1]:
            previous_address, previous_end, previous_offset, previous_identity = merged[-1]
            if identity != previous_identity or offset - address != previous_offset - previous_address:
                raise ValueError("Conflicting Windows mapping observations overlap")
            merged[-1] = (previous_address, max(previous_end, end), previous_offset, identity)
        else:
            merged.append((address, end, offset, identity))
    return merged


def read_process(pid: int, mappings: list[dict]) -> tuple[int, dict[tuple, np.ndarray]]:
    """Read unique resident bytes and registered shared backing byte offsets.

    Unique pages outside registered mappings follow psutil's Windows USS rule:
    non-shareable pages, plus shareable pages used by at most one process. Inside
    registered mappings, only non-shareable (including copy-on-write) pages are
    counted privately. Shareable pages instead enter their allocation's union,
    even when their reported share count is one or saturated at seven.

    There is one successful working-set enumeration per process. Errors propagate
    so the owning collector can mark incomplete coverage rather than assume zero.
    """
    pid = index(pid)
    if not 0 < pid < 1 << 32:
        raise ValueError("Windows process identifiers must be positive 32-bit integers")
    ranges = _mapping_ranges(mappings)
    entries = _working_set(pid)
    addresses = (entries >> np.uint64(12)) * np.uint64(PAGE_SIZE)
    shared = (entries & _SHARED) != 0
    share_count = (entries >> np.uint64(5)) & np.uint64(7)
    tracked = np.zeros(entries.size, dtype=bool)
    allocation_pages = {}
    for address, end, offset, identity in ranges:
        within = (addresses >= address) & (addresses < end)
        tracked |= within
        offsets = addresses[within & shared] - np.uint64(address) + np.uint64(offset)
        allocation_pages.setdefault(identity, []).append(offsets)
    unique = (~shared) | ((share_count <= 1) & ~tracked)
    private_bytes = int(np.count_nonzero(unique)) * PAGE_SIZE
    resident = {
        identity: np.unique(parts[0] if len(parts) == 1 else np.concatenate(parts))
        for identity, parts in allocation_pages.items()
    }
    return private_bytes, resident

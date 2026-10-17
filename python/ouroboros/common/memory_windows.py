"""Windows private-resident counter for the operation memory collector.

Shared allocations are accounted for separately by their manager. Reading the
native private working-set counter needs no per-page working-set enumeration.
"""

import ctypes
from functools import lru_cache
from operator import index
import sys


_POINTER_BYTES = ctypes.sizeof(ctypes.c_void_p)
_PROCESS_QUERY_INFORMATION = 0x0400
_PROCESS_VM_READ = 0x0010
_UNWRITTEN = 0xDEADBEEFDEADBEEF


class _ProcessMemoryCountersEX2(ctypes.Structure):
    # https://learn.microsoft.com/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex2
    _fields_ = [
        ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t), ("PrivateWorkingSetSize", ctypes.c_size_t),
        ("SharedCommitUsage", ctypes.c_uint64),
    ]


class _WindowsAPI:
    def __init__(self):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.OpenProcess = kernel32.OpenProcess
        self.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
        self.OpenProcess.restype = ctypes.c_void_p
        self.GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
        self.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCountersEX2),
                                             ctypes.c_uint32]
        self.GetProcessMemoryInfo.restype = ctypes.c_int32
        self.CloseHandle = kernel32.CloseHandle
        self.CloseHandle.argtypes = [ctypes.c_void_p]
        self.CloseHandle.restype = ctypes.c_int32
        self.last_error = ctypes.get_last_error


@lru_cache(maxsize=1)
def _windows_api():
    if sys.platform != "win32":
        raise NotImplementedError("Windows private-resident collection requires Windows")
    if _POINTER_BYTES != 8:
        raise NotImplementedError("Windows private-resident collection requires 64-bit Python")
    return _WindowsAPI()


def _api_error(api, operation):
    code = api.last_error()
    return OSError(code, f"{operation} failed (Windows error {code})")


def private_resident_bytes(pid: int) -> int:
    """Return current private working-set bytes, excluding shared allocations.

    EX2 requires Windows 10/11 22H2 with the September 2023 cumulative update or
    newer. Older systems may accept the larger structure without filling its
    extra fields; reject that result rather than changing the reported metric.
    """
    pid = index(pid)
    if not 0 < pid < 1 << 32:
        raise ValueError("Windows process identifiers must be positive 32-bit integers")
    api = _windows_api()
    handle = api.OpenProcess(_PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid)
    if not handle:
        raise _api_error(api, "OpenProcess")
    completed = False
    try:
        counters = _ProcessMemoryCountersEX2()
        counters.cb = ctypes.sizeof(counters)
        counters.PrivateWorkingSetSize = counters.SharedCommitUsage = _UNWRITTEN
        if not api.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise _api_error(api, "GetProcessMemoryInfo")
        if _UNWRITTEN in (counters.PrivateWorkingSetSize, counters.SharedCommitUsage):
            raise NotImplementedError("Windows did not provide PROCESS_MEMORY_COUNTERS_EX2 fields")
        completed = True
        return int(counters.PrivateWorkingSetSize)
    finally:
        if not api.CloseHandle(handle) and completed:
            raise _api_error(api, "CloseHandle")

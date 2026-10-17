"""macOS resident-memory queries for the common run-memory collector.

Private memory follows psutil's unreleased libproc-based USS implementation:
https://github.com/giampaolo/psutil/blob/f277b9325a80ab51d789adf0cbff2a439f72d72e/psutil/arch/osx/proc.c
Tracked mappings are excluded before the collector adds their resident pages
once by backing identity. The metric is private plus tracked backing residency:
file-cache pages may be resident without belonging to a process's working set.
Untracked shared pages and compressed pages are not included. Copy-on-write
tracked files are unsupported, because their private copies differ from the
backing object. Native handles and mappings live only for a query.

The adapted psutil accounting code is distributed under the BSD 3-Clause License:

Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the psutil authors nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import ctypes
import errno
from functools import lru_cache
import mmap
import os
import platform

import numpy as np


PAGE_SIZE = mmap.PAGESIZE


class _ProcRegionInfo(ctypes.Structure):
    # Darwin <sys/proc_info.h>, PROC_PIDREGIONINFO. This is an ABI boundary,
    # not a second owner of the collector's mapping or allocation state.
    _fields_ = [
        ("protection", ctypes.c_uint32),
        ("max_protection", ctypes.c_uint32),
        ("inheritance", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
        ("behavior", ctypes.c_uint32),
        ("user_wired_count", ctypes.c_uint32),
        ("user_tag", ctypes.c_uint32),
        ("pages_resident", ctypes.c_uint32),
        ("pages_shared_now_private", ctypes.c_uint32),
        ("pages_swapped_out", ctypes.c_uint32),
        ("pages_dirtied", ctypes.c_uint32),
        ("ref_count", ctypes.c_uint32),
        ("shadow_depth", ctypes.c_uint32),
        ("share_mode", ctypes.c_uint32),
        ("private_pages_resident", ctypes.c_uint32),
        ("shared_pages_resident", ctypes.c_uint32),
        ("obj_id", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("address", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
    ]


class _Native:
    def __init__(self):
        # Do not load Darwin libraries at import time: the common collector is
        # imported on every supported platform.
        proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        self.proc_pidinfo = proc.proc_pidinfo
        self.proc_pidinfo.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ctypes.POINTER(_ProcRegionInfo), ctypes.c_int,
        ]
        self.proc_pidinfo.restype = ctypes.c_int
        self.shm_open = libc.shm_open
        self.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
        self.shm_open.restype = ctypes.c_int
        self.mmap = libc.mmap
        self.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int64,
        ]
        self.mmap.restype = ctypes.c_void_p
        self.mincore = libc.mincore
        self.mincore.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.mincore.restype = ctypes.c_int
        self.munmap = libc.munmap
        self.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.munmap.restype = ctypes.c_int
        self.page_size = PAGE_SIZE
        # Match psutil's exclusion of non-private Intel shared-cache regions.
        self.shared_region = (
            (0x00007FF800000000, 0x00007FFFFE000000)
            if platform.machine() == "x86_64" else (0, 0)
        )


@lru_cache(maxsize=1)
def _native():
    return _Native()


def _error(operation):
    code = ctypes.get_errno() or errno.EIO
    return OSError(code, f"{operation}: {os.strerror(code)}")


def _check_shared_mapping(mapping):
    if mapping.get("kind") == "file" and mapping.get("mode") == "c":
        raise OSError(errno.ENOTSUP, "Copy-on-write tracked files are unsupported on macOS")


def private_resident_bytes(pid: int, mappings: list[dict]) -> int:
    """Read private resident memory outside this PID's tracked mappings.

    libproc permits ordinary same-user inspection without a Mach task port.
    Region residency is aggregate: a tracked range must cover an entire region
    before it can be excluded. Partial overlap fails instead of estimating how
    the resident pages are distributed within that region.
    """
    native = _native()
    page_size = native.page_size
    ranges = []
    for mapping in mappings:
        _check_shared_mapping(mapping)
        address, size = int(mapping["address"]), int(mapping["size"])
        if address < 0 or size < 0:
            raise ValueError("Mapping address and size must be nonnegative")
        if size:
            ranges.append((address // page_size * page_size,
                           (address + size + page_size - 1) // page_size * page_size))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))

    address = pages = regions = range_index = 0
    region = _ProcRegionInfo()
    while True:
        ctypes.set_errno(0)
        result = native.proc_pidinfo(pid, 7, address, ctypes.byref(region), ctypes.sizeof(region))
        if result <= 0:
            if regions and ctypes.get_errno() == errno.EINVAL:
                break  # libproc's end-of-address-space result.
            raise _error("proc_pidinfo(PROC_PIDREGIONINFO)")
        if result != ctypes.sizeof(region):
            raise OSError(errno.EIO, "Truncated proc_pidinfo region result")
        end = region.address + region.size
        if region.address < address or end <= address or end >= 1 << 64:
            raise OSError(errno.EAGAIN, "Process mappings changed during region walk")
        regions += 1
        address = end

        if region.share_mode in (2, 8):  # SM_PRIVATE, SM_LARGE_PAGE
            private_pages = region.private_pages_resident + region.shared_pages_resident
        elif region.share_mode == 1:  # SM_COW
            private_pages = region.private_pages_resident
            if region.ref_count == 1:
                private_pages += region.shared_pages_resident
        else:
            continue
        if not private_pages:
            continue
        low, high = native.shared_region
        if region.share_mode != 2 and low <= region.address < high:
            continue

        while range_index < len(merged) and merged[range_index][1] <= region.address:
            range_index += 1
        if range_index < len(merged):
            start, stop = merged[range_index]
            if start < end:
                if start > region.address or stop < end:
                    raise OSError(errno.EAGAIN, "Tracked mapping partially overlaps a VM region")
                continue
        pages += private_pages
    return pages * page_size


def resident_pages(mapping: dict) -> np.ndarray:
    """Return resident backing-page byte offsets for one tracked allocation.

    The temporary read-only mapping queries the backing object's residency with
    mincore; it never reads or locks data pages. The caller deduplicates these
    offsets by its canonical backing identity. Python SharedMemory is deliberately
    not used, because attaching would enlist its resource tracker as another owner.
    """
    _check_shared_mapping(mapping)
    native = _native()
    page_size = native.page_size
    size, offset = int(mapping["size"]), int(mapping["offset"])
    if size < 0 or offset < 0:
        raise ValueError("Mapping size and backing offset must be nonnegative")
    if not size:
        return np.empty(0, dtype=np.int64)
    identity = tuple(mapping["identity"])
    if mapping["kind"] == "shm":
        if identity != ("shm", mapping["name"]):
            raise ValueError("Shared-memory identity does not match its name")
        ctypes.set_errno(0)
        fd = native.shm_open(os.fsencode("/" + mapping["name"].lstrip("/")), os.O_RDONLY, 0)
        if fd < 0:
            raise _error("shm_open")
    elif mapping["kind"] == "file":
        fd = os.open(mapping["name"], os.O_RDONLY)
    else:
        raise ValueError("Unknown tracked mapping kind")
    try:
        stat = os.fstat(fd)
        if mapping["kind"] == "file" and identity != ("file", stat.st_dev, stat.st_ino):
            raise OSError(errno.ESTALE, "Tracked file identity changed")
        start = offset // page_size * page_size
        length = (offset + size + page_size - 1) // page_size * page_size - start
        if start + length > (stat.st_size + page_size - 1) // page_size * page_size:
            raise OSError(errno.EAGAIN, "Tracked backing allocation was truncated")
        ctypes.set_errno(0)
        address = native.mmap(None, length, 1, 1, fd, start)  # PROT_READ, MAP_SHARED
        if address == ctypes.c_void_p(-1).value:
            raise _error("mmap")
        try:
            chunks = []
            # Bound temporary residency-vector storage to 64 KiB.
            for first in range(0, length // page_size, 65536):
                count = min(65536, length // page_size - first)
                flags = np.empty(count, dtype=np.uint8)
                ctypes.set_errno(0)
                result = native.mincore(
                    address + first * page_size, count * page_size,
                    flags.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                )
                if result:
                    raise _error("mincore")
                pages = np.flatnonzero(flags & 1).astype(np.int64, copy=False)
                chunks.append(start + (first + pages) * page_size)
            return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        finally:
            if native.munmap(address, length):
                raise _error("munmap")
    finally:
        os.close(fd)

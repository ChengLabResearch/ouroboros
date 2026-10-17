"""macOS resident-memory queries for the common run-memory collector.

Private memory follows psutil's unreleased libproc-based USS implementation:
https://github.com/giampaolo/psutil/blob/f277b9325a80ab51d789adf0cbff2a439f72d72e/psutil/arch/osx/proc.c
Only ranges supplied by the canonical shared-memory manager are excluded from
its own process count before managed residency is added once. Worker mappings
remain shared while the manager owns its mapping, so they require no tracking.
This reader does not query TIFF backing files or compressed pages.

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
        self.proc_pidinfo = proc.proc_pidinfo
        self.proc_pidinfo.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ctypes.POINTER(_ProcRegionInfo), ctypes.c_int,
        ]
        self.proc_pidinfo.restype = ctypes.c_int
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


def private_resident_bytes(pid: int, owned_segments=()) -> int:
    """Read libproc private residency, excluding this manager's owned segments.

    ``owned_segments`` is a one-way observation derived from the manager's live
    handles. Ordinary processes supply none. A named segment can become private
    in libproc when its owner is the last mapper, so those owner ranges must be
    excluded before the collector adds managed residency.

    Region residency is aggregate: a supplied range must cover the whole region
    before it can be excluded. Partial overlap fails rather than estimating.
    """
    native = _native()
    page_size = native.page_size
    ranges = []
    for segment in owned_segments:
        address, size = int(segment["address"]), int(segment["size"])
        if address < 0 or size < 0:
            raise ValueError("Owned segment address and size must be nonnegative")
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
                    raise OSError(errno.EAGAIN, "Owned segment partially overlaps a VM region")
                continue
        pages += private_pages
    return pages * page_size

"""One-way observations of the shared-memory manager's allocation owner.

No worker/array registry is maintained. The owner publishes its live handles at
allocation/release boundaries; only the sampler reads the resulting snapshots.
"""

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import json
import mmap
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import RLock
import time

import psutil

from ouroboros.common.memory_ipc import observation_file_io


@dataclass(frozen=True)
class ManagerContext:
    directory: str
    failure_flag: object


_pid = os.getpid()
_lock = RLock()
_context = None
_session_count = 0
_RESIDENT_BITS = bytes(value & 1 for value in range(256))


def _ensure_process():
    global _pid, _lock, _context, _session_count
    if _pid != os.getpid():
        _pid, _lock, _context, _session_count = os.getpid(), RLock(), None, 0


def current_context():
    _ensure_process()
    return _context


def configure_manager(context):
    global _context
    _ensure_process()
    _context = context


class ManagerSession:
    """Share one observation channel across overlapping runs in a backend."""

    def __init__(self):
        self.context = None
        self._acquired = False

    def acquire(self):
        global _context, _session_count
        _ensure_process()
        with _lock:
            if not self._acquired:
                if _context is None:
                    failure = multiprocessing.get_context("spawn").Value("b", 0)
                    _context = ManagerContext(tempfile.mkdtemp(prefix="ouroboros-managers-"), failure)
                self.context = _context
                self._acquired = True
                _session_count += 1
        return self

    def release(self):
        global _context, _session_count
        _ensure_process()
        with _lock:
            if not self._acquired:
                return
            self._acquired = False
            _session_count -= 1
            if not _session_count:
                _context = None
                shutil.rmtree(self.context.directory)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()


def publish_owned_segments(segments):
    """Derive a snapshot under the owner's lock; never retain allocation state."""
    context = current_context()
    if context is None:
        return
    temporary = Path(context.directory, f"{os.getpid()}.tmp")
    try:
        records = []
        for memory, _termed in segments.values():
            record = {"name": memory.name, "size": memory.size}
            if sys.platform == "darwin":
                # The manager can be a segment's sole remaining mapper. Exclude
                # its range from private residency before adding the segment.
                record["address"] = ctypes.addressof(ctypes.c_char.from_buffer(memory.buf))
            records.append(record)
        payload = json.dumps({"pid": os.getpid(), "create_time": psutil.Process().create_time(),
                              "generation": time.monotonic_ns(), "segments": records})

        def publish():
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(temporary.with_suffix(".json"))

        observation_file_io(publish)
    except Exception:
        # Observation failure must not turn a scientific allocation into a
        # failure or leave the caller unaware of its newly allocated segment.
        try:
            context.failure_flag.value = 1
        except Exception:
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


@lru_cache(maxsize=1)
def _posix_api():
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib" if sys.platform == "darwin" else None,
                       use_errno=True)
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_int64]
    libc.mmap.restype = ctypes.c_void_p
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    return libc


def _native_error(operation):
    error = ctypes.get_errno()
    return OSError(error, f"{operation}: {os.strerror(error)}")


def resident_bytes(segment):
    """Read managed POSIX backing residency without faulting in its data.

    A temporary read-only mapping avoids Python's resource tracker and conveys
    no unlink authority. No file/TIFF cache is queried. Residency-vector storage
    is bounded to 64 KiB regardless of allocation size.
    """
    # Reuse CPython's POSIX binding (including its librt linkage on older
    # Linux), without constructing a resource-tracked SharedMemory attachment.
    from multiprocessing.shared_memory import _posixshmem

    native = _posix_api()
    size = segment["size"]
    name = segment["name"]
    fd = _posixshmem.shm_open("/" + name.lstrip("/"), os.O_RDONLY, mode=0)
    try:
        if size <= 0 or os.fstat(fd).st_size != size:
            raise OSError("Managed segment size changed during memory sweep")
        address = native.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if address == ctypes.c_void_p(-1).value:
            raise _native_error("mmap")
        try:
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            resident = 0
            for first in range(0, pages, 65536):
                count = min(65536, pages - first)
                flags = (ctypes.c_ubyte * count)()
                if native.mincore(address + first * mmap.PAGESIZE, count * mmap.PAGESIZE, flags):
                    raise _native_error("mincore")
                resident += bytes(flags).translate(_RESIDENT_BITS).count(1)
            return resident * mmap.PAGESIZE
        finally:
            if native.munmap(address, size):
                raise _native_error("munmap")
    finally:
        os.close(fd)

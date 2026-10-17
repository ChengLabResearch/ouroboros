"""Observe live scientific mappings without owning or retaining their allocations.

The allocation owners supply addresses and backing identities. Per-process JSON
files are one-way observations for the memory sampler; execution never reads them.
Windows and macOS retain weak owner references and original backing identities
outside a reporting session, so a later run can observe buffers that already
exist. Addresses and manifests are produced only during a session. Linux uses
process metrics and does not register mappings.
"""

import ctypes
from dataclasses import dataclass
import json
import mmap
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from threading import RLock
import weakref

import psutil

from ouroboros.common.memory_ipc import observation_file_io


@dataclass(frozen=True)
class MappingContext:
    directory: str
    failure_flag: object


@dataclass
class _Observation:
    reference: object
    kind: str
    name: str | None = None
    identity: tuple | None = None
    error: str | None = None
    record: dict | None = None


_SUPPORTED = sys.platform in ("win32", "darwin")
_pid = os.getpid()
_lock = RLock()
_context = None
_session_context = None
_session_count = 0
_registry = {}
_errors = []
_create_time = None


def _ensure_process():
    """Do not reuse a forked parent's locks, identities, or observations."""
    global _pid, _lock, _context, _session_context, _session_count, _registry, _errors, _create_time
    if _pid != os.getpid():
        _pid = os.getpid()
        _lock = RLock()
        _context = _session_context = None
        _session_count = 0
        _registry = {}
        _errors = []
        _create_time = None


def _note_error(error):
    message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    if message not in _errors and len(_errors) < 8:
        _errors.append(message)
    if _context is not None:
        try:
            _context.failure_flag.value = 1
        except Exception:
            pass


def _publish():
    """Called under the process lock; failed observation never replaces science."""
    global _create_time
    if _context is None:
        return
    temporary = None
    try:
        if _create_time is None:
            _create_time = psutil.Process(_pid).create_time()
        mappings = []
        for key, observation in list(_registry.items()):
            try:
                owner = observation.reference()
                backing = getattr(owner, "_mmap", owner)
                if backing is None or backing.closed:
                    _registry.pop(key, None)
                else:
                    if observation.error is not None:
                        _note_error(observation.error)
                        continue
                    if observation.record is None:
                        observation.record = _describe(
                            owner, observation.kind, observation.name, observation.identity,
                        )
                    mappings.append(observation.record)
            except Exception as error:
                _note_error(error)
        payload = {"pid": _pid, "create_time": _create_time,
                   "mappings": mappings, "errors": list(_errors)}
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf8", dir=_context.directory,
            prefix=f"{_pid}-", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(payload, output)
        observation_file_io(lambda: os.replace(temporary, Path(_context.directory, f"{_pid}.json")))
        temporary = None
    except Exception as error:
        _note_error(error)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:
                pass


def refresh_mappings():
    if not _SUPPORTED:
        return
    _ensure_process()
    if _context is not None:
        with _lock:
            _publish()


def configure_mappings(context):
    """ProcessPoolExecutor initializer, also used by the coordinating process."""
    global _context, _errors
    if not _SUPPORTED:
        return
    _ensure_process()
    with _lock:
        if _context != context:
            _errors = []
        _context = context
        _publish()


def executor_mapping_options(context):
    # A process tree sampler also observes concurrent unreported work. Pools
    # created during an existing session publish into that same owned channel.
    if context is None and _SUPPORTED:
        _ensure_process()
        context = _context
    return {} if context is None else {"initializer": configure_mappings, "initargs": (context,)}


class MappingSession:
    """Share the observation channel across overlapping runs in one process."""

    def __init__(self):
        self.context = None
        self._acquired = False

    def acquire(self):
        global _session_context, _session_count
        if not _SUPPORTED:
            return self
        _ensure_process()
        with _lock:
            if self._acquired:
                return self
            if _session_context is None:
                # Spawn-compatible synchronization also works with fork workers.
                failure = multiprocessing.get_context("spawn").Value("b", 0)
                directory = tempfile.mkdtemp(prefix="ouroboros-mappings-")
                _session_context = MappingContext(directory, failure)
            self.context = _session_context
            self._acquired = True
            _session_count += 1
            configure_mappings(self.context)
            return self

    def release(self):
        global _context, _session_context, _session_count
        _ensure_process()
        with _lock:
            if not self._acquired:
                return
            self._acquired = False
            _session_count -= 1
            if _session_count:
                return
            try:
                shutil.rmtree(self.context.directory)
            except Exception as error:
                _note_error(error)
            _context = _session_context = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()


def _describe(owner, kind, name, identity):
    if kind == "shm":
        return {
            "address": ctypes.addressof(ctypes.c_char.from_buffer(owner)),
            "size": len(owner), "offset": 0,
            "identity": ["shm", name], "kind": kind, "name": name,
        }
    offset = int(owner.offset)
    mapping_offset = offset - offset % mmap.ALLOCATIONGRANULARITY
    return {
        "address": int(owner.ctypes.data) - (offset - mapping_offset),
        "size": len(owner._mmap), "offset": mapping_offset,
        "identity": list(identity), "kind": kind,
        "name": os.fspath(owner.filename), "mode": owner.mode,
    }


def _register(owner, kind, name=None, identity=None, error=None):
    key = id(owner)

    def released(reference):
        try:
            _ensure_process()
            with _lock:
                existing = _registry.get(key)
                if existing is not None and existing.reference is reference:
                    _registry.pop(key)
                    _publish()
        except BaseException:
            # Weakref callbacks can run during interpreter teardown.
            pass

    _registry[key] = _Observation(weakref.ref(owner, released), kind, name, identity, error)
    _publish()


def observe_shared_memory(shared):
    """Derive the whole mapping from its live SharedMemory handle, not a view."""
    if not _SUPPORTED:
        return
    _ensure_process()
    with _lock:
        try:
            backing = shared._mmap
            if backing is None or backing.closed:
                return
            _register(backing, "shm", shared.name)
        except Exception as error:
            _note_error(error)
            _publish()


def tracked_tiff_memmap(*args, **kwargs):
    """Delegate allocation unchanged, then observe the returned mapping owner."""
    import tifffile

    array = tifffile.memmap(*args, **kwargs)
    if not _SUPPORTED:
        return array
    _ensure_process()
    with _lock:
        try:
            identity = error = None
            try:
                # NumPy's mmap does not expose its original file descriptor.
                # Capture the identity now: a later session must not reidentify
                # an old mapping using a replacement at the same pathname.
                file = os.stat(array.filename)
                identity = ("file", file.st_dev, file.st_ino)
            except Exception as failure:
                error = f"{type(failure).__name__}: {failure}"
            _register(array, "file", identity=identity, error=error)
        except Exception as error:
            _note_error(error)
            _publish()
    return array

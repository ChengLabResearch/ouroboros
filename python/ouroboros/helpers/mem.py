from dataclasses import astuple, replace, asdict, fields
from functools import cached_property
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing import parent_process, util as multiprocessing_util
from multiprocessing.connection import wait
from multiprocessing.managers import SharedMemoryManager, SharedMemoryServer, BaseManager, dispatch
import os
import signal
from sys import stdout
import threading
from time import sleep
from typing import TextIO

import numpy as np

from .shapes import DataShape
from .log import log, LOG
from ouroboros.common.memory_managed import (
    configure_manager, current_context, publish_owned_segments,
)

# Existing pre-exit grace period in ``exit_cleanly``.
SHUTDOWN_GRACE_S = 1.0


def is_advanced_index(index):
    if isinstance(index, tuple):
        return any(isinstance(i, (list, np.ndarray)) and not isinstance(i, slice) for i in index)
    return isinstance(index, (list, np.ndarray)) and not isinstance(index, slice)


class SharedNPArray:
    __shutdown = True

    def __init__(self, name: str, shape: DataShape, dtype: np.dtype,
                 views: list = None, *, allocate: bool = False):
        self.__name = name
        self.__shape = shape
        self.__dtype = np.dtype(dtype)
        self.__views = [] if views is None else views
        self.__shutdown = False
        self.__allocated = allocate
        self.__sm = None
        if allocate:
            self.__sm = SharedMemory(name=name, create=True, size=self.size())
        else:
            self.__sm = SharedMemory(name=name)

    def size(self, swap=None):
        """ Size of array in memory.

            :param swap: Tuple overriding shape parameters, for a custom size.
            :retval: Size in bytes.
        """
        calc_shape = self.shape if swap is None else replace(self.shape, **asdict(swap))
        return int(np.prod(astuple(calc_shape), dtype=np.float64) * self.__dtype.itemsize)

    def __load(self) -> np.ndarray:
        """Allocates a numpy array in the shared memory."""
        if self.__sm.buf is not None:
            self.__ar = np.ndarray(shape=astuple(self.__shape), dtype=self.dtype, buffer=self.__sm.buf)
            if self.has_views:
                ar = self.__ar
                for view in self.__views:
                    ar = ar[view]
                return ar
            else:
                return self.__ar
        else:
            log.write("Shared Memory", f"Shared Memory {self.__name} Was Destroyed or Never Created",
                      log_level=LOG.ERROR)
            raise FileNotFoundError(f"Shared Memory {self.__name} Was Destroyed or Never Created")

    def array(self):
        return self.__load()

    def __enter__(self):
        return self.__load()

    def shutdown(self):
        # Lifecycle guards and non-owning serialization adapted from
        # den-sq/lftomo, lftomo/util/mem.py (8d4c884).
        if self.__shutdown:
            return
        if self.__sm is None:
            self.__shutdown = True
            return
        try:
            self.__sm.close()
        finally:
            if self.__allocated:
                try:
                    self.__sm.unlink()
                except FileNotFoundError:
                    pass
            self.__shutdown = True

    close = shutdown

    def __del__(self):
        if not getattr(self, "_SharedNPArray__shutdown", True):
            try:
                self.shutdown()
            except (AttributeError, BufferError, FileNotFoundError):
                pass

    def __exit__(self, *args, **kwargs):
        if hasattr(self, "_SharedNPArray__ar"):
            del self.__ar

    def __reduce__(self):
        """Serialize an attachment, never the creator's unlink authority or view."""
        return type(self), (self.__name, self.__shape, self.__dtype, self.__views)

    @property
    def owns_unlink(self) -> bool:
        return self.__allocated

    def __getitem__(self, view):
        # Test if it would create a copy. non-ideal but we shouldn't be calling this much anwyay!
        if is_advanced_index(view):
            raise IndexError("This would be an advanced index and create copies, so should not be used here.")
        else:
            return SharedNPArray(self.__name, self.__shape, self.__dtype, self.__views + [view])

    @cached_property
    def has_views(self):
        return len(self.__views) > 0

    @property
    def name(self):
        return self.__name

    @property
    def dtype(self):
        return self.__dtype

    @cached_property
    def shape(self):
        if len(self.__views) == 0:
            return self.__shape
        else:
            # non-ideal but we shouldn't be calling this much anwyay!
            shape = self.__load().shape
            if len(shape) == len(fields(self.__shape)):
                shape = type(self.__shape)(*shape)
            else:
                shape = self.__shape.gen(fields(self.__shape)[-len(shape):])(*shape)
            return shape


class _OwnedSharedMemoryTracker:
    """The server's sole allocation registry: live handles and release policy.

    Implements the tracker interface used by CPython's SharedMemoryServer.
    Unlike its names-only tracker, this owner keeps Windows mappings alive.
    All registry operations run in server request threads under one lock.
    """

    def __init__(self):
        self._segments: dict[str, tuple[SharedMemory, bool]] = {}
        self._lock = threading.RLock()
        self._closing = False
        publish_owned_segments(self._segments)

    @property
    def segment_names(self):
        with self._lock:
            return list(self._segments)

    def allocate(self, size, termed=False):
        with self._lock:
            if self._closing:
                raise RuntimeError("Shared memory manager is shutting down")
            memory = SharedMemory(create=True, size=size)
            self._segments[memory.name] = (memory, termed)
            publish_owned_segments(self._segments)
            return memory.name

    def register_segment(self, name):
        # Compatibility with inherited ShareableList / track_segment clients.
        # The caller retains its handle until this synchronous RPC completes.
        with self._lock:
            if self._closing:
                raise RuntimeError("Shared memory manager is shutting down")
            if name not in self._segments:
                self._segments[name] = (SharedMemory(name), False)
                publish_owned_segments(self._segments)

    def destroy_segment(self, name, termed_only=False):
        with self._lock:
            record = self._segments.get(name)
            if record is None or (termed_only and not record[1]):
                kind = "termed shared memory array" if termed_only else "managed shared memory segment"
                raise FileNotFoundError(f"{name} is not a {kind}")
            memory, _ = record
            try:
                # POSIX removes the name now; existing attachments remain valid.
                # Windows unlink is a no-op: close drops the owner's handle.
                try:
                    memory.unlink()
                except FileNotFoundError:
                    pass
            finally:
                memory.close()
                del self._segments[name]
                publish_owned_segments(self._segments)

    def unlink(self):
        with self._lock:
            self._closing = True
            for name in list(self._segments):
                self.destroy_segment(name)


class _OwnedSharedMemoryServer(SharedMemoryServer):
    public = SharedMemoryServer.public + ["allocate_segment"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the empty stdlib tracker, rather than tracking allocations in
        # parallel. Inherited list/track/shutdown operations use this owner too.
        self.shared_memory_context = _OwnedSharedMemoryTracker()

    def allocate_segment(self, connection, size, termed=False):
        return self.shared_memory_context.allocate(size, termed)

    def release_segment(self, connection, name, termed_only=False):
        try:
            self.shared_memory_context.destroy_segment(name, termed_only)
        except FileNotFoundError:
            # Server-level dispatch turns raised exceptions into RemoteError.
            # Return the atomic validation outcome to preserve the public API.
            return False
        return True


def _initialize_manager_observation(context, initializer, initargs):
    configure_manager(context)
    if initializer is not None:
        initializer(*initargs)


class SharedNPManager(SharedMemoryManager):
    """ Manages shared memory numpy arrays. """
    _Server = _OwnedSharedMemoryServer

    @classmethod
    def _run_server(cls, registry, address, authkey, serializer, writer,
                    initializer=None, initargs=()):
        """Adapt lftomo's detached owner-death cleanup to all supported OSes.

        Mirrors BaseManager._run_server because CPython exposes no public
        server/owner-death hook. A process sentinel, unlike getppid polling,
        also detects owner death on Windows. Native CI covers 3.11 and 3.12.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if hasattr(os, "setsid"):
            os.setsid()
        if initializer is not None:
            initializer(*initargs)
        server = cls._Server(registry, address, authkey, serializer)

        def cleanup_and_exit():
            try:
                server.shared_memory_context.unlink()
            finally:
                os._exit(0)

        def cleanup_after_sigterm(_signum, _frame):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            cleanup_and_exit()

        signal.signal(signal.SIGTERM, cleanup_after_sigterm)
        owner_sentinel = parent_process().sentinel

        def cleanup_after_owner_exit():
            wait([owner_sentinel])
            cleanup_and_exit()

        threading.Thread(target=cleanup_after_owner_exit, daemon=True).start()
        writer.send(server.address)
        writer.close()
        multiprocessing_util.info("manager serving at %r", server.address)
        server.serve_forever()

    def __init__(self, *args,
                 queue_mem: list[tuple[DataShape, np.dtype] | tuple[tuple[DataShape, np.dtype]]] | None = None,
                 **kwargs):
        SharedMemoryManager.__init__(self, *args, **kwargs)

        self.__mem_queue = []
        for mem in queue_mem or ():
            if isinstance(mem[0], tuple):
                self.__mem_queue.append([mem[0][0], mem[0][1]] + list(mem[1:]))
            else:
                self.__mem_queue.append([mem[0], mem[1]])

    def start(self, initializer=None, initargs=()):
        # Only the canonical manager publishes allocation observations.
        # Worker pools and array attachments need no observation initializer.
        context = current_context()
        if context is not None:
            if initializer is not None and not callable(initializer):
                raise TypeError("initializer must be a callable")
            super().start(_initialize_manager_observation, (context, initializer, initargs))
        else:
            super().start(initializer, initargs)

    def _request(self, method, *args):
        with self._Client(self._address, authkey=self._authkey) as connection:
            return dispatch(connection, None, method, args)

    def SharedMemory(self, size, *, termed=False):
        """Allocate in the owning server before returning a local attachment."""
        name = self._request("allocate_segment", int(size), termed)
        try:
            return SharedMemory(name)
        except BaseException:
            self._request("release_segment", name)
            raise

    def SharedNPArray(self, shape: DataShape, dtype: np.dtype, *create_with: tuple[DataShape, np.dtype]):
        return self._array(shape, dtype, create_with, termed=False)

    def _array(self, shape, dtype, create_with, *, termed):
        full_set = [(shape, dtype)] + list(create_with)
        size = max([np.prod(astuple(shape), dtype=object) * np.dtype(dtype).itemsize for (shape, dtype) in full_set])
        mem = self.SharedMemory(int(size), termed=termed)
        result = []
        try:
            for array_shape, array_dtype in full_set:
                result.append(SharedNPArray(mem.name, array_shape, array_dtype))
            return result[0] if len(result) == 1 else result
        except BaseException:
            for array in result:
                array.shutdown()
            self._request("release_segment", mem.name)
            raise
        finally:
            mem.close()

    def TermedNPArray(self, shape: DataShape, dtype: np.dtype, *create_with: tuple[DataShape, np.dtype]):
        return self._array(shape, dtype, create_with, termed=True)

    def clear_queue(self):  # pragma: no cover - IPC lifecycle.
        ar_mem = []
        while len(self.__mem_queue) > 0:
            new_mem = self.SharedNPArray(*self.__mem_queue.pop(0))
            ar_mem += new_mem if isinstance(new_mem, list) else [new_mem]
        return ar_mem

    def remove_termed(self, mem):
        name = mem.name if isinstance(mem, SharedNPArray) else mem
        # Validate/release at the owner before closing the supplied attachment.
        # Rejected removal must not invalidate an unrelated run-scoped array.
        if not self._request("release_segment", name, True):
            raise FileNotFoundError(f"{name} is not a termed shared memory array")
        if isinstance(mem, SharedNPArray):
            mem.shutdown()

    def __enter__(self):  # pragma: no cover - IPC lifecycle.
        this = [BaseManager.__enter__(self)]
        return tuple(this + self.clear_queue())

    def __exit__(self, *args, **kwargs):  # pragma: no cover - IPC lifecycle.
        super().__exit__(*args, **kwargs)


def exit_cleanly(step: str, *shm_objects, return_code: int = 0, statement: str = '', log_level: LOG = LOG.TIME,
                 out: TextIO = stdout, throw: Exception = None):
    """ Exit while cleaning up shared memory.

        :param step: Step of reconstruction process we are exiting during.
        :param shm_objects: Shared memory objects to shut down.
        :param return_code: Process return code to send.
    """
    log.write(step, statement, log_level, out)
    cleanup_mem(*shm_objects)

    sleep(SHUTDOWN_GRACE_S)
    log.footer(error=throw)

    exit(return_code)


def cleanup_mem(*shm_objects):
    """ Close and unlink shared memory objects.

        :param shm_objects: Shared memory objects to shut down.
    """
    for shm in shm_objects:
        if isinstance(shm, SharedNPArray):
            shm.shutdown()
            del shm
        elif isinstance(shm, ShareableList):
            shm.shm.close()
            shm.shm.unlink()
        elif isinstance(shm, SharedMemory):
            shm.close()
            shm.unlink()

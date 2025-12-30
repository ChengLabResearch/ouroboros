from dataclasses import astuple, replace, asdict, fields
from functools import cached_property
from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing.managers import SharedMemoryManager, BaseManager, ListProxy
from sys import stdout
from time import sleep
from typing import TextIO

import numpy as np

from .shapes import DataShape
from .log import log, LOG

MEM_INTERVAL_TIMER = 1.0


def is_advanced_index(index):
    if isinstance(index, tuple):
        return any(isinstance(i, (list, np.ndarray)) and not isinstance(i, slice) for i in index)
    return isinstance(index, (list, np.ndarray)) and not isinstance(index, slice)


class SharedNPArray:
    def __init__(self, name: str, shape: DataShape, dtype: np.dtype,
                 views: list = None, *, allocate: bool = False):
        self.__name = name
        self.__shape = shape
        self.__dtype = np.dtype(dtype)
        self.__views = [] if views is None else views
        if allocate:
            self.__sm = SharedMemory(name=name, create=True, size=self.size())
            self.__allocated = True
        else:
            self.__sm = SharedMemory(name=name)
            self.__allocated = False
        self.__shutdown = False

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
        self.__sm.close()
        if self.__allocated:
            self.__sm.unlink()
        self.__shutdown = True

    def __del__(self):
        if not self.__shutdown:
            self.shutdown()

    def __exit__(self, *args, **kwargs):
        del self.__ar

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


_termed_mem = []


def get_termed_mem():
    """Returns the existing global list rather than creating a new one."""
    return _termed_mem


class SharedNPManager(SharedMemoryManager):
    SharedMemoryManager.register(
        '_TermedMem',
        callable=get_termed_mem,
        proxytype=ListProxy
    )

    """ Manages shared memory numpy arrays. """
    def __init__(self, *args,
                 queue_mem: list[tuple[DataShape, np.dtype] | tuple[tuple[DataShape, np.dtype]]] = [],
                 **kwargs):
        SharedMemoryManager.__init__(self, *args, **kwargs)

        self.__mem_queue = []
        for mem in queue_mem:
            if isinstance(mem[0], tuple):
                self.__mem_queue.append([mem[0][0], mem[0][1]] + list(mem[1:]))
            else:
                self.__mem_queue.append([mem[0], mem[1]])
        self.__termed_mem = None

    def SharedNPArray(self, shape: DataShape, dtype: np.dtype, *create_with: tuple[DataShape, np.dtype]):
        full_set = [(shape, dtype)] + list(create_with)
        size = max([np.prod(astuple(shape), dtype=object) * np.dtype(dtype).itemsize for (shape, dtype) in full_set])
        mem = self.SharedMemory(int(size))
        result = [SharedNPArray(mem.name, shape, dtype) for (shape, dtype) in full_set]
        return result[0] if len(result) == 1 else result

    def TermedNPArray(self, shape: DataShape, dtype: np.dtype, *create_with: tuple[DataShape, np.dtype]):
        full_set = [(shape, dtype)] + list(create_with)
        size = max([np.prod(astuple(shape), dtype=object) * np.dtype(dtype).itemsize for (shape, dtype) in full_set])
        mem = SharedMemory(create=True, size=int(size))
        result = [SharedNPArray(mem.name, shape, dtype) for (shape, dtype) in full_set]
        self.__termed_mem.append(mem.name)
        return result[0] if len(result) == 1 else result

    def clear_queue(self):
        ar_mem = []
        while len(self.__mem_queue) > 0:
            new_mem = self.SharedNPArray(*self.__mem_queue.pop(0))
            ar_mem += new_mem if isinstance(new_mem, list) else [new_mem]
        return ar_mem

    def remove_termed(self, mem):
        if isinstance(mem, SharedNPArray):
            name = mem.name
            mem.shutdown()
        else:
            name = mem
        if name in self.__termed_mem:
            self.__termed_mem.pop(self.__termed_mem.index(name))
            t = SharedMemory(name)
            t.close()
            t.unlink()
        else:
            raise FileNotFoundError(f"{name} is not a termed shared memory array. {self.__termed_mem}")

    def shutdown(self):
        for name in self.__termed_mem:
            t = SharedMemory(name)
            t.close()
            t.unlink()
        super().shutdown()

    def start(self, *args, **kwargs):
        super().start(*args, **kwargs)
        # Initialize the proxy immediately upon start
        self.__termed_mem = self._TermedMem()

    def connect(self):
        super().connect()
        # Initialize the proxy immediately upon connect
        self.__termed_mem = self._TermedMem()

    def __enter__(self):
        this = [BaseManager.__enter__(self)]
        return tuple(this + self.clear_queue())

    def __exit__(self, *args, **kwargs):
        print("Exiting! SHM!")
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

    sleep(MEM_INTERVAL_TIMER)
    log.footer(error=throw)

    exit(return_code)


def mem_monitor(mem_file, mem_store, pid):
    with open(mem_file, "w") as out, mem_store as mem_branch:
        with mem_branch as last_step_arr:
            last_step = last_step_arr.tobytes().decode()

            while last_step.strip() not in ["COMPLETED", "ERRORED"]:
                last_step = last_step_arr.tobytes().decode()
                log.write(last_step, out=out, pid=pid)
                sleep(MEM_INTERVAL_TIMER)


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

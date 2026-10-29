"""Small native IPC regressions, including creator-exits-before-receiver-opens.

These test allocation lifetime, not numerical performance.
"""

import gc
import multiprocessing
from multiprocessing.managers import RemoteError
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.util import Finalize
import os
import pickle
import signal
import time
from uuid import uuid4

import numpy as np
import psutil
import pytest

from ouroboros.helpers.mem import SharedNPArray, SharedNPManager
from ouroboros.helpers.shapes import ProjOrder


SHAPE = ProjOrder(Y=2, Theta=3, X=4)
ORDER_SHAPE = ProjOrder(Y=5, Theta=6, X=7)
SPAWN = multiprocessing.get_context("spawn")
START_METHODS = multiprocessing.get_all_start_methods()


def _produce(address, authkey, sender, termed, crash):
    manager = SharedNPManager(address=address, authkey=authkey)
    manager.connect()
    allocate = manager.TermedNPArray if termed else manager.SharedNPArray
    array = allocate(SHAPE, np.uint16)
    with array as pixels:
        pixels[:] = 37
    # Send bytes so the receiver controls exactly when attachment occurs.
    payload = pickle.dumps(array)
    if not crash:
        array.shutdown()
    sender.send_bytes(payload)
    sender.close()
    if crash:
        os._exit(7)


def _produce_ordered(address, authkey, sender, termed, order):
    manager = SharedNPManager(address=address, authkey=authkey)
    manager.connect()
    allocate = manager.TermedNPArray if termed else manager.SharedNPArray
    array = allocate(ORDER_SHAPE, np.uint16, order=order)
    first_view = array[::2, 1:, :]
    nested_view = first_view[1:, :, ::-2]
    try:
        with array as pixels:
            pixels[:] = np.arange(210, dtype=np.uint16).reshape(6, 5, 7)
        sender.send_bytes(pickle.dumps((array, nested_view)))
    finally:
        nested_view.shutdown()
        first_view.shutdown()
        array.shutdown()
        sender.close()


def _consume_ordered(payload, sender, order):
    array, view = pickle.loads(payload)
    expected = np.arange(210, dtype=np.uint16).reshape(6, 5, 7)
    try:
        assert array.order == view.order == order
        assert not array.owns_unlink
        assert not view.owns_unlink
        with array as pixels, view as subset:
            assert pixels.flags.c_contiguous == (order == "C")
            assert pixels.flags.f_contiguous == (order == "F")
            assert not subset.flags.c_contiguous
            assert not subset.flags.f_contiguous
            np.testing.assert_array_equal(pixels, expected)
            np.testing.assert_array_equal(subset, expected[::2, 1:, :][1:, :, ::-2])
            subset += 1000
            sender.send((array.name, np.array(pixels)))
    finally:
        view.shutdown()
        array.shutdown()
        sender.close()


def _owner_until_killed(sender, process_group, start_method):
    if process_group:
        os.setsid()
    with SharedNPManager(ctx=multiprocessing.get_context(start_method)) as (manager,):
        normal = manager.SharedNPArray(SHAPE, np.uint16)
        termed = manager.TermedNPArray(SHAPE, np.uint16)
        names = (normal.name, termed.name)
        normal.shutdown()
        termed.shutdown()
        sender.send((manager._process.pid, names))
        sender.recv()  # Parent kills us after receiving the names.


def _record_initializer(sender, marker):
    sender.send((os.getpid(), multiprocessing.parent_process().pid, marker))
    sender.close()


def _assert_absent(name):
    with pytest.raises(FileNotFoundError):
        handle = SharedMemory(name)
        handle.close()  # Do not leak a handle if the assertion fails.


def _join(process):
    process.join(15)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("Shared-memory subprocess did not finish")


@pytest.mark.parametrize("termed", [False, True])
@pytest.mark.parametrize("crash", [False, True])
@pytest.mark.parametrize("start_method", START_METHODS)
def test_manager_owns_buffer_across_producer_exit(termed, crash, start_method):
    context = multiprocessing.get_context(start_method)
    with SharedNPManager(ctx=context) as (manager,):
        receiver, sender = context.Pipe(duplex=False)
        producer = context.Process(
            target=_produce,
            args=(manager.address, bytes(manager._authkey), sender, termed, crash),
        )
        producer.start()
        sender.close()
        try:
            assert receiver.poll(15)
            payload = receiver.recv_bytes()
            _join(producer)
            assert producer.exitcode == (7 if crash else 0)
            # No producer handle exists now. Only the server can keep this alive.
            array = pickle.loads(payload)
            name = array.name
            try:
                assert not array.owns_unlink
                with array as pixels:
                    np.testing.assert_array_equal(pixels, np.full((3, 2, 4), 37))
                assert name in manager._request("list_segments")
            finally:
                array.shutdown()
            if termed:
                manager.remove_termed(name)
                assert name not in manager._request("list_segments")
                _assert_absent(name)
        finally:
            receiver.close()
            if producer.is_alive():
                producer.kill()
                producer.join(5)
    _assert_absent(name)


@pytest.mark.parametrize("termed", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
def test_array_order_and_nested_views_survive_spawned_producer_and_consumer(termed, order):
    with SharedNPManager(ctx=SPAWN) as (manager,):
        receiver, sender = SPAWN.Pipe(duplex=False)
        producer = SPAWN.Process(
            target=_produce_ordered,
            args=(manager.address, bytes(manager._authkey), sender, termed, order),
        )
        producer.start()
        sender.close()
        try:
            assert receiver.poll(15)
            payload = receiver.recv_bytes()
            _join(producer)
            assert producer.exitcode == 0
        finally:
            receiver.close()
            if producer.is_alive():
                producer.kill()
                producer.join(5)

        # Attach in another spawned process only after the producer has exited.
        receiver, sender = SPAWN.Pipe(duplex=False)
        consumer = SPAWN.Process(target=_consume_ordered, args=(payload, sender, order))
        consumer.start()
        sender.close()
        try:
            assert receiver.poll(15)
            name, pixels = receiver.recv()
            _join(consumer)
            assert consumer.exitcode == 0
            expected = np.arange(210, dtype=np.uint16).reshape(6, 5, 7)
            expected[::2, 1:, :][1:, :, ::-2] += 1000
            np.testing.assert_array_equal(pixels, expected)
            assert manager._request("list_segments") == [name]
            if termed:
                manager.remove_termed(name)
                _assert_absent(name)
        finally:
            receiver.close()
            if consumer.is_alive():
                consumer.kill()
                consumer.join(5)
    _assert_absent(name)


@pytest.mark.parametrize("start_method", START_METHODS)
def test_cpython_startup_dispatch_and_shutdown_contract(start_method):
    context = multiprocessing.get_context(start_method)
    receiver, sender = context.Pipe(duplex=False)
    manager = SharedNPManager(ctx=context)
    marker = "initializer arguments survived startup"
    try:
        manager.start(initializer=_record_initializer, initargs=(sender, marker))
        sender.close()
        assert receiver.poll(15)
        assert receiver.recv() == (manager._process.pid, os.getpid(), marker)
        # BaseManager.start must install the finalizer; cleanup must therefore
        # be implemented at the server's shared_memory_context boundary.
        assert isinstance(manager.shutdown, Finalize)
        with pytest.raises(RemoteError, match="ValueError"):
            manager.SharedMemory(0)
        assert manager._request("list_segments") == []
        with pytest.raises(FileNotFoundError, match="not a termed shared memory array"):
            manager.remove_termed("unknown-segment")
        raw = manager.SharedMemory(48)
        name = raw.name
        raw.close()
        assert manager._request("list_segments") == [name]
    finally:
        sender.close()
        receiver.close()
        if hasattr(manager, "shutdown"):
            manager.shutdown()
    _assert_absent(name)


def test_remove_termed_releases_last_array_handle():
    with SharedNPManager(ctx=SPAWN) as (manager,):
        array = manager.TermedNPArray(SHAPE, np.uint16)
        name = array.name
        manager.remove_termed(array)
        assert manager._request("list_segments") == []
        _assert_absent(name)
        with pytest.raises(FileNotFoundError):
            array.array()
        with pytest.raises(FileNotFoundError, match="not a termed shared memory array"):
            manager.remove_termed(name)
        array.shutdown()  # Repeated local cleanup is harmless.


def test_remove_termed_rejects_unregistered_array_without_closing_it():
    with SharedNPManager(ctx=SPAWN) as (manager,):
        array = manager.SharedNPArray(SHAPE, np.uint16)
        try:
            with array as pixels:
                pixels[:] = 17
            with pytest.raises(FileNotFoundError, match="not a termed shared memory array"):
                manager.remove_termed(array)
            with array as pixels:
                assert np.all(pixels == 17)
            assert manager._request("list_segments") == [array.name]
        finally:
            array.shutdown()


def test_release_preserves_existing_attachment_but_drops_manager_ownership():
    with SharedNPManager(ctx=SPAWN) as (manager,):
        array = manager.TermedNPArray(SHAPE, np.uint16)
        attachment = SharedNPArray(array.name, SHAPE, np.uint16)
        try:
            with array as pixels:
                pixels[:] = 29
            manager.remove_termed(array)
            assert manager._request("list_segments") == []
            if os.name == "posix":
                _assert_absent(array.name)  # Unlinked before manager shutdown.
            with attachment as pixels:
                assert np.all(pixels == 29)  # Already-mapped consumers still work.
        finally:
            attachment.shutdown()
            array.shutdown()
        _assert_absent(array.name)


def test_tracker_release_closes_owning_handle_and_shutdown_rejects_allocation():
    from ouroboros.helpers.mem import _OwnedSharedMemoryTracker

    tracker = _OwnedSharedMemoryTracker()
    name = tracker.allocate(48, termed=True)
    owning_handle = tracker._segments[name][0]
    tracker.destroy_segment(name, termed_only=True)
    assert owning_handle.buf is None
    assert tracker.segment_names == []
    _assert_absent(name)
    tracker.unlink()
    tracker.unlink()
    with pytest.raises(RuntimeError, match="shutting down"):
        tracker.allocate(48)
    with pytest.raises(RuntimeError, match="shutting down"):
        tracker.register_segment("unused")


def test_standard_manager_apis_share_the_same_owner():
    with SharedNPManager(ctx=SPAWN) as (manager,):
        raw = manager.SharedMemory(48)
        shared_list = manager.ShareableList([1, 2, 3])
        names = {raw.name, shared_list.shm.name}
        raw.close()
        shared_list.shm.close()
        assert set(manager._request("list_segments")) == names
        for name in names:
            with_attachment = SharedMemory(name)
            with_attachment.close()
    for name in names:
        _assert_absent(name)


def test_exception_shutdown_releases_both_allocation_lifetimes():
    names = []
    with pytest.raises(RuntimeError, match="deliberate"):
        with SharedNPManager(ctx=SPAWN) as (manager,):
            for allocate in (manager.SharedNPArray, manager.TermedNPArray):
                array = allocate(SHAPE, np.uint16)
                names.append(array.name)
                array.shutdown()
            raise RuntimeError("deliberate")
    for name in names:
        _assert_absent(name)
    manager.shutdown()


def test_failed_client_attachment_releases_server_allocation(monkeypatch):
    import ouroboros.helpers.mem as mem_module

    with SharedNPManager(ctx=SPAWN) as (manager,):
        names = []

        def fail_attachment(name):
            names.append(name)
            raise OSError("deliberate attachment failure")

        # Spawned server uses the real allocator; only this client's opening
        # step fails, after the server has established ownership.
        with monkeypatch.context() as patch:
            patch.setattr(mem_module, "SharedMemory", fail_attachment)
            with pytest.raises(OSError, match="deliberate attachment failure"):
                manager.SharedMemory(48)
        assert len(names) == 1
        assert manager._request("list_segments") == []
        _assert_absent(names[0])


def test_failed_second_view_closes_first_and_releases_server_allocation(monkeypatch):
    import ouroboros.helpers.mem as mem_module

    with SharedNPManager(ctx=SPAWN) as (manager,):
        views = []

        def fail_second_view(*args, **kwargs):
            if views:
                raise OSError("deliberate view failure")
            array = SharedNPArray(*args, **kwargs)
            views.append(array)
            return array

        with monkeypatch.context() as patch:
            patch.setattr(mem_module, "SharedNPArray", fail_second_view)
            with pytest.raises(OSError, match="deliberate view failure"):
                manager.TermedNPArray(SHAPE, np.uint16, (SHAPE, np.uint16))
        assert len(views) == 1
        assert manager._request("list_segments") == []
        _assert_absent(views[0].name)
        with pytest.raises(FileNotFoundError):
            views[0].array()


@pytest.mark.parametrize("termed", [False, True])
def test_invalid_order_releases_server_allocation(termed):
    with SharedNPManager(ctx=SPAWN) as (manager,):
        allocate = manager.TermedNPArray if termed else manager.SharedNPArray
        with pytest.raises(ValueError, match="order"):
            allocate(SHAPE, np.uint16, order="K")
        assert manager._request("list_segments") == []


def test_pickle_transfers_attachment_without_unlink_authority_or_cached_pixels():
    # Keep explicit test names below macOS's shared-memory name limit.
    owner = SharedNPArray(f"ob-{uuid4().hex[:16]}", SHAPE, np.uint16, allocate=True)
    try:
        # Leave a cached ndarray in the owner; serialization must ignore it.
        owner.array()[:] = 13
        clone = pickle.loads(pickle.dumps(owner))
        try:
            assert owner.owns_unlink
            assert not clone.owns_unlink
            assert "_SharedNPArray__ar" not in clone.__dict__
            with clone as pixels:
                pixels[:] = 19
        finally:
            clone.shutdown()
            clone.shutdown()
        with owner as pixels:
            assert np.all(pixels == 19)
        check = SharedMemory(owner.name)
        check.close()
    finally:
        owner.shutdown()
        owner.shutdown()
    _assert_absent(owner.name)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_partially_initialized_array_cleanup():
    partial = object.__new__(SharedNPArray)
    partial.__del__()
    partial.__exit__()
    with pytest.raises(FileNotFoundError):
        SharedNPArray(f"missing-{uuid4().hex[:16]}", SHAPE, np.uint16)
    gc.collect()


@pytest.mark.parametrize("stop", ["terminate", "kill", "kill_group"])
@pytest.mark.parametrize("start_method", START_METHODS)
def test_manager_cleans_up_after_owner_death(stop, start_method):
    if stop == "kill_group" and not hasattr(os, "killpg"):
        pytest.skip("POSIX process groups only")
    receiver, sender = SPAWN.Pipe()
    owner = SPAWN.Process(
        target=_owner_until_killed,
        args=(sender, stop == "kill_group", start_method),
    )
    owner.start()
    sender.close()
    manager_pid = None
    try:
        assert receiver.poll(15)
        manager_pid, names = receiver.recv()
        manager_process = psutil.Process(manager_pid)
        if stop == "kill_group":
            os.killpg(owner.pid, signal.SIGKILL)
        else:
            getattr(owner, stop)()
        _join(owner)
        # Wait for the owning manager to close its handles before inspecting
        # names. Polling open/close itself could extend Windows buffer lifetime.
        deadline = time.monotonic() + 10
        while manager_process.is_running():
            try:
                if manager_process.status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.NoSuchProcess:
                break
            assert time.monotonic() < deadline, "Manager survived its owning process"
            time.sleep(0.02)
        for name in names:
            _assert_absent(name)
    finally:
        receiver.close()
        if owner.is_alive():
            owner.kill()
            owner.join(5)
        if manager_pid is not None:
            try:
                psutil.Process(manager_pid).terminate()
            except psutil.NoSuchProcess:
                pass

from dataclasses import astuple
import gc
from multiprocessing.shared_memory import SharedMemory, ShareableList
import pytest
from sys import getsizeof
from time import sleep
from uuid import uuid4

import numpy as np

from ouroboros.helpers.mem import SharedNPManager, SharedNPArray, cleanup_mem, exit_cleanly
from ouroboros.helpers.shapes import SinoOrder, ProjOrder, ReconOrder


def test_alt_creation():
    po = ProjOrder(Y=12, Theta=1501, X=2048)
    with SharedNPManager(queue_mem=[(po.to(SinoOrder), np.uint16), ((po, np.uint16), (ReconOrder.of(po), np.uint16))]) \
         as (_snm, sino_mem, proj_mem, recon_mem):
        assert sino_mem.shape == SinoOrder(Y=12, Theta=1501, X=2048)
        assert proj_mem.shape == ProjOrder(Y=12, Theta=1501, X=2048)
        assert recon_mem.shape == ReconOrder(Y=12, X=2048, Z=2048)
        assert sino_mem.order == proj_mem.order == recon_mem.order == "C"

        assert recon_mem.name == proj_mem.name
        assert recon_mem.name != sino_mem.name

        with sino_mem as si, proj_mem as pj, recon_mem as rc:
            si[0, 0, 0] = 6
            pj[0, 0, 0] = 5
            assert rc[0, 0, 0] == pj[0, 0, 0]
            assert pj[0, 0, 0] != si[0, 0, 0]


@pytest.mark.parametrize("order", [None, "C", "F"])
def test_direct_array_order_matches_shared_buffer_layout(order):
    shape = ProjOrder(Y=2, Theta=3, X=4)
    options = {} if order is None else {"order": order}
    expected_order = "C" if order is None else order
    owner = SharedNPArray(
        f"ob-{uuid4().hex[:16]}", shape, "uint16", allocate=True, **options
    )
    attachment = None
    try:
        expected = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
        with owner as pixels:
            pixels[:] = expected
            assert pixels.flags.c_contiguous == (expected_order == "C")
            assert pixels.flags.f_contiguous == (expected_order == "F")
        assert owner.order == expected_order
        assert owner.dtype == np.dtype(np.uint16)
        raw = SharedMemory(owner.name)
        try:
            assert bytes(raw.buf) == expected.tobytes(order=expected_order)
        finally:
            raw.close()
        attachment = SharedNPArray(owner.name, shape, np.uint16, **options)
        with attachment as pixels:
            np.testing.assert_array_equal(pixels, expected)
            pixels[1, 0, 2] = 100
        with owner as pixels:
            assert pixels[1, 0, 2] == 100
        assert not attachment.owns_unlink
        assert owner.owns_unlink
    finally:
        if attachment is not None:
            attachment.shutdown()
        owner.shutdown()


@pytest.mark.parametrize("order", ["A", "K", "c", "f", "", None])
@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_invalid_array_order_is_rejected_before_creation_or_attachment(order):
    name = f"ob-{uuid4().hex[:16]}"
    shape = ProjOrder(Y=2, Theta=3, X=4)
    for allocate in (False, True):
        with pytest.raises(ValueError, match="order"):
            SharedNPArray(name, shape, np.uint16, order=order, allocate=allocate)
    with pytest.raises(FileNotFoundError):
        handle = SharedMemory(name)
        handle.close()
    gc.collect()


@pytest.mark.parametrize("order", [None, "F"])
@pytest.mark.parametrize("termed", [False, True])
def test_manager_aliases_share_order_and_preserve_shape_and_dtype(order, termed):
    specifications = (
        (ProjOrder(Y=2, Theta=3, X=4), np.dtype(np.uint16)),
        (SinoOrder(Y=2, Theta=3, X=4), np.dtype(np.uint16)),
        (ReconOrder(Y=2, X=3, Z=2), np.dtype(np.uint32)),
    )
    options = {} if order is None else {"order": order}
    expected_order = "C" if order is None else order
    with SharedNPManager() as (manager,):
        allocate = manager.TermedNPArray if termed else manager.SharedNPArray
        arrays = allocate(*specifications[0], *specifications[1:], **options)
        try:
            expected = np.arange(24, dtype=np.uint16).reshape(3, 2, 4)
            with arrays[0] as pixels:
                pixels[:] = expected
            buffer = expected.tobytes(order=expected_order)
            assert len({array.name for array in arrays}) == 1
            for array, (shape, dtype) in zip(arrays, specifications):
                assert array.order == expected_order
                assert array.shape == shape
                assert array.dtype == dtype
                assert not array.owns_unlink
                expected_view = np.ndarray(
                    astuple(shape), dtype=dtype, buffer=buffer, order=expected_order
                )
                with array as pixels:
                    np.testing.assert_array_equal(pixels, expected_view)
                    assert pixels.flags.c_contiguous == (expected_order == "C")
                    assert pixels.flags.f_contiguous == (expected_order == "F")
        finally:
            for array in arrays:
                array.shutdown()
        if termed:
            manager.remove_termed(arrays[0].name)
            assert manager._request("list_segments") == []


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
def test_partial_shared_np_array_destruction_is_already_shutdown():
    po = ProjOrder(Y=12, Theta=1501, X=2048)

    partial = object.__new__(SharedNPArray)
    partial.__del__()

    with pytest.raises(FileNotFoundError):
        SharedNPArray("MissingTestMem", po, np.uint16)
    gc.collect()


def test_direct_create():
    po = ProjOrder(Y=12, Theta=1501, X=2048)

    # Creates memory and array, deletes when object is deleted on exit.
    with SharedNPArray("TestMem", po, np.uint16, allocate=True) as po_ar:
        assert po_ar.shape == (1501, 12, 2048)
        assert isinstance(po_ar, np.ndarray)

    # Createss new array with same mory.
    test_mem = SharedNPArray("TestMem", po, np.float32, allocate=True)
    assert test_mem.size() == 2048 * 1501 * 12 * 4

    cleanup_mem(test_mem)

    # Cannot create array after memory is deleted.
    with pytest.raises(FileNotFoundError):
        _ = test_mem.array()

    sleep(1.0)

    # FileNotFound error when memory hasn't been created.
    with pytest.raises(FileNotFoundError):
        SharedNPArray("MyHat", po, np.uint16)

    # Deletes multiple objects on exit cleanly call.
    test_mem = SharedNPArray("AnotherTest", po, np.uint16, allocate=True)
    shl = ShareableList([5, 1, 2])
    shm = SharedMemory("TestThree", create=True, size=512)

    with pytest.raises(SystemExit) as se:
        exit_cleanly("TEST", test_mem, shl, shm, return_code=5, statement="Mem Test Suite",
                     throw=FileNotFoundError("Ignore Me"))

    with pytest.raises(FileNotFoundError):
        SharedNPArray("AnotherTest", po, np.uint16)
    with pytest.raises(FileNotFoundError):
        SharedNPArray("TestThree", po, np.uint16)

    assert se.value.code == 5


def test_mem():
    snm = SharedNPManager()
    snm.start()

    snp = snm.SharedNPArray(SinoOrder(Y=24, Theta=1501, X=2048), dtype=np.uint16)

    snpss = snp[1:10, :, :]

    snpv = snp[1, :, :]

    snpvv = snpv[5:10, :]

    with pytest.raises(IndexError):
        _ = snp[[5, 4, 1], :, :]

    with pytest.raises(IndexError):
        _ = snp[[5, 4, 1, 4]]

    assert snp.shape == SinoOrder(Y=24, Theta=1501, X=2048)
    assert snpss.shape == SinoOrder(Y=9, Theta=1501, X=2048)
    assert astuple(snpv.shape) == (1501, 2048)
    assert astuple(snpvv.shape) == (5, 2048)
    assert snp.size() == 147554304
    assert snpv.size() == 6148096
    assert snpvv.size() == 20480
    assert snp.dtype == snpv.dtype == snpvv.dtype == np.dtype(np.uint16)
    assert snp.name == snpv.name == snpvv.name
    assert not snp.has_views
    assert snpss.has_views
    assert snpv.has_views
    assert snpvv.has_views

    print(f"Shapes: {snp.shape} | {snpv.shape} | {snpvv.shape}")

    print(f"Sizes: {getsizeof(snm)} | {getsizeof(snpv)} | {getsizeof(snpvv)}")

    with snpv as var, snp as ar, snpvv as vvar:
        print(f"MSM: {np.shares_memory(var, ar[1, :, :])}")

        assert np.all(var[:, :] == ar[1, :, :])

        print(f"{type(ar)} - {ar.shape} | {type(var)} - {var.shape} | {type(vvar)} - {var.shape}")

        ar[1, 5, 1] = 80

        print(vvar[:, 1])

        assert vvar[0, 1] == 80

        var[0:10, :] = 77

        assert all(ar[1, 0:10, 5] == 77)

        print(ar[1, 0:10, 0:5])

        assert np.all(snpss.array() == ar[1:10, :, :])

    snm.shutdown()

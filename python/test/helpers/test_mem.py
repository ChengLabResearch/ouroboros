from dataclasses import astuple
from multiprocessing.shared_memory import SharedMemory, ShareableList
import pytest
from sys import getsizeof
from time import sleep

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

        assert recon_mem.name == proj_mem.name
        assert recon_mem.name != sino_mem.name

        with sino_mem as si, proj_mem as pj, recon_mem as rc:
            si[0, 0, 0] = 6
            pj[0, 0, 0] = 5
            assert rc[0, 0, 0] == pj[0, 0, 0]
            assert pj[0, 0, 0] != si[0, 0, 0]


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

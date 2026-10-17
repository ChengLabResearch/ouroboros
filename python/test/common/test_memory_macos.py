import ctypes
import errno
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.common import memory_macos


PAGE = 4096


def region(address, pages=1, *, mode=2, private=0, shared=0, refs=1):
    return memory_macos._ProcRegionInfo(
        address=address, size=pages * PAGE, share_mode=mode,
        private_pages_resident=private, shared_pages_resident=shared,
        ref_count=refs,
    )


def region_reader(monkeypatch, regions, *, terminal_error=errno.EINVAL):
    pending = iter(regions)

    def proc_pidinfo(pid, flavor, address, output, size):
        assert pid == 123 and flavor == 7
        item = next(pending, None)
        if item is None:
            ctypes.set_errno(terminal_error)
            return 0
        ctypes.memmove(output, ctypes.byref(item), size)
        return size

    native = SimpleNamespace(
        page_size=PAGE, shared_region=(0, 0), proc_pidinfo=proc_pidinfo,
    )
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    return native


def test_region_abi_matches_darwin_header():
    assert ctypes.sizeof(memory_macos._ProcRegionInfo) == 96
    assert memory_macos._ProcRegionInfo.offset.offset == 16
    assert memory_macos._ProcRegionInfo.private_pages_resident.offset == 64
    assert memory_macos._ProcRegionInfo.address.offset == 80


def test_private_counts_exclude_shared_and_tracked_unique_pages(monkeypatch):
    region_reader(monkeypatch, [
        region(PAGE, 4, private=2, shared=1),
        region(5 * PAGE, 4, mode=1, private=1, shared=2, refs=2),
        region(9 * PAGE, 4, mode=1, private=1, shared=2, refs=1),
        region(13 * PAGE, 2, mode=4, shared=2),
        region(15 * PAGE, 2, private=2),
        region(17 * PAGE, 2, mode=8, private=2),
    ])
    assert memory_macos.private_resident_bytes(123, [
        {"address": 15 * PAGE, "size": 2 * PAGE},
    ]) == 9 * PAGE


def test_adjacent_tracked_ranges_cover_a_merged_region(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, 2, private=2)])
    assert memory_macos.private_resident_bytes(123, [
        {"address": PAGE, "size": PAGE},
        {"address": 2 * PAGE, "size": PAGE},
    ]) == 0


def test_partial_private_region_coverage_is_not_guessed(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, 2, private=1)])
    with pytest.raises(OSError, match="partially overlaps"):
        memory_macos.private_resident_bytes(123, [{"address": PAGE, "size": PAGE}])


@pytest.mark.parametrize("regions", [[], [region(PAGE, private=1)]])
def test_permission_errors_do_not_become_zero_or_partial_samples(monkeypatch, regions):
    region_reader(monkeypatch, regions, terminal_error=errno.EPERM)
    with pytest.raises(PermissionError):
        memory_macos.private_resident_bytes(123, [])


def test_region_walk_rejects_nonprogressing_results(monkeypatch):
    region_reader(monkeypatch, [region(PAGE, private=1), region(PAGE, private=1)])
    with pytest.raises(OSError, match="changed during"):
        memory_macos.private_resident_bytes(123, [])


def test_truncated_region_reply_is_rejected(monkeypatch):
    native = region_reader(monkeypatch, [])
    native.proc_pidinfo = lambda *args: ctypes.sizeof(memory_macos._ProcRegionInfo) - 1
    with pytest.raises(OSError, match="Truncated"):
        memory_macos.private_resident_bytes(123, [])


def test_intel_shared_cache_exclusion_preserves_private_pages(monkeypatch):
    native = region_reader(monkeypatch, [
        region(PAGE, mode=1, private=1), region(2 * PAGE, private=1),
    ])
    native.shared_region = (PAGE, 3 * PAGE)
    assert memory_macos.private_resident_bytes(123, []) == PAGE


class ResidencyReader:
    page_size = PAGE

    def __init__(self, flags, *, fail=False):
        self.flags = flags
        self.fail = fail
        self.calls = []
        self.address = 0x100000

    def mmap(self, address, length, protection, flags, fd, offset):
        self.fd = fd
        self.calls.append(("mmap", length, protection, flags, offset))
        return self.address

    def mincore(self, address, length, output):
        self.calls.append(("mincore", address, length))
        if self.fail:
            ctypes.set_errno(errno.EACCES)
            return -1
        first = (address - self.address) // PAGE
        count = length // PAGE
        for i in range(count):
            output[i] = self.flags[first + i]
        return 0

    def munmap(self, address, length):
        self.calls.append(("munmap", address, length))
        return 0


def file_mapping(path, size, offset=0):
    stat = path.stat()
    return {"kind": "file", "identity": ["file", stat.st_dev, stat.st_ino],
            "name": str(path), "offset": offset, "size": size}


def test_residency_uses_incore_bit_and_returns_backing_byte_offsets(monkeypatch, tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes(8 * PAGE))
    native = ResidencyReader([1, 2, 4, 0x81])
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    mapping = file_mapping(path, 3 * PAGE, PAGE + 17)
    np.testing.assert_array_equal(memory_macos.resident_pages(mapping), [PAGE, 4 * PAGE])
    assert native.calls[0] == ("mmap", 4 * PAGE, 1, 1, PAGE)
    assert native.calls[-1] == ("munmap", native.address, 4 * PAGE)
    with pytest.raises(OSError):
        os.fstat(native.fd)


def test_failed_residency_query_releases_mapping_and_file(monkeypatch, tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes(PAGE))
    native = ResidencyReader([], fail=True)
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    with pytest.raises(PermissionError, match="mincore"):
        memory_macos.resident_pages(file_mapping(path, PAGE))
    assert native.calls[-1][0] == "munmap"
    with pytest.raises(OSError):
        os.fstat(native.fd)


def test_replaced_file_is_rejected_before_mapping(monkeypatch, tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes(PAGE))
    mapping = file_mapping(path, PAGE)
    mapping["identity"][2] += 1
    native = ResidencyReader([])
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    with pytest.raises(OSError, match="identity changed"):
        memory_macos.resident_pages(mapping)
    assert not native.calls


def test_failed_mmap_closes_backing_file_without_unmapping(monkeypatch, tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes(PAGE))
    native = ResidencyReader([])

    def mmap(*args):
        native.fd = args[4]
        ctypes.set_errno(errno.ENOMEM)
        return ctypes.c_void_p(-1).value

    native.mmap = mmap
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    with pytest.raises(OSError, match="mmap"):
        memory_macos.resident_pages(file_mapping(path, PAGE))
    assert not native.calls
    with pytest.raises(OSError):
        os.fstat(native.fd)


def test_truncated_backing_is_not_reported_as_zero_residency(monkeypatch, tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes(PAGE))
    native = ResidencyReader([])
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    with pytest.raises(OSError, match="was truncated"):
        memory_macos.resident_pages(file_mapping(path, 2 * PAGE))
    assert not native.calls


def test_copy_on_write_mapping_is_explicitly_unsupported(monkeypatch, tmp_path):
    path = tmp_path / "mapped.bin"
    path.write_bytes(bytes(PAGE))
    mapping = {**file_mapping(path, PAGE), "address": PAGE, "mode": "c"}
    region_reader(monkeypatch, [region(PAGE, private=1)])
    with pytest.raises(OSError, match="Copy-on-write") as error:
        memory_macos.private_resident_bytes(123, [mapping])
    assert error.value.errno == errno.ENOTSUP
    with pytest.raises(OSError, match="Copy-on-write"):
        memory_macos.resident_pages(mapping)


def test_shm_query_uses_native_readonly_open_without_a_resource_tracker(monkeypatch, tmp_path):
    backing = tmp_path / "shm"
    backing.write_bytes(bytes(PAGE))
    native = ResidencyReader([1])

    def shm_open(name, flags, mode):
        assert (name, flags, mode) == (b"/tracked-buffer", os.O_RDONLY, 0)
        return os.open(backing, flags)

    native.shm_open = shm_open
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    np.testing.assert_array_equal(memory_macos.resident_pages({
        "kind": "shm", "identity": ["shm", "tracked-buffer"],
        "name": "tracked-buffer", "offset": 0, "size": PAGE,
    }), [0])
    assert backing.exists()
    with pytest.raises(OSError):
        os.fstat(native.fd)


def test_residency_vector_is_bounded_for_large_mappings(monkeypatch, tmp_path):
    path = tmp_path / "sparse.bin"
    with path.open("wb") as stream:
        stream.truncate(65537 * PAGE)
    native = ResidencyReader(np.zeros(65537, dtype=np.uint8))
    monkeypatch.setattr(memory_macos, "_native", lambda: native)
    assert not memory_macos.resident_pages(file_mapping(path, 65537 * PAGE)).size
    calls = [call for call in native.calls if call[0] == "mincore"]
    assert [call[2] for call in calls] == [65536 * PAGE, PAGE]


@pytest.mark.skipif(sys.platform != "darwin", reason="Native macOS smoke test")
def test_native_same_user_child_and_untouched_shared_observer(tmp_path):
    from multiprocessing.shared_memory import SharedMemory

    code = ("import sys; data=bytearray(8*1024*1024); "
            "data[::4096]=b'x'*(len(data)//4096); "
            "print('ready',flush=True); sys.stdin.read(1)")
    with subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            assert memory_macos.private_resident_bytes(child.pid, []) >= 8 * 1024 * 1024
        finally:
            child.communicate("x", timeout=10)

    page_size = os.sysconf("SC_PAGE_SIZE")
    shm = SharedMemory(create=True, size=2 * page_size)
    try:
        shm.buf[0] = 1
        shm.buf[page_size] = 1
        pages = memory_macos.resident_pages({
            "kind": "shm", "identity": ["shm", shm.name], "name": shm.name,
            "offset": 0, "size": shm.size,
        })
        np.testing.assert_array_equal(pages, [0, page_size])
        assert shm.buf[0] == 1  # Query neither unlinked nor modified the allocation.
    finally:
        shm.close()
        shm.unlink()

    # These file pages reside in the cache after writing; the observer's new
    # read-only view has not touched them. Nonzero offsets exercise mmap's ABI.
    path = tmp_path / "cached.bin"
    path.write_bytes(bytes(3 * page_size))
    np.testing.assert_array_equal(
        memory_macos.resident_pages(file_mapping(path, page_size, page_size + 17)),
        [page_size, 2 * page_size],
    )

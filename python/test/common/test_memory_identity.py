"""Exercise psutil's native identity through clock changes, without setting the OS clock."""

from collections import Counter
import json
import mmap
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from ouroboros.common import memory_ipc as ipc, memory_managed as managed
from ouroboros.common import memory_statistics as memory


@pytest.fixture
def clock_offset(monkeypatch):
    if sys.platform == "linux":
        from psutil import _pslinux as native
    elif sys.platform == "darwin":
        from psutil import _psosx as native
    else:
        pytest.skip("Native boot-time conversion applies to Linux and macOS")
    baseline = native.boot_time()
    if sys.platform == "darwin":
        # Exercise adjust_proc_create_time(), including machines whose import
        # baseline predates another adjustment. Native identity is untouched.
        monkeypatch.setattr(native, "INIT_BOOT_TIME", baseline)
    offset = [0]
    monkeypatch.setattr(native, "boot_time", lambda: baseline + offset[0])
    return offset


@pytest.fixture
def process_family(tmp_path):
    """An isolated real tree; an optional production observer waits for start."""
    script = """
import json, subprocess, sys
children = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
            for _ in range(2)]
print(json.dumps([child.pid for child in children]), flush=True)
try:
    for line in sys.stdin:
        child = subprocess.Popen(json.loads(line), stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        children.append(child)
        print(child.pid, flush=True)
finally:
    for child in children:
        child.terminate()
    for child in children:
        child.wait(timeout=5)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH"),
    ]))
    family = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, env=environment)
    try:
        children = json.loads(family.stdout.readline())
        root = psutil.Process(family.pid)

        def start_observer():
            directory = tmp_path / "observer"
            directory.mkdir()
            memory._write_json(directory / "config.json", {
                "root_identity": ipc.export_process_identity(root), "interval": 1,
                "managers_directory": None,
            })
            family.stdin.write(json.dumps([
                sys.executable, "-m", memory._OBSERVER_MODULE, str(directory),
            ]) + "\n")
            family.stdin.flush()
            observer = psutil.Process(int(family.stdout.readline()))
            deadline = time.monotonic() + 10
            while not (directory / "ready").exists():
                assert observer.is_running() and time.monotonic() < deadline
                time.sleep(0.01)
            return observer

        yield SimpleNamespace(root=root, children=children, start_observer=start_observer)
    finally:
        try:
            family.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            for process in psutil.Process(family.pid).children(recursive=True):
                process.kill()
            family.kill()
            family.communicate(timeout=5)
        assert family.returncode == 0


def record_native_reads(monkeypatch, after_read=lambda: None):
    if sys.platform == "linux":
        module, name = memory, "_linux_nonshmem_pss"
    elif sys.platform == "darwin":
        from ouroboros.common import memory_macos as module
        name = "private_resident_bytes"
    else:
        from ouroboros.common import memory_windows as module
        name = "private_resident_bytes"
    native = getattr(module, name)
    reads = []

    def record(pid, *args):
        value = native(pid, *args)
        reads.append((pid, value))
        after_read()
        return value

    monkeypatch.setattr(module, name, record)
    return reads


def test_native_identity_serialization_preserves_psutil_equality_and_hash():
    # An explicit upgrade guard for the one private psutil boundary adapter.
    assert psutil.__version__ == "7.1.3"
    original = psutil.Process()
    token = json.loads(json.dumps(ipc.export_process_identity(original)))
    assert token == {"psutil_version": psutil.__version__, "platform": sys.platform,
                     "identity": list(original._ident)}
    restored = ipc.process_from_identity(token)
    assert restored is not original
    assert restored == original and hash(restored) == hash(original)
    assert len({original, restored, psutil.Process(original.pid)}) == 1
    assert ipc.matches_process_identity(restored, token)
    if sys.platform == "win32":
        assert tuple(token["identity"]) == (original.pid, original.create_time())


def test_native_clock_changes_preserve_root_token_and_process_identity(clock_offset):
    original = psutil.Process()
    original_created = original.create_time()
    token = json.loads(json.dumps(ipc.export_process_identity(original)))
    for offset in (4, -4, 30, -30, 0):
        clock_offset[0] = offset
        fresh = psutil.Process(original.pid)
        if offset:
            assert fresh.create_time() != original_created  # The old key really changes.
        assert fresh == original and hash(fresh) == hash(original)
        assert ipc.export_process_identity(fresh) == token
        assert ipc.process_from_identity(token) == original
        assert memory.TreeSampler(token).root == original


@pytest.mark.parametrize("change_during_sweep", [False, True])
def test_clock_changes_never_duplicate_process_reads_or_reinclude_observers(
        monkeypatch, clock_offset, process_family, change_during_sweep):
    family = process_family
    observer = family.start_observer()
    reader = memory.TreeSampler(ipc.export_process_identity(family.root))
    pids = {family.root.pid, *family.children}
    initial = reader._discover()
    assert {process.pid for process in initial} == pids
    assert reader.excluded == {observer}

    def drift():
        if change_during_sweep:
            clock_offset[0] += 2

    reads = record_native_reads(monkeypatch, drift)
    for offset in (0, 4, -4, 30, -30):
        clock_offset[0] = offset
        reads.clear()
        total, count = reader.sample()
        assert count == len(pids)
        assert Counter(pid for pid, _ in reads) == Counter({pid: 1 for pid in pids})
        assert total == sum(value for _, value in reads)
        assert reader.known == initial
        assert reader.excluded == {observer}


def test_managed_publication_survives_clock_change_before_discovery_and_updates(
        monkeypatch, clock_offset):
    from ouroboros.helpers.mem import SharedNPManager

    with managed.ManagerSession() as session:
        manager = SharedNPManager(ctx=multiprocessing.get_context("spawn"))
        manager.start()
        allocations = []
        try:
            first = manager.SharedMemory(4 * mmap.PAGESIZE)
            allocations.append(first)
            first.buf[:] = b"a" * first.size
            path = Path(session.context.directory, f"{manager._process.pid}.json")
            before = json.loads(path.read_text())
            token = ipc.export_process_identity(psutil.Process())
            clock_offset[0] = 4
            reader = memory.TreeSampler(token, managers_directory=session.context.directory)
            reads = record_native_reads(monkeypatch)
            resident = managed.resident_bytes
            observed_segments = []

            def observe_segment(segment):
                value = resident(segment)
                observed_segments.append((segment["name"], value))
                return value

            monkeypatch.setattr(managed, "resident_bytes", observe_segment)
            for offset in (4, -4):
                clock_offset[0] = offset
                reads.clear()
                observed_segments.clear()
                if offset < 0:
                    second = manager.SharedMemory(2 * mmap.PAGESIZE)
                    allocations.append(second)
                    second.buf[:] = b"b" * second.size
                    after = json.loads(path.read_text())
                    assert after["generation"] != before["generation"]
                    assert after["owner"] == before["owner"]
                total, count = reader.sample()
                assert count == len({pid for pid, _ in reads}) == len(reads)
                assert Counter(name for name, _ in observed_segments) == Counter(
                    {allocation.name: 1 for allocation in allocations})
                assert sum(value for _, value in observed_segments) == sum(
                    allocation.size for allocation in allocations)
                assert total == sum(value for _, value in reads + observed_segments)
                owner = next(process for process in reader.known if process.pid == manager._process.pid)
                assert ipc.matches_process_identity(owner, before["owner"])
            assert session.context.failure_flag.value == 0
        finally:
            for allocation in allocations:
                allocation.close()
            manager.shutdown()


def test_retained_process_identity_access_failure_is_incomplete_and_recovers(
        monkeypatch, process_family):
    reader = memory.TreeSampler(ipc.export_process_identity(process_family.root))
    initial = reader._discover()
    retained = next(process for process in initial if process.pid in process_family.children)
    # Simulate reparenting: the already-observed process remains alive but is no
    # longer rediscovered through this root. Its identity still must be checked.
    monkeypatch.setattr(reader.root, "children", lambda recursive: [
        process for process in initial if process != retained and process != reader.root])
    native_create_time = psutil._psplatform.Process.create_time

    def denied(process, *args, **kwargs):
        if process.pid == retained.pid:
            raise psutil.AccessDenied(process.pid)
        return native_create_time(process, *args, **kwargs)

    with monkeypatch.context() as denied_context:
        denied_context.setattr(psutil._psplatform.Process, "create_time", denied)
        assert psutil.Process(retained.pid)._ident == (retained.pid, None)
        with pytest.raises(ValueError, match="identity"):
            reader.sample()
        assert retained in reader.known
        assert not retained._pid_reused  # A transient access failure is not PID reuse.
    assert reader.sample()[1] == len(initial)
    assert reader.known == initial


@pytest.mark.parametrize("owner", [None, {"identity": [123, None]}])
def test_unverified_manager_owner_cannot_become_an_empty_contribution(tmp_path, owner):
    process = psutil.Process()
    memory._write_json(tmp_path / f"{process.pid}.json", {"owner": owner, "segments": []})
    reader = memory.TreeSampler(ipc.export_process_identity(process), managers_directory=tmp_path)
    with pytest.raises(ValueError, match="identity"):
        reader._segments(process)


@pytest.mark.parametrize("identity", [None, [], [1], [True, 1], [1, None],
                                      [1, float("nan")], [1, float("inf")], [1, -1]])
def test_unverified_serialized_identity_is_rejected(identity):
    token = ipc.export_process_identity(psutil.Process())
    token["identity"] = identity
    with pytest.raises(ValueError, match="identity"):
        ipc.process_from_identity(token)


@pytest.mark.parametrize("field,value", [("psutil_version", "7.2.0"), ("platform", "other")])
def test_unverified_token_contract_is_rejected(field, value):
    token = ipc.export_process_identity(psutil.Process())
    token[field] = value
    with pytest.raises(ValueError, match="identity"):
        ipc.process_from_identity(token)


@pytest.mark.parametrize("field", [None, "identity", "platform", "psutil_version"])
def test_missing_token_or_required_field_is_rejected(field):
    token = ipc.export_process_identity(psutil.Process())
    if field is None:
        token = None
    else:
        token.pop(field)
    with pytest.raises(ValueError, match="identity"):
        ipc.process_from_identity(token)


def test_missing_psutil_private_identity_is_explicit(monkeypatch):
    process = psutil.Process()
    monkeypatch.delattr(process, "_ident")
    with pytest.raises(ValueError, match="Process._ident"):
        ipc.export_process_identity(process)


@pytest.mark.parametrize("change", ["reused", "exited"])
def test_checked_liveness_rejects_native_pid_reuse_and_exit(monkeypatch, change):
    process = psutil.Process()
    native_create_time = psutil._psplatform.Process.create_time
    assert ipc.process_is_running(process)

    def changed(native_process, *args, **kwargs):
        if native_process.pid == process.pid:
            if change == "exited":
                raise psutil.NoSuchProcess(process.pid)
            return native_create_time(native_process, *args, **kwargs) + 1
        return native_create_time(native_process, *args, **kwargs)

    with monkeypatch.context() as native_query:
        native_query.setattr(psutil._psplatform.Process, "create_time", changed)
        if change == "reused":
            assert psutil.Process(process.pid) != process  # Real psutil equality.
        assert not ipc.process_is_running(process)
    assert ipc.process_is_running(process)


def test_unverified_native_identity_and_dependency_upgrade_are_explicit(monkeypatch):
    process = psutil.Process()
    monkeypatch.setattr(process, "_ident", (process.pid, None))
    with pytest.raises(ValueError, match="identity"):
        ipc.export_process_identity(process)
    monkeypatch.setattr(psutil, "__version__", "7.2.0")
    with pytest.raises(RuntimeError, match="requires psutil"):
        ipc.export_process_identity(psutil.Process())

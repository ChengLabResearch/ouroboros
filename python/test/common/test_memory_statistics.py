import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import weakref

import numpy as np
import psutil
import pytest

from ouroboros.common import memory_observer, memory_statistics as memory


def retry_clock(monkeypatch, *, system="win32"):
    elapsed = [0.0]
    sleeps = []

    def sleep(delay):
        sleeps.append(delay)
        elapsed[0] += delay

    monkeypatch.setattr(memory.sys, "platform", system)
    monkeypatch.setattr(memory.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(memory.time, "sleep", sleep)
    return elapsed, sleeps


def test_control_read_retries_transient_windows_publication_denial(monkeypatch, tmp_path):
    path = tmp_path / "start.json"
    path.write_text("12.5", encoding="utf-8")
    elapsed, sleeps = retry_clock(monkeypatch)
    read = Path.read_text
    attempts = []

    def briefly_locked(self, *args, **kwargs):
        attempts.append(self)
        if len(attempts) < 3:
            raise PermissionError("publication handle has not closed")
        return read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", briefly_locked)
    assert memory._read_json(path) == 12.5
    assert attempts == [path] * 3
    assert sleeps == [0.01, 0.01]
    assert elapsed[0] == pytest.approx(0.02)


def test_control_write_retries_windows_replace_without_exposing_partial_json(monkeypatch, tmp_path):
    path = tmp_path / "result.json"
    path.write_text('{"old": true}', encoding="utf-8")
    _, sleeps = retry_clock(monkeypatch)
    replace = Path.replace
    attempts = []

    def briefly_locked(self, target):
        attempts.append(self)
        if len(attempts) == 1:
            assert json.loads(path.read_text()) == {"old": True}
            raise PermissionError("reader still holds the destination")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", briefly_locked)
    memory._write_json(path, {"new": "complete"})
    assert memory._read_json(path) == {"new": "complete"}
    assert sleeps == [0.01]
    assert not path.with_suffix(".tmp").exists()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_control_file_persistent_windows_denial_has_a_short_bound(monkeypatch, tmp_path, operation):
    elapsed, sleeps = retry_clock(monkeypatch)
    denied = PermissionError("actual access denied")

    def fail(*args, **kwargs):
        raise denied

    monkeypatch.setattr(Path, "read_text" if operation == "read" else "write_text", fail)
    with pytest.raises(PermissionError) as caught:
        if operation == "read":
            memory._read_json(tmp_path / "config.json")
        else:
            memory._write_json(tmp_path / "stop.json", 12.5)
    assert caught.value is denied
    assert elapsed[0] == pytest.approx(0.5)
    assert len(sleeps) <= 51


@pytest.mark.parametrize("system,error", [
    ("linux", PermissionError("denied")),
    ("darwin", PermissionError("denied")),
    ("win32", FileNotFoundError("missing")),
])
def test_control_file_other_failures_do_not_retry(monkeypatch, tmp_path, system, error):
    _, sleeps = retry_clock(monkeypatch, system=system)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(type(error)):
        memory._read_json(tmp_path / "start.json")
    assert sleeps == []


def test_control_file_malformed_json_is_not_hidden_by_retry(monkeypatch, tmp_path):
    path = tmp_path / "config.json"
    path.write_text("invalid", encoding="utf-8")
    _, sleeps = retry_clock(monkeypatch)
    with pytest.raises(json.JSONDecodeError):
        memory._read_json(path)
    assert sleeps == []


class Process:
    def __init__(self, pid, created=1.0, pss=10, command=None):
        self.pid, self.created, self.pss = pid, created, pss
        self.command = command or ["python", "worker.py"]
        self.running = True
        self.descendants = []
        self.on_read = lambda: None

    def create_time(self):
        return self.created

    def is_running(self):
        return self.running

    def children(self, recursive):
        assert recursive
        return self.descendants

    def cmdline(self):
        return self.command

    def memory_full_info(self):
        self.on_read()
        return SimpleNamespace(pss=self.pss, rss=999999)


def sampler(monkeypatch, *, system="Linux", directory=None):
    root = Process(123)
    monkeypatch.setattr(memory.psutil, "Process", lambda pid=None: root)
    monkeypatch.setattr(memory.platform, "system", lambda: system)
    return root, memory.TreeSampler(root.pid, root.created, mappings_directory=directory)


def manifest(directory, process, mappings=(), errors=()):
    memory._write_json(Path(directory) / f"{process.pid}.json", {
        "pid": process.pid, "create_time": process.created,
        "mappings": list(mappings), "errors": list(errors),
    })


def test_constructor_rejects_reused_backend_pid(monkeypatch):
    monkeypatch.setattr(memory.psutil, "Process", lambda pid: Process(pid, created=2))
    with pytest.raises(psutil.NoSuchProcess, match="reused"):
        memory.TreeSampler(123, 1)


def test_tree_sums_current_pss_and_excludes_all_memory_observers(monkeypatch):
    root, reader = sampler(monkeypatch)
    child = Process(124, pss=20)
    root.descendants = [child,
                        Process(125, pss=9999, command=["python", "-m", memory._OBSERVER_MODULE, "/tmp"]),
                        Process(126, pss=9999, command=["frozen", memory._OBSERVER_FLAG, "/tmp"])]
    assert reader.sample() == (30, 2)
    child.pss = 2
    root.descendants = []  # Already-observed live children survive reparenting.
    assert reader.sample() == (12, 2)
    child.running = False
    assert reader.sample() == (10, 1)
    assert (child.pid, child.created) not in reader.known


def test_reused_child_pid_is_a_new_identity(monkeypatch):
    root, reader = sampler(monkeypatch)
    old = Process(124, created=1, pss=20)
    root.descendants = [old]
    assert reader.sample() == (30, 2)
    old.running = False
    new = Process(124, created=2, pss=3)
    root.descendants = [new]
    assert reader.sample() == (13, 2)
    assert set(reader.known) == {(123, 1.0), (124, 2)}


@pytest.mark.parametrize("change", ["appeared", "exited"])
def test_process_churn_during_a_sweep_is_incomplete(monkeypatch, change):
    root, reader = sampler(monkeypatch)
    if change == "appeared":
        root.on_read = lambda: root.descendants.append(Process(124))
    else:
        root.on_read = lambda: setattr(root, "running", False)
    with pytest.raises((OSError, psutil.NoSuchProcess)):
        reader.sample()


def test_access_denied_does_not_become_a_zero_sample(monkeypatch):
    root, reader = sampler(monkeypatch)

    def denied():
        raise psutil.AccessDenied(root.pid)

    root.on_read = denied
    with pytest.raises(psutil.AccessDenied):
        reader.sample()


def test_windows_shared_offsets_are_unioned_across_processes(monkeypatch, tmp_path):
    from ouroboros.common import memory_windows

    root, reader = sampler(monkeypatch, system="Windows", directory=tmp_path)
    root.descendants = [Process(124)]
    monkeypatch.setattr(memory_windows, "PAGE_SIZE", 4096)
    monkeypatch.setattr(memory_windows, "read_process", lambda pid, mappings: (
        10, {("shm", "allocation"): np.array([0, 4096]) if pid == 123 else np.array([4096, 8192])},
    ))
    assert reader.sample() == (20 + 3 * 4096, 2)


def test_macos_reuses_identical_queries_and_unions_backing_offsets(monkeypatch, tmp_path):
    from ouroboros.common import memory_macos

    root, reader = sampler(monkeypatch, system="Darwin", directory=tmp_path)
    child = Process(124)
    root.descendants = [child]
    mapping = {"identity": ["shm", "allocation"], "offset": 0, "size": 8192}
    manifest(tmp_path, root, [mapping])
    manifest(tmp_path, child, [mapping])
    calls = []
    monkeypatch.setattr(memory_macos, "PAGE_SIZE", 4096)
    monkeypatch.setattr(memory_macos, "private_resident_bytes", lambda pid, mappings: 10)

    def resident(value):
        calls.append(value)
        return np.array([0, 4096])

    monkeypatch.setattr(memory_macos, "resident_pages", resident)
    assert reader.sample() == (20 + 8192, 2)
    assert calls == [mapping]


def test_mapping_change_during_a_sweep_is_incomplete(monkeypatch, tmp_path):
    root, reader = sampler(monkeypatch, directory=tmp_path)
    manifest(tmp_path, root)
    root.on_read = lambda: manifest(tmp_path, root, [{"new": "mapping"}])
    with pytest.raises(OSError, match="observations changed"):
        reader.sample()


def test_stale_manifest_cannot_supply_another_process_mapping(monkeypatch, tmp_path):
    root, reader = sampler(monkeypatch, directory=tmp_path)
    manifest(tmp_path, Process(root.pid, created=0), [{"stale": "mapping"}])
    mappings, generation = reader._mappings((root.pid, root.created))
    assert mappings == [] and generation is not None


def test_failed_mapping_observation_is_explicit(monkeypatch, tmp_path):
    root, reader = sampler(monkeypatch, directory=tmp_path)
    manifest(tmp_path, root, errors=["cannot publish mapping"])
    with pytest.raises(OSError, match="observation failed"):
        reader.sample()


def test_samples_use_only_complete_sweeps_inside_final_window():
    samples = memory.Samples(1)
    samples.add(0.9, 1.1, 0.1, (999, 1))  # Crosses the start boundary.
    samples.add(1.2, 1.4, 0.1, (20, 2))
    samples.add(2.2, 2.4, 0.1, (10, 1))
    samples.add(3.2, 3.8, 0.3, (9999, 5))  # Stop marker arrived after add().
    result = samples.record(1, 3.5, collector_cpu_seconds=0.9)
    assert result["status"] == "sampled"
    assert result["sampled_peak_bytes"] == 20
    assert result["last_complete_sample_bytes"] == 10
    details = result["sampling"]
    assert details["complete_samples"] == 2
    assert details["discarded_samples"] == 2
    assert details["collector_cpu_seconds"] == 0.9
    assert details["max_process_count"] == 2
    assert details["observed_interval_seconds"] == pytest.approx({"min": 1, "max": 1})


def test_no_sample_is_unavailable_but_measured_zero_is_valid():
    samples = memory.Samples(1)
    assert samples.record(0, 0.1)["sampled_peak_bytes"] is None
    samples.add(0, 0.05, 0, (0, 1))
    result = samples.record(0, 0.1)
    assert result["sampled_peak_bytes"] == 0
    assert result["status"] == "sampled"


def test_failed_samples_preserve_coverage_without_retaining_tracebacks():
    class Payload:
        pass

    def failure():
        payload = Payload()
        try:
            raise OSError("x" * 1000)
        except OSError as error:
            return error, weakref.ref(payload)

    samples = memory.Samples(1)
    error, reference = failure()
    samples.add(0, 0.1, 0.02, error=error)
    del error
    gc.collect()
    assert reference() is None
    samples.add(1, 1.1, 0.02, (10, 1))
    result = samples.record(0, 2)
    assert result["status"] == "partial"
    assert result["sampling"]["incomplete_samples"] == 1
    assert len(result["errors"][0]["example"]) == 500


def test_observer_discards_sweep_when_stop_publication_was_delayed(monkeypatch, tmp_path):
    memory._write_json(tmp_path / "config.json", {
        "root_pid": 123, "root_created": 1, "interval": 1, "mappings_directory": None,
    })
    memory._write_json(tmp_path / "start.json", 0)
    reader = SimpleNamespace(system="Linux", root=Process(123), sample=lambda: (999, 1))
    monkeypatch.setattr(memory, "TreeSampler", lambda *args, **kwargs: reader)

    class DelayedStop(memory.Samples):
        def add(self, *args):
            super().add(*args)
            # Represents a parent write whose timestamp predates the sweep's
            # completion but whose atomic publication arrives afterward.
            memory._write_json(tmp_path / "stop.json", 1.5)

    monkeypatch.setattr(memory, "Samples", DelayedStop)
    clocks = iter([1.0, 2.0])
    cpu = iter([10.0, 10.1, 10.4, 10.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clocks))
    monkeypatch.setattr(time, "process_time", lambda: next(cpu))
    memory_observer.main(tmp_path)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["sampled_peak_bytes"] is None
    assert result["sampling"]["discarded_samples"] == 1
    assert result["sampling"]["collector_cpu_seconds"] == pytest.approx(0.5)


class Child:
    def __init__(self, directory, *, timeout=False):
        self.directory = Path(directory)
        self.timeout = timeout
        self.returncode = None
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        if self.timeout and not self.killed:
            raise subprocess.TimeoutExpired("observer", timeout)
        self.returncode = 0
        memory._write_json(self.directory / "result.json", {
            "status": "sampled", "sampled_peak_bytes": 10,
            "last_complete_sample_bytes": 10, "errors": [],
        })
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def start_collector(monkeypatch, *, system="Linux", timeout=False, failure_flag=0):
    monkeypatch.setattr(memory.platform, "system", lambda: system)
    session = SimpleNamespace(context=SimpleNamespace(
        directory="unused", failure_flag=SimpleNamespace(value=failure_flag)), released=False)
    session.acquire = lambda: session
    session.release = lambda: setattr(session, "released", True)
    if system != "Linux":
        from ouroboros.common import memory_mappings
        monkeypatch.setattr(memory_mappings, "MappingSession", lambda: session)
    invocation = {}

    def popen(command, **kwargs):
        invocation.update(command=command, options=kwargs)
        child = Child(command[-1], timeout=timeout)
        (child.directory / "ready").touch()
        invocation["child"] = child
        return child

    monkeypatch.setattr(memory.subprocess, "Popen", popen)
    collector = memory.MemoryCollector()
    collector.start()
    return collector, invocation, session


def test_collector_lifecycle_uses_light_observer_and_cleans_workspace(monkeypatch):
    collector, invocation, _ = start_collector(monkeypatch)
    directory = Path(collector.workspace.name)
    assert (directory / "start.json").exists()
    assert invocation["command"][1:3] == ["-m", memory._OBSERVER_MODULE]
    assert invocation["options"]["env"]["OPENBLAS_NUM_THREADS"] == "1"
    collector.stop()
    assert collector.summary["sampled_peak_bytes"] == 10
    assert collector.process is collector.workspace is None
    assert not directory.exists()
    collector.stop()
    assert collector.summary["sampled_peak_bytes"] == 10


def test_double_start_does_not_orphan_the_original_observer(monkeypatch):
    collector, invocation, _ = start_collector(monkeypatch)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            collector.start()
        assert collector.process is invocation["child"]
    finally:
        collector.stop()


def test_observer_timeout_kills_child_and_releases_all_resources(monkeypatch):
    collector, invocation, session = start_collector(monkeypatch, system="Darwin", timeout=True)
    directory = Path(collector.workspace.name)
    collector.stop()
    assert collector.summary["status"] == "unavailable"
    assert invocation["child"].terminated and invocation["child"].killed
    assert session.released and not directory.exists()
    assert collector.mapping_context is None


def test_unpublished_mapping_failure_invalidates_sampled_peak(monkeypatch):
    collector, _, session = start_collector(monkeypatch, system="Darwin", failure_flag=1)
    collector.stop()
    assert collector.summary["status"] == "unavailable"
    assert collector.summary["sampled_peak_bytes"] is None
    assert collector.summary["errors"][-1]["kind"] == "mapping_observation"
    assert session.released


def test_spawn_failure_releases_mapping_session_and_workspace(monkeypatch):
    from ouroboros.common import memory_mappings

    monkeypatch.setattr(memory.platform, "system", lambda: "Darwin")
    session = SimpleNamespace(context=SimpleNamespace(directory="unused"), released=False)
    session.acquire = lambda: session
    session.release = lambda: setattr(session, "released", True)
    monkeypatch.setattr(memory_mappings, "MappingSession", lambda: session)

    def fail(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(memory.subprocess, "Popen", fail)
    collector = memory.MemoryCollector()
    with pytest.raises(OSError, match="cannot spawn"):
        collector.start()
    assert session.released
    assert collector.process is collector.workspace is collector.mapping_session is None


def test_startup_failure_preserves_the_native_error(monkeypatch):
    monkeypatch.setattr(memory.platform, "system", lambda: "Linux")
    directories = []

    def failed_start(command, **kwargs):
        child = Child(command[-1])
        child.returncode = 1
        directories.append(child.directory)
        memory._write_json(child.directory / "result.json", memory.unavailable("native access denied"))
        return child

    monkeypatch.setattr(memory.subprocess, "Popen", failed_start)
    collector = memory.MemoryCollector()
    with pytest.raises(RuntimeError, match="native access denied"):
        collector.start()
    assert collector.summary["errors"][0]["example"] == "native access denied"
    assert collector.process is collector.workspace is None
    assert not directories[0].exists()


def test_importing_observer_does_not_import_the_application():
    script = ("import sys; import ouroboros.common.memory_observer; "
              "import ouroboros.common.memory_statistics; "
              "assert 'ouroboros.cli' not in sys.modules; "
              "assert 'ouroboros.server' not in sys.modules; "
              "assert 'ouroboros.pipeline' not in sys.modules; "
              "assert 'numpy' not in sys.modules")
    subprocess.run([sys.executable, "-c", script],
                   cwd=Path(__file__).resolve().parents[2], check=True, timeout=10)


@pytest.mark.skipif(sys.platform not in ("linux", "win32", "darwin"), reason="Supported native backends")
def test_native_collector_samples_and_cleans_up():
    # An isolated backend keeps this a short controller/adapter smoke test,
    # independent of memory retained by the rest of the pytest session.
    script = """
import json
from pathlib import Path
import time
from ouroboros.common.memory_statistics import MemoryCollector

collector = MemoryCollector(interval=0.1)
collector.start()
observer = collector.process
workspace = Path(collector.workspace.name)
mappings = Path(collector.mapping_context.directory) if collector.mapping_context else None
try:
    time.sleep(0.3)
finally:
    collector.stop()
assert observer.poll() is not None
assert collector.process is collector.workspace is collector.mapping_session is None
assert not workspace.exists()
assert mappings is None or not mappings.exists()
print(json.dumps(collector.summary))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
        check=True, timeout=25, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    assert result["metric"] == memory._METRICS[memory.platform.system()][0]
    assert result["status"] == "sampled", result
    assert result["sampled_peak_bytes"] > 0
    assert result["sampling"]["complete_samples"] > 0
    assert result["sampling"]["collector_cpu_seconds"] > 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PSS and fork semantics")
def test_linux_tree_pss_counts_a_live_shared_mapping_once(tmp_path):
    size = 16 * 1024 * 1024
    backing = tmp_path / "shared-pages.bin"
    with backing.open("wb") as output:
        output.truncate(size)
    # Both live processes reuse and fault the same mapping. Readiness pipes
    # prevent a timing-dependent sample of a child that has not touched it yet.
    script = """
import mmap, os, sys
backing = open(sys.argv[1], 'r+b')
mapping = mmap.mmap(backing.fileno(), 0)
for index in range(0, len(mapping), mmap.PAGESIZE):
    mapping[index] = 1
ready_read, ready_write = os.pipe()
finish_read, finish_write = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(ready_read)
    os.close(finish_write)
    assert sum(mapping[index] for index in range(0, len(mapping), mmap.PAGESIZE)) > 0
    os.write(ready_write, b'1')
    os.read(finish_read, 1)
    os._exit(0)
os.close(ready_write)
os.close(finish_read)
assert os.read(ready_read, 1) == b'1'
print(pid, flush=True)
sys.stdin.read(1)
os.write(finish_write, b'1')
os.waitpid(pid, 0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(backing)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    child = None
    try:
        child = psutil.Process(int(process.stdout.readline()))
        root = psutil.Process(process.pid)
        reader = memory.TreeSampler(root.pid, root.create_time())
        value, count = reader.sample()
        assert count == 2
        details = [item.memory_full_info() for item in (root, child)]
        assert value == pytest.approx(sum(item.pss for item in details), abs=1024 * 1024)
        assert sum(item.rss for item in details) - value >= 0.75 * size
        mapped = [region for item in (root, child) for region in item.memory_maps(grouped=False)
                  if region.path == str(backing)]
        assert len(mapped) == 2
        assert sum(region.rss for region in mapped) == 2 * size
        assert sum(region.pss for region in mapped) == size
    finally:
        try:
            process.communicate("x", timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
        if child is not None and child.is_running():
            child.kill()
        assert process.returncode == 0

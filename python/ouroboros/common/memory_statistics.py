"""Operation-scoped observations of current process-tree memory.

No task, allocation, or executor state is owned here. Native objects and the OS
remain authoritative; the collector owns only samples and their coverage.
"""

from collections import Counter
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import psutil

from ouroboros.common.memory_ipc import observation_file_io


DEFAULT_INTERVAL_SECONDS = 1.0
_OBSERVER_MODULE = "ouroboros.common.memory_observer"
_OBSERVER_FLAG = "--ouroboros-memory-observer"
_METRICS = {
    "Linux": ("process_tree_nonshmem_pss_plus_managed_resident_bytes", "smaps_rollup Pss - Pss_Shmem + managed segment mincore"),
    "Windows": ("process_tree_private_resident_plus_managed_allocated_bytes", "PrivateWorkingSetSize + manager-owned allocation sizes"),
    "Darwin": ("process_tree_private_plus_managed_resident_bytes", "libproc private residency + managed segment mincore"),
}


def unavailable(reason, *, status="unavailable"):
    metric, method = _METRICS.get(platform.system(), (None, None))
    return {"status": status, "metric": metric, "unit": "bytes", "method": method,
            "scope": "backend parent and observed descendants; sampler excluded; overlapping jobs not isolated",
            "sampled_peak_bytes": None, "last_complete_sample_bytes": None,
            "sampling": None, "errors": [{"kind": "collector", "count": 1, "example": reason}]}



def _read_json(path):
    path = Path(path)
    return observation_file_io(lambda: json.loads(path.read_text(encoding="utf-8")))


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    payload = json.dumps(value)

    def publish():
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)

    observation_file_io(publish)


def _identity(process):
    return process.pid, process.create_time()


def _observer(process):
    command = process.cmdline()
    return (len(command) >= 3 and command[1:3] == ["-m", _OBSERVER_MODULE]
            or len(command) >= 2 and command[1] == _OBSERVER_FLAG)


def _linux_nonshmem_pss(pid):
    # PSS includes single-reader shared pages as well as multiply mapped ones.
    # Subtract the entire shared-memory contribution before adding each live
    # managed backing once, including buffers queued without a reader.
    values = {}
    with Path(psutil.PROCFS_PATH, str(pid), "smaps_rollup").open() as source:
        for line in source:
            field, _, value = line.partition(":")
            if field in ("Pss", "Pss_Shmem"):
                values[field] = int(value.split()[0]) * 1024
    if values.keys() != {"Pss", "Pss_Shmem"}:
        raise NotImplementedError("Linux smaps_rollup requires Pss and Pss_Shmem")
    return values["Pss"] - values["Pss_Shmem"]


class TreeSampler:
    """Retain observed identities, never individual-process lifetime peaks."""

    def __init__(self, root_pid, root_created, *, managers_directory=None):
        self.root = psutil.Process(root_pid)
        if self.root.create_time() != root_created:
            raise psutil.NoSuchProcess(root_pid, msg="Backend PID was reused")
        self.known = {_identity(self.root): self.root}
        self.excluded = set()
        self.managers_directory = Path(managers_directory) if managers_directory else None
        self.system = platform.system()
        if self.system not in _METRICS:
            raise NotImplementedError(f"Memory collection is unavailable on {self.system}")

    def _discover(self):
        if not self.root.is_running():
            raise psutil.NoSuchProcess(self.root.pid)
        discovered = set()
        for child in self.root.children(recursive=True):
            identity = _identity(child)
            discovered.add(identity)
            if identity in self.excluded:
                continue
            if identity not in self.known:
                if _observer(child):
                    self.excluded.add(identity)
                    continue
                self.known[identity] = child
        # A child that ended between samples is normal; one ending during a read
        # makes that sweep incomplete. Already observed reparented children stay.
        self.known = {identity: process for identity, process in self.known.items() if process.is_running()}
        self.excluded.intersection_update(discovered)
        return dict(self.known)

    def _segments(self, identity):
        if self.managers_directory is None:
            return [], None
        path = self.managers_directory / f"{identity[0]}.json"
        try:
            raw = observation_file_io(path.read_bytes)
        except FileNotFoundError:
            return [], None  # Only shared-memory managers publish a manifest.
        record = json.loads(raw)
        if (record["pid"], record["create_time"]) != identity:
            return [], raw  # Stale file from a different process identity.
        if record.get("errors"):
            raise OSError(f"Manager observation failed: {record['errors']}")
        return record["segments"], raw

    def sample(self):
        from ouroboros.common.memory_managed import resident_bytes

        processes = self._discover()
        total = 0
        observed = {}
        segments = {}
        if self.system == "Windows":
            from ouroboros.common import memory_windows as native
        elif self.system == "Darwin":
            from ouroboros.common import memory_macos as native
        for identity, process in processes.items():
            if not process.is_running():
                raise psutil.NoSuchProcess(process.pid)
            owned, generation = self._segments(identity)
            observed[identity] = generation
            for segment in owned:
                previous = segments.setdefault(segment["name"], segment)
                if previous["size"] != segment["size"]:
                    raise OSError("Conflicting managed segment observations")
            if self.system == "Linux":
                total += _linux_nonshmem_pss(process.pid)
            elif self.system == "Windows":
                total += native.private_resident_bytes(process.pid)
            else:
                total += native.private_resident_bytes(process.pid, owned)
            if not process.is_running():
                raise psutil.NoSuchProcess(process.pid)
        total += sum(segment["size"] if self.system == "Windows" else resident_bytes(segment)
                     for segment in segments.values())
        # Allocation/release and process churn invalidate a sweep, rather than
        # mixing observations from different owners or recycled process IDs.
        for identity in processes:
            if self._segments(identity)[1] != observed[identity]:
                raise OSError("Manager observations changed during memory sweep")
        after = self._discover()
        if set(after) != set(processes):
            raise OSError("Process tree changed during memory sweep")
        return int(total), len(processes)


class Samples:
    """Own scalar observations; reduce them only after the window is known.

    The stop timestamp is published by another process. Keeping observations
    until it arrives prevents a delayed write from admitting a post-window
    sweep. No page arrays or exception tracebacks are retained here.
    """

    def __init__(self, interval):
        self.interval = interval
        self.observations = []

    def add(self, start, end, cpu, result=None, error=None):
        failure = None if error is None else (type(error).__name__, str(error)[:500])
        self.observations.append((start, end, cpu, result if failure is None else None, failure))

    def record(self, start, end, *, collector_cpu_seconds=None):
        included = [row for row in self.observations if start <= row[0] <= row[1] <= end]
        discarded = [row for row in self.observations if not start <= row[0] <= row[1] <= end]
        results = [row[3] for row in included if row[4] is None]
        failures = [row[4] for row in included if row[4] is not None]
        intervals = [right[0] - left[0] for left, right in zip(included, included[1:])]
        errors = Counter(failure[0] for failure in failures)
        examples = {}
        for kind, message in failures:
            examples.setdefault(kind, message)
        record = unavailable("No complete sample within operation window")
        record.update(
            status="partial" if results and failures else "sampled" if results else "unavailable",
            sampled_peak_bytes=max((value for value, _ in results), default=None),
            last_complete_sample_bytes=results[-1][0] if results else None,
            sampling={
                "requested_interval_seconds": self.interval,
                "observed_interval_seconds": {"min": min(intervals, default=None),
                                               "max": max(intervals, default=None)},
                "complete_samples": len(results), "incomplete_samples": len(failures),
                "discarded_samples": len(discarded),
                "max_process_count": max((count for _, count in results), default=0),
                "max_sweep_seconds": max((row[1] - row[0] for row in self.observations), default=0),
                "collector_cpu_seconds": (sum(row[2] for row in self.observations)
                                          if collector_cpu_seconds is None else collector_cpu_seconds),
                "window_seconds": end - start,
                "first_sample_start_seconds": included[0][0] - start if included else None,
                "last_sample_end_seconds": included[-1][1] - start if included else None,
            },
        )
        if results or errors:
            record["errors"] = [{"kind": key, "count": count, "example": examples[key]}
                                for key, count in sorted(errors.items())]
        return record


class MemoryCollector:
    """RunStatistics-owned controller; no collection thread in the backend."""

    def __init__(self, interval=DEFAULT_INTERVAL_SECONDS):
        if interval <= 0:
            raise ValueError("Memory sampling interval must be positive")
        self.interval = interval
        self.process = self.workspace = self.manager_session = None
        self.summary = unavailable("Pipeline has not started", status="not_started")

    @property
    def manager_context(self):
        return self.manager_session.context if self.manager_session is not None else None

    def start(self):
        if self.process is not None or self.workspace is not None:
            raise RuntimeError("Memory collector is already running")
        try:
            from ouroboros.common.memory_managed import ManagerSession
            self.manager_session = ManagerSession().acquire()
            self.workspace = tempfile.TemporaryDirectory(prefix="ouroboros-memory-")
            directory = Path(self.workspace.name)
            _write_json(directory / "config.json", {
                "root_pid": os.getpid(), "root_created": psutil.Process().create_time(),
                "interval": self.interval,
                "managers_directory": self.manager_context.directory if self.manager_context else None,
            })
            command = ([sys.executable, _OBSERVER_FLAG] if getattr(sys, "frozen", False)
                       else [sys.executable, "-m", _OBSERVER_MODULE])
            command.append(str(directory))
            environment = os.environ.copy()
            # Source checkouts/tests need the same import root in the light child.
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
                str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH"),
            ]))
            # Avoid importing NumPy with a new BLAS thread pool in the observer.
            for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                environment[name] = "1"
            options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                       if os.name == "nt" else {"start_new_session": True})
            with (directory / "stderr.txt").open("wb") as error_log:
                self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                                stdout=subprocess.DEVNULL, stderr=error_log,
                                                env=environment, **options)
            deadline = time.monotonic() + 10
            while not (directory / "ready").exists():
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    result_path = directory / "result.json"
                    if result_path.exists():
                        self.summary = _read_json(result_path)
                        reason = "; ".join(error["example"] for error in self.summary["errors"])
                    else:
                        reason = (directory / "stderr.txt").read_text(errors="replace")[-1000:]
                    raise RuntimeError("Memory observer did not become ready: " +
                                       reason)
                time.sleep(0.01)
            _write_json(directory / "start.json", time.perf_counter())
        except BaseException:
            self._cleanup()
            raise

    def stop(self):
        if self.process is None:
            return
        directory = Path(self.workspace.name)
        try:
            _write_json(directory / "stop.json", time.perf_counter())
            self.process.wait(timeout=10)
            self.summary = _read_json(directory / "result.json")
            if self.manager_context is not None and self.manager_context.failure_flag.value:
                self.summary.update(status="unavailable", sampled_peak_bytes=None, last_complete_sample_bytes=None)
                self.summary["errors"].append({"kind": "manager_observation", "count": 1,
                                               "example": "A manager snapshot could not be published; coverage is unknown"})
        except Exception as error:
            self.summary = unavailable(f"Memory observer failed: {type(error).__name__}: {error}")
        finally:
            self._cleanup()

    def _cleanup(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        finally:
            self.process = None
            try:
                if self.manager_session is not None:
                    self.manager_session.release()
            finally:
                self.manager_session = None
                if self.workspace is not None:
                    workspace = self.workspace
                    self.workspace = None
                    workspace.cleanup()

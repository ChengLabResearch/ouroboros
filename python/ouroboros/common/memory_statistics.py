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
    "Linux": ("process_tree_pss_bytes", "psutil memory_full_info PSS"),
    "Windows": ("process_tree_private_plus_tracked_shared_resident_bytes", "QueryWorkingSet and tracked allocation/page union"),
    "Darwin": ("process_tree_private_plus_tracked_backing_resident_bytes", "libproc private residency and tracked backing mincore union"),
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


class TreeSampler:
    """Retain observed identities, never individual-process lifetime peaks."""

    def __init__(self, root_pid, root_created, *, mappings_directory=None):
        self.root = psutil.Process(root_pid)
        if self.root.create_time() != root_created:
            raise psutil.NoSuchProcess(root_pid, msg="Backend PID was reused")
        self.known = {_identity(self.root): self.root}
        self.excluded = set()
        self.mappings_directory = Path(mappings_directory) if mappings_directory else None
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

    def _mappings(self, identity):
        if self.mappings_directory is None:
            return [], None
        path = self.mappings_directory / f"{identity[0]}.json"
        try:
            raw = observation_file_io(path.read_bytes)
        except FileNotFoundError:
            return [], None  # Helpers without mapping owners need no manifest.
        record = json.loads(raw)
        if (record["pid"], record["create_time"]) != identity:
            return [], raw  # Stale file from a different process identity.
        if record.get("errors"):
            raise OSError(f"Mapping observation failed: {record['errors']}")
        return record["mappings"], raw

    def sample(self):
        processes = self._discover()
        total = 0
        shared = {}
        observed = {}
        if self.system != "Linux":
            import numpy as np
        if self.system == "Windows":
            from ouroboros.common import memory_windows as native
        elif self.system == "Darwin":
            from ouroboros.common import memory_macos as native
        for identity, process in processes.items():
            if not process.is_running():
                raise psutil.NoSuchProcess(process.pid)
            mappings, generation = self._mappings(identity)
            if self.system == "Linux":
                total += process.memory_full_info().pss
            elif self.system == "Windows":
                private, resident = native.read_process(process.pid, mappings)
                total += private
                for key, pages in resident.items():
                    shared.setdefault(key, []).append(pages)
            else:
                total += native.private_resident_bytes(process.pid, mappings)
                # Query each live mapping range once. Aliases with different
                # ranges still union by the allocation's backing byte offsets.
                for mapping in mappings:
                    key = tuple(mapping["identity"])
                    query = (key, mapping["offset"], mapping["size"])
                    if query not in observed:
                        shared.setdefault(key, []).append(native.resident_pages(mapping))
                        observed[query] = True
            observed[identity] = generation
            if not process.is_running():
                raise psutil.NoSuchProcess(process.pid)
        # A mapping or process created/closed during a sweep invalidates its
        # provenance; do not silently attribute recycled addresses or omit it.
        for identity in processes:
            if self._mappings(identity)[1] != observed[identity]:
                raise OSError("Mapping observations changed during memory sweep")
        after = self._discover()
        if set(after) != set(processes):
            raise OSError("Process tree changed during memory sweep")
        if self.system != "Linux":
            total += sum(int(np.unique(parts[0] if len(parts) == 1 else np.concatenate(parts)).size)
                         * native.PAGE_SIZE for parts in shared.values())
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
        self.process = self.workspace = self.mapping_session = None
        self.summary = unavailable("Pipeline has not started", status="not_started")

    @property
    def mapping_context(self):
        return self.mapping_session.context if self.mapping_session is not None else None

    def start(self):
        if self.process is not None or self.workspace is not None:
            raise RuntimeError("Memory collector is already running")
        try:
            if platform.system() in ("Windows", "Darwin"):
                from ouroboros.common.memory_mappings import MappingSession
                self.mapping_session = MappingSession().acquire()
            self.workspace = tempfile.TemporaryDirectory(prefix="ouroboros-memory-")
            directory = Path(self.workspace.name)
            _write_json(directory / "config.json", {
                "root_pid": os.getpid(), "root_created": psutil.Process().create_time(),
                "interval": self.interval,
                "mappings_directory": self.mapping_context.directory if self.mapping_context else None,
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
            if self.mapping_context is not None and self.mapping_context.failure_flag.value:
                self.summary.update(status="unavailable", sampled_peak_bytes=None, last_complete_sample_bytes=None)
                self.summary["errors"].append({"kind": "mapping_observation", "count": 1,
                                               "example": "A mapping manifest could not be published; coverage is unknown"})
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
                if self.mapping_session is not None:
                    self.mapping_session.release()
            finally:
                self.mapping_session = None
                if self.workspace is not None:
                    workspace = self.workspace
                    self.workspace = None
                    workspace.cleanup()

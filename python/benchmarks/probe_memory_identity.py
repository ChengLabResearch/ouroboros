#!/usr/bin/env python3
"""Linux-only identity diagnostic; not an operation-overhead benchmark.

Keep an isolated root and 50 children alive while changing only psutil's
wall-clock conversion. The observer and its evidence remain outside that tree.
Production TreeSampler owns discovery and memory reads. Identity evidence and
JSON serialization happen after each timed sweep; the host clock is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

from benchmark_run_statistics import PYTHON_ROOT, production_digest, write_json


_PROCESS_FAMILY = """
import json, subprocess, sys
children = []
try:
    for _ in range(int(sys.argv[1])):
        children.append(subprocess.Popen(
            [sys.executable, '-c', "import sys; print('ready', flush=True); sys.stdin.read()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
    for child in children:
        if child.stdout.readline().strip() != 'ready':
            raise RuntimeError('Child did not become ready')
        child.stdout.close()
    print(json.dumps([child.pid for child in children]), flush=True)
    sys.stdin.read()
finally:
    for child in children:
        child.stdin.close()
    for child in children:
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
"""


def probe(args):
    import psutil
    import psutil._pslinux as linux
    from ouroboros.common import memory_statistics as memory
    from ouroboros.common.memory_ipc import export_process_identity

    revision = os.environ.get("OUROBOROS_BUILD_REVISION")
    if not revision:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(PYTHON_ROOT), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            revision = None
    result = {
        "purpose": "Controlled-clock identity diagnostic, not workload overhead evidence",
        "source_revision": revision,
        "production_python_sha256": production_digest(),
        "probe_python_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_module": str(Path(memory.__file__).resolve()),
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "psutil": psutil.__version__, "logical_cpus": os.cpu_count()},
        "protocol": {"children": args.children, "sweeps": args.sweeps,
                     "delay_between_sweeps_seconds": args.delay,
                     "mid_sweep_shift": args.mid_sweep_shift,
                     "injected_boundary": "psutil._pslinux.boot_time only",
                     "sampled_tree": "Isolated root and waiting children; observer and evidence ledger excluded",
                     "memory_read": "production Linux Pss minus Pss_Shmem; no managed segments",
                     "identity_evidence": "Production sampler's retained Process objects, after each sweep",
                     "cost_boundary": "TreeSampler.sample only; excludes identity evidence and serialization"},
        "sweeps": [], "errors": [],
    }
    family = None
    try:
        family = subprocess.Popen(
            [sys.executable, "-c", _PROCESS_FAMILY, str(args.children)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        children = json.loads(family.stdout.readline())
        if len(children) != args.children:
            raise RuntimeError("Isolated family did not start the requested children")
        result["sampled_root_pid"] = family.pid
        result["sampled_child_pids"] = children
        result["observer_pid"] = os.getpid()
        sampler = memory.TreeSampler(export_process_identity(psutil.Process(family.pid)))
        baseline = sampler._discover()
        if {process.pid for process in baseline} != {family.pid, *children}:
            raise RuntimeError("Sampled tree differs from the isolated process family")
        result["baseline_identities"] = [export_process_identity(process)
                                         for process in sorted(baseline, key=lambda process: process.pid)]
        boot_time = linux.boot_time()
        state = {"offset": 0.0, "shift": 0.0, "first_read": False}
        native_read = memory._linux_nonshmem_pss

        def shifted_read(pid):
            value = native_read(pid)
            if state["first_read"]:
                state["offset"] += state["shift"]
                state["first_read"] = False
            return value

        with patch.object(linux, "boot_time", lambda: boot_time + state["offset"]), \
                patch.object(memory, "_linux_nonshmem_pss", shifted_read):
            for index in range(args.sweeps):
                # Walk forwards then backwards; fresh wrappers must remain equal.
                state["offset"] = float(min(index, args.sweeps - 1 - index))
                initial_offset = state["offset"]
                state["shift"] = (0.5 if index % 2 == 0 else -0.5) if args.mid_sweep_shift else 0.0
                state["first_read"] = True
                public_time_before = psutil.Process(family.pid).create_time()
                value = failure = None
                started = time.perf_counter()
                cpu = time.process_time()
                try:
                    value = sampler.sample()
                except Exception as error:
                    failure = {"kind": type(error).__name__, "message": str(error)}
                elapsed_cpu = time.process_time() - cpu
                elapsed = time.perf_counter() - started
                retained = set(sampler.known)
                identities = [export_process_identity(process)
                              for process in sorted(retained, key=lambda process: process.pid)]
                fresh = {psutil.Process(process.pid) for process in retained}
                result["sweeps"].append({
                    "epoch": index, "offset_before_seconds": initial_offset,
                    "offset_after_seconds": state["offset"],
                    "public_root_create_time_before": public_time_before,
                    "public_root_create_time_after": psutil.Process(family.pid).create_time(),
                    "elapsed_seconds": elapsed, "cpu_seconds": elapsed_cpu,
                    "reported_process_count": value[1] if value else None,
                    "sampled_bytes": value[0] if value else None,
                    "retained_process_count": len(retained),
                    "unique_pid_count": len({process.pid for process in retained}),
                    "unique_identity_count": len(set(retained)),
                    "same_identities_as_baseline": retained == baseline,
                    "fresh_wrappers_match_retained": fresh == retained,
                    "identities": identities, "error": failure,
                })
                if args.delay and index + 1 < args.sweeps:
                    time.sleep(args.delay)
    except Exception as error:
        result["errors"].append({"kind": type(error).__name__, "message": str(error)})
    finally:
        if family is not None:
            try:
                # Closing the root's stdin releases its children and reaps them.
                family.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                for process in psutil.Process(family.pid).children(recursive=True):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                family.kill()
                family.communicate(timeout=5)
            if family.returncode:
                result["errors"].append({"kind": "FamilyExit", "returncode": family.returncode})

    rows = result["sweeps"]
    accepted = [row for row in rows if row["error"] is None]
    expected_count = args.children + 1
    count_fields = ("reported_process_count", "retained_process_count", "unique_pid_count",
                    "unique_identity_count")
    part = max(1, len(rows) // 4)
    result["summary"] = {
        "expected_process_count": expected_count,
        "complete_sweeps": len(accepted), "failed_sweeps": len(rows) - len(accepted),
        "identity_invariants_passed": bool(accepted) and all(
            all(row[field] == expected_count for field in count_fields)
            and row["same_identities_as_baseline"] and row["fresh_wrappers_match_retained"]
            for row in accepted),
        "public_creation_timestamp_changed": len({row["public_root_create_time_after"] for row in rows}) > 1,
        "early_sweeps": [row["epoch"] for row in rows[:part]],
        "late_sweeps": [row["epoch"] for row in rows[-part:]],
        "early_median_sweep_seconds": statistics.median(row["elapsed_seconds"] for row in rows[:part]) if rows else None,
        "late_median_sweep_seconds": statistics.median(row["elapsed_seconds"] for row in rows[-part:]) if rows else None,
        "early_median_cpu_seconds": statistics.median(row["cpu_seconds"] for row in rows[:part]) if rows else None,
        "late_median_cpu_seconds": statistics.median(row["cpu_seconds"] for row in rows[-part:]) if rows else None,
        "interpretation": "Early/late costs describe this diagnostic only; they are not pipeline overhead acceptance.",
    }
    result["production_code_unchanged"] = production_digest() == result["production_python_sha256"]
    result["passed"] = (len(accepted) == args.sweeps and not result["errors"]
                        and result["summary"]["identity_invariants_passed"]
                        and result["summary"]["public_creation_timestamp_changed"]
                        and result["production_code_unchanged"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--children", type=int, default=50)
    parser.add_argument("--sweeps", type=int, default=64)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--mid-sweep-shift", action="store_true",
                        help="Also shift conversion after the first native memory read of each sweep.")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This diagnostic requires Linux psutil boot-time conversion and native PSS reads")
    if args.children < 1 or args.sweeps < 4 or args.delay < 0:
        parser.error("Require at least one child, four sweeps, and a nonnegative delay")
    if args.output.exists():
        parser.error("Output already exists; preserve prior attempts and choose a new filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = probe(args)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "passed": result["passed"],
                      "summary": result["summary"]}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

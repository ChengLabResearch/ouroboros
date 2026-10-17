"""Lightweight subprocess entry; also dispatched before frozen app imports."""

import sys


def dispatch_frozen_observer():
    if getattr(sys, "frozen", False) and len(sys.argv) > 1 and sys.argv[1] == "--ouroboros-memory-observer":
        main(sys.argv[2])
        raise SystemExit(0)


def main(directory):
    from pathlib import Path
    import time
    from ouroboros.common.memory_statistics import TreeSampler, Samples, _read_json, _write_json, unavailable

    directory = Path(directory)
    try:
        config = _read_json(directory / "config.json")
        sampler = TreeSampler(config["root_pid"], config["root_created"],
                              mappings_directory=config["mappings_directory"])
        samples = Samples(config["interval"])
        # Load native adapters before declaring readiness, outside data operations.
        if sampler.system == "Windows":
            from ouroboros.common.memory_windows import _windows_api
            _windows_api()
        elif sampler.system == "Darwin":
            from ouroboros.common.memory_macos import _native
            _native()
        (directory / "ready").touch()
        while not (directory / "start.json").exists():
            if not sampler.root.is_running():
                return
            time.sleep(0.01)
        started = _read_json(directory / "start.json")
        observation_cpu_started = time.process_time()
        next_sample = started
        while not (directory / "stop.json").exists():
            now = time.perf_counter()
            if now < next_sample:
                if not sampler.root.is_running():
                    return
                time.sleep(min(0.02, next_sample - now))
                continue
            cpu = time.process_time()
            value = error = None
            try:
                value = sampler.sample()
            except Exception as failure:
                error = failure
            ended = time.perf_counter()
            cpu = time.process_time() - cpu
            samples.add(now, ended, cpu, value, error)
            next_sample = max(now + config["interval"], ended)
        stopped = _read_json(directory / "stop.json")
        result = samples.record(
            started, stopped,
            # Includes polling and completion of any discarded in-flight sweep.
            collector_cpu_seconds=time.process_time() - observation_cpu_started,
        )
    except Exception as error:
        result = unavailable(f"{type(error).__name__}: {error}")
    _write_json(directory / "result.json", result)


if __name__ == "__main__":
    main(sys.argv[1])

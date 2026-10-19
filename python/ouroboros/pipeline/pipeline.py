import numpy as np
import time
from abc import ABC, abstractmethod
from itertools import chain
from threading import RLock
from typing import Callable

from tqdm import tqdm

from ouroboros.pipeline.pipeline_input import BasePipelineInput


class Pipeline:
    def __init__(self, steps: list["PipelineStep"]) -> None:
        self.steps = steps
        self.statistics_enabled = False
        self.elapsed_seconds = 0.0
        self.status = "pending"

    def set_statistics_enabled(self, enabled: bool) -> "Pipeline":
        self.statistics_enabled = bool(enabled)
        for step in self.steps:
            step.statistics_enabled = self.statistics_enabled
        return self

    def process(
        self, input_data: BasePipelineInput
    ) -> tuple[BasePipelineInput, None] | tuple[None, str]:
        """Run each step and retain the pipeline wall span, including failures."""
        for step in self.steps:
            step.reset_statistics()
        self.status = "running"
        start = time.perf_counter()
        try:
            data = input_data
            for step in self.steps:
                data, errors = step.process(data)
                if errors is not None:
                    self.status = "failed"
                    return None, errors
            self.status = "succeeded"
            return data, None
        except BaseException as error:
            self.status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            raise
        finally:
            self.elapsed_seconds = time.perf_counter() - start

    def get_steps_progress(self) -> list[list[any]]:
        return [[step.step_name, step.get_progress()] for step in self.steps]

    def get_steps_progress_and_durations(self) -> list[list[any]]:
        return [
            [step.step_name, step.get_progress(), step.get_duration()]
            for step in self.steps
        ]

    def get_step_statistics(self) -> list[dict]:
        return [step.get_time_statistics() for step in self.steps]

    def get_steps_status(self) -> list[dict]:
        """Serialize the live step owners without enabling optional statistics."""
        return [step.get_status() for step in self.steps]


class PipelineStep(ABC):
    """Own step duration, progress, and optional detailed elapsed-time samples.

    Worker samples can overlap each other and coordinator work. Their sums are
    accumulated elapsed time, never pipeline wall time. ``includes`` identifies
    nested totals so consumers do not add them to their constituent phases.
    """

    def __init__(self, inputs: tuple[str]) -> None:
        self.step_name = type(self).__name__
        self.statistics_enabled = False
        self._statistics_lock = RLock()
        self.reset_statistics()
        self.progress_listener_callables = []
        self.show_progress_bar = False
        self.progress_bar = None
        self.inputs = inputs

    def reset_statistics(self):
        self.timing = {
            "pipeline": self.step_name,
            "custom_times": {},
            "duration_seconds": 0.0,
            "status": "pending",
        }
        self._timing_metadata = {}
        self.byte_counts = {}
        self.execution_resources = {}
        self.progress = 0
        self.stage = None
        self._stage_determinate = True

    def process(
        self, input_data: BasePipelineInput
    ) -> tuple[BasePipelineInput, None] | tuple[BasePipelineInput, str]:
        self.reset_statistics()
        self.timing["status"] = "running"
        start = time.perf_counter()
        try:
            if self.show_progress_bar:
                tqdm.write(f"Starting Step: {self.step_name}")
                self.progress_bar = tqdm(total=100)
            self.update_progress(0)
            result = self._process(chain(input_data[self.inputs], (input_data,)))
            self.timing["status"] = "succeeded" if result is None else "failed"
            if result is None:
                self.update_progress(1)
            return input_data, result
        except BaseException as error:
            self.timing["status"] = (
                "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            )
            raise
        finally:
            self.timing["duration_seconds"] = time.perf_counter() - start
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None

    @abstractmethod
    def _process(self, input_data: tuple[any]) -> None | str:
        pass

    def get_time_statistics(self) -> dict:
        """Return an independent JSON-safe presentation without replacing samples."""
        with self._statistics_lock:
            custom_times = {}
            for key, samples in self.timing["custom_times"].items():
                if not samples:
                    continue
                custom_times[key] = {
                    "mean": float(np.mean(samples)),
                    "std": float(np.std(samples)),
                    "min": float(min(samples)),
                    "max": float(max(samples)),
                    "accumulated_elapsed_seconds": float(sum(samples)),
                    "gap": float(max(samples) - min(samples)),
                    "loops": len(samples),
                    "unit": "seconds",
                    "scope": self._timing_metadata[key][0],
                    "includes": list(self._timing_metadata[key][1]),
                }
            return {
                "pipeline": self.step_name,
                "duration_seconds": self.get_duration(),
                "status": self.timing["status"],
                "custom_times": custom_times,
                "byte_counts": dict(self.byte_counts),
            }

    def get_duration(self) -> float:
        return self.timing["duration_seconds"]

    def set_stage(self, stage: str, *, determinate: bool = False):
        """Name the current work; a fraction is valid only with a known denominator.

        A stage describes work inside this step, including concurrent workers.
        Its lifecycle remains exclusively owned by ``timing['status']``.
        """
        with self._statistics_lock:
            self.stage = stage
            self._stage_determinate = determinate

    def get_status(self) -> dict:
        with self._statistics_lock:
            status = self.timing["status"]
            return {
                "name": self.step_name,
                "status": status,
                "progress": (float(self.get_progress())
                             if self._stage_determinate or status == "succeeded" else None),
                "duration_seconds": self.get_duration(),
                "stage": self.stage,
            }

    def add_timing(self, key: str, value: float, *, scope="worker", includes=()):
        if self.statistics_enabled:
            self.add_timing_list(key, (value,), scope=scope, includes=includes)

    def add_timing_list(self, key: str, values: list[float], *, scope="worker", includes=()):
        if not self.statistics_enabled:
            return
        with self._statistics_lock:
            self.timing["custom_times"].setdefault(key, []).extend(float(value) for value in values)
            previous_includes = self._timing_metadata.get(key, (scope, ()))[1]
            self._timing_metadata[key] = (scope, tuple(dict.fromkeys((*previous_includes, *includes))))

    def add_byte_count(self, key: str, value: int):
        if self.statistics_enabled:
            with self._statistics_lock:
                self.byte_counts[key] = self.byte_counts.get(key, 0) + int(value)

    def with_progress_bar(self) -> "PipelineStep":
        self.show_progress_bar = True
        return self

    def update_progress(self, progress: float):
        self.progress = progress
        for progress_callable in self.progress_listener_callables:
            progress_callable(progress)
        if self.show_progress_bar and self.progress_bar is not None:
            self.progress_bar.update(progress * 100 - self.progress_bar.n)

    def get_progress(self) -> float:
        return self.progress

    def listen_for_progress(self, progress_callable: Callable[[float], None]):
        """Subscribe to this step's progress, expressed as a fraction from 0 to 1."""
        self.progress_listener_callables.append(progress_callable)

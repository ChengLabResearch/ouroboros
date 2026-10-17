"""One output adapter for the existing task, pipeline and volume measurement owners."""

from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import time

from ouroboros.common.file_system import publish_statistics
from ouroboros.helpers.files import format_statistics_file
from ouroboros.helpers.models import pretty_json_output
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline.volume_statistics import collect_volume_statistics


@lru_cache(maxsize=1)
def runtime_metadata() -> dict:
    """Cheap process-invariant provenance, gathered outside the measured operation."""
    try:
        version = metadata.version("ouroboros")
    except metadata.PackageNotFoundError:
        version = None
    revision = os.environ.get("OUROBOROS_BUILD_REVISION")
    working_tree_dirty = None
    if not revision:
        # Installed wheels may have no checkout. Never infer revision from the caller's cwd.
        repository = Path(__file__).resolve().parents[3]
        if (repository / ".git").exists():
            try:
                revision = subprocess.check_output(
                    ["git", "-C", str(repository), "rev-parse", "HEAD"],
                    text=True, stderr=subprocess.DEVNULL, timeout=1,
                ).strip()
                working_tree_dirty = bool(subprocess.check_output(
                    ["git", "-C", str(repository), "status", "--porcelain"],
                    text=True, stderr=subprocess.DEVNULL, timeout=1,
                ).strip())
            except (OSError, subprocess.SubprocessError):
                revision = None
    return {
        "software": {"version": version, "revision": revision,
                     "working_tree_dirty": working_tree_dirty,
                     "build": os.environ.get("OUROBOROS_BUILD_ID")},
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "os_release": platform.release(), "architecture": platform.machine(),
                        "logical_cpu_count": os.cpu_count(),
                        "thread_environment": {key: os.environ[key] for key in
                                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
                                               if key in os.environ}},
    }


class RunStatistics:
    """Own operation timing and publication; derive all other facts from their live owners.

    Task owns run identity/outcome, Pipeline owns step/worker timings, and native data
    objects own geometry. The volume snapshot is a one-way serialization boundary taken
    before Docker removes outputs; it is never an input to scientific execution.
    """

    def __init__(self, task, *, docker: bool = False):
        self.task = task
        self.docker = docker
        self.started = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.elapsed_seconds = None
        self.finished_at = None
        self.primary_options = None
        self.slice_options = None
        self.output_folder = None
        self.input_identifiers = {"options_path": task.options}
        self.volumes = None
        self.collection_errors = []
        self.publication = {"status": "pending", "path": None,
                            "recovery_path": None, "error": None,
                            "presentation_error": None, "presentation_interrupted": False}

    @property
    def operation(self):
        from ouroboros.common.server_types import SliceTask
        return "slice" if isinstance(self.task, SliceTask) else "backproject"

    def configure(self, options):
        """Observe options once at the input boundary, before Docker remaps paths."""
        if isinstance(options, SliceOptions):
            if self.operation == "backproject":
                self.slice_options = options
            self.input_identifiers["annotation_path"] = options.neuroglancer_json
        primary = isinstance(options, BackprojectOptions) or self.operation == "slice"
        if primary:
            self.primary_options = options
            self.output_folder = options.output_file_folder
            if isinstance(options, BackprojectOptions):
                self.input_identifiers.update(straightened_volume_path=options.straightened_volume_path,
                                              slice_options_path=options.slice_options_path)

    def capture_volumes(self):
        if self.task.pipeline_input is None:
            return
        try:
            self.volumes = collect_volume_statistics(self.task.pipeline_input, self.operation)
        except Exception as error:
            # Instrumentation failure must be explicit without replacing the scientific outcome.
            self.collection_errors.append(f"Volume statistics: {type(error).__name__}: {error}")
            self.volumes = {"status": "unavailable", "reason": self.collection_errors[-1]}

    def finish(self):
        self.elapsed_seconds = time.perf_counter() - self.started
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_record(self) -> dict:
        pipeline = self.task.pipeline
        options = self.primary_options
        input_data = self.task.pipeline_input
        cache = getattr(input_data, "volume_cache", None)
        resources = [{"step": step.step_name, **step.execution_resources}
                     for step in (pipeline.steps if pipeline else [])
                     if getattr(step, "execution_resources", None)]
        outcomes = {"done": "succeeded", "error": "failed", "interrupted": "interrupted"}
        return {
            "schema_version": 1,
            "run_id": self.task.task_id,
            "operation": self.operation,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": outcomes.get(self.task.status, "incomplete"),
            "error": self.task.error,
            "operation_elapsed_seconds": self.elapsed_seconds,
            "pipeline_elapsed_seconds": (pipeline.elapsed_seconds
                                         if pipeline and pipeline.status != "pending" else None),
            "phase_wall_spans": None,
            "steps": pipeline.get_step_statistics() if pipeline else [],
            "volumes": self.volumes or {"status": "unavailable", "reason": "pipeline inputs unavailable"},
            "memory": None,
            "inputs": self.input_identifiers,
            "effective_configuration": {
                "operation": options.model_dump(mode="json") if options else None,
                **({"slicing": self.slice_options.model_dump(mode="json")} if self.slice_options else {}),
            },
            **runtime_metadata(),
            "execution": {"deployment": "docker" if self.docker else "native",
                          "workers": resources},
            "cache": {"flush_requested": cache.flush_cache if cache else None,
                      "cloudvolume_cache_enabled": getattr(getattr(getattr(getattr(cache, "cv", None), "cv", None),
                                                           "cache", None), "enabled", None),
                      "warmth": "unknown", "network_conditions": "unknown"},
            "collection_errors": self.collection_errors,
        }

    def publish(self) -> str:
        formatted = pretty_json_output(self.to_record())
        if self.output_folder is None or self.primary_options is None:
            self.publication = {"status": "unavailable", "path": None, "recovery_path": None,
                                "error": "Statistics destination unavailable: options could not be loaded.",
                                "presentation_error": None, "presentation_interrupted": False}
        else:
            filename = format_statistics_file(self.primary_options.output_file_name,
                                              self.operation, self.task.task_id)
            self.publication = publish_statistics(formatted, filename, self.output_folder, self.docker)
        return formatted

    def presentation_failed(self, error: BaseException):
        """Keep console/log failure separate from the already completed file delivery."""
        previous = self.publication.get("presentation_error")
        message = f"Statistics presentation failed: {type(error).__name__}: {error}"
        self.publication["presentation_error"] = f"{previous}; {message}" if previous else message
        self.publication["presentation_interrupted"] = (
            self.publication.get("presentation_interrupted", False)
            or isinstance(error, (KeyboardInterrupt, SystemExit))
        )

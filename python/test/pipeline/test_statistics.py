import concurrent.futures
import json
import threading

import numpy as np
import pytest

from ouroboros.common.logging_mode import LoggingMode
from ouroboros.helpers import files as files_module
from ouroboros.pipeline import pipeline as pipeline_module
from ouroboros.pipeline.pipeline import Pipeline, PipelineStep
from ouroboros.pipeline.pipeline_input import BasePipelineInput
from test.harness import clock


class FunctionStep(PipelineStep):
    def __init__(self, action=lambda step: None):
        super().__init__(())
        self.action = action

    def _process(self, inputs):
        return self.action(self)


def test_overlapping_worker_samples_have_explicit_nested_semantics_and_repeatable_summaries(monkeypatch):
    barrier = threading.Barrier(2)

    def work(step):
        def worker():
            barrier.wait(timeout=2)
            step.add_timing("compute", np.float64(0.75))
            step.add_timing("total_worker", 1.0, includes=("compute",))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: worker(), range(2)))

    step = FunctionStep(work)
    pipeline = Pipeline([step]).set_logging_mode(LoggingMode.STANDARD)
    monkeypatch.setattr(pipeline_module, "time", clock([0, 0, 1, 1]))
    pipeline.process(BasePipelineInput())
    original = pipeline.get_step_statistics()
    assert original == pipeline.get_step_statistics()
    assert json.loads(json.dumps(original)) == original
    detail = original[0]["custom_times"]
    assert detail["compute"]["accumulated_elapsed_seconds"] == 1.5
    assert detail["compute"]["accumulated_elapsed_seconds"] > pipeline.elapsed_seconds
    assert detail["compute"]["scope"] == "worker"
    assert detail["total_worker"]["includes"] == ["compute"]
    assert detail["total_worker"]["loops"] == 2
    assert "total" not in detail["compute"]
    original[0]["custom_times"]["compute"]["mean"] = -1
    original[0].pop("pipeline")
    assert pipeline.get_step_statistics()[0]["custom_times"]["compute"]["mean"] == 0.75
    assert step.timing["custom_times"]["compute"] == [0.75, 0.75]
    assert pipeline.get_steps_progress_and_durations()[0][2] == step.get_duration() == 1


@pytest.mark.parametrize("failure, status", [("bad input", "failed"), (ValueError("worker failure"), "failed"),
                                             (KeyboardInterrupt(), "interrupted")])
def test_failed_or_interrupted_steps_finalize_owned_durations_without_success_progress(monkeypatch, failure, status):
    def work(step):
        step.update_progress(0.25)
        if isinstance(failure, BaseException):
            raise failure
        return failure
    step, skipped = FunctionStep(work), FunctionStep()
    pipeline = Pipeline([step, skipped])
    observed = []
    step.listen_for_progress(observed.append)
    monkeypatch.setattr(pipeline_module, "time", clock([10, 11, 14, 15]))
    if isinstance(failure, BaseException):
        with pytest.raises(type(failure)):
            pipeline.process(BasePipelineInput())
    else:
        assert pipeline.process(BasePipelineInput()) == (None, failure)
    assert pipeline.elapsed_seconds == 5
    assert step.get_duration() == 3
    assert pipeline.status == step.timing["status"] == status
    assert skipped.timing["status"] == "pending"
    assert observed == [0, 0.25]
    assert pipeline.get_steps_progress_and_durations()[0] == [step.step_name, 0.25, 3]


def test_disabled_statistics_keep_progress_and_step_duration_but_no_detailed_samples():
    def work(step):
        step.add_timing("worker", 12)
        step.add_timing_list("other", [3, 5])
        step.add_byte_count("intermediate_payload_bytes", 100)
    step = FunctionStep(work)
    pipeline = Pipeline([step])
    pipeline.process(BasePipelineInput())
    assert pipeline.get_step_statistics()[0]["custom_times"] == {}
    assert pipeline.get_step_statistics()[0]["byte_counts"] == {}
    assert step.timing["custom_times"] == {} and step.byte_counts == {}
    assert step.get_progress() == 1
    assert pipeline.elapsed_seconds >= step.get_duration() > 0
    pipeline.set_logging_mode(LoggingMode.STANDARD).process(BasePipelineInput())
    assert step.get_time_statistics()["custom_times"]["worker"]["loops"] == 1
    assert step.byte_counts == {"intermediate_payload_bytes": 100}
    pipeline.process(BasePipelineInput())
    assert step.get_time_statistics()["custom_times"]["worker"]["loops"] == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_status_snapshot_derives_lifecycle_from_step_owner_and_keeps_unknown_work_indeterminate(enabled):
    observed = []

    def work(step):
        step.set_stage("downloading_and_slicing", determinate=True)
        step.update_progress(0.75)
        observed.append(step.get_status())
        step.set_stage("normalizing_and_writing")
        observed.append(step.get_status())
        return "rewrite failed"

    step, skipped = FunctionStep(work), FunctionStep()
    pipeline = Pipeline([step, skipped]).set_logging_mode(LoggingMode.STANDARD if enabled else LoggingMode.OFF)
    assert pipeline.get_steps_status()[0]["status"] == "pending"
    pipeline.process(BasePipelineInput())
    assert observed[0]["status"] == observed[1]["status"] == "running"
    assert observed[0]["progress"] == 0.75 and observed[1]["progress"] is None
    status, pending = pipeline.get_steps_status()
    assert status["status"] == step.timing["status"] == "failed"
    assert status["stage"] == "normalizing_and_writing" and status["progress"] is None
    assert status["duration_seconds"] == step.get_duration() > 0
    assert pending["status"] == skipped.timing["status"] == "pending"
    assert pipeline.get_steps_progress_and_durations()[0][1] == step.get_progress() == 0.75
    status["status"] = "succeeded"
    assert step.timing["status"] == "failed", "Serialized status must never become a runtime authority"


def test_statistics_filename_keeps_operation_and_run_identity():
    assert files_module.format_statistics_file("scan", "slice", "abc") == "scan-slice-abc-statistics.json"
    assert files_module.format_statistics_file("scan", "backproject", "abc") != \
        files_module.format_statistics_file("scan", "slice", "abc")

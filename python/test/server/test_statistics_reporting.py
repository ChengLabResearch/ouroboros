"""Exercise the shared lifecycle through CLI, server, and Docker boundaries."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from threading import Event

from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
import pytest
from sse_starlette.sse import AppStatus

from ouroboros import cli
from ouroboros.common import file_system, run_statistics, server_handlers
from ouroboros.common.server_api import create_api
from ouroboros.common.server_types import BackProjectTask, SliceTask
from ouroboros.helpers.files import format_slice_output_file, format_slice_output_multiple
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline import Pipeline, PipelineInput, PipelineStep


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    output = tmp_path / "host-output"
    output.mkdir()
    annotation = tmp_path / "annotations.json"
    annotation.write_text("{}")
    straightened = tmp_path / "mask.tif"
    straightened.write_bytes(b"fixture input")
    slice_options_path = tmp_path / "slice-options.json"
    backproject_options_path = tmp_path / "backproject-options.json"
    state = SimpleNamespace(
        output=output, annotation=annotation, volume=tmp_path / "volume",
        slice_path=slice_options_path, backproject_path=backproject_options_path,
        events=[], pipelines=[], inputs=[], collected=[], memory_events=[], behavior="success",
        transfer_failure=None, import_failure=None,
    )

    def configure(single=True):
        slicing = SliceOptions(
            output_file_folder=str(output), output_file_name="sample",
            neuroglancer_json=str(annotation), slice_width=4, slice_height=3,
            make_single_file=single,
        )
        backprojecting = BackprojectOptions(
            output_file_folder=str(output), output_file_name="sample-bp",
            straightened_volume_path=str(straightened), slice_options_path=str(slice_options_path),
            make_single_file=single, process_count=4,
        )
        slicing.save_to_json(slice_options_path)
        backprojecting.save_to_json(backproject_options_path)

    state.configure = configure
    configure()

    class SyntheticOutputStep(PipelineStep):
        def __init__(self, operation, options):
            super().__init__(inputs=())
            self.operation = operation
            self.options = options

        def _process(self, values):
            data, = values
            options = self.options
            if self.operation == "slice":
                name = (format_slice_output_file if options.make_single_file else format_slice_output_multiple)(
                    options.output_file_name
                )
            else:
                name = "actual-bp.tif" if options.make_single_file else "actual-bp"
            path = Path(options.output_file_folder, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if options.make_single_file:
                path.write_bytes(b"unchanged scientific output")
            else:
                path.mkdir(exist_ok=True)
                (path / "0.tif").write_bytes(b"unchanged scientific output")
            data.output_file_path = str(path) if self.operation == "slice" else name
            self.add_timing("compute", 0.00123456789)
            state.events.append("scientific-output")
            if state.behavior == "returned":
                return "returned failure"
            if state.behavior == "raised":
                raise RuntimeError("raised failure")
            if state.behavior == "interrupted":
                raise KeyboardInterrupt()
            return None

    def build(operation, options, slicing):
        data = PipelineInput(slice_options=slicing, json_path=slicing.neuroglancer_json)
        if operation == "backproject":
            data.backproject_options = options
        pipeline = Pipeline([SyntheticOutputStep(operation, options)])
        state.pipelines.append(pipeline)
        state.inputs.append(data)
        return pipeline, data

    monkeypatch.setattr(server_handlers, "slice_pipeline", lambda options, verbose: build("slice", options, options))
    monkeypatch.setattr(
        server_handlers, "backproject_pipeline",
        lambda options, slicing, verbose: build("backproject", options, slicing),
    )

    def collect(data, operation):
        assert data is state.inputs[-1]
        path = Path(data.output_file_path)
        if operation == "backproject":
            path = Path(data.backproject_options.output_file_folder, path)
        assert path.exists(), "Volume snapshot must precede scientific scratch cleanup"
        state.collected.append((data, operation))
        state.events.append("collect-volumes")
        return {"fixture_output": str(path), "operation": operation}

    monkeypatch.setattr(run_statistics, "collect_volume_statistics", collect)

    class FixtureMemoryCollector:

        def __init__(self):
            self.active = False
            self.summary = {"status": "not_started"}

        def start(self):
            self.active = True
            state.memory_events.append("start")
            state.events.append("memory-start")

        def stop(self):
            if self.active:
                self.active = False
                state.memory_events.append("stop")
                state.events.append("memory-stop")
                self.summary = {"status": "sampled", "sampled_peak_bytes": 12345}

    monkeypatch.setattr(run_statistics, "MemoryCollector", FixtureMemoryCollector)
    return state


def _task(state, operation, **kwargs):
    cls = SliceTask if operation == "slice" else BackProjectTask
    path = state.slice_path if operation == "slice" else state.backproject_path
    return cls(task_id="fixture-run", options=str(path), **kwargs)


def _record(task):
    return json.loads(Path(task.statistics.publication["path"]).read_text())


@pytest.mark.parametrize("method", ["start", "stop"])
def test_memory_observation_failure_keeps_scientific_outcome(run_fixture, monkeypatch, method):
    def failed(self):
        raise OSError("fixture memory API unavailable")

    monkeypatch.setattr(run_statistics.MemoryCollector, method, failed)
    task = _task(run_fixture, "slice")
    server_handlers.handle_task(task)
    record = _record(task)
    assert task.status == "done"
    assert record["memory"]["status"] == "unavailable"
    assert record["memory"]["sampled_peak_bytes"] is None
    assert "fixture memory API unavailable" in record["memory"]["errors"][0]["example"]


@pytest.mark.parametrize("operation", ["slice", "backproject"])
@pytest.mark.parametrize(
    ("behavior", "task_status", "outcome"),
    [("success", "done", "succeeded"), ("returned", "error", "failed"),
     ("raised", "error", "failed"), ("interrupted", "interrupted", "interrupted")],
)
def test_native_outcomes_finalize_one_record_from_the_canonical_pipeline(
    run_fixture, operation, behavior, task_status, outcome,
):
    state = run_fixture
    state.behavior = behavior
    task = _task(state, operation)
    assert server_handlers.handle_task(task) is task
    assert task.status == task_status
    assert task.pipeline is state.pipelines[0]
    assert task.pipeline_input is state.inputs[0]
    assert task.statistics.task is task
    assert len(state.collected) == 1
    record = _record(task)
    assert record["outcome"] == outcome
    assert record["operation"] == operation
    assert record["run_id"] == task.task_id
    assert record["operation_elapsed_seconds"] >= record["pipeline_elapsed_seconds"] > 0
    assert record["started_at"].endswith("+00:00")
    assert record["finished_at"] is not None
    assert record["steps"] == task.pipeline.get_step_statistics()
    assert record["steps"][0]["custom_times"]["compute"]["mean"] == 0.00123456789
    assert record["memory"] == task.statistics.memory_collector.summary
    assert state.memory_events == ["start", "stop"]
    assert state.events.index("memory-start") < state.events.index("scientific-output")
    assert state.events.index("memory-stop") < state.events.index("collect-volumes")
    assert record["phase_wall_spans"] is None
    assert ("slicing" in record["effective_configuration"]) == (operation == "backproject")
    assert task.statistics.publication["status"] == "published"
    assert len(list(state.output.glob("*statistics*.json"))) == 1


@pytest.mark.parametrize("operation", ["slice", "backproject"])
@pytest.mark.parametrize("enabled", [True, False])
def test_cli_verbose_prints_exact_saved_json_and_disabled_skips_collection(
    run_fixture, capsys, monkeypatch, operation, enabled,
):
    state = run_fixture
    tasks = []

    def execute(task, **kwargs):
        tasks.append(task)
        return server_handlers.handle_task(task, **kwargs)

    monkeypatch.setattr(cli, "handle_task", execute)
    path = state.slice_path if operation == "slice" else state.backproject_path
    monkeypatch.setattr(sys, "argv", ["ouroboros-cli", operation, str(path)] + (["--verbose"] if enabled else []))
    cli.main()
    task, = tasks
    stdout = capsys.readouterr().out
    assert task.pipeline is state.pipelines[0]
    assert task.reporting_enabled is enabled
    assert task.show_progress is True
    name, progress, duration = task.pipeline.get_steps_progress_and_durations()[0]
    assert name == task.pipeline.steps[0].step_name
    assert progress == 1
    assert duration > 0
    if enabled:
        file_text = Path(task.statistics.publication["path"]).read_text()
        assert stdout.split("Calculation Statistics:", 1)[1].strip() == file_text
        assert json.loads(file_text)["outcome"] == "succeeded"
        assert len(state.collected) == 1
    else:
        assert state.memory_events == []
        assert "Calculation Statistics:" not in stdout
        assert task.statistics is None
        assert state.collected == []
        assert not state.pipelines[0].statistics_enabled
        assert state.pipelines[0].get_step_statistics()[0]["custom_times"] == {}
        assert list(state.output.glob("*statistics*.json")) == []


def test_repeated_cli_output_name_preserves_distinct_runs_and_prior_payload(run_fixture):
    state = run_fixture
    args = SimpleNamespace(options=str(state.slice_path), verbose=True)
    first = cli.handle_slice(args)
    first_path = Path(first.statistics.publication["path"])
    first_payload = first_path.read_bytes()
    first_step = first.pipeline.steps[0]
    first_samples = first_step.timing["custom_times"]["compute"]

    second = cli.handle_slice(args)
    second_path = Path(second.statistics.publication["path"])
    assert first.task_id != second.task_id
    assert first_path != second_path
    assert first_path.read_bytes() == first_payload
    assert first.pipeline is state.pipelines[0]
    assert second.pipeline is state.pipelines[1]
    assert first.statistics.task.pipeline.steps[0] is first_step
    assert first_step.timing["custom_times"]["compute"] is first_samples
    assert first_samples == [0.00123456789]
    assert _record(first)["steps"] == first.pipeline.get_step_statistics()
    assert _record(second)["steps"] == second.pipeline.get_step_statistics()
    assert _record(first)["run_id"] == first.task_id
    assert _record(second)["run_id"] == second.task_id
    assert len(list(state.output.glob("*statistics*.json"))) == 2


def test_early_options_failure_has_no_unintended_cwd_report(run_fixture, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task = SliceTask(task_id="bad-options", options=str(tmp_path / "missing.json"))
    server_handlers.handle_task(task)
    assert task.status == "error"
    assert task.pipeline is None
    assert task.statistics.to_record()["outcome"] == "failed"
    assert task.statistics.publication["status"] == "unavailable"
    assert task.statistics.publication["path"] is None
    assert list(tmp_path.glob("*statistics*.json")) == []
    assert run_fixture.collected == []


def test_native_relative_output_report_path_matches_published_file(run_fixture, tmp_path, monkeypatch):
    state = run_fixture
    monkeypatch.chdir(tmp_path)
    options = SliceOptions.load_from_json(state.slice_path)
    options.output_file_folder = "output"
    options.save_to_json(state.slice_path)
    task = _task(state, "slice")
    server_handlers.handle_task(task)
    publication = task.statistics.publication
    assert task.status == "done"
    assert publication["status"] == "published"
    actual_files = list((tmp_path / "output").glob("*statistics*.json"))
    assert len(actual_files) == 1
    assert Path(publication["path"]).is_file()
    assert Path(publication["path"]).resolve() == actual_files[0]
    assert _record(task)["run_id"] == task.task_id


def _docker_transport(state, monkeypatch):
    def volume_path(plugin="main"):
        return str(state.volume / plugin) + "/"

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(source, target)

    def import_files(files):
        for item in files:
            source = Path(item["sourcePath"])
            if state.import_failure == source.name:
                return False, "fixture import failed"
            copy(source, Path(volume_path(), source.name))
        return True, ""

    def export_files(files, plugin_folder_name="main"):
        state.events.append(f"export-{plugin_folder_name}")
        if state.transfer_failure == plugin_folder_name:
            return False, "fixture transfer failed"
        for item in files:
            target = Path(item["sourcePath"])
            copy(Path(volume_path(plugin_folder_name), target.name), target)
        return True, ""

    def clear():
        state.events.append("cleanup")
        if state.transfer_failure == "cleanup":
            return False, "fixture cleanup failed"
        shutil.rmtree(state.volume / "main", ignore_errors=True)
        return True, ""

    monkeypatch.setattr(file_system, "get_volume_path", volume_path)
    monkeypatch.setattr(file_system, "copy_to_volume", import_files)
    monkeypatch.setattr(file_system, "copy_to_host", export_files)
    monkeypatch.setattr(file_system, "clear_plugin_folder", clear)
    monkeypatch.setattr(server_handlers, "clear_plugin_folder", clear)


@pytest.mark.parametrize("operation", ["slice", "backproject"])
@pytest.mark.parametrize("single", [True, False])
def test_docker_exports_science_then_cleans_before_report_delivery(run_fixture, monkeypatch, operation, single):
    state = run_fixture
    state.configure(single)
    _docker_transport(state, monkeypatch)
    task = _task(state, operation)
    server_handlers.handle_task_docker(task)
    assert task.status == "done"
    assert state.events == [
        "memory-start", "scientific-output", "memory-stop", "collect-volumes", "export-main", "cleanup", "export-main-statistics",
    ]
    record = _record(task)
    assert record["outcome"] == "succeeded"
    assert record["execution"]["deployment"] == "docker"
    assert record["inputs"]["annotation_path"] == str(state.annotation)
    assert record["effective_configuration"]["operation"]["output_file_folder"] == str(state.volume / "main") + "/"
    assert Path(task.statistics.publication["path"]).parent == state.output
    assert not (state.volume / "main").exists()
    assert list((state.volume / "main-statistics").glob("*.json")) == []
    science = [path for path in state.output.iterdir() if not path.name.endswith(".json")]
    assert len(science) == 1
    sample = science[0] if single else science[0] / "0.tif"
    assert sample.read_bytes() == b"unchanged scientific output"


@pytest.mark.parametrize("failure", ["main", "cleanup"])
def test_docker_scientific_delivery_failure_finalizes_failed_operation(run_fixture, monkeypatch, failure):
    state = run_fixture
    _docker_transport(state, monkeypatch)
    state.transfer_failure = failure
    task = _task(state, "slice")
    server_handlers.handle_task_docker(task)
    assert task.status == "error"
    assert _record(task)["outcome"] == "failed"
    assert "fixture" in _record(task)["error"]
    assert task.statistics.publication["status"] == "published"


@pytest.mark.parametrize("behavior", ["success", "returned"])
def test_docker_cleanup_interrupt_still_publishes_interrupted_record(run_fixture, monkeypatch, behavior):
    state = run_fixture
    state.behavior = behavior
    _docker_transport(state, monkeypatch)

    def interrupted_cleanup():
        state.events.append("cleanup-interrupted")
        raise KeyboardInterrupt()

    monkeypatch.setattr(server_handlers, "clear_plugin_folder", interrupted_cleanup)
    task = _task(state, "slice")
    server_handlers.handle_task_docker(task)
    assert task.status == "interrupted"
    assert task.statistics.publication["status"] == "published"
    record = _record(task)
    assert record["outcome"] == "interrupted"
    assert record["finished_at"] is not None
    assert record["operation_elapsed_seconds"] >= record["pipeline_elapsed_seconds"]
    assert state.events[-2:] == ["cleanup-interrupted", "export-main-statistics"]
    if behavior == "returned":
        assert "returned failure" in record["error"]


def test_failed_report_transfer_retains_recovery_outside_scientific_cleanup(run_fixture, monkeypatch):
    state = run_fixture
    _docker_transport(state, monkeypatch)
    state.transfer_failure = "main-statistics"
    task = _task(state, "slice")
    server_handlers.handle_task_docker(task)
    assert task.status == "done"
    assert task.error is None
    publication = task.statistics.publication
    assert publication["status"] == "failed"
    recovery = Path(publication["recovery_path"])
    assert recovery.parent == state.volume / "main-statistics"
    assert json.loads(recovery.read_text())["outcome"] == "succeeded"
    assert not Path(publication["path"]).exists()
    server_handlers.clear_plugin_folder()
    assert recovery.exists()


def test_native_report_write_failure_preserves_scientific_outcome(run_fixture, monkeypatch):
    def fail_write(*args):
        raise OSError("fixture disk full")

    monkeypatch.setattr(file_system, "write_statistics_file", fail_write)
    task = _task(run_fixture, "slice")
    server_handlers.handle_task(task)
    assert task.status == "done"
    assert task.error is None
    assert task.statistics.publication["status"] == "failed"
    assert "fixture disk full" in task.statistics.publication["error"]
    assert task.statistics.to_record()["outcome"] == "succeeded"


def test_docker_input_failure_after_options_retains_host_report_destination(run_fixture, monkeypatch):
    state = run_fixture
    _docker_transport(state, monkeypatch)
    state.import_failure = state.annotation.name
    task = _task(state, "slice")
    server_handlers.handle_task_docker(task)
    assert task.status == "error"
    assert task.pipeline is None
    assert task.statistics.output_folder == str(state.output)
    assert Path(task.statistics.publication["path"]).parent == state.output
    assert _record(task)["outcome"] == "failed"


def test_operation_clock_includes_delivery_cleanup_and_excludes_report_publication(run_fixture, monkeypatch):
    state = run_fixture
    _docker_transport(state, monkeypatch)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(run_statistics.time, "perf_counter", lambda: clock.now)

    def advance_before(function, seconds):
        def timed(*args, **kwargs):
            clock.now += seconds
            return function(*args, **kwargs)
        return timed

    factory = server_handlers.slice_pipeline

    def timed_pipeline(*args):
        pipeline, data = factory(*args)
        step = pipeline.steps[0]
        step._process = advance_before(step._process, 7)
        return pipeline, data

    monkeypatch.setattr(server_handlers, "slice_pipeline", timed_pipeline)
    monkeypatch.setattr(
        run_statistics, "collect_volume_statistics",
        advance_before(run_statistics.collect_volume_statistics, 3),
    )
    monkeypatch.setattr(
        server_handlers, "save_output_for_slice_docker",
        advance_before(server_handlers.save_output_for_slice_docker, 5),
    )
    monkeypatch.setattr(server_handlers, "clear_plugin_folder", advance_before(server_handlers.clear_plugin_folder, 11))
    monkeypatch.setattr(file_system, "write_statistics_file", advance_before(file_system.write_statistics_file, 13))

    task = _task(state, "slice")
    server_handlers.handle_task_docker(task)
    record = _record(task)
    assert record["pipeline_elapsed_seconds"] == 7
    assert record["operation_elapsed_seconds"] == 7 + 3 + 5 + 11
    assert clock.now == 7 + 3 + 5 + 11 + 13


def test_volume_collection_error_is_explicit_without_replacing_scientific_success(run_fixture, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("fixture metadata unavailable")

    monkeypatch.setattr(run_statistics, "collect_volume_statistics", unavailable)
    task = _task(run_fixture, "slice")
    server_handlers.handle_task(task)
    record = _record(task)
    assert task.status == "done"
    assert record["outcome"] == "succeeded"
    assert record["volumes"]["status"] == "unavailable"
    assert "fixture metadata unavailable" in record["collection_errors"][0]


@pytest.mark.parametrize(("behavior", "exit_code"), [("returned", 1), ("interrupted", 130)])
def test_cli_failed_and_interrupted_runs_publish_before_nonzero_exit(run_fixture, behavior, exit_code):
    state = run_fixture
    state.behavior = behavior
    with pytest.raises(SystemExit) as error:
        cli.handle_slice(SimpleNamespace(options=str(state.slice_path), verbose=True))
    assert error.value.code == exit_code
    reports = list(state.output.glob("*statistics*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["outcome"] == ("failed" if behavior == "returned" else "interrupted")


@pytest.mark.parametrize("operation", ["slice", "backproject"])
def test_api_reporting_default_opt_out_and_publication_status(run_fixture, operation, monkeypatch):
    # sse-starlette retains a process-global event; each asyncio.run owns a new loop.
    monkeypatch.setattr(AppStatus, "should_exit_event", None)
    async def scenario():
        app = FastAPI()
        queued = []

        @app.middleware("http")
        async def queue_middleware(request: Request, call_next):
            request.state.queue = SimpleNamespace(put_nowait=queued.append)
            return await call_next(request)

        create_api(app)
        path = run_fixture.slice_path if operation == "slice" else run_fixture.backproject_path
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/{operation}/", params={"options": str(path)})
            assert response.status_code == 200
            task = queued[-1]
            assert task.reporting_enabled
            status = await client.get(f"/status/{task.task_id}")
            assert status.json()["statistics"]["status"] == "pending"
            server_handlers.handle_task(task)
            status = await client.get(f"/status/{task.task_id}")
            assert status.status_code == 200
            assert status.json()["status"] == "done"
            assert status.json()["statistics"] == task.statistics.publication
            assert status.json()["statistics"]["status"] == "published"

            task.statistics.publication["status"] = "pending"
            status = await client.get(f"/status/{task.task_id}")
            assert status.json()["status"] == "started"
            assert status.json()["error"] is None
            duplicate = await client.post(f"/{operation}/", params={"options": str(path)})
            assert duplicate.status_code == 409
            deleted = await client.post("/delete/", params={"task_id": task.task_id})
            assert deleted.status_code == 409
            assert len(queued) == 1
            task.statistics.publication["status"] = "published"

            def failed_progress():
                raise RuntimeError("fixture progress presentation failed")

            monkeypatch.setattr(task.pipeline, "get_steps_progress_and_durations", failed_progress)
            status = await client.get(f"/status/{task.task_id}")
            assert status.json()["status"] == "done"
            assert "fixture progress presentation failed" in status.json()["progress_error"]
            assert task.status == "done"
            assert task.statistics.to_record()["outcome"] == "succeeded"

            response = await client.post(
                f"/{operation}/", params={"options": str(path), "reporting_enabled": "false"},
            )
            assert response.status_code == 200
            disabled = queued[-1]
            assert not disabled.reporting_enabled
            server_handlers.handle_task(disabled)
            status = await client.get(f"/status/{disabled.task_id}")
            assert status.json()["statistics"]["status"] == "disabled"
            assert disabled.statistics is None
            assert len(run_fixture.collected) == 1

            task.statistics.publication.update(status="failed", error="fixture report error")
            status = await client.get(f"/status/{task.task_id}")
            assert status.status_code == 500
            assert status.json()["status"] == "done"
            assert status.json()["error"] == "fixture report error"
            stream = await client.get(
                f"/{operation}_status_stream/", params={"task_id": task.task_id, "update_freq": 1},
            )
            assert stream.text.count("event: error_event") == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["slice", "backproject"])
def test_reporting_toggle_preserves_live_progress_durations_and_terminal_stream(
    run_fixture, operation, monkeypatch,
):
    monkeypatch.setattr(AppStatus, "should_exit_event", None)
    entered, release = Event(), Event()
    progress_updates = []
    factory_name = f"{operation}_pipeline"
    factory = getattr(server_handlers, factory_name)

    def observed_pipeline(*args):
        pipeline, data = factory(*args)
        step = pipeline.steps[0]
        original_process = step._process
        step.listen_for_progress(progress_updates.append)

        def blocked_process(values):
            step.update_progress(0.4)
            entered.set()
            assert release.wait(5)
            return original_process(values)

        monkeypatch.setattr(step, "_process", blocked_process)
        return pipeline, data

    monkeypatch.setattr(server_handlers, factory_name, observed_pipeline)

    async def scenario():
        app = FastAPI()
        queued = []

        @app.middleware("http")
        async def queue_middleware(request: Request, call_next):
            request.state.queue = SimpleNamespace(put_nowait=queued.append)
            return await call_next(request)

        create_api(app)
        path = run_fixture.slice_path if operation == "slice" else run_fixture.backproject_path
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for enabled in (False, True, False):
                entered.clear()
                release.clear()
                progress_updates.clear()
                collected_before = len(run_fixture.collected)
                memory_before = len(run_fixture.memory_events)
                reports_before = len(list(run_fixture.output.glob("*statistics*.json")))
                response = await client.post(
                    f"/{operation}/", params={"options": str(path), "reporting_enabled": enabled},
                )
                assert response.status_code == 200
                task = queued[-1]
                assert task.reporting_enabled is enabled
                with ThreadPoolExecutor(1) as pool:
                    running = pool.submit(server_handlers.handle_task, task)
                    try:
                        assert entered.wait(2)
                        assert task.pipeline is run_fixture.pipelines[-1]
                        assert task.pipeline.statistics_enabled is enabled
                        status = await client.get(f"/status/{task.task_id}")
                        assert status.status_code == 200
                        live = status.json()
                        assert live["status"] == "started"
                        assert live["progress"] == task.pipeline.get_steps_progress_and_durations()
                        assert live["progress"][0][1] == 0.4
                        assert live["statistics"]["status"] == ("pending" if enabled else "disabled")
                    finally:
                        release.set()
                    assert running.result(timeout=2) is task
                status = (await client.get(f"/status/{task.task_id}")).json()
                assert status["status"] == "done"
                assert status["error"] is None and status["progress_error"] is None
                assert status["progress"] == task.pipeline.get_steps_progress_and_durations()
                assert status["progress"][0][1] == 1
                assert status["progress"][0][2] > 0
                assert progress_updates == [0, 0.4, 1]
                stream = await client.get(f"/{operation}_status_stream/", params={"task_id": task.task_id})
                assert stream.text.count("event: done_event") == 1
                payload, = [line.removeprefix("data: ") for line in stream.text.splitlines() if line.startswith("data: ")]
                assert json.loads(payload) == status
                assert len(run_fixture.collected) == collected_before + int(enabled)
                assert len(run_fixture.memory_events) == memory_before + 2 * int(enabled)
                assert len(list(run_fixture.output.glob("*statistics*.json"))) == reports_before + int(enabled)
                if not enabled:
                    assert task.statistics is None
                    assert task.pipeline.get_step_statistics()[0]["custom_times"] == {}

    asyncio.run(scenario())


def test_disabled_reporting_keeps_failed_task_active_through_docker_cleanup(run_fixture, monkeypatch):
    state = run_fixture
    state.behavior = "returned"
    _docker_transport(state, monkeypatch)
    entered, release = Event(), Event()
    original_cleanup = server_handlers.clear_plugin_folder

    def blocked_cleanup():
        entered.set()
        assert release.wait(5)
        return original_cleanup()

    monkeypatch.setattr(server_handlers, "clear_plugin_folder", blocked_cleanup)

    async def scenario():
        app = FastAPI()
        queued = []

        @app.middleware("http")
        async def queue_middleware(request: Request, call_next):
            request.state.queue = SimpleNamespace(put_nowait=queued.append)
            return await call_next(request)

        create_api(app, docker=True)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/slice/", params={"options": str(state.slice_path),
                                                          "reporting_enabled": "false"})
            assert response.status_code == 200
            task = queued[0]
            with ThreadPoolExecutor(1) as pool:
                running = pool.submit(server_handlers.handle_task_docker, task)
                try:
                    assert entered.wait(2)
                    assert not running.done()
                    assert task.status == "started" and task.error is None
                    assert task.is_running and task.is_active
                    status = await client.get(f"/status/{task.task_id}")
                    assert status.status_code == 200
                    assert status.json()["status"] == "started"
                    assert status.json()["statistics"]["status"] == "disabled"
                    duplicate = await client.post("/slice/", params={"options": str(state.slice_path)})
                    assert duplicate.status_code == 409
                    deleted = await client.post("/delete/", params={"task_id": task.task_id})
                    assert deleted.status_code == 409
                finally:
                    release.set()
                assert running.result(timeout=2) is task
            assert not task.is_active
            status = await client.get(f"/status/{task.task_id}")
            assert status.status_code == 500
            assert status.json()["status"] == "error"
            assert status.json()["error"] == "returned failure"

    asyncio.run(scenario())


@pytest.mark.parametrize(("failure", "exit_code"), [(BrokenPipeError("closed stdout"), 1), (KeyboardInterrupt(), 130)])
def test_cli_presentation_failure_preserves_delivered_report(run_fixture, monkeypatch, capsys, failure, exit_code):
    tasks = []

    def execute(task, **kwargs):
        tasks.append(task)
        return server_handlers.handle_task(task, **kwargs)

    def failed_print(*args):
        raise failure

    monkeypatch.setattr(cli, "handle_task", execute)
    monkeypatch.setattr(server_handlers, "print", failed_print, raising=False)
    with pytest.raises(SystemExit) as exited:
        cli.handle_slice(SimpleNamespace(options=str(run_fixture.slice_path), verbose=True))
    assert exited.value.code == exit_code
    task = tasks[0]
    assert task.status == "done" and task.error is None
    assert not task.is_active
    assert task.statistics.publication["status"] == "published"
    assert task.statistics.publication["error"] is None
    assert type(failure).__name__ in task.statistics.publication["presentation_error"]
    assert task.statistics.publication["presentation_interrupted"] is (exit_code == 130)
    assert _record(task)["outcome"] == "succeeded"
    assert type(failure).__name__ in capsys.readouterr().err


def test_logging_failure_cannot_replace_delivered_file_or_escape_handler(run_fixture, monkeypatch):
    def failed_log(*args):
        raise OSError("logging unavailable")

    monkeypatch.setattr(server_handlers.logger, "info", failed_log)
    monkeypatch.setattr(server_handlers.logger, "error", failed_log)
    task = _task(run_fixture, "slice")
    assert server_handlers.handle_task(task) is task
    assert task.status == "done" and not task.is_active
    assert task.statistics.publication["status"] == "published"
    assert "logging unavailable" in task.statistics.publication["presentation_error"]
    assert _record(task)["outcome"] == "succeeded"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_docker_report_transfer_retains_recovery_and_scientific_outcome(
    run_fixture, monkeypatch, interruption,
):
    state = run_fixture
    _docker_transport(state, monkeypatch)
    original_export = file_system.copy_to_host

    def interrupted_export(files, plugin_folder_name="main"):
        if plugin_folder_name == "main-statistics":
            raise interruption()
        return original_export(files, plugin_folder_name)

    monkeypatch.setattr(file_system, "copy_to_host", interrupted_export)
    task = _task(state, "slice")
    assert server_handlers.handle_task_docker(task) is task
    assert task.status == "done" and task.error is None
    assert not task.is_running and not task.is_active
    publication = task.statistics.publication
    assert publication["status"] == "interrupted"
    assert publication["error"] == interruption.__name__
    assert publication["path"] is not None
    recovery = Path(publication["recovery_path"])
    assert json.loads(recovery.read_text())["outcome"] == "succeeded"
    assert not (state.volume / "main").exists()
    server_handlers.clear_plugin_folder()
    assert recovery.exists()


def test_report_serialization_interrupt_finalizes_publication_without_changing_science(run_fixture, monkeypatch):
    def interrupted_record(self):
        raise KeyboardInterrupt()

    monkeypatch.setattr(run_statistics.RunStatistics, "to_record", interrupted_record)
    task = _task(run_fixture, "slice")
    assert server_handlers.handle_task(task) is task
    assert task.status == "done" and task.error is None
    assert not task.is_active
    assert task.statistics.elapsed_seconds > 0
    assert task.statistics.publication["status"] == "interrupted"
    assert task.statistics.publication["recovery_path"] is None
    assert "KeyboardInterrupt" in task.statistics.publication["error"]
    assert list(run_fixture.output.glob("*statistics*.json")) == []

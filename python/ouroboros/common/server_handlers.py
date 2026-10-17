"""Shared CLI/server operation lifecycle, including scientific output delivery."""

from ouroboros.common.file_system import (
    load_options_for_backproject, load_options_for_backproject_docker,
    load_options_for_slice, load_options_for_slice_docker,
    save_output_for_backproject_docker, save_output_for_slice_docker,
)
from ouroboros.common.logging import logger
from ouroboros.common.pipelines import backproject_pipeline, slice_pipeline
from ouroboros.common.run_statistics import RunStatistics
from ouroboros.common.server_types import BackProjectTask, SliceTask, Task
from ouroboros.common.volume_server_interface import clear_plugin_folder
from ouroboros.helpers.files import combine_unknown_folder
from ouroboros.helpers.options import BackprojectOptions, SliceOptions


def _configure(task, options):
    if task.statistics is not None:
        task.statistics.configure(options)


def _process(task, pipeline, input_data):
    task.pipeline = pipeline
    task.pipeline_input = input_data
    pipeline.set_statistics_enabled(task.reporting_enabled)
    try:
        return pipeline.process(input_data)
    finally:
        if task.statistics is not None:
            task.statistics.capture_volumes()


def handle_slice_core(task: SliceTask, slice_options: SliceOptions):
    pipeline, input_data = slice_pipeline(slice_options, task.show_progress)
    _, error = _process(task, pipeline, input_data)
    return error


def handle_backproject_core(task: BackProjectTask, options: BackprojectOptions, slice_options: SliceOptions):
    pipeline, input_data = backproject_pipeline(options, slice_options, task.show_progress)
    output, error = _process(task, pipeline, input_data)
    return error if error is not None else output


def handle_slice(task: SliceTask):
    options = load_options_for_slice(task.options)
    if isinstance(options, str):
        return options
    _configure(task, options)
    return handle_slice_core(task, options)


def handle_slice_docker(task: SliceTask):
    loaded = load_options_for_slice_docker(task.options, on_options=lambda options: _configure(task, options))
    if isinstance(loaded, str):
        return loaded
    options, host_output_file, host_output_slices = loaded
    error = handle_slice_core(task, options)
    if error is not None:
        return error
    return save_output_for_slice_docker(host_output_file, host_output_slices=host_output_slices,
                                        cleanup=False)


def handle_backproject(task: BackProjectTask):
    options = load_options_for_backproject(task.options)
    if isinstance(options, str):
        return options
    _configure(task, options)
    slice_options = load_options_for_slice(options.slice_options_path)
    if isinstance(slice_options, str):
        return slice_options
    _configure(task, slice_options)
    result = handle_backproject_core(task, options, slice_options)
    return result if isinstance(result, str) else None


def handle_backproject_docker(task: BackProjectTask):
    loaded = load_options_for_backproject_docker(task.options,
                                                on_options=lambda options: _configure(task, options))
    if isinstance(loaded, str):
        return loaded
    options, slice_options, host_output_file, host_output_slices, host_output_folder = loaded
    result = handle_backproject_core(task, options, slice_options)
    if isinstance(result, str):
        return result
    if options.make_single_file:
        host_output_file = combine_unknown_folder(host_output_folder, result.output_file_path)
    else:
        host_output_slices = combine_unknown_folder(host_output_folder, result.output_file_path)
    return save_output_for_backproject_docker(host_output_file, host_output_slices=host_output_slices,
                                              cleanup=False)


def _execute_task(task: Task, *, docker: bool = False, print_statistics: bool = False):
    # Task owns lifecycle state for CLI and queued server execution.
    task.status = "started"
    task.error = None
    task.statistics = RunStatistics(task, docker=docker) if task.reporting_enabled else None
    outcome, failure = "done", None
    try:
        if isinstance(task, SliceTask):
            error = (handle_slice_docker if docker else handle_slice)(task)
        elif isinstance(task, BackProjectTask):
            error = (handle_backproject_docker if docker else handle_backproject)(task)
        else:
            raise ValueError("Invalid task type")
        if error is not None:
            outcome, failure = "error", error
    except (KeyboardInterrupt, SystemExit) as error:
        outcome, failure = "interrupted", str(error) or type(error).__name__
    except Exception as error:
        outcome, failure = "error", str(error) or type(error).__name__
    finally:
        if docker:
            try:
                success, error = clear_plugin_folder()
                if not success:
                    raise OSError(error or "Scientific output cleanup failed.")
            except (KeyboardInterrupt, SystemExit) as error:
                outcome = "interrupted"
                failure = f"{failure + '; ' if failure else ''}Cleanup interrupted: {str(error) or type(error).__name__}"
            except Exception as error:
                failure = f"{failure + '; ' if failure else ''}Cleanup failed: {error}"
                if outcome != "interrupted":
                    outcome = "error"
        # Cleanup belongs to execution even when optional reporting is disabled.
        # Commit the terminal task state only after the operation boundary closes.
        task.status, task.error = outcome, failure
        if task.statistics is not None:
            task.statistics.finish()
            formatted = None
            try:
                formatted = task.statistics.publish()
            except (Exception, KeyboardInterrupt, SystemExit) as error:
                task.statistics.publication.update(
                    status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                    error=f"Statistics publication failed: {str(error) or type(error).__name__}",
                )
            if formatted is not None:
                try:
                    if print_statistics:
                        print("\nCalculation Statistics:\n")
                        print(formatted)
                    else:
                        logger.info("%s Pipeline Statistics:\n%s", task.statistics.operation, formatted)
                except (Exception, KeyboardInterrupt, SystemExit) as error:
                    task.statistics.presentation_failed(error)
            publication = task.statistics.publication
            if publication["error"] or publication.get("presentation_error"):
                try:
                    logger.error("%s", publication["error"] or publication["presentation_error"])
                except (Exception, KeyboardInterrupt, SystemExit) as error:
                    task.statistics.presentation_failed(error)
    return task


def handle_task(task: Task, *, print_statistics: bool = False):
    return _execute_task(task, print_statistics=print_statistics)


def handle_task_docker(task: Task):
    return _execute_task(task, docker=True)

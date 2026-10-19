import json
from fastapi import FastAPI, Request

from sse_starlette.sse import EventSourceResponse
from fastapi.responses import JSONResponse
import asyncio
import uuid

from ouroboros.common.file_system import (
    load_options_for_slice,
    load_options_for_slice_docker,
)
from ouroboros.common.pipelines import visualization_pipeline
from ouroboros.common.server_types import BackProjectTask, SliceTask
from ouroboros.common.step_names import step_names_payload


def create_api(app: FastAPI, docker: bool = False):
    """
    Create the API for the Ouroboros server.

    Parameters
    ----------
    app : FastAPI
        The FastAPI server.
    docker : bool, optional
        Whether the server is running in a Docker container, by default False
    """

    tasks = {}

    def has_active_task(task_type):
        return any(
            isinstance(task, task_type)
            and task.is_active
            for task in tasks.values()
        )

    @app.get("/")
    async def server_active():
        return JSONResponse("Server is active")

    @app.get("/step-names")
    async def get_step_names():
        """Return the canonical pipeline step names.

        The renderer uses these to identify progress rows (see
        ``SLICE_STEP_NAME`` in ``SlicesPage.tsx``). Exposing them through
        the API means a rename on the Python side surfaces as a drift
        warning in the renderer instead of silently blanking the row.
        """

        return JSONResponse(step_names_payload())

    @app.post("/slice/")
    async def add_slice_task(options: str, request: Request, reporting_enabled: bool = True):
        if has_active_task(SliceTask):
            return JSONResponse(
                {"detail": "A slice task is already queued or running."},
                status_code=409,
            )

        task_id = str(uuid.uuid4())
        task = SliceTask(task_id=task_id, options=options, reporting_enabled=reporting_enabled)
        tasks[task_id] = task
        request.state.queue.put_nowait(task)  # Add request to the queue
        return {"task_id": task_id}

    @app.get("/slice_visualization/")
    async def get_slice_visualization(task_id: str):
        """
        Get the slice visualization data for an existing task.

        Parameters
        ----------
        task_id : str
            The ID of the task.

        Returns
        -------
        JSONResponse
            The slice visualization data.
        """

        result = {
            "data": None,
            "error": None,
        }

        if task_id in tasks:
            task = tasks[task_id]
            if task.status == "done":
                data = {
                    "rects": task.pipeline_input.slice_rects.tolist(),
                    "bounding_boxes": [
                        {
                            "min": [bbox.x_min, bbox.y_min, bbox.z_min],
                            "max": [bbox.x_max, bbox.y_max, bbox.z_max],
                        }
                        for bbox in task.pipeline_input.volume_cache.bounding_boxes
                    ],
                    "link_rects": task.pipeline_input.volume_cache.link_rects,
                }
                result["data"] = data

                return JSONResponse(result, status_code=200)
            else:
                result["error"] = "Task is not done."
                return JSONResponse(result, status_code=400)
        else:
            result["error"] = "Item ID Not Found"
            return JSONResponse(result, status_code=404)

    @app.get("/create_slice_visualization/")
    async def on_demand_slice_visualization(options: str):
        """
        Create the slice visualization data from the options.

        Parameters
        ----------
        options : str
            The options for slicing the volume.

        Returns
        -------
        JSONResponse
            The slice visualization data.
        """

        result = {
            "data": None,
            "error": None,
        }

        try:
            load_result = (
                load_options_for_slice_docker(options)
                if docker
                else load_options_for_slice(options)
            )
        except BaseException as e:
            result["error"] = f"Error loading options: {str(e)}"
            return JSONResponse(result, status_code=400)

        if isinstance(load_result, str):
            result["error"] = load_result
            return JSONResponse(result, status_code=400)

        slice_options = load_result[0] if docker else load_result

        pipeline, input_data = visualization_pipeline(slice_options)

        _, error = pipeline.process(input_data)

        if error:
            result["error"] = error
            return JSONResponse(result, status_code=400)

        data = {
            "rects": input_data.slice_rects.tolist(),
            "bounding_boxes": [
                {
                    "min": [bbox.x_min, bbox.y_min, bbox.z_min],
                    "max": [bbox.x_max, bbox.y_max, bbox.z_max],
                }
                for bbox in input_data.volume_cache.bounding_boxes
            ],
            "link_rects": input_data.volume_cache.link_rects,
        }
        result["data"] = data

        return JSONResponse(result, status_code=200)

    @app.post("/backproject/")
    async def add_backproject_task(
        request: Request,
        options: str,
        reporting_enabled: bool = True,
    ):
        if has_active_task(BackProjectTask):
            return JSONResponse(
                {"detail": "A backprojection task is already queued or running."},
                status_code=409,
            )

        task_id = str(uuid.uuid4())
        task = BackProjectTask(
            task_id=task_id,
            options=options,
            reporting_enabled=reporting_enabled,
        )
        tasks[task_id] = task
        request.state.queue.put_nowait(task)  # Add request to the queue
        return {"task_id": task_id}

    def get_status(task_id: str):
        if task_id in tasks:
            task = tasks[task_id]
            progress_error = None
            progress = []
            steps = []
            if task.pipeline is not None:
                try:
                    progress = task.pipeline.get_steps_progress_and_durations()
                    steps = task.pipeline.get_steps_status()
                except BaseException as e:
                    progress_error = f"Error occurred while getting progress details: {e}."
            publication = (task.statistics.publication if task.statistics is not None else
                           {"status": "pending" if task.reporting_enabled else "disabled",
                            "path": None, "recovery_path": None, "error": None,
                            "presentation_error": None, "presentation_interrupted": False})
            publishing = task.publication_pending
            return {
                "status": "started" if publishing else task.status,
                "phase": task.phase,
                "failure_phase": task.failure_phase,
                "steps": steps,
                "progress": progress,
                "error": None if publishing else (
                    task.error or publication["error"] or publication.get("presentation_error")
                ),
                "progress_error": progress_error,
                "statistics": publication,
            }
        else:
            return {
                "status": "error",
                "phase": "complete",
                "failure_phase": None,
                "steps": [],
                "progress": [],
                "error": "Item ID Not Found",
            }

    @app.get("/status/{task_id}")
    async def check_status(task_id: str):
        result = get_status(task_id)

        if result["error"] == "Item ID Not Found":
            return JSONResponse(result, status_code=404)
        elif result["error"]:
            return JSONResponse(result, status_code=500)
        else:
            return JSONResponse(result, status_code=200)

    @app.get("/slice_status_stream/")
    async def slice_status_stream(
        request: Request, task_id: str, update_freq: int = 1000
    ):
        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break

                result = get_status(task_id)

                event = ""

                match result["status"]:
                    case "error" | "interrupted":
                        event = "error_event"
                    case "done":
                        event = "done_event"
                    case _:
                        event = "update_event"

                if result["error"]:
                    event = "error_event"

                yield {
                    "event": event,
                    "id": task_id,
                    "retry": update_freq,
                    "data": json.dumps(result),
                }

                if event in {"done_event", "error_event"}:
                    break

                await asyncio.sleep(update_freq / 1000.0)

        return EventSourceResponse(event_generator())

    @app.get("/backproject_status_stream/")
    async def backproject_status_stream(
        request: Request, task_id: str, update_freq: int = 2000
    ):
        async def event_generator():
            while True:
                if await request.is_disconnected():
                    break

                result = get_status(task_id)

                event = ""

                match result["status"]:
                    case "error" | "interrupted":
                        event = "error_event"
                    case "done":
                        event = "done_event"
                    case _:
                        event = "update_event"

                if result["error"]:
                    event = "error_event"

                yield {
                    "event": event,
                    "id": task_id,
                    "retry": update_freq,
                    "data": json.dumps(result),
                }

                if event in {"done_event", "error_event"}:
                    break

                await asyncio.sleep(update_freq / 1000.0)

        return EventSourceResponse(event_generator())

    @app.post("/delete/")
    async def delete_task(task_id: str):
        if task_id in tasks:
            task = tasks[task_id]
            if task.is_running:
                return JSONResponse(
                    {
                        "success": False,
                        "detail": "A running task cannot be deleted.",
                    },
                    status_code=409,
                )

            if task.status == "enqueued":
                task.status = "cancelled"

            del tasks[task_id]
            return JSONResponse({"success": True}, status_code=200)
        else:
            return JSONResponse({"success": False}, status_code=404)

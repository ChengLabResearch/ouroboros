import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from ouroboros.common.server import process_requests
from ouroboros.common.server_api import create_api
from ouroboros.common.server_types import BackProjectTask, SliceTask


class RecordingQueue:
    def __init__(self):
        self.tasks = []

    def put_nowait(self, task):
        self.tasks.append(task)


def create_test_app():
    app = FastAPI()
    queue = RecordingQueue()

    @app.middleware("http")
    async def add_queue_to_request(request: Request, call_next):
        request.state.queue = queue
        return await call_next(request)

    create_api(app)
    return app, queue


def run_scenario(scenario):
    asyncio.run(scenario())


def test_duplicate_task_is_rejected_until_active_task_finishes():
    async def scenario():
        app, queue = create_test_app()
        transport = ASGITransport(app=app)
        task_types = [
            (
                SliceTask,
                "/slice/",
                "A slice task is already queued or running.",
            ),
            (
                BackProjectTask,
                "/backproject/",
                "A backprojection task is already queued or running.",
            ),
        ]

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for task_type, start_path, expected_detail in task_types:
                first = await client.post(
                    start_path, params={"options": "options.json"}
                )
                assert first.status_code == 200
                task = queue.tasks[-1]
                assert isinstance(task, task_type)

                duplicate = await client.post(
                    start_path, params={"options": "options.json"}
                )
                assert duplicate.status_code == 409
                assert duplicate.json() == {"detail": expected_detail}

                task.status = "started"
                running_delete = await client.post(
                    "/delete/", params={"task_id": first.json()["task_id"]}
                )
                assert running_delete.status_code == 409

                task.status = "done"
                next_task = await client.post(
                    start_path, params={"options": "options.json"}
                )
                assert next_task.status_code == 200
                queue.tasks[-1].status = "done"

    run_scenario(scenario)


def test_deleted_queued_task_is_skipped_by_worker():
    async def scenario():
        app, recording_queue = create_test_app()
        transport = ASGITransport(app=app)

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/backproject/", params={"options": "backproject.json"}
            )
            assert response.status_code == 200
            task = recording_queue.tasks[-1]

            deleted = await client.post(
                "/delete/", params={"task_id": response.json()["task_id"]}
            )
            assert deleted.status_code == 200
            assert task.status == "cancelled"

        handled_tasks = []
        queue = asyncio.Queue()
        queue.put_nowait(task)
        with ThreadPoolExecutor() as pool:
            worker = asyncio.create_task(
                process_requests(queue, pool, handled_tasks.append)
            )
            await asyncio.wait_for(queue.join(), timeout=1)
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker

        assert handled_tasks == []

    run_scenario(scenario)


def test_error_status_streams_terminate():
    async def scenario():
        app, queue = create_test_app()
        transport = ASGITransport(app=app)
        stream_paths = [
            ("/slice/", "/slice_status_stream/"),
            ("/backproject/", "/backproject_status_stream/"),
        ]

        async with AsyncClient(transport=transport, base_url="http://test") as client:
            for start_path, stream_path in stream_paths:
                response = await client.post(
                    start_path, params={"options": "options.json"}
                )
                assert response.status_code == 200
                task = queue.tasks[-1]
                task.status = "error"
                task.error = "Pipeline failed"

                stream = await client.get(
                    stream_path,
                    params={"task_id": task.task_id, "update_freq": 1},
                )

                assert stream.status_code == 200
                assert stream.text.count("event: error_event") == 1
                assert "Pipeline failed" in stream.text

    run_scenario(scenario)

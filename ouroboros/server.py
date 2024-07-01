from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from concurrent.futures import Executor, ThreadPoolExecutor
import asyncio
import uvicorn
import uuid

from ouroboros.helpers.config import Config
from ouroboros.pipeline import (
    BackprojectPipelineStep,
    LoadConfigPipelineStep,
    ParseJSONPipelineStep,
    Pipeline,
    PipelineInput,
    SaveConfigPipelineStep,
    SaveParallelPipelineStep,
    SlicesGeometryPipelineStep,
    VolumeCachePipelineStep,
)

HOST = "0.0.0.0"
PORT = 8000


# Note: kw_only=True is used to make the fields keyword-only,
# which is required for the parent dataclass to have default values
@dataclass(kw_only=True)
class Task:
    task_id: int
    pipeline_input: PipelineInput = None
    pipeline: Pipeline = None
    last_progress: list[tuple[str, float]] = field(default_factory=list)
    status: str = "enqueued"
    error: str = None


@dataclass(kw_only=True)
class SliceTask(Task):
    neuroglancer_json: str
    options: str


@dataclass(kw_only=True)
class BackProjectTask(Task):
    straightened_volume_path: str
    config: str
    options: str | None = None


def handle_slice(task: SliceTask):
    neuroglancer_json_path = task.neuroglancer_json
    options_path = task.options

    config = Config.from_json(options_path)

    pipeline = Pipeline(
        [
            ParseJSONPipelineStep(),
            SlicesGeometryPipelineStep(),
            VolumeCachePipelineStep(),
            SaveParallelPipelineStep(),
            SaveConfigPipelineStep(),
        ]
    )

    # Store the pipeline in the task
    task.pipeline = pipeline

    input_data = PipelineInput(config=config, json_path=neuroglancer_json_path)

    # Store the input data in the task
    task.pipeline_input = input_data

    _, error = pipeline.process(input_data)

    if error:
        task.error = error
        task.status = "error"


def handle_backproject(task: BackProjectTask):
    straightened_volume_path = task.straightened_volume_path
    config_path = task.config
    options_path = task.options

    options = None

    if options_path:
        options = Config.from_json(options_path)

    pipeline = Pipeline(
        [
            LoadConfigPipelineStep()
            .with_custom_output_file_path(straightened_volume_path)
            .with_custom_options(options),
            BackprojectPipelineStep(),
            SaveConfigPipelineStep(),
        ]
    )

    # Store the pipeline in the task
    task.pipeline = pipeline

    input_data = PipelineInput(config_file_path=config_path)

    # Store the input data in the task
    task.pipeline_input = input_data

    _, error = pipeline.process(input_data)

    if error:
        task.error = error
        task.status = "error"


def handle_task(task: Task):
    task.status = "started"

    if isinstance(task, SliceTask):
        handle_slice(task)
    elif isinstance(task, BackProjectTask):
        handle_backproject(task)
    else:
        raise ValueError("Invalid task type")

    if task.status != "error":
        task.status = "done"


async def process_requests(queue: asyncio.Queue, pool: Executor):
    while True:
        task = await queue.get()
        loop = asyncio.get_running_loop()
        task.status = "started"
        await loop.run_in_executor(pool, handle_task, task)
        queue.task_done()
        task.status = "done"


@asynccontextmanager
async def lifespan(app: FastAPI):
    queue = asyncio.Queue()  # note that asyncio.Queue() is not thread safe
    pool = ThreadPoolExecutor()
    asyncio.create_task(process_requests(queue, pool))
    yield {"queue": queue, "pool": pool}
    pool.shutdown()


tasks = {}
app = FastAPI(lifespan=lifespan)


@app.post("/slice/")
async def add_slice_task(neuroglancer_json: str, options: str, request: Request):
    task_id = str(uuid.uuid4())
    task = SliceTask(
        task_id=task_id, neuroglancer_json=neuroglancer_json, options=options
    )
    tasks[task_id] = task
    request.state.queue.put_nowait(task)  # Add request to the queue
    return task_id


@app.post("/backproject/")
async def add_backproject_task(
    request: Request,
    straightened_volume_path: str,
    config: str,
    options: str | None = None,
):
    task_id = str(uuid.uuid4())
    task = BackProjectTask(
        task_id=task_id,
        straightened_volume_path=straightened_volume_path,
        config=config,
        options=options,
    )
    tasks[task_id] = task
    request.state.queue.put_nowait(task)  # Add request to the queue
    return task_id


@app.get("/status/{task_id}")
async def check_status(task_id: str):
    if task_id in tasks:
        task = tasks[task_id]
        if task.status == "started" or task.status == "done":
            try:
                task.last_progress = tasks[task_id].pipeline.get_steps_progress()
            except BaseException as e:
                task.status = "error"
                task.error = f"Error occurred while getting progress details: {str(e)}."
                task.last_progress = []
        return {
            "status": task.status,
            "progress": task.last_progress,
            "error": task.error,
        }
    else:
        return JSONResponse("Item ID Not Found", status_code=404)


@app.delete("/delete/{task_id}")
async def delete_task(task_id: str):
    if task_id in tasks:
        del tasks[task_id]
        return JSONResponse("Task Deleted", status_code=200)
    else:
        return JSONResponse("Item ID Not Found", status_code=404)


def main():
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()

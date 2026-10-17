from dataclasses import dataclass
from typing import TYPE_CHECKING
from ouroboros.pipeline import PipelineInput, Pipeline

if TYPE_CHECKING:
    from ouroboros.common.run_statistics import RunStatistics


# Note: kw_only=True is used to make the fields keyword-only,
# which is required for the parent dataclass to have default values
@dataclass(kw_only=True)
class Task:
    task_id: str
    pipeline_input: PipelineInput = None
    pipeline: Pipeline = None
    status: str = "enqueued"
    error: str = None
    reporting_enabled: bool = True
    show_progress: bool = False
    statistics: "RunStatistics | None" = None

    @property
    def publication_pending(self) -> bool:
        return self.statistics is not None and self.statistics.publication["status"] == "pending"

    @property
    def is_running(self) -> bool:
        return self.status == "started" or self.publication_pending

    @property
    def is_active(self) -> bool:
        return self.status == "enqueued" or self.is_running


@dataclass(kw_only=True)
class SliceTask(Task):
    options: str


@dataclass(kw_only=True)
class BackProjectTask(Task):
    options: str

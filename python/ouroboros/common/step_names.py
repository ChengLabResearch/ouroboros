"""Canonical names for pipeline steps, exposed both to Python callers and to
the FastAPI client through the ``/step-names`` endpoint.

The renderer identifies progress rows by the *class name* of the pipeline
step (see ``SLICE_STEP_NAME`` in
``src/renderer/src/routes/SlicesPage/SlicesPage.tsx``). Keeping the canonical
list in one Python module and exposing it through the API means a rename on
either side surfaces immediately instead of silently blanking the progress
UI.

Follow-up: the renderer should consume ``/step-names`` at runtime in
development builds to warn on drift. Tracked as
https://github.com/ChengLabResearch/ouroboros/issues/107.
"""

from enum import Enum


class StepName(str, Enum):
    """Canonical pipeline step names.

    Values match the ``__class__.__name__`` of each ``PipelineStep``
    subclass in ``ouroboros.pipeline``. Any rename on the class side must
    be reflected here in the same commit.
    """

    PARSE_JSON = "ParseJSONPipelineStep"
    SLICES_GEOMETRY = "SlicesGeometryPipelineStep"
    VOLUME_CACHE = "VolumeCachePipelineStep"
    SLICE_PARALLEL = "SliceParallelPipelineStep"
    BACKPROJECT = "BackprojectPipelineStep"
    SAVE_CONFIG = "SaveConfigPipelineStep"
    LOAD_CONFIG = "LoadConfigPipelineStep"


# Convenience alias kept for readability in call sites that only reference
# one specific step name.
SLICE_PARALLEL_PIPELINE_STEP = StepName.SLICE_PARALLEL.value


def step_names_payload() -> dict[str, str]:
    """Return the enum as a ``{member_name: class_name}`` mapping.

    The renderer receives a stable JSON object; new steps appended to
    ``StepName`` show up as new keys without breaking existing consumers.
    """

    return {member.name: member.value for member in StepName}

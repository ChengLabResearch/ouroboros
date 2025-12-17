from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline import (
    Pipeline,
    PipelineInput,
    ParseJSONPipelineStep,
    SlicesGeometryPipelineStep,
    VolumeCachePipelineStep,
    SliceParallelPipelineStep,
    BackprojectPipelineStep,
)


def geometry_pipeline(slice_options: SliceOptions) -> tuple[Pipeline, PipelineInput]:
    """
    Creates a pipeline for slicing a volume, as well as the default input data for the pipeline.

    Parameters
    ----------
    slice_options : SliceOptions
        The options for slicing the volume.
    include-ng : bool, optional
        Whether to include neuroglancer output data, default false.

    Returns
    -------
    tuple[Pipeline, PipelineInput]
        The pipeline for slicing the volume and the default input data for the pipeline
    """

    pipeline = Pipeline(
        [
            ParseJSONPipelineStep(),
            SlicesGeometryPipelineStep()
        ]
    )

    default_input_data = PipelineInput(
        slice_options=slice_options, json_path=slice_options.neuroglancer_json
    )

    return pipeline, default_input_data


def slice_pipeline(slice_options: SliceOptions, verbose: bool = False) -> tuple[Pipeline, PipelineInput]:
    """
    Creates a pipeline for slicing a volume, as well as the default input data for the pipeline.

    Parameters
    ----------
    slice_options : SliceOptions
        The options for slicing the volume.
    verbose : bool, optional
        Whether to show a progress bar for the pipeline, by default False

    Returns
    -------
    tuple[Pipeline, PipelineInput]
        The pipeline for slicing the volume and the default input data for the pipeline
    """

    pipeline = Pipeline(
        [
            ParseJSONPipelineStep(),
            SlicesGeometryPipelineStep(),
            VolumeCachePipelineStep(),
            (
                SliceParallelPipelineStep().with_progress_bar()
                if verbose
                else SliceParallelPipelineStep()
            ),
        ]
    )

    default_input_data = PipelineInput(
        slice_options=slice_options, json_path=slice_options.neuroglancer_json
    )

    return pipeline, default_input_data


def backproject_pipeline(
    backproject_options: BackprojectOptions,
    slice_options: SliceOptions,
    verbose: bool = False,
) -> tuple[Pipeline, PipelineInput]:
    """
    Creates a pipeline for backprojecting a volume, as well as the default input data for the pipeline.

    Parameters
    ----------
    backproject_options : BackprojectOptions
        The options for backprojecting the volume.
    slice_options : SliceOptions
        The options for slicing the volume.
    verbose : bool, optional
        Whether to show a progress bar for the pipeline, by default False

    Returns
    -------
    tuple[Pipeline, PipelineInput]
        The pipeline for backprojecting the volume and the default input data for the pipeline
    """

    pipeline = Pipeline(
        [
            ParseJSONPipelineStep(),
            SlicesGeometryPipelineStep(),
            VolumeCachePipelineStep(),
            (
                BackprojectPipelineStep().with_progress_bar()
                if verbose
                else BackprojectPipelineStep()
            ),
        ]
    )

    default_input_data = PipelineInput(
        slice_options=slice_options,
        backproject_options=backproject_options,
        json_path=slice_options.neuroglancer_json,
    )

    return pipeline, default_input_data


def visualization_pipeline(slice_options: SliceOptions) -> tuple[Pipeline, PipelineInput]:
    """
    Creates a pipeline for visualizing the slicing process.

    Parameters
    ----------
    slice_options : SliceOptions
        The options for slicing the volume.
    verbose : bool, optional
        Whether to show a progress bar for the pipeline, by default False

    Returns
    -------
    tuple[Pipeline, PipelineInput]
        The pipeline for slicing the volume and the default input data for the pipeline
    """

    pipeline = Pipeline(
        [
            ParseJSONPipelineStep(),
            SlicesGeometryPipelineStep(),
            VolumeCachePipelineStep(),
        ]
    )

    default_input_data = PipelineInput(
        slice_options=slice_options, json_path=slice_options.neuroglancer_json
    )

    return pipeline, default_input_data

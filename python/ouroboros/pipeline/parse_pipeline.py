from ouroboros.helpers.parse import (
    parse_neuroglancer_json,
    neuroglancer_config_to_annotation,
    neuroglancer_config_to_source,
)
from .pipeline import PipelineStep
from ouroboros.helpers.options import SliceOptions
import numpy as np


# TODO: Consider making an abstract parse step
class ParseJSONPipelineStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(inputs=("slice_options", "json_path"))

    def _process(self, input_data: tuple[any]) -> None | str:
        config, json_path, pipeline_input = input_data

        # Verify that a config object is provided
        if not isinstance(config, SliceOptions):
            return "Input data must contain a SliceOptions object."

        # Verify that input_data is a string containing a path to a JSON file
        if not isinstance(json_path, str):
            return "Input data must contain a string containing a path to a JSON file."

        ng_config, error = parse_neuroglancer_json(json_path)

        self.update_progress(0.5)

        if error:
            return error

        annotation, error = neuroglancer_config_to_annotation(
            ng_config, config.neuroglancer_annotation_layer
        )

        if error:
            return error

        sample_points = annotation.points
        # Connect the end point to the first couple points if that option is enabled
        if config.connect_start_and_end:
            sample_points = np.concatenate((sample_points, sample_points[0:1]), axis=0)

        source_url, error = neuroglancer_config_to_source(
            ng_config, config.neuroglancer_image_layer,
            require_untransformed=annotation.resolution_nm is not None,
        )

        if error:
            return error

        # Store the source url in pipeline input for later use
        pipeline_input.source_url = source_url

        # Update the pipeline input with the sample points
        pipeline_input.sample_points = sample_points
        pipeline_input.sample_points_resolution_nm = annotation.resolution_nm

        return None

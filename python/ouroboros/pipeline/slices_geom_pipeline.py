from ouroboros.helpers.coordinates import convert_points_between_volumes
from ouroboros.helpers.slice import calculate_slice_rects
from ouroboros.helpers.spline import Spline
from ouroboros.helpers.volume_cache import get_mip_volume_sizes
from .pipeline import PipelineStep
from ouroboros.helpers.options import SliceOptions
import numpy as np


class SlicesGeometryPipelineStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(inputs=("slice_options", "sample_points", "source_url"))

    def _process(self, input_data: tuple[any]) -> None | str:
        config, sample_points, source_url, pipeline_input = input_data

        # Verify that a config object is provided
        if not isinstance(config, SliceOptions):
            return "Input data must contain a SliceOptions object."

        # Verify that sample points is given
        if not isinstance(sample_points, np.ndarray):
            return "Input data must contain an array of sample points."

        # Remove duplicates from sample points.
        _, indicies = np.unique(sample_points.round(decimals=2), axis=0, return_index=True)
        sample_points = sample_points[sorted(indicies)]

        # Rescale the sample points if the option is enabled
        if config.annotation_mip_level != config.output_mip_level:
            mip_sizes = get_mip_volume_sizes(source_url)

            if len(mip_sizes) == 0:
                return "Failed to get the mip sizes from the volume."

            if (
                config.annotation_mip_level not in mip_sizes
                or config.output_mip_level not in mip_sizes
            ):
                return "The specified mip levels are not present in the volume."

            sample_points = convert_points_between_volumes(
                sample_points,
                mip_sizes[config.annotation_mip_level],
                mip_sizes[config.output_mip_level],
            )

            # Update the pipeline input with the rescaled sample points
            # Note: this ensures that the rescaled sample points are saved to the JSON file,
            # which is important for the backprojection step
            pipeline_input.sample_points = sample_points

        spline = Spline(sample_points, degree=3)

        # Plot equidistant points along the spline
        if config.slicing_params.use_adaptive_slicing:
            equidistant_params = spline.calculate_adaptive_parameters(
                config.slicing_params.dist_between_slices,
                ratio=config.slicing_params.adaptive_slicing_ratio,
            )
        else:
            equidistant_params = spline.calculate_equidistant_parameters(
                config.slicing_params.dist_between_slices
            )

        equidistant_points = spline(equidistant_params)

        self.update_progress(0.5)

        # Calculate the slice rects for each t value
        slice_rects = calculate_slice_rects(
            equidistant_params,
            spline,
            config.slice_width,
            config.slice_height,
            spline_points=equidistant_points,
        )

        # Update the pipeline input with the slice rects
        pipeline_input.slice_rects = slice_rects

        # Remove sample_points from the pipeline input (it is no longer needed, and it is a large object)
        pipeline_input.clear_entry("sample_points")

        return None

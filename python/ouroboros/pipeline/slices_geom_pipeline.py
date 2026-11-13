from ouroboros.helpers.coordinates import convert_points_between_volumes
from ouroboros.helpers.slice import calculate_slice_rects
from ouroboros.helpers.spline import Spline
from ouroboros.helpers.volume_cache import CloudVolumeInterface, get_mip_volume_sizes
from .pipeline import PipelineStep
from ouroboros.helpers.options import SliceOptions
import numpy as np


def _remove_duplicate_points(sample_points: np.ndarray) -> np.ndarray:
    _, indices = np.unique(sample_points.round(decimals=2), axis=0, return_index=True)
    return sample_points[sorted(indices)]


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

        image_transform = pipeline_input.image_source_transform
        if image_transform is None:
            # Preserve the established filtering frame for untransformed inputs.
            sample_points = _remove_duplicate_points(sample_points)

        # Explicit physical units replace the legacy annotation-MIP assumption.
        # Use source resolution directly: volume-size ratios include rounding.
        annotation_resolution = pipeline_input.sample_points_resolution_nm
        if image_transform is not None and annotation_resolution is None:
            return "Image source transforms require explicit annotation coordinate scales."
        if annotation_resolution is not None:
            try:
                source = CloudVolumeInterface(source_url)
                if config.output_mip_level not in source.available_mips:
                    return "The specified output mip level is not present in the volume."
                output_resolution = source.get_resolution_nm(config.output_mip_level)
                if image_transform is not None and image_transform.input_resolution_nm is not None:
                    native_resolution = (
                        output_resolution if config.output_mip_level == 0 else source.get_resolution_nm(0)
                    )
            except Exception as error:
                return f"Failed to get the voxel resolution from the volume: {str(error) or type(error).__name__}"
            if image_transform is None:
                sample_points = sample_points * (
                    np.asarray(annotation_resolution) / output_resolution
                )
            else:
                # Neuroglancer's linear map acts in physical units, while its
                # translation was converted from outputDimensions to nm at import.
                physical_points = sample_points * np.asarray(annotation_resolution)
                image_points_nm = np.linalg.solve(
                    image_transform.linear,
                    (physical_points - image_transform.translation_nm).T,
                ).T
                if image_transform.input_resolution_nm is None:
                    sample_points = image_points_nm / output_resolution
                else:
                    sample_points = (image_points_nm / image_transform.input_resolution_nm) * (
                        np.asarray(native_resolution) / output_resolution
                    )
                # These are absolute source voxels; voxel offsets already define
                # source bounds and must not be added or subtracted here.
                # Filter after inverse placement: image scaling can compress
                # distinct source voxels below the global rounding precision.
                sample_points = _remove_duplicate_points(sample_points)
            pipeline_input.sample_points = sample_points
            pipeline_input.sample_points_resolution_nm = None
            pipeline_input.image_source_transform = None
        elif config.annotation_mip_level != config.output_mip_level:
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

        # Convert annotation points to straightened-volume coordinates.
        # `spline.u` are the fit parameters corresponding to these input points.
        annotation_projection_points = spline(spline.u).T

        _, normal_vectors, binormal_vectors = spline.calculate_rotation_minimizing_vectors(spline.u)

        annotation_offsets = sample_points - annotation_projection_points
        lateral_offsets_x = np.einsum("ij,ij->i", annotation_offsets, normal_vectors.T)
        lateral_offsets_y = np.einsum("ij,ij->i", annotation_offsets, binormal_vectors.T)

        straightened_depth = np.interp(spline.u, equidistant_params,
                                       np.arange(len(equidistant_params), dtype=np.float32))

        straightened_annotation_points = np.column_stack(
            (
                (config.slice_width - 1) / 2.0 + lateral_offsets_x,
                (config.slice_height - 1) / 2.0 - lateral_offsets_y,
                straightened_depth,
            )
        )
        pipeline_input.annotation_points = straightened_annotation_points

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
        pipeline_input.clear_entry("sample_points_resolution_nm")
        pipeline_input.clear_entry("image_source_transform")

        return None

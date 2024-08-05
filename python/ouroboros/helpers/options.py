from pydantic import BaseModel, field_serializer, field_validator

from ouroboros.helpers.bounding_boxes import BoundingBoxParams

from ouroboros.helpers.models import model_with_json


@model_with_json
class CommonOptions(BaseModel):
    output_file_folder: str  # Folder to save the output file
    output_file_name: str  # Name of the output file
    flush_cache: bool = False  # Whether to flush the cache after processing
    make_single_file: bool = True  # Whether to save the output to a single file
    max_ram_gb: int = 0  # Maximum amount of RAM to use in GB (0 means no limit)


@model_with_json
class SliceOptions(CommonOptions):
    slice_width: int  # Width of the slice
    slice_height: int  # Height of the slice
    neuroglancer_json: str  # Path to the neuroglancer JSON file
    neuroglancer_image_layer: (
        str  # Name of the image layer in the neuroglancer JSON file
    ) = ""
    neuroglancer_annotation_layer: (
        str  # Name of the annotation layer in the neuroglancer JSON file
    ) = ""
    dist_between_slices: int | float = 1  # Distance between slices
    bounding_box_params: BoundingBoxParams = (
        BoundingBoxParams()
    )  # Parameters for generating bounding boxes
    connect_start_and_end: bool = (
        False  # Whether to connect the start and end of the given annotation points
    )

    @field_serializer("bounding_box_params")
    def serialize_bounding_box_params(self, value: BoundingBoxParams):
        return value.to_dict()

    @field_validator("bounding_box_params", mode="before")
    @classmethod
    def validate_bounding_box_params(cls, value: any):
        if isinstance(value, dict):
            return BoundingBoxParams.from_dict(value)
        return value


@model_with_json
class BackprojectOptions(CommonOptions):
    straightened_volume_path: str  # Path to the straightened volume
    config_path: str  # Path to the config file
    backproject_min_bounding_box: bool = (
        True  # Whether to backproject to a minimum bounding box or the entire volume
    )
    make_backprojection_binary: bool = (
        False  # Whether to make the backprojection binary (values of 0 or 1)
    )
    backprojection_compression: str = (
        "zlib"  # Compression type for the backprojected file
    )


DEFAULT_SLICE_OPTIONS = SliceOptions(
    slice_width=100,
    slice_height=100,
    output_file_folder="./output/",
    output_file_name="sample",
    neuroglancer_json="./neuroglancer.json",
)

DEFAULT_BACKPROJECT_OPTIONS = BackprojectOptions(
    output_file_folder="./output/",
    output_file_name="sample",
    straightened_volume_path="./sample.tif",
    config_path="./sample-configuration.json",
    make_single_file=False,
)

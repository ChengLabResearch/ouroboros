import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from ouroboros.helpers.models import model_with_json
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.volume_cache import CloudVolumeInterface, VolumeCache


@model_with_json
class BasePipelineInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getitem__(self, keys) -> iter:
        return iter(getattr(self, k) for k in keys)

    def clear_entry(self, key) -> None:
        """
        Clear the entry with the given key.
        """
        setattr(self, key, None)


@model_with_json
class PipelineInput(BasePipelineInput):
    """
    Dataclass for the input to the pipeline.
    """

    json_path: str | None = None
    slice_options: SliceOptions | None = None
    backproject_options: BackprojectOptions | None = None
    source_url: str | None = None
    sample_points: np.ndarray | None = None
    # Units of sample_points, when explicitly supplied by the annotation source.
    # None preserves the legacy annotation_mip_level interpretation.
    sample_points_resolution_nm: tuple[float, float, float] | None = None
    annotation_points: np.ndarray | None = None
    slice_rects: np.ndarray | None = None
    volume_cache: VolumeCache | None = None
    # Backprojection owns one envelope and source metadata, not slicing partitions.
    backprojection_box: BoundingBox | None = None
    backprojection_source: CloudVolumeInterface | None = None
    output_file_path: str | None = None
    backprojected_folder_path: str | None = None
    config_file_path: str | None = None
    backprojection_offset: str | None = None

    @field_serializer("sample_points", "annotation_points", "slice_rects")
    def to_list(self, value) -> list | None:
        return value.tolist() if value is not None else None

    @field_validator("sample_points", "annotation_points", "slice_rects", mode="before")
    @classmethod
    def validate_list(cls, value: any) -> np.ndarray | None:
        if isinstance(value, list):
            return np.array(value)
        return value

    @field_serializer("volume_cache", "backprojection_box", "backprojection_source")
    def to_dict(self, value) -> dict | None:
        return value.to_dict() if value is not None else None

    @field_validator("volume_cache", mode="before")
    @classmethod
    def validate_volume_cache(cls, value: any) -> VolumeCache | None:
        if isinstance(value, dict):
            return VolumeCache.from_dict(value)
        return value

    @field_validator("backprojection_box", mode="before")
    @classmethod
    def validate_backprojection_box(cls, value: any) -> BoundingBox | None:
        return BoundingBox.from_dict(value) if isinstance(value, dict) else value

    @field_validator("backprojection_source", mode="before")
    @classmethod
    def validate_backprojection_source(cls, value: any) -> CloudVolumeInterface | None:
        return CloudVolumeInterface.from_dict(value) if isinstance(value, dict) else value

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from ouroboros.helpers.models import model_with_json
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers.volume_cache import VolumeCache


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
    annotation_points: np.ndarray | None = None
    slice_rects: np.ndarray | None = None
    volume_cache: VolumeCache | None = None
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

    @field_serializer("volume_cache")
    def to_dict(self, value) -> dict | None:
        return value.to_dict() if value is not None else None

    @field_validator("volume_cache", mode="before")
    @classmethod
    def validate_volume_cache(cls, value: any) -> VolumeCache | None:
        if isinstance(value, dict):
            return VolumeCache.from_dict(value)
        return value

from enum import Enum
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ValidatorFunctionWrapHandler, field_validator

from ouroboros.helpers.models import model_with_json


# Supported cloudvolume formats to map from NG adapter format to cloudvolume format.
class CV_FORMAT(Enum):
    PRECOMPUTED = ["neuroglancer-precomputed"]
    ZARR = ["zarr", "zarr2", "zarr3"]
    N5 = ["n5"]

    def __str__(self):
        return f"{self.name.lower()}://"

    @classmethod
    def get(cls, suffix):
        for e in cls:
            if suffix in e.value:
                return e
        raise ValueError(f"No cloudvolume format type found for {suffix}")


class LayerModel(BaseModel):
    type: str
    name: str


class AnnotationModel(BaseModel):
    point: list[float]
    type: str


class AnnotationLayerModel(LayerModel):
    annotations: list[AnnotationModel]
    type: str
    name: str
    # Keep the boundary value intact until the selected layer is interpreted.
    # Validating it in this union member could silently fall back to LayerModel.
    source: Any = None

    def resolution_nm(self) -> tuple[float, float, float] | None:
        """Read local XYZ identity-transform units, not global viewer units."""
        if isinstance(self.source, list) and any(
            isinstance(source, dict) and source.get("transform") for source in self.source
        ):
            raise ValueError("Annotation coordinate transforms in source arrays are not supported.")
        if not isinstance(self.source, dict) or self.source.get("transform") is None:
            return None
        transform = self.source["transform"]
        if transform == {}:
            return None
        if not isinstance(transform, dict) or self.source.get("url") != "local://annotations":
            raise ValueError("Only local XYZ annotation coordinate scales are supported.")
        dimensions = transform.get("outputDimensions")
        if not isinstance(dimensions, dict) or list(dimensions) != ["x", "y", "z"]:
            raise ValueError("Annotation outputDimensions must be ordered x, y, z length dimensions.")
        input_dimensions = transform.get("inputDimensions", dimensions)
        if input_dimensions != dimensions or list(input_dimensions) != list(dimensions):
            raise ValueError("Different annotation inputDimensions and outputDimensions are not supported.")
        if "matrix" in transform and not np.array_equal(transform["matrix"], np.eye(3, 4)):
            raise ValueError("Only identity annotation transform matrices are supported.")
        resolution = []
        for scale_unit in dimensions.values():
            if (not isinstance(scale_unit, list) or len(scale_unit) != 2
                    or scale_unit[1] != "m"):
                raise ValueError("Annotation coordinate dimensions must use meters (m).")
            scale = float(scale_unit[0]) * 1e9
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Annotation coordinate scales must be finite and positive.")
            resolution.append(scale)
        return tuple(resolution)


class SourceModel(BaseModel):
    url: str
    transform: Any = None


class ImageLayerModel(LayerModel):
    source: str | SourceModel
    type: str
    name: str
    transform: Any = None

    @field_validator("source", mode="wrap")
    @classmethod
    def parse_source(cls, value, handler: ValidatorFunctionWrapHandler) -> str:
        source = handler(value)
        base_source = source.url if isinstance(source, SourceModel) else source
        split_source = base_source.split("|")
        if len(split_source) > 1:
            kv_store = split_source[1].split(":")
            base_source = f"{CV_FORMAT.get(kv_store[0])}{split_source[0]}{kv_store[1]}"

        return source.model_copy(update={"url": base_source}) if isinstance(source, SourceModel) else base_source


@model_with_json
class NeuroglancerJSONModel(BaseModel):
    layers: list[AnnotationLayerModel | ImageLayerModel | LayerModel]


Result = tuple[any, None] | tuple[None, str]

ParseResult = tuple[NeuroglancerJSONModel | None, None | str]


@dataclass(frozen=True)
class AnnotationCoordinates:
    """Parsed input adapter; the pipeline owns these facts after import."""

    points: np.ndarray
    resolution_nm: tuple[float, float, float] | None


def parse_neuroglancer_json(json_path: str) -> ParseResult:
    """
    Open and parse a neuroglancer state JSON string and return a dictionary of the parsed data.

    Parameters
    ----------
    json_string : str
        The JSON string to parse.

    Returns
    -------
    ParseResult
        A tuple containing the parsed JSON (NeuroglancerJSONModel) and an error if one occurred.
    """

    neuroglancer_json_or_error = NeuroglancerJSONModel.load_from_json(json_path)

    if isinstance(neuroglancer_json_or_error, str):
        return None, neuroglancer_json_or_error

    return neuroglancer_json_or_error, None


def neuroglancer_config_to_annotation(
    config: NeuroglancerJSONModel, neuroglancer_annotation_layer: str
) -> Result:
    """
    Extract ordered points and their optional physical units from the selected layer.

    Parameters
    ----------
    config : NeuroglancerJSONModel
        The neuroglancer state JSON object.
    neuroglancer_annotation_layer : str
        The name of the annotation layer.

    Returns
    -------
    Result
        A tuple containing the annotation coordinates and an error if one occurred.
    """

    valid_layer_name = neuroglancer_annotation_layer != ""

    try:
        for layer in config.layers:
            if layer.type == "annotation" and (
                layer.name == neuroglancer_annotation_layer or not valid_layer_name
            ):
                annotations = layer.annotations

                return (
                    AnnotationCoordinates(
                        np.array([data.point for data in annotations if data.type == "point"]),
                        layer.resolution_nm(),
                    ),
                    None,
                )
    except BaseException as e:
        return None, f"Error extracting annotations: {e}"

    return None, "The selected annotation layer was not found in the file."


def neuroglancer_config_to_source(
    config: NeuroglancerJSONModel, neuroglancer_image_layer: str, *,
    require_untransformed: bool = False,
) -> Result:
    """
    Extract the source URL from a neuroglancer state JSON dictionary.

    Parameters
    ----------
    config : NeuroglancerJSONModel
        The neuroglancer state JSON object.
    neuroglancer_image_layer : str
        The name of the image layer.

    Returns
    -------
    Result
        A tuple containing the source URL and an error if one occurred.
    """

    valid_layer_name = neuroglancer_image_layer != ""

    try:
        for layer in config.layers:
            if layer.type == "image" and (
                layer.name == neuroglancer_image_layer or not valid_layer_name
            ):
                if require_untransformed and layer.transform:
                    return None, "Image layer transforms are not supported with physical annotation coordinates."
                if isinstance(layer.source, str):
                    return layer.source, None
                elif isinstance(layer.source, SourceModel):
                    if require_untransformed and layer.source.transform:
                        return None, (
                            "Image source transforms are not supported with physical annotation coordinates."
                        )
                    return layer.source.url, None
                else:  # pragma: no cover - ImageLayerModel constrains source to str or SourceModel.
                    # Don't think you can hit this as image layer types only build
                    # from str or SourceModel
                    return None, "Invalid source format in the file."
    except BaseException as e:
        return None, f"Error extracting source URL: {e}"

    return None, "The selected image layer was not found in the file."

import json

import numpy as np
import pytest

from ouroboros.helpers.parse import (
    CV_FORMAT,
    NeuroglancerJSONModel,
    parse_neuroglancer_json,
    neuroglancer_config_to_annotation,
    neuroglancer_config_to_source
)
from test.sample_data import generate_sample_neuroglancer_json, generate_novel_neuroglancer_json


def test_cloudvolume_format_rejects_unknown_suffix():
    with pytest.raises(ValueError, match="No cloudvolume format type found"):
        CV_FORMAT.get("made-up-format")


def test_parse_neuroglancer_json(tmp_path):
    # Generate a sample neuroglancer JSON file
    json_path = generate_sample_neuroglancer_json(tmp_path)

    # Parse the generated JSON file
    parsed_data, error = parse_neuroglancer_json(json_path)

    # Assert that there is no error
    assert error is None

    # Assert that the parsed data is a dictionary
    assert isinstance(parsed_data, NeuroglancerJSONModel)

    # Assert that the parsed data contains expected keys
    data = parsed_data.model_dump()

    assert "layers" in data
    assert len(data["layers"]) == 3
    assert data["layers"][0]["type"] == "annotation"
    assert data["layers"][1]["type"] == "image"
    assert data["layers"][2]["type"] == "image"

    # Extract the annotation layer
    annotations, error = neuroglancer_config_to_annotation(parsed_data, "annotations")

    # Assert that there is no error
    assert error is None

    # The parsed coordinates retain ordered points and optional physical units.
    assert annotations is not None
    assert annotations.points.shape == (100, 3)
    assert annotations.resolution_nm is None

    # Extract the image layer
    source, error = neuroglancer_config_to_source(parsed_data, "image_layer")

    # Assert that there is no error
    assert error is None

    # Assert that the source is a string
    assert source is not None
    assert isinstance(source, str)

    # Extract the source model layer
    source, error = neuroglancer_config_to_source(parsed_data, "other_image_layer")

    # Assert that there is no error
    assert error is None

    # Assert that the source returned is a string even from source model.
    assert source is not None
    assert isinstance(source, str)


def test_parse_annotation_errors(tmp_path):
    # Generate a sample neuroglancer JSON file
    json_path = generate_sample_neuroglancer_json(tmp_path)

    # Parse the generated JSON file
    parsed_data, error = parse_neuroglancer_json(json_path)

    # Assert that there is no error
    assert error is None

    # Assert that the parsed data is a dictionary
    assert isinstance(parsed_data, NeuroglancerJSONModel)

    # Attempt to extract a non-existent annotation layer.
    annotations, error = neuroglancer_config_to_annotation(parsed_data, "kumquat")

    assert annotations is None
    assert error == "The selected annotation layer was not found in the file."

    # Attempt to extract an annotation layer from a non-JSON object
    annotations, error = neuroglancer_config_to_annotation("kumquat", "annotations")

    assert annotations is None
    assert error == "Error extracting annotations: 'str' object has no attribute 'layers'"

    # Attempt to extract an image layer that doesn't exist.
    annotations, error = neuroglancer_config_to_source(parsed_data, "kumquat")

    assert annotations is None
    assert error == "The selected image layer was not found in the file."

    # Attempt to extract an image layer from a non-JSON object
    annotations, error = neuroglancer_config_to_source("kumquat", "image_layer")

    assert annotations is None
    assert error == "Error extracting source URL: 'str' object has no attribute 'layers'"


def test_parse_neuroglancer_json_invalid():
    # Parse an invalid JSON file
    parsed_data, error = parse_neuroglancer_json("invalid.json")

    # Assert that there is an error
    assert error is not None

    # Assert that the parsed data is None
    assert parsed_data is None


def test_parse_newv_image_layers(tmp_path):
    json_path = generate_novel_neuroglancer_json(tmp_path)

    parsed_data, error = parse_neuroglancer_json(json_path)
    print(parsed_data)

    assert error is None

    assert parsed_data.layers[-1].source == "n5://http://sourcewebsite.com/image/another/path/"
    assert parsed_data.layers[-3].source.url == "zarr://http://sourcewebsite.com/image2/path/to/"
    assert parsed_data.layers[0].source == "precomputed://http://sourcewebsite.com/image"
    assert parsed_data.layers[2].source.url == "precomputed://http://sourcewebsite.com/image2"

    assert parsed_data.layers[1].source == "precomputed://http://sourcewebsite.com/image"


@pytest.fixture
def coordinate_document(tmp_path):
    with open(generate_sample_neuroglancer_json(tmp_path), encoding="utf-8") as handle:
        document = json.load(handle)
    document["layers"][0]["source"] = {
        "url": "local://annotations",
        "transform": {"outputDimensions": {axis: [scale, "m"] for axis, scale in
                                           zip("xyz", (7e-7, 1.1e-6, 1.3e-6))}},
    }
    return document


@pytest.mark.parametrize("explicit_identity", [False, True])
def test_selected_annotation_preserves_order_and_physical_units(coordinate_document, explicit_identity):
    layer = coordinate_document["layers"][0]
    transform = layer["source"]["transform"]
    if explicit_identity:
        transform["matrix"] = np.eye(3, 4).tolist()
        transform["inputDimensions"] = transform["outputDimensions"]
    # The viewer's coordinate units must not override this layer's own units.
    coordinate_document["dimensions"] = {axis: [2e-6, "m"] for axis in "xyz"}
    parsed = NeuroglancerJSONModel.model_validate(coordinate_document)

    coordinates, error = neuroglancer_config_to_annotation(parsed, "annotations")

    assert error is None
    np.testing.assert_array_equal(coordinates.points, [entry["point"] for entry in layer["annotations"]])
    np.testing.assert_allclose(coordinates.resolution_nm, (700, 1100, 1300), rtol=1e-15)


@pytest.mark.parametrize("transform,error", [
    pytest.param("invalid", "Only local XYZ", id="malformed-transform"),
    pytest.param({"outputDimensions": {"z": [7e-7, "m"], "y": [7e-7, "m"], "x": [7e-7, "m"]}},
                 "ordered x, y, z", id="axis-order"),
    pytest.param({"outputDimensions": {axis: [7e-7, "m"] for axis in "xyz"},
                  "inputDimensions": {axis: [7e-7, "m"] for axis in "zyx"}},
                 "Different annotation inputDimensions", id="input-axis-order"),
    pytest.param({"outputDimensions": {axis: [700, "nm"] for axis in "xyz"}},
                 "must use meters", id="unsupported-units"),
    pytest.param({"outputDimensions": {axis: [0, "m"] for axis in "xyz"}},
                 "finite and positive", id="zero-scale"),
    pytest.param({"outputDimensions": {axis: [float("nan"), "m"] for axis in "xyz"}},
                 "finite and positive", id="nonfinite-scale"),
])
def test_invalid_explicit_coordinate_metadata_is_not_treated_as_legacy(coordinate_document, transform, error):
    # Each case would otherwise silently use a different coordinate frame.
    coordinate_document["layers"][0]["source"]["transform"] = transform
    parsed = NeuroglancerJSONModel.model_validate(coordinate_document)

    coordinates, actual_error = neuroglancer_config_to_annotation(parsed, "annotations")

    assert coordinates is None
    assert error in actual_error


def test_image_url_normalization_retains_transform_for_coordinate_validation(coordinate_document):
    transform = {"matrix": [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]]}
    coordinate_document["layers"][1]["source"] = {
        "url": "http://sourcewebsite.com/image|neuroglancer-precomputed:",
        "transform": transform,
    }
    parsed = NeuroglancerJSONModel.model_validate(coordinate_document)

    assert parsed.layers[1].source.transform == transform
    source, error = neuroglancer_config_to_source(parsed, "image_layer", require_untransformed=True)
    assert source is None
    assert "Image source transforms are not supported" in error
    source, error = neuroglancer_config_to_source(parsed, "image_layer")
    assert error is None
    assert source == "precomputed://http://sourcewebsite.com/image"


def test_annotation_source_array_does_not_hide_explicit_coordinate_metadata(coordinate_document):
    layer = coordinate_document["layers"][0]
    layer["source"] = [layer["source"]]
    parsed = NeuroglancerJSONModel.model_validate(coordinate_document)

    coordinates, error = neuroglancer_config_to_annotation(parsed, "annotations")

    assert coordinates is None
    assert "Annotation coordinate transforms in source arrays are not supported" in error

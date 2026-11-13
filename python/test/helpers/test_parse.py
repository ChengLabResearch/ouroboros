import json

import numpy as np
import pytest

from ouroboros.helpers.parse import (
    CV_FORMAT,
    ImageSourceCoordinates,
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

    # Both source forms expose one normalized boundary adapter.
    assert source is not None
    assert isinstance(source, ImageSourceCoordinates)
    assert source.url == "precomputed://http://sourcewebsite.com/image"
    assert source.transform is None

    # Extract the source model layer
    source, error = neuroglancer_config_to_source(parsed_data, "other_image_layer")

    # Assert that there is no error
    assert error is None

    assert source is not None
    assert isinstance(source, ImageSourceCoordinates)
    assert source.url == "precomputed://http://sourcewebsite.com/image2"
    assert source.transform is None


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


def test_image_url_normalization_preserves_placement_in_physical_units(coordinate_document):
    transform = {
        "outputDimensions": {axis: [scale, "m"] for axis, scale in zip("xyz", (7e-7, 1.1e-6, 1.3e-6))},
        "inputDimensions": {axis: [scale, "m"] for axis, scale in zip("xyz", (6.37e-7, 8e-7, 1.1e-6))},
        "matrix": [[0, -2, 0, 10], [1, 0, 0, -20], [0, 0, 0.5, 30]],
    }
    coordinate_document["layers"][1]["source"] = {
        "url": "http://sourcewebsite.com/image|neuroglancer-precomputed:",
        "transform": transform,
    }
    parsed = NeuroglancerJSONModel.model_validate(coordinate_document)

    assert parsed.layers[1].source.transform == transform
    source, error = neuroglancer_config_to_source(parsed, "image_layer")
    assert error is None
    assert source.url == "precomputed://http://sourcewebsite.com/image"
    np.testing.assert_array_equal(source.transform.linear, [[0, -2, 0], [1, 0, 0], [0, 0, 0.5]])
    np.testing.assert_allclose(source.transform.translation_nm, (7000, -22000, 39000), rtol=1e-15)
    np.testing.assert_allclose(source.transform.input_resolution_nm, (637, 800, 1100), rtol=1e-15)
    parsed.layers[1].source.transform["matrix"][0][3] = -100
    assert source.transform.translation_nm[0] == pytest.approx(7000), "Raw JSON must not own the imported placement"


def test_annotation_source_array_does_not_hide_explicit_coordinate_metadata(coordinate_document):
    layer = coordinate_document["layers"][0]
    layer["source"] = [layer["source"]]
    parsed = NeuroglancerJSONModel.model_validate(coordinate_document)

    coordinates, error = neuroglancer_config_to_annotation(parsed, "annotations")

    assert coordinates is None
    assert "Annotation coordinate transforms in source arrays are not supported" in error


@pytest.fixture
def image_transform_document(coordinate_document):
    coordinate_document["layers"][1]["source"] = {
        "url": "precomputed://http://sourcewebsite.com/image",
        "transform": {"outputDimensions": {axis: [7e-7, "m"] for axis in "xyz"}},
    }
    return coordinate_document


@pytest.mark.parametrize("matrix", [None, np.eye(3, 4).tolist()])
@pytest.mark.parametrize("explicit_input", [False, True])
def test_image_identity_only_collapses_when_native_input_scales_are_implied(
    image_transform_document, matrix, explicit_input,
):
    transform = image_transform_document["layers"][1]["source"]["transform"]
    if matrix is not None:
        transform["matrix"] = matrix
    if explicit_input:
        transform["inputDimensions"] = {axis: [7e-7, "m"] for axis in "xyz"}

    source, error = neuroglancer_config_to_source(NeuroglancerJSONModel.model_validate(image_transform_document), "")

    assert error is None
    if explicit_input:
        assert source.transform.input_resolution_nm == (700, 700, 700)
        assert source.transform.translation_nm == (0, 0, 0)
        np.testing.assert_array_equal(source.transform.linear, np.eye(3))
    else:
        assert source.transform is None, "Output units alone do not reposition the image"


def test_omitted_image_input_dimensions_remain_native_source_dependent(image_transform_document):
    image_transform_document["layers"][1]["source"]["transform"]["matrix"] = [
        [1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0],
    ]

    source, error = neuroglancer_config_to_source(NeuroglancerJSONModel.model_validate(image_transform_document), "")

    assert error is None
    assert source.transform.input_resolution_nm is None, "Image inputs must not inherit the 700-nm output units"
    assert source.transform.translation_nm == (7000, 0, 0)


@pytest.mark.parametrize("matrix", [
    pytest.param([[1, 0, 0], [0, 1, 0], [0, 0, 1]], id="missing-translation-column"),
    pytest.param([[1, 0, 0, 0], [0, 1], [0, 0, 1, 0]], id="ragged"),
    pytest.param([[1, 0, 0, float("inf")], [0, 1, 0, 0], [0, 0, 1, 0]], id="nonfinite"),
])
def test_invalid_image_matrix_cannot_be_imported_as_identity(image_transform_document, matrix):
    image_transform_document["layers"][1]["source"]["transform"]["matrix"] = matrix

    source, error = neuroglancer_config_to_source(NeuroglancerJSONModel.model_validate(image_transform_document), "")

    assert source is None
    assert "matrix must be a finite 3 x 4 array" in error


@pytest.mark.parametrize("x_scale,valid", [(0, False), (1e-20, True)])
def test_image_linear_component_requires_invertibility_without_an_arbitrary_threshold(
    image_transform_document, x_scale, valid,
):
    image_transform_document["layers"][1]["source"]["transform"]["matrix"] = [
        [x_scale, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
    ]

    source, error = neuroglancer_config_to_source(NeuroglancerJSONModel.model_validate(image_transform_document), "")

    if valid:
        assert error is None
        assert source.transform.linear[0][0] == x_scale
    else:
        assert source is None
        assert "matrix must have an invertible linear component" in error


@pytest.mark.parametrize("replacement,error", [
    pytest.param("invalid", "Image source transform must be an object", id="non-object"),
    pytest.param({"matrix": np.eye(3, 4).tolist()}, "Image outputDimensions", id="missing-output-units"),
    pytest.param({"outputDimensions": {axis: [7e-7, "m"] for axis in "zyx"}},
                 "Image outputDimensions", id="output-axis-order"),
    pytest.param({"outputDimensions": {axis: [700, "nm"] for axis in "xyz"}},
                 "must use meters", id="unsupported-units"),
    pytest.param({"outputDimensions": {axis: [-7e-7, "m"] for axis in "xyz"}},
                 "finite and positive", id="negative-scale"),
    pytest.param({"outputDimensions": {axis: [7e-7, "m"] for axis in "xyz"},
                  "inputDimensions": {axis: [7e-7, "m"] for axis in "zyx"}},
                 "Image inputDimensions", id="input-axis-order"),
    pytest.param({"outputDimensions": {axis: [7e-7, "m"] for axis in "xyz"}, "sourceRank": 2},
                 "must have sourceRank 3", id="synthetic-dimension"),
    pytest.param({"outputDimensions": {axis: [1, "m"] for axis in "xyz"},
                  "matrix": [[1, 0, 0, 1e308], [0, 1, 0, 0], [0, 0, 1, 0]]},
                 "translation must be finite in nanometers", id="physical-translation-overflow"),
])
def test_unsupported_image_coordinates_report_errors_for_the_selected_source(
    image_transform_document, replacement, error,
):
    image_transform_document["layers"][1]["source"]["transform"] = replacement
    parsed = NeuroglancerJSONModel.model_validate(image_transform_document)

    other_source, other_error = neuroglancer_config_to_source(parsed, "other_image_layer")
    assert other_error is None, "An unselected placement must not block another image layer"
    assert other_source.transform is None
    source, actual_error = neuroglancer_config_to_source(parsed, "image_layer")
    assert source is None
    assert error in actual_error


def test_legacy_image_layer_transform_is_not_silently_ignored(coordinate_document):
    coordinate_document["layers"][1]["transform"] = {"translation": [10, 0, 0]}

    source, error = neuroglancer_config_to_source(NeuroglancerJSONModel.model_validate(coordinate_document), "image_layer")

    assert source is None
    assert error == "Legacy image layer transforms are not supported; use source.transform."

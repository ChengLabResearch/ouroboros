from ouroboros.helpers.parse import (
    NeuroglancerJSONModel,
    parse_neuroglancer_json,
    neuroglancer_config_to_annotation,
    neuroglancer_config_to_source,
    SourceModel
)
from test.sample_data import generate_sample_neuroglancer_json, generate_novel_neuroglancer_json


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

    # Assert that the annotations are a numpy array
    assert annotations is not None
    assert annotations.shape == (100, 3)

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

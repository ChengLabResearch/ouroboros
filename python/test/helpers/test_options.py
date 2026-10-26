import pytest
from pydantic import ValidationError
from ouroboros.helpers.bounding_boxes import BoundingBoxParams
from ouroboros.helpers.options import CommonOptions, SliceOptions, BackprojectOptions


def test_common_options_validation():
    with pytest.raises(ValidationError):
        CommonOptions()


def test_slice_options_validation():
    with pytest.raises(ValidationError):
        SliceOptions(
            slice_width=100,
            slice_height=-1,
            output_file_folder="./output/",
            output_file_name="output",
        )


def test_backproject_options_validation():
    with pytest.raises(ValidationError):
        BackprojectOptions(
            output_file_folder="./output/",
            output_file_name="output",
        )


def test_common_options_to_json(tmp_path):
    options = CommonOptions(
        slice_width=100,
        slice_height=100,
        output_file_folder="./output/",
        output_file_name="output",
    )
    json_path = tmp_path / "common_options.json"
    options.save_to_json(json_path)
    assert json_path.exists()


def test_slice_options_to_json(tmp_path):
    options = SliceOptions(
        slice_width=100,
        slice_height=100,
        output_file_folder="./output/",
        output_file_name="output",
        neuroglancer_json="./neuroglancer.json",
    )
    json_path = tmp_path / "slice_options.json"
    options.save_to_json(json_path)
    assert json_path.exists()


def test_slice_options_from_json():
    options = SliceOptions(
        slice_width=100,
        slice_height=100,
        output_file_folder="./output/",
        output_file_name="output",
        neuroglancer_json="./neuroglancer.json",
        bounding_box_params=BoundingBoxParams(max_depth=10, target_slices_per_box=120),
    )
    json_output = options.to_json()
    loaded_options = SliceOptions.from_json(json_output)

    assert isinstance(loaded_options.bounding_box_params, BoundingBoxParams)
    assert not isinstance(loaded_options, str)
    assert loaded_options == options


def test_backproject_options_to_json(tmp_path):
    options = BackprojectOptions(
        output_file_folder="./output/",
        output_file_name="output",
        straightened_volume_path="./volume.tif",
        slice_options_path="./slice-options.json",
    )
    json_path = tmp_path / "backproject_options.json"
    options.save_to_json(json_path)
    assert json_path.exists()


@pytest.mark.parametrize("model, required", [
    (SliceOptions, dict(slice_width=10, slice_height=10, neuroglancer_json="state.json")),
    (BackprojectOptions, dict(straightened_volume_path="mask.tif", slice_options_path="slice.json")),
])
def test_worker_budget_is_owned_by_serialized_options(model, required):
    import os
    common = dict(output_file_folder="output", output_file_name="image", **required)
    assert model(**common).process_count == os.cpu_count()
    for count in (4, 8, 12, 48):
        options = model(**common, process_count=count)
        restored = model.from_json(options.to_json())
        assert restored.process_count == count
    for count in (0, -4, 3, 4.5):
        with pytest.raises(ValidationError):
            model(**common, process_count=count)


@pytest.mark.parametrize("docker", [False, True])
def test_api_option_schema_is_derived_from_owning_models(docker):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ouroboros.common.server_api import create_api
    app = FastAPI()
    create_api(app, docker=docker)
    with TestClient(app) as client:
        response = client.get("/options-schema")
    assert response.status_code == 200
    assert response.json() == {
        "slice": SliceOptions.model_json_schema(),
        "backproject": BackprojectOptions.model_json_schema(),
    }

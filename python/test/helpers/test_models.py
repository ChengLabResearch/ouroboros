
import pytest
from pydantic import BaseModel
from ouroboros.helpers.models import (
    model_with_json,
    pretty_json_output,
)


@model_with_json
class SampleModel(BaseModel):
    field1: int
    field2: str


@pytest.mark.filterwarnings("error::pydantic.PydanticDeprecatedSince211")
def test_model_with_json(tmp_path):
    # Create an instance of SampleModel
    sample = SampleModel(field1=123, field2="test")

    # Test to_dict method
    assert sample.to_dict() == {"field1": 123, "field2": "test"}

    # Test from_dict method
    new_sample = SampleModel.from_dict({"field1": 456, "field2": "new_test"})
    assert new_sample.field1 == 456
    assert new_sample.field2 == "new_test"

    from_dict_err = SampleModel.from_dict({"field1": "invalid", "field2": 123})
    assert from_dict_err[:35] == "2 validation errors for SampleModel"

    # Test save_to_json and load_from_json methods
    json_path = tmp_path / "sample_model.json"
    sample.save_to_json(json_path)
    loaded_sample = SampleModel.load_from_json(json_path)
    assert loaded_sample.field1 == 123
    assert loaded_sample.field2 == "test"

    # Test copy_values_from_other method
    another_sample = SampleModel(field1=789, field2="another_test")
    sample.copy_values_from_other(another_sample)
    assert sample.field1 == 789
    assert sample.field2 == "another_test"


def test_model_from_json():
    json_data = '{"field1": 42, "field2": "example"}'
    sample = SampleModel.from_json(json_data)
    assert sample.field1 == 42
    assert sample.field2 == "example"

    json_err = SampleModel.from_json('{"field1": "invalid", "field2": 123}')
    assert json_err[:35] == "2 validation errors for SampleModel"


def test_model_generic_error_branches(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        def raise_from_dict(cls, class_dict):
            raise PermissionError("dict blocked")

        m.setattr(SampleModel, "model_validate", classmethod(raise_from_dict))
        assert SampleModel.from_dict({"field1": 1, "field2": "ok"}) == "dict blocked"

    with monkeypatch.context() as m:
        def raise_from_json(cls, json_data):
            raise RuntimeError("json blocked")

        m.setattr(SampleModel, "model_validate_json", classmethod(raise_from_json))
        assert SampleModel.from_json('{"field1": 1, "field2": "ok"}') == "json blocked"

    bad_encoding = tmp_path / "bad-encoding.json"
    bad_encoding.write_bytes(b"\xff")

    assert "invalid start byte" in SampleModel.load_from_json(bad_encoding)


def test_model_with_json_invalid_class():
    class InvalidClass:
        pass

    try:
        model_with_json(InvalidClass)
    except TypeError as e:
        assert str(e) == "model_with_json must be applied to a BaseModel type"


def test_invalid_load_json(tmp_path):
    json_path = tmp_path / "invalid.json"
    with open(json_path, "w") as f:
        f.write("invalid json")

    loaded_sample = SampleModel.load_from_json(json_path)
    assert "Invalid JSON" in loaded_sample


def test_file_not_found(tmp_path):
    json_path = tmp_path / "file_not_found.json"

    loaded_sample = SampleModel.load_from_json(json_path)
    assert "No such file or directory" in loaded_sample


def test_pretty_json_output():
    obj = {"key": "value"}
    pretty_json = pretty_json_output(obj)
    assert pretty_json == '{\n    "key": "value"\n}'

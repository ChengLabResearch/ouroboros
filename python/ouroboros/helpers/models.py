import json
from pydantic import BaseModel, ValidationError
import sys


def pretty_json_output(obj: object) -> str:
    return json.dumps(obj, indent=4)


def model_with_json(cls: BaseModel) -> BaseModel:
    """
    Add methods to the given class to serialize and deserialize the class to and from
    a dictionary, JSON string, and JSON file.

    Parameters
    ----------
    cls : BaseModel
        The class to add the methods to.

    Returns
    -------
    BaseModel
        The class with the added methods.
    """

    if not issubclass(cls, BaseModel):
        raise TypeError("model_with_json must be applied to a BaseModel type")

    cls.to_dict = cls.model_dump
    cls.from_dict = classmethod(from_dict)
    cls.to_json = cls.model_dump_json
    cls.from_json = classmethod(from_json)
    cls.save_to_json = save_to_json
    cls.load_from_json = classmethod(load_from_json)
    cls.copy_values_from_other = copy_values_from_other

    return cls


def save_to_json(self: BaseModel, json_path: str):
    # Also specify encoding here for consistency
    with open(json_path, "w", encoding='utf-8') as f:
        # Add indent here for consistency? Or handle in Pydantic model
        f.write(self.to_json(indent=4))


@classmethod
def from_dict(cls: type[BaseModel], class_dict: dict) -> BaseModel | str:
    try:
        result = cls.model_validate(class_dict)
        return result
    except (ValidationError, json.JSONDecodeError) as vse:
        # Catch specific Pydantic validation errors and JSON syntax errors
        print(f"Error in validation of dict data for {cls.__name__}:\n{vse}", file=sys.stderr)
        return str(vse)
    except Exception as e:
        # Catch other potential errors like permission denied, unicode issues etc.
        print(f"Error parsing dict data: {e}", file=sys.stderr)
        return str(e)


@classmethod
def from_json(cls: type[BaseModel], json_data: str) -> BaseModel | str:
    """Loads a Pydantic model from a JSON string.

    Args:
        cls: The Pydantic model class.
        json_data: JSON-format string of the object.

    Returns:
        An instance of the model, or the exception if loading fails.
    """
    try:
        result = cls.model_validate_json(json_data)
        return result
    except (ValidationError, json.JSONDecodeError) as vse:
        # Catch specific Pydantic validation errors and JSON syntax errors
        print(f"Error in validation of JSON for {cls.__name__}:\n{vse}", file=sys.stderr)
        return str(vse)
    except Exception as e:
        # Catch other potential errors like permission denied, unicode issues etc.
        print(f"Error parsing string json: {e}", file=sys.stderr)
        return str(e)


@classmethod
def load_from_json(cls: type[BaseModel], json_path: str) -> BaseModel | str:
    """Loads a Pydantic model from a JSON file.

    Args:
        cls: The Pydantic model class.
        json_path: Path to the JSON file.

    Returns:
        An instance of the model, or the exception if loading fails.
    """
    try:
        # Explicitly use utf-8 encoding
        with open(json_path, "r", encoding='utf-8') as f:
            # Use model_validate_json directly for better error context from Pydantic
            result = cls.model_validate_json(f.read())
            return result
    except FileNotFoundError as fe:
        print(f"Error: File not found at {json_path}", file=sys.stderr)
        return str(fe)
    except (ValidationError, json.JSONDecodeError) as vse:
        # Catch specific Pydantic validation errors and JSON syntax errors
        print(f"Error loading {cls.__name__} from JSON file '{json_path}':\n{vse}", file=sys.stderr)
        return str(vse)
    except Exception as e:
        # Catch other potential errors like permission denied, unicode issues etc.
        print(f"Error reading or parsing file '{json_path}': {e}", file=sys.stderr)
        return str(e)


def copy_values_from_other(self: BaseModel, other: BaseModel):
    for field in type(self).model_fields.keys():
        setattr(self, field, getattr(other, field))

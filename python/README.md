[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Coverage](./coverage.svg)

# Ourobors Python Package

This python package organizes all of the core functionality of the Ouroboros segmentation pipeline.

### CLI Usage

Installing this package automatically provides a CLI called `ouroboros-cli`. It is simple, but wraps most core functionality.

**Commands**

Slice the original volume along a path and save to a tiff file.

`ouroboros-cli slice <options.json> [--verbose]` 

Project the straightened slices back into the space of the original volume.

`ouroboros-cli backproject <options.json> [--verbose]`

Export sample options files into the current folder.

`ouroboros-cli sample-options`

### Server Usage

This package also comes with a FastAPI server that can be run with `ouroboros-server`. Internally, this is compiled using PyInstaller and run in the electron app. 

The usage is very similar to the cli, so to try it out, I recommend going to the `docs` website for the server once you run it. That is `http://127.0.0.1:8000/docs`.

### Development

Poetry must be run from the python/ directory, rather than repository root, or it will not be able to identify the pyproject.toml.  You may open the python folder separately or just make sure you always switch directories in the CLI.

[Poetry](https://python-poetry.org/) is the virtual environment and dependency manager for this project.

- `poetry install` - install all dependencies of this project into a local virtual environment
- `poetry add [package]` - add a new dependency to the project
- `poetry run [file.py]` - run a python file locally without activating the virtual environment
- `poetry shell` - active the virtual environment in this shell

**To make sure that you can run PyInstaller locally (requires Python with shared libraries):**

Run the line from the link below before creating a pyenv Python version. Then, use that version for the poetry environment. 

Use --enabled-shared if on Linux.

https://stackoverflow.com/questions/60917013/how-to-build-python-with-enable-framework-enable-shared-on-macos

**To have VSCode recognize the poetry venv, follow these instructions:**

You just need to type in your shell:

```bash
poetry config virtualenvs.in-project true
```

The virtualenv will be created inside the project path and vscode will recognize. Consider adding this to your .bashrc or .zshrc.

Note: If you install poetry via conda, this will likely not have any effect and poetry will ouroboros into the conda environment. 

If you already have created your project, you need to re-create the virtualenv to make it appear in the correct place:

```bash
poetry env list  # shows the name of the current environment
poetry env remove <current environment>
poetry install  # will create a new environment using your updated configuration
```

Instructions From: https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option


### Testing

In VSCode, use the testing tab. Otherwise, use `pytest`.

To run code coverage, use the following command: 
```
coverage run --source=ouroboros/helpers -m pytest && coverage report -m
```

_Only `helpers` is unit tested right now, primarily because the rest of the package fundamentally depends on `helpers`._

Commits to main and PR's should trigger coverage reports automatically on GitHub.

## Using the Python Package

The Ouroboros CLI and Server internally use the Python package for slicing and backprojecting. 

It is designed to be easy to use at a high level, but also extensible for more custom processing when needed.

#### Using Predefined Pipelines

The package includes predefined pipelines for common tasks such as backprojection and slicing. Here is an example of how to use the backprojection pipeline:

```python
from ouroboros.common.pipelines import backproject_pipeline
from ouroboros.helpers.options import BackprojectOptions

# Load options from a JSON file
backproject_options = BackprojectOptions.load_from_json('path/to/options.json')

# Create the pipeline and input data
pipeline, input_data = backproject_pipeline(backproject_options, verbose=True)

# Process the pipeline
output, error = pipeline.process(input_data)

if error:
    print(f"Error: {error}")
else:
    print("Backprojection completed successfully.")
```

The two predefined pipelines are `backproject_pipeline` and `slice_pipeline`.

For more example usage, see [Server Handlers](./ouroboros/common/server_handlers.py).

#### Creating a Custom Pipeline

You can create custom pipelines by defining your own pipeline steps and combining them. Here is an example of how to create a custom pipeline:

1. **Define Custom Pipeline Steps**

```python
from ouroboros.pipeline import PipelineStep
from ouroboros.helpers.options import SliceOptions

class CustomPipelineStep(PipelineStep):
    def __init__(self) -> None:
        # List the desired inputs 
        # Note: see PipelineInput for default inputs
        # If you use a custom pipeline input, these names
        # must match the field names in the input class
        super().__init__(inputs=("slice_options",))

    def _process(self, input_data: tuple[any]) -> None | str:
        # A tuple/iterator is automatically provided with the
        # desired inputs. `pipeline_input` is always included
        options, pipeline_input = input_data

        if not isinstance(options, SliceOptions):
            # Return an error when something goes wrong
            return "Input data must contain a SliceOptions object."

        # Do some custom processing
        # Note: set existing fields of pipeline_input
        # to persist data to the next pipeline step
        pipeline_input.sample_points = custom_points()

        # Return None when everything succeeds
        return None
```

2. **Create and Configure the Pipeline**

```python
from ouroboros.pipeline import Pipeline, PipelineInput
from . import CustomPipelineStep

# Create a custom pipeline with our custom step
pipeline = Pipeline(
    [
        CustomPipelineStep()
    ]
)

# Create input data for the pipeline
input_data = PipelineInput(json_path="./data/myjson.json")

# Process the pipeline
output, error = pipeline.process(input_data)

if error:
    print(f"Error: {error}")
else:
    print("Custom pipeline completed successfully.")
```

3. **Custom Pipeline Input**

The default pipeline input class, `PipelineInput`, is the one used for both the `slice_pipeline` and the `backproject_pipeline`.

Creating a new pipeline input class for your custom pipeline is relatively simple. The default pipeline input is a good template.

This is the code for the default pipeline input:

```python
from ouroboros.pipeline import BasePipelineInput
from pydantic import field_serializer, field_validator

from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.helpers.volume_cache import VolumeCache
from ouroboros.helpers.models import model_with_json

import numpy as np

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
    slice_rects: np.ndarray | None = None
    volume_cache: VolumeCache | None = None
    output_file_path: str | None = None
    backprojected_folder_path: str | None = None
    config_file_path: str | None = None
    backprojection_offset: str | None = None

    # A custom serializer is needed to correctly convert 
    # numpy arrays to a JSON-compatible form
    @field_serializer("sample_points", "slice_rects")
    def to_list(self, value):
        return value.tolist() if value is not None else None

    # A custom validator is needed to convert parsed lists
    # into numpy arrays before the type is validated
    @field_validator("sample_points", "slice_rects", mode="before")
    @classmethod
    def validate_list(cls, value: any):
        if isinstance(value, list):
            return np.array(value)
        return value

    # A custom serializer is needed to correctly convert a
    # class with a `to_dict` method to a JSON-compatible format.
    @field_serializer("volume_cache")
    def to_dict(self, value):
        return value.to_dict() if value is not None else None

    # A custom validator is needed to convert a parsed dict
    # to a class with a `from_dict` method.
    @field_validator("volume_cache", mode="before")
    @classmethod
    def validate_volume_cache(cls, value: any):
        if isinstance(value, dict):
            return VolumeCache.from_dict(value)
        return value

```
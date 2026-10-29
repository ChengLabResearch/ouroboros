[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Coverage](./coverage.svg)

# Ourobors Python Package

This python package organizes all of the core functionality of the Ouroboros segmentation pipeline.

### CLI Usage

Installing this package automatically provides a CLI called `ouroboros-cli`. It is simple, but wraps most core functionality.

**Commands**

Slice the original volume along a path and save to a tiff file.

`ouroboros-cli slice <options.json> [--logging-mode {off,standard,detailed}]`

Project the straightened slices back into the space of the original volume.

`ouroboros-cli backproject <options.json> [--logging-mode {off,standard,detailed}]`

Logging defaults to `off`, which keeps progress and basic step durations but
does not generate performance reports. `standard` prints and saves step timings,
byte counts, and volume statistics without RAM monitoring. `detailed` also
monitors RAM, which adds overhead. `--verbose` remains an alias for
`--logging-mode detailed`; the two options are mutually exclusive. CLI logging
is selected by the command line, independently of saved desktop preferences.

Reporting modes print and save the same formatted JSON beside the configured
outputs. Reports are named
`<output_file_name>-<operation>-<run_id>-statistics.json`, so repeated runs retain
separate records. They include operation and pipeline timings, volume/storage
metrics, effective configuration, and explicit success/failure/interruption
status. Detailed includes sampled process-tree memory; Standard explicitly
records RAM monitoring as disabled.

Server reporting also defaults to `off`; pass `logging_mode=standard` or
`logging_mode=detailed` when submitting a task to enable it. Legacy API
`reporting_enabled=false/true` maps to Off/Detailed when `logging_mode` is absent;
an explicit mode takes precedence. Task status exposes report delivery and recovery
information. See the [run-statistics reference](../documentation/docs/reference/run-statistics.md)
for timing boundaries, schema definitions, Docker behavior, and limitations, and
the [benchmark guide](../documentation/docs/development/benchmarks/index.md) for reproducible overhead measurements.

Export sample options files into the current folder.

`ouroboros-cli sample-options`

### Straightened TIFF Annotation Metadata

Slice outputs embed straightened annotation prompts in the first TIFF
page's `ImageDescription` JSON under the `annotation_points` key:

```json
{
  "annotation_points": [[x, y, z], ...]
}
```

Each row is ordered as `x, y, z` in straightened-volume coordinates:
`x` is the column coordinate within a straightened slice, `y` is the
row coordinate within that slice, and `z` is the straightened slice
index/depth coordinate. These are not original source-volume
coordinates.

Single-stack TIFF outputs and directory-of-TIFF outputs both use this
same metadata schema. Directory outputs write the same metadata to each
per-slice TIFF; consumers can read the first sorted TIFF file to recover
the annotation prompts.

### Server Usage

This package also comes with a FastAPI server that can be run with `ouroboros-server`. Internally, this is compiled using PyInstaller and run in the electron app. 

The usage is very similar to the cli, so to try it out, I recommend going to the `docs` website for the server once you run it. That is `http://127.0.0.1:8000/docs`.

### Development

This setup is for native Python development and tests. The [Electron development workflow](../README.md#development-quick-start) runs the backend in Docker and does not require a host Python environment.

Run Poetry commands from the repository's `python/` directory, which contains `pyproject.toml`. If you use VS Code, opening that folder in a separate window keeps the Python environment separate from the Electron workspace; its integrated terminal is a convenient place to run these commands.

#### 1. Install and select Python

Use Python **3.11** to match the Docker backend and coverage CI. The package supports Python 3.11 and 3.12 (`>=3.11,<3.13` in [pyproject.toml](pyproject.toml)). Download Python from [python.org](https://www.python.org/downloads/), or use [pyenv](https://github.com/pyenv/pyenv#installation) on Linux/macOS after installing its build prerequisites:

```bash
pyenv install 3.11
```

If you want it as your user-wide pyenv default, run `pyenv global 3.11`. Otherwise, select the interpreter explicitly with `poetry env use 3.11` in the setup below. If it is not discoverable, pass the full path to the installed Python executable to `poetry env use`.

For local PyInstaller builds, a source-built Python must provide its shared library. Current pyenv builds enable `--enable-shared` by default. If customizing an older build, pass `--enable-shared` on Linux or `--enable-framework` for a macOS framework build through `PYTHON_CONFIGURE_OPTS` **when running `pyenv install`**, then select that interpreter for Poetry. See [Python's shared-library option](https://docs.python.org/3.11/using/configure.html#cmdoption-enable-shared), [macOS framework option](https://docs.python.org/3.11/using/configure.html#cmdoption-enable-framework), and [pyenv build configuration](https://github.com/pyenv/pyenv/tree/master/plugins/python-build#special-environment-variables). This is a prerequisite for local executable builds, not an extra step for Docker development.

#### 2. Install Poetry and dependencies

[Poetry](https://python-poetry.org/) manages this package's dependencies and virtual environment. Follow its [official installer instructions](https://python-poetry.org/docs/#installing-with-the-official-installer), including adding the executable to `PATH`, and confirm `poetry --version` works. CI uses Poetry 2.1.3.

From the repository root, enter the Python directory and create the environment:

```bash
cd python
poetry config virtualenvs.in-project true
poetry env use 3.11
poetry install
poetry env info --path
```

The environment should be `python/.venv`, making it easy for VS Code to discover. Select its interpreter with **Python: Select Interpreter**, then open a new terminal. If it is not activated automatically, use the commands below.

#### 3. Run commands in the environment

Use `poetry run` without activating a shell:

```bash
poetry run ouroboros-cli --help
poetry run ouroboros-server
```

The first command describes the slicing/backprojection CLI; the second starts the native API server. Do not run it alongside the Docker backend on the same port. To run a script, use `poetry run python your_script.py`; to add a dependency, use `poetry add package-name`.

For an interactive environment, run `poetry env activate` and execute the activation command it prints for your shell. Once activated, `ouroboros-cli` and `ouroboros-server` can be run directly. In Poetry 2, `poetry shell` requires the optional shell plugin; it is no longer built in. See [Poetry's environment activation instructions](https://python-poetry.org/docs/managing-environments/#activating-the-environment).

#### Existing environments and VS Code discovery

Setting `virtualenvs.in-project` does not move an existing Poetry environment into `.venv`. If you want to recreate this project's cached environment there, run `poetry env list` to identify it, remove **that environment** with `poetry env remove "environment-name-from-the-list"`, and repeat the interpreter selection and installation commands above. Removal deletes that environment's installed packages. See [Poetry's in-project environment behavior](https://python-poetry.org/docs/configuration/#virtualenvsin-project).

Poetry can reuse an already active virtual environment, including Conda. Deactivate it first if you want Poetry to create this project's `.venv`. Use `poetry env info --path` to check which environment is actually selected rather than relying only on the terminal prompt.


### Testing

In VSCode, use the testing tab. Otherwise, use `pytest`.

To run code coverage, use the following command: 
```
coverage erase && coverage run -m pytest && coverage combine && coverage report -m
```

_Only `helpers` is unit tested right now, primarily because the rest of the package fundamentally depends on `helpers`._

Coverage includes multiprocessing workers and combines their result files. The
small shared-memory ownership regressions can also run independently with
`pytest test/test_shared_memory_lifecycle.py -q`; CI runs these on Linux, Windows,
and macOS with Python 3.11 and 3.12.
See the [shared-memory lifecycle contract](../documentation/docs/development/shared-memory-lifecycle.md)
for ownership, release semantics, and the RAM-reporting integration boundary.

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

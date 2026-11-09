![Coverage](https://raw.githubusercontent.com/ChengLabResearch/ouroboros/main/python/coverage.svg)

# Ouroboros Python Package

This python package organizes all of the core functionality of the Ouroboros segmentation pipeline.

The Python distribution is named `ouroboros-roi`; the import remains `ouroboros`
and the commands remain `ouroboros-cli`, `ouroboros-server`, and
`ouroboros-docker-server`. It supports Python 3.11 and 3.12.

To install from a local checkout, run `python -m pip install ./python` from the
repository root. To build an installable wheel, run `poetry build` from `python/`;
the wheel is named `ouroboros_roi-<version>-py3-none-any.whl`.

The unrelated `ouroboros` distribution on PyPI uses the same import name.
Use a fresh environment for `ouroboros-roi`, including when moving from an
earlier local installation named `ouroboros`; the distributions share module
files and cannot coexist safely. PyPI publication of `ouroboros-roi` is tracked in
[issue #244](https://github.com/ChengLabResearch/ouroboros/issues/244).

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
information. See the [run-statistics reference](https://github.com/ChengLabResearch/ouroboros/blob/main/documentation/docs/reference/run-statistics.md)
for timing boundaries, schema definitions, Docker behavior, and limitations, and
the [benchmark guide](https://github.com/ChengLabResearch/ouroboros/blob/main/documentation/docs/development/benchmarks/index.md) for reproducible overhead measurements.

Export sample options files into the current folder.

`ouroboros-cli sample-options`

Write the slice rectangles for a slicing configuration to `<output_file_name>_rects.npy` in its output folder. `--include-ng` also writes Neuroglancer points and angles to `<output_file_name>_ng.json`.

`ouroboros-cli geometry <options.json> [--include-ng]`

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

This package also comes with a FastAPI server that can be run natively with `ouroboros-server`. The desktop app does not use this entry point: it runs the server in Docker, as described in the [root README](https://github.com/ChengLabResearch/ouroboros/blob/main/README.md).

The usage is very similar to the CLI. To try it out, open the server's interactive API docs once it is running, at `http://127.0.0.1:8000/docs`.

### Development

This setup is for native Python development and tests. The [Electron development workflow](https://github.com/ChengLabResearch/ouroboros/blob/main/README.md#development-quick-start) runs the backend in Docker and does not require a host Python environment.

Run Poetry commands from the repository's `python/` directory, which contains `pyproject.toml`. If you use VS Code, opening that folder in a separate window keeps the Python environment separate from the Electron workspace; its integrated terminal is a convenient place to run these commands.

#### 1. Install and select Python

Use Python **3.11** to match the Docker backend and coverage CI. The package supports Python 3.11 and 3.12 (`>=3.11,<3.13` in [pyproject.toml](https://github.com/ChengLabResearch/ouroboros/blob/main/python/pyproject.toml)). Download Python from [python.org](https://www.python.org/downloads/), or use [pyenv](https://github.com/pyenv/pyenv#installation) on Linux/macOS after installing its build prerequisites:

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

#### Build standalone executables

After `poetry install`, run PyInstaller from the checkout's `python/` directory
with the existing build specifications:

```bash
poetry run pyinstaller cli.spec
poetry run pyinstaller server.spec
```

These produce `dist/ouroboros-cli` and `dist/ouroboros-server` (with `.exe` on
Windows). They replace the development aliases `build-cli-executable` and
`build-server-executable`, which were installed by earlier wheels but could not
run outside the checkout. PyInstaller remains a development dependency.


### Testing

Run tests from this directory. In VS Code, use the testing tab; otherwise, use `pytest`.

To run code coverage, use the following command:
```
coverage erase && coverage run -m pytest && coverage combine && coverage report -m
```

The tests in `test/` cover the helpers, pipeline steps, server reporting, shared-memory lifecycle, and native memory readers. The coverage report and its 98% threshold measure only `ouroboros/helpers`; see `[tool.coverage.run]` and `[tool.coverage.report]` in [pyproject.toml](https://github.com/ChengLabResearch/ouroboros/blob/main/python/pyproject.toml). Coverage includes multiprocessing workers and combines their result files.

Tests of a shared path use the `canary` and `needs` markers from [test/conftest.py](https://github.com/ChengLabResearch/ouroboros/blob/main/python/test/conftest.py). When a canary fails, the tests that depend on it skip and name that canary, so fix it first.

The shared-memory and native memory tests can run on their own with a reduced set of dependencies, for example `pytest test/test_shared_memory_lifecycle.py -q`. CI runs that set on Linux, Windows, and macOS with Python 3.11 and 3.12; see [pr-coverage.yml](https://github.com/ChengLabResearch/ouroboros/blob/main/.github/workflows/pr-coverage.yml) for the list.
See the [shared-memory lifecycle contract](https://github.com/ChengLabResearch/ouroboros/blob/main/documentation/docs/development/shared-memory-lifecycle.md)
for ownership, release semantics, and the RAM-reporting integration boundary.

Commits to main and PR's should trigger coverage reports automatically on GitHub.

## Using the Python Package

The CLI and server use this package for slicing and backprojecting, and it can be used directly. See [Python pipelines](https://github.com/ChengLabResearch/ouroboros/blob/main/documentation/docs/development/python-pipelines.md) for predefined pipelines and writing custom ones.

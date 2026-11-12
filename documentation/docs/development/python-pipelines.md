# Python pipelines

The Ouroboros CLI and Server internally use the Python package for slicing and backprojecting. 

It is designed to be easy to use at a high level, but also extensible for more custom processing when needed.

## Using Predefined Pipelines

The package includes predefined pipelines for common tasks such as backprojection and slicing. Here is an example of how to use the backprojection pipeline:

```python
from ouroboros.common.pipelines import backproject_pipeline
from ouroboros.helpers.options import BackprojectOptions, SliceOptions

# Load the backprojection options and the slice options they refer to
backproject_options = BackprojectOptions.load_from_json('path/to/options.json')
slice_options = SliceOptions.load_from_json(backproject_options.slice_options_path)

# Create the pipeline and input data
pipeline, input_data = backproject_pipeline(backproject_options, slice_options, verbose=True)

# Process the pipeline
output, error = pipeline.process(input_data)

if error:
    print(f"Error: {error}")
else:
    print("Backprojection completed successfully.")
```

The predefined pipelines in [pipelines.py](https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/common/pipelines.py) are `slice_pipeline`, `backproject_pipeline`, `geometry_pipeline`, and `visualization_pipeline`.

For more example usage, see [Server Handlers](https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/common/server_handlers.py).

## Creating a Custom Pipeline

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

When supplying `sample_points` to `SlicesGeometryPipelineStep`, leave
`sample_points_resolution_nm` as `None` for coordinates in
`SliceOptions.annotation_mip_level`. For points with explicit physical units,
set it to the three XYZ scales in nanometers. The geometry step converts those
points directly to output-MIP voxels, ignoring the annotation MIP setting, and
releases both fields after creating the slice rectangles. The built-in parser
sets these fields together from its `AnnotationCoordinates` input adapter.

2. **Create and Configure the Pipeline**

```python
from ouroboros.helpers.options import SliceOptions
from ouroboros.pipeline import Pipeline, PipelineInput
from . import CustomPipelineStep

# Create a custom pipeline with our custom step
pipeline = Pipeline(
    [
        CustomPipelineStep()
    ]
)

# Create input data for the pipeline
slice_options = SliceOptions.load_from_json("./data/slice-options.json")
input_data = PipelineInput(slice_options=slice_options)

# Process the pipeline
output, error = pipeline.process(input_data)

if error:
    print(f"Error: {error}")
else:
    print("Custom pipeline completed successfully.")
```

3. **Custom Pipeline Input**

The default pipeline input class, `PipelineInput`, is the one used for both the `slice_pipeline` and the `backproject_pipeline`.

To create your own, subclass `BasePipelineInput` and apply `@model_with_json`, as [`PipelineInput`](https://github.com/ChengLabResearch/ouroboros/blob/main/python/ouroboros/pipeline/pipeline_input.py) does. That class shows the two patterns a custom input usually needs: a `field_serializer` and `field_validator` pair converts fields such as NumPy arrays to and from a JSON-compatible form. Field names must match the names your steps list in `inputs`.

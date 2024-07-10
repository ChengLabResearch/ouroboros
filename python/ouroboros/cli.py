from multiprocessing import freeze_support

import argparse

from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline import (
    Pipeline,
    PipelineInput,
    ParseJSONPipelineStep,
    SlicesGeometryPipelineStep,
    VolumeCachePipelineStep,
    SaveParallelPipelineStep,
    BackprojectPipelineStep,
    SaveConfigPipelineStep,
    LoadConfigPipelineStep,
)
import json


def main():
    # Create a cli parser with the built-in library argparse
    parser = argparse.ArgumentParser(
        prog="Ouroboros CLI",
        description="A CLI for extracting ROIs from cloud-hosted 3D volumes for segmentation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Create the parser for the slice command
    parser_slice = subparsers.add_parser(
        "slice", help="Slice the original volume along a path and save to a tiff file."
    )
    parser_slice.add_argument(
        "options",
        type=str,
        help="The path to the options json file.",
    )
    parser_slice.add_argument(
        "--verbose",
        action="store_true",
        help="Output timing statistics for the calculations.",
    )

    # Create the parser for the backproject command
    parser_backproject = subparsers.add_parser(
        "backproject",
        help="Project the straightened slices back into the space of the original volume.",
    )
    parser_backproject.add_argument(
        "options",
        type=str,
        help="The path to the options json file.",
    )
    parser_backproject.add_argument(
        "--verbose",
        action="store_true",
        help="Output timing statistics for the calculations.",
    )

    # Create the parser for the sample-options command
    subparsers.add_parser(
        "sample-options",
        help="Export sample options files into the current folder.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Dispatch to the appropriate function
    match args.command:
        case "slice":
            handle_slice(args)
        case "backproject":
            handle_backproject(args)
        case "sample-options":
            handle_sample_options()
        case _:
            parser.print_help()


def handle_slice(args):
    slice_options = SliceOptions.load_from_json(args.options)

    pipeline = Pipeline(
        [
            ParseJSONPipelineStep(),
            SlicesGeometryPipelineStep(),
            VolumeCachePipelineStep(),
            SaveParallelPipelineStep().with_progress_bar(),
            SaveConfigPipelineStep(),
        ]
    )

    input_data = PipelineInput(
        slice_options=slice_options, json_path=slice_options.neuroglancer_json
    )

    _, error = pipeline.process(input_data)

    if error:
        print(error)

    if args.verbose:
        print("\nCalculation Statistics:\n")

        for stat in pipeline.get_step_statistics():
            print(json.dumps(stat, indent=4), "\n")


def handle_backproject(args):
    backproject_options = BackprojectOptions.load_from_json(args.options)

    pipeline = Pipeline(
        [
            LoadConfigPipelineStep()
            .with_custom_output_file_path(backproject_options.straightened_volume_path)
            .with_custom_options(backproject_options),
            BackprojectPipelineStep().with_progress_bar(),
            SaveConfigPipelineStep(),
        ]
    )

    input_data = PipelineInput(config_file_path=backproject_options.config_path)

    _, error = pipeline.process(input_data)

    if error:
        print(error)

    if args.verbose:
        print("\nCalculation Statistics:\n")

        for stat in pipeline.get_step_statistics():
            print(json.dumps(stat, indent=4), "\n")


def handle_sample_options():
    sample_options = SliceOptions(
        slice_width=100,
        slice_height=100,
        output_file_folder="./output/",
        output_file_name="sample",
        neuroglancer_json="",
    )

    sample_options.save_to_json("./sample-slice-options.json")

    sample_options = BackprojectOptions(
        slice_width=100,
        slice_height=100,
        output_file_folder="./output/",
        output_file_name="sample",
        straightened_volume_path="./sample.tif",
        config_path="./sample-configuration.json",
    )

    sample_options.save_to_json("./sample-backproject-options.json")


if __name__ == "__main__":
    # Necessary to run multiprocessing in child processes
    freeze_support()

    main()

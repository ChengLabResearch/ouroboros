from ouroboros.common.memory_observer import dispatch_frozen_observer

dispatch_frozen_observer()

import argparse
import json
from multiprocessing import freeze_support
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation as R

from ouroboros.common.pipelines import geometry_pipeline
from ouroboros.common.logging_mode import LoggingMode
from ouroboros.common.server_handlers import handle_task
from ouroboros.common.server_types import BackProjectTask, SliceTask
from ouroboros.helpers.options import (
    DEFAULT_BACKPROJECT_OPTIONS,
    DEFAULT_SLICE_OPTIONS,
    SliceOptions,
)


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
    for operation_parser in (parser_slice, parser_backproject):
        logging = operation_parser.add_mutually_exclusive_group()
        logging.add_argument(
            "--logging-mode", type=LoggingMode, choices=list(LoggingMode),
            default=None,
            help="Off (default): no report; standard: print/save step and volume statistics; "
                 "detailed: also monitor RAM (adds overhead). Progress remains available in all modes.",
        )
        logging.add_argument(
            "--verbose", dest="logging_mode", action="store_const", const=LoggingMode.DETAILED,
            help="Compatibility alias for --logging-mode detailed.",
        )

    # Create the parser for the sample-options command
    subparsers.add_parser(
        "sample-options",
        help="Export sample options files into the current folder.",
    )

    # Create the parser for the geometry command
    parser_geometry = subparsers.add_parser(
        "geometry", help="Output the geometry of the spline."
    )
    parser_geometry.add_argument(
        "options",
        type=str,
        help="The path to the options json file.",
    )
    parser_geometry.add_argument(
        "--include-ng",
        action="store_true",
        help="Include Neuroglancer points and angles."
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.command in ("slice", "backproject") and args.logging_mode is None:
        # Resolve the default after argparse checks mutual exclusion, so an
        # explicit Off cannot be mistaken for an absent --logging-mode option.
        args.logging_mode = LoggingMode.OFF

    # Dispatch to the appropriate function
    match args.command:
        case "slice":
            handle_slice(args)
        case "backproject":
            handle_backproject(args)
        case "sample-options":
            handle_sample_options()
        case "geometry":
            handle_geometry(args)
        case _:
            parser.print_help()


def handle_geometry(args):
    print(f"Loading slice options from: {args.options}")
    slice_options = SliceOptions.load_from_json(args.options)

    if isinstance(slice_options, str):
        print("Exiting due to errors loading slice options.", file=sys.stderr)
        sys.exit(1)

    print("Slice options loaded successfully.")
    pipeline, input_data = geometry_pipeline(slice_options)

    _, error = pipeline.process(input_data)

    if error:
        print(f"Pipeline Error: {error}", file=sys.stderr)

    rects_output = Path(slice_options.output_file_folder, f"{slice_options.output_file_name}_rects").with_suffix(".npy")
    np.save(rects_output, input_data.slice_rects)

    if args.include_ng:
        ng_output = Path(slice_options.output_file_folder, f"{slice_options.output_file_name}_ng").with_suffix(".json")
        rot_matrix = np.empty((3, len(input_data.slice_rects), 3))

        rot_matrix[0] = (input_data.slice_rects[:, 1, :] - input_data.slice_rects[:, 0, :]) / slice_options.slice_width
        rot_matrix[1] = (input_data.slice_rects[:, 3, :] - input_data.slice_rects[:, 0, :]) / slice_options.slice_height
        rot_matrix[2] = np.divide(np.cross(rot_matrix[0], rot_matrix[1], axis=1),
                                  (slice_options.slice_width * slice_options.slice_height))
        centers = (input_data.slice_rects[:, 0, :] + input_data.slice_rects[:, 2, :]) / 2
        quats = R.from_matrix(rot_matrix.transpose(1, 0, 2)).as_quat()

        ng_points = []
        for point in range(len(input_data.slice_rects)):
            ng_points.append({"position": centers[point].tolist(), "crossSectionOrientation": quats[point].tolist()})

        with open(ng_output, "w") as handle:
            json.dump(ng_points, handle)


def _handle_operation(args, task_type):
    print(f"Loading options from: {args.options}")
    task = task_type(task_id=str(uuid4()), options=args.options,
                     logging_mode=args.logging_mode, show_progress=True)
    handle_task(task, print_statistics=True)
    if task.error:
        print(f"Pipeline Error: {task.error}", file=sys.stderr)
    publication = task.statistics.publication if task.statistics else {}
    publication_error = publication.get("error")
    presentation_error = publication.get("presentation_error")
    if publication_error:
        print(f"Statistics Error: {publication_error}", file=sys.stderr)
    if presentation_error:
        print(presentation_error, file=sys.stderr)
    if (task.status == "interrupted" or publication.get("status") == "interrupted"
            or publication.get("presentation_interrupted")):
        raise SystemExit(130)
    if task.error or publication_error or presentation_error:
        raise SystemExit(1)
    return task


def handle_slice(args):
    return _handle_operation(args, SliceTask)


def handle_backproject(args):
    return _handle_operation(args, BackProjectTask)


def handle_sample_options():
    # Create sample options files
    sample_slice_options = DEFAULT_SLICE_OPTIONS
    sample_backproject_options = DEFAULT_BACKPROJECT_OPTIONS

    # Save the sample options files
    sample_slice_options.save_to_json("./sample-slice-options.json")
    sample_backproject_options.save_to_json("./sample-backproject-options.json")


if __name__ == "__main__":
    # Necessary to run multiprocessing in child processes
    freeze_support()

    main()

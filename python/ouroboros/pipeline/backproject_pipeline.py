import concurrent.futures
from dataclasses import astuple
from functools import partial
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from filelock import FileLock
import numpy as np
import psutil
import tifffile

from ouroboros.helpers.memory_usage import (
    calculate_gigabytes_from_dimensions
)
from ouroboros.helpers.slice import (        # noqa: F401
    detect_color_channels_shape,
    FrontProjStack,
    backproject_box,
    BackProjectIter
)
from ouroboros.helpers.volume_cache import VolumeCache, get_mip_volume_sizes, update_writable_rects
from ouroboros.helpers.bounding_boxes import BoundingBox
from .pipeline import PipelineStep
from .pipeline_input import PipelineInput
from ouroboros.helpers.options import BackprojectOptions
from ouroboros.helpers.files import (
    format_backproject_resave_volume,
    get_sorted_tif_files,
    join_path,
    generate_tiff_write,
    write_conv_vol,
    write_raw_intermediate
)
from ouroboros.helpers.shapes import DataRange, ImgSliceC


class BackprojectPipelineStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(
            inputs=(
                "backproject_options",
                "volume_cache",
                "slice_rects",
            )
        )

    def _process(self,
                 input_data: tuple[BackprojectOptions, VolumeCache, np.ndarray, PipelineInput]
                 ) -> tuple[any, None] | tuple[None, any]:
        config, volume_cache, slice_rects, pipeline_input = input_data

        # Verify that a config object is provided
        if not isinstance(config, BackprojectOptions):
            return "Input data must contain a BackprojectOptions object."

        # Verify that input_data is a string containing a path to a tif file
        if not isinstance(config.straightened_volume_path, str):
            return "Input data must contain a string containing a path to a tif file."

        # Verify that a volume cache is given
        if not isinstance(volume_cache, VolumeCache):
            return "Input data must contain a VolumeCache object."

        # Verify that slice rects is given
        if not isinstance(slice_rects, np.ndarray):
            return "Input data must contain an array of slice rects."

        straightened_volume_path = config.straightened_volume_path

        # Make sure the straightened volume exists
        if not os.path.exists(straightened_volume_path):
            return (f"The straightened volume does not exist at {straightened_volume_path}.")

        if Path(straightened_volume_path).is_dir():
            with tifffile.TiffFile(next(Path(straightened_volume_path).iterdir())) as tif:
                is_compressed = bool(tif.pages[0].compression)
                # tiff format check to add
                FPShape = FrontProjStack(D=len(list(Path(straightened_volume_path).iterdir())),
                                         V=tif.pages[0].shape[0], U=tif.pages[0].shape[1])
                channels = 1 if len(tif.pages[0].shape) < 3 else tif.pages[0].shape[-1]
        else:
            with tifffile.TiffFile(straightened_volume_path) as tif:
                is_compressed = bool(tif.pages[0].compression)
                FPShape = FrontProjStack(D=len(tif.pages), V=tif.pages[0].shape[0], U=tif.pages[0].shape[1])
                channels = 1 if len(tif.pages[0].shape) < 3 else tif.pages[0].shape[-1]

        # Make sure the config dimensions match the straightened volume dimensions.
        ESShape = FrontProjStack(D=len(slice_rects), U=np.round(np.linalg.norm(slice_rects[0][1]-slice_rects[0][0])),
                                 V=np.round(np.linalg.norm(slice_rects[0][3]-slice_rects[0][0])))

        if ESShape != FPShape:
            raise ValueError("Straightened volume file does not match sliced shape:\n"
                             f" ({FPShape} vs {ESShape}, respectively).")

        cannot_memmap = False
        try:
            if Path(straightened_volume_path).is_dir():
                _ = tifffile.memmap(next(Path(straightened_volume_path).iterdir()), mode="r")
            else:
                _ = tifffile.memmap(straightened_volume_path, mode="r")
        except:     # noqa: E722
            # Check here is because memmap needs certain types of dataoffsets, which aren't always there
            # separate to compression.
            cannot_memmap = True

        if is_compressed or cannot_memmap:
            print("Input data compressed; Rewriting.")

            # Create a new path for the straightened volume
            new_straightened_volume_path = join_path(
                config.output_file_folder,
                format_backproject_resave_volume(config.output_file_name),
            )

            # Save the straightened volume to a new tif file
            with tifffile.TiffWriter(new_straightened_volume_path) as tif:
                if straightened_volume_path.endswith((".tif", ".tiff")):
                    # Read the single tif file
                    with tifffile.TiffFile(straightened_volume_path) as tif_in:
                        for i in range(len(tif_in.pages)):
                            tif.save(tif_in.pages[i].asarray(), contiguous=True, compression=None)
                else:
                    # Read the tif files from the folder
                    images = get_sorted_tif_files(straightened_volume_path)
                    for image in images:
                        tif.save(tifffile.imread(join_path(straightened_volume_path, image)),
                                 contiguous=True, compression=None)

            straightened_volume_path = new_straightened_volume_path

        full_bounding_box = BoundingBox.bound_boxes(volume_cache.bounding_boxes)
        write_shape = np.flip(full_bounding_box.get_shape()).tolist()

        pipeline_input.output_file_path = (f"{config.output_file_name}_"
                                           f"{'_'.join(map(str, full_bounding_box.get_min(np.uint32)))}")
        folder_path = Path(config.output_file_folder, pipeline_input.output_file_path)
        folder_path.mkdir(exist_ok=True, parents=True)

        # Intermediate Path
        i_path = Path(config.output_file_folder, f"{os.getpid()}_{config.output_file_name}")

        if config.make_single_file:
            is_big_tiff = calculate_gigabytes_from_dimensions(
                            np.prod(write_shape),
                            np.uint8 if config.make_backprojection_binary else np.uint16) > 4
        else:
            is_big_tiff = calculate_gigabytes_from_dimensions(
                            np.prod(write_shape[1:]),
                            np.uint8 if config.make_backprojection_binary else np.uint16) > 4

        # Generate image writing function
        # Combining compression with binary images can cause issues.
        bp_offset = pipeline_input.backprojection_offset if config.backproject_min_bounding_box else None
        tif_write = partial(generate_tiff_write,
                            compression=config.backprojection_compression,
                            micron_resolution=volume_cache.get_resolution_um(),
                            backprojection_offset=bp_offset,
                            photometric="rgb" if channels > 1 else "minisblack")

        if pipeline_input.slice_options.output_mip_level != config.output_mip_level:
            scaling_factors, _ = calculate_scaling_factors(
                pipeline_input.source_url,
                pipeline_input.slice_options.output_mip_level,
                config.output_mip_level,
                write_shape
            )
        else:
            scaling_factors = None

        # Allocate procs equally between BP math and writing if we're rescaling, otherwise 3-1 favoring
        # the BP calculation.
        exec_procs = config.process_count // 4 * (2 if scaling_factors is not None else 3)
        write_procs = config.process_count // 4 * (2 if scaling_factors is not None else 1)

        # Process each bounding box in parallel, writing the results to the backprojected volume
        try:
            with (concurrent.futures.ProcessPoolExecutor(exec_procs) as executor,
                 concurrent.futures.ProcessPoolExecutor(write_procs) as write_executor):
                bp_futures = []
                write_futures = []

                chunk_range = DataRange(FPShape.make_with(0), FPShape, FPShape.make_with(config.chunk_size))
                chunk_iter = partial(BackProjectIter, shape=FPShape, slice_rects=np.array(slice_rects))
                processed = np.zeros(astuple(chunk_range.length))
                z_sources = np.zeros((write_shape[0], ) + astuple(chunk_range.length), dtype=bool)

                for chunk, _, chunk_rects, _, index in chunk_range.get_iter(chunk_iter):
                    bp_futures.append(executor.submit(
                        process_chunk,
                        config=config,
                        straightened_volume_path=straightened_volume_path,
                        chunk_rects=chunk_rects,
                        chunk=chunk,
                        index=index,
                        full_bounding_box=full_bounding_box
                    ))

                # Track what's written.
                min_dim = full_bounding_box.get_min(int)[2]
                num_pages = full_bounding_box.get_shape()[2]
                writeable = np.zeros(num_pages)
                pages_written = 0

                def note_written(write_future):
                    self.add_timing("Free Memory", psutil.virtual_memory().available)
                    nonlocal pages_written
                    pages_written += 1
                    self.update_progress((np.sum(processed) / len(chunk_range)) * (exec_procs / config.process_count)
                                         + (pages_written / num_pages) * (write_procs / config.process_count))
                    for key, value in write_future.result().items():
                        self.add_timing(key, value)

                for bp_future in concurrent.futures.as_completed(bp_futures):
                    self.add_timing("Free Memory", psutil.virtual_memory().available)
                    start = time.perf_counter()
                    # Store the durations for each bounding box
                    durations, index, z_stack = bp_future.result()
                    for key, value in durations.items():
                        self.add_timing_list(key, value)

                    z_sources[(z_stack, ) + index] = True

                    # Update the progress bar
                    processed[index] = 1
                    self.update_progress((np.sum(processed) / len(chunk_range)) * (exec_procs / config.process_count)
                                         + (pages_written / num_pages) * (write_procs / config.process_count))

                    update_writable_rects(processed, slice_rects, min_dim, writeable, config.chunk_size)

                    if np.any(writeable == 1):
                        write = np.flatnonzero(writeable == 1)
                        # Single File needs to be in order
                        for index in write:
                            write_futures.append(write_executor.submit(
                                write_conv_vol,
                                writer=tif_write(tifffile.imwrite),
                                source_path=i_path.joinpath(f"i_{index:05}.dat"),
                                shape=ImgSliceC(*write_shape[1:], channels),
                                dtype=bool if config.make_backprojection_binary else np.uint16,
                                scaling=scaling_factors,
                                target_folder=folder_path,
                                index=index,
                                interpolation=config.upsample_order,
                                discrete=config.make_backprojection_discrete
                            ))
                            write_futures[-1].add_done_callback(note_written)

                        writeable[write] = 2

                    self.add_timing("Process Backproject Future", time.perf_counter() - start)

        except BaseException as e:
            traceback.print_tb(e.__traceback__, file=sys.stderr)
            return f"An error occurred while processing the bounding boxes: {e}"

        for write_future in concurrent.futures.as_completed(write_futures):
            # Consume them to make sure they're finished.
            pass

        start = time.perf_counter()

        if config.make_single_file:
            pipeline_input.output_file_path += ".tif"
            writer = tif_write(tifffile.TiffWriter(folder_path.with_suffix(".tif"), bigtiff=is_big_tiff).write)
            for fname in get_sorted_tif_files(folder_path):
                writer(tifffile.imread(folder_path.joinpath(fname)))

        # Update the pipeline input with the output file path
        pipeline_input.backprojected_folder_path = folder_path

        self.add_timing("export", time.perf_counter() - start)

        if config.make_single_file:
            shutil.rmtree(folder_path)
        shutil.rmtree(i_path)

        return None


def process_chunk(
    config: BackprojectOptions,
    straightened_volume_path: str,
    chunk_rects: list[np.ndarray],
    chunk: tuple[slice],
    index: tuple[int],
    full_bounding_box: BoundingBox
) -> tuple[dict, str, int]:
    durations = {}

    start_total = time.perf_counter()

    # Load the straightened volume
    try:
        straightened_volume = tifffile.memmap(straightened_volume_path, mode="r")
        durations["memmap"] = [time.perf_counter() - start_total]
    except BaseException as be:
        print(f"Error loading Volume: {be} : {straightened_volume_path}")
        traceback.print_tb(be.__traceback__, file=sys.stderr)
        raise be

    # Get the slices from the straightened volume  Dumb but maybe bugfix?
    start = time.perf_counter()
    slices = straightened_volume[chunk].squeeze()
    bounding_box = BoundingBox.from_rects(chunk_rects)

    # Close the memmap
    del straightened_volume
    durations["get_slices"] = [time.perf_counter() - start]

    start = time.perf_counter()
    try:
        lookup, values, weights = backproject_box(bounding_box, chunk_rects, slices)
    except BaseException as be:
        print(f"Error on BP: {be}")
        traceback.print_tb(be.__traceback__, file=sys.stderr)
        raise be

    durations["back_project"] = [time.perf_counter() - start]
    durations["total_bytes"] = [int(lookup.nbytes + values.nbytes + weights.nbytes)]

    if values.nbytes == 0:
        # No data to write from this chunk, so return as such.
        durations["total_chunk_process"] = [time.perf_counter() - start_total]
        return durations, index, []

    # Save the data
    try:
        start = time.perf_counter()

        z_vals, yx_vals = np.divmod(lookup, np.prod(bounding_box.get_shape()[:2], dtype=np.uint32))
        offset = np.flip(bounding_box.get_min(np.int64) - full_bounding_box.get_min(np.int64)).astype(np.uint32)

        offset_dict = {
            # Columns are Y, Rows are X;  Offset is ZYX; Bounding Box Shapes are XYZ
            "source_rows": bounding_box.get_shape()[0],
            "target_rows": full_bounding_box.get_shape()[0],
            "offset_columns": offset[1],
            "offset_rows": offset[2],
            "channel_count": np.uint32(1 if len(slices.shape) < 4 else slices.shape[-1]),
        }
        type_ar = np.array([yx_vals.dtype.str, values.dtype.str, weights.dtype.str], dtype='S8')
        durations["split"] = [time.perf_counter() - start]

        # Gets slices off full array corresponding to each Z value.
        z_idx = [0] + list(np.where(z_vals[:-1] != z_vals[1:])[0] + 1) + [len(z_vals)]
        z_stack = z_vals[z_idx[:-1]]
        z_slices = [np.s_[z_idx[i]: z_idx[i + 1]] for i in range(len(z_idx) - 1)]

        durations["stack"] = [time.perf_counter() - start]
        start = time.perf_counter()

        i_path = Path(config.output_file_folder, f"{os.getppid()}_{config.output_file_name}")
        i_path.mkdir(exist_ok=True, parents=True)

        def write_z(target, z_slice):
            write_raw_intermediate(target,
                                   np.fromiter(offset_dict.values(), dtype=np.uint32, count=5).tobytes(),
                                   np.uint32(len(yx_vals[z_slice])).tobytes(),
                                   type_ar.tobytes(),
                                   yx_vals[z_slice].tobytes(), values[z_slice].tobytes(), weights[z_slice].tobytes())

        def make_z(i, z_slice):
            offset_z = z_stack[i] + offset[0]
            z_path = i_path.joinpath(f"i_{offset_z:05}.dat")
            with FileLock(z_path.with_suffix(".lock")):
                write_z(open(z_path, "ab"), z_slice)

        with ThreadPool(12) as pool:
            pool.starmap(make_z, enumerate(z_slices))

        durations["write_intermediate"] = [time.perf_counter() - start]
    except BaseException as be:
        print(f"Error on BP: {be}")
        traceback.print_tb(be.__traceback__, file=sys.stderr)
        raise be

    durations["total_chunk_process"] = [time.perf_counter() - start_total]

    return durations, index, z_stack + offset[0]


def calculate_scaling_factors(
    source_url: str, current_mip: int, target_mip: int, tif_shape: tuple
) -> tuple[tuple, tuple]:
    # Determine the current and target resolutions
    mip_sizes = get_mip_volume_sizes(source_url)

    current_resolution = mip_sizes[current_mip]
    target_resolution = mip_sizes[target_mip]

    # Determine the scaling factor for each axis as a tuple
    resolution_factors = tuple(
        max(target_resolution[i] / current_resolution[i], 1)
        for i in range(len(target_resolution))
    )

    has_color_channels, num_channels = detect_color_channels_shape(tif_shape)

    # Determine the scaling factor for each axis as a tuple
    scaling_factors = resolution_factors + (
        (num_channels,) if has_color_channels else ()
    )

    return scaling_factors, resolution_factors

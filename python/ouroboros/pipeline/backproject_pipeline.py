import concurrent.futures
from dataclasses import astuple
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import shutil
import sys
import tifffile
import time
import traceback

import numpy as np
import scipy
import scipy.ndimage

from ouroboros.helpers.memory_usage import (
    calculate_chunk_size,
    calculate_gigabytes_from_dimensions
)
from ouroboros.helpers.slice import (        # noqa: F401
    detect_color_channels_shape,
    make_volume_binary,
    FrontProjStack,
    backproject_box,
    BackProjectIter
)
from ouroboros.helpers.volume_cache import VolumeCache, get_mip_volume_sizes, update_writable_rects
from ouroboros.helpers.bounding_boxes import BoundingBox
from .pipeline import PipelineStep
from ouroboros.helpers.options import BackprojectOptions
from ouroboros.helpers.files import (
    format_backproject_resave_volume,
    format_tiff_name,
    get_sorted_tif_files,
    join_path,
    num_digits_for_n_files,
    parse_tiff_name,
    generate_tiff_write,
    write_conv_vol,
    write_small_intermediate
)
from ouroboros.helpers.shapes import DataRange, ImgSlice


DEFAULT_CHUNK_SIZE = 160
AXIS = 0


class BackprojectPipelineStep(PipelineStep):
    def __init__(self, processes=cpu_count()) -> None:
        super().__init__(
            inputs=(
                "backproject_options",
                "volume_cache",
                "slice_rects",
            )
        )

        self.num_processes = processes

    def _process(self, input_data: any) -> tuple[any, None] | tuple[None, any]:
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
        else:
            with tifffile.TiffFile(straightened_volume_path) as tif:
                is_compressed = bool(tif.pages[0].compression)
                FPShape = FrontProjStack(D=len(tif.pages), V=tif.pages[0].shape[0], U=tif.pages[0].shape[1])

        if is_compressed:
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

        # Write huge temp files (need to address)
        full_bounding_box = BoundingBox.bound_boxes(volume_cache.bounding_boxes)
        write_shape = np.flip(full_bounding_box.get_shape()).tolist()
        print(f"\nFront Projection Shape: {FPShape}")
        print(f"\nBack Projection Shape (Z/Y/X):{write_shape}")

        pipeline_input.output_file_path = f"{config.output_file_name}_{'_'.join(map(str, full_bounding_box.get_min(np.uint32)))}"
        folder_path = Path(config.output_file_folder, pipeline_input.output_file_path)
        folder_path.mkdir(exist_ok=True, parents=True)

        i_path = Path(config.output_file_folder,
                      f"{config.output_file_name}_t_{'_'.join(map(str, full_bounding_box.get_min(np.uint32)))}")

        if config.make_single_file:
            is_big_tiff = calculate_gigabytes_from_dimensions(np.prod(write_shape), np.uint16) > 4     # Check Dtype
        else:
            is_big_tiff = calculate_gigabytes_from_dimensions(np.prod(write_shape[1:]), np.uint16) > 4     # Check Dtype

        bp_offset = pipeline_input.backprojection_offset if config.backproject_min_bounding_box else None
        tif_write = partial(generate_tiff_write,
                            compression=config.backprojection_compression,
                            micron_resolution=volume_cache.get_resolution_um(),
                            backprojection_offset=bp_offset)

        # Process each bounding box in parallel, writing the results to the backprojected volume
        try:
            with (concurrent.futures.ProcessPoolExecutor((self.num_processes // 4) * 3) as executor,
                 concurrent.futures.ProcessPoolExecutor(self.num_processes // 4) as write_executor):
                bp_futures = []
                write_futures = []

                chunk_range = DataRange(FPShape.make_with(0), FPShape, FPShape.make_with(DEFAULT_CHUNK_SIZE))
                chunk_iter = partial(BackProjectIter, shape=FPShape, slice_rects=np.array(slice_rects))
                processed = np.zeros(astuple(chunk_range.length))
                z_sources = np.zeros((write_shape[0], ) + astuple(chunk_range.length), dtype=bool)

                for chunk, _, chunk_rects, _, index in chunk_range.get_iter(chunk_iter):
                    bp_futures.append(executor.submit(
                        process_chunk,
                        config,
                        straightened_volume_path,
                        chunk_rects,
                        chunk,
                        index,
                        full_bounding_box
                    ))

                # Track what's written.
                min_dim = full_bounding_box.get_min(int)[2]
                num_pages = full_bounding_box.get_shape()[2]
                writeable = np.zeros(num_pages)
                pages_written = 0

                def note_written(write_future):
                    nonlocal pages_written
                    pages_written += 1
                    self.update_progress((np.sum(processed) / len(chunk_range)) * (2 / 3)
                                         + (pages_written / num_pages) * (1 / 3))
                    for key, value in write_future.result().items():
                        self.add_timing(key, value)

                for bp_future in concurrent.futures.as_completed(bp_futures):
                    start = time.perf_counter()
                    # Store the durations for each bounding box
                    durations, index, z_stack = bp_future.result()
                    for key, value in durations.items():
                        self.add_timing_list(key, value)

                    z_sources[(z_stack, ) + index] = True

                    # Update the progress bar
                    processed[index] = 1
                    self.update_progress((np.sum(processed) / len(chunk_range)) * (2 / 3)
                                         + (pages_written / num_pages) * (1 / 3))

                    update_writable_rects(processed, slice_rects, min_dim, writeable, DEFAULT_CHUNK_SIZE)

                    if np.any(writeable == 1):
                        write = np.flatnonzero(writeable == 1)
                        # Single File needs to be in order
                        for index in write:
                            write_futures.append(write_executor.submit(
                                write_conv_vol,
                                tif_write(tifffile.imwrite), i_path.joinpath(f"i_{index:05}"),
                                ImgSlice(*write_shape[1:]), np.uint16, folder_path.joinpath(f"{index:05}.tif")
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
            writer = tif_write(tifffile.TiffWriter(folder_path.with_suffix(".tiff"), bigtiff=is_big_tiff).write)
            for fname in get_sorted_tif_files(folder_path):
                writer(tifffile.imread(folder_path.joinpath(fname)))

        # Rescale the backprojected volume to the output mip level
        if pipeline_input.slice_options.output_mip_level != config.output_mip_level:
            output_name = f"{folder_path}-temp"

            error = rescale_mip_volume(
                pipeline_input.source_url,
                pipeline_input.slice_options.output_mip_level,
                config.output_mip_level,
                single_path=(None if config.make_single_file is False else folder_path + ".tif"),
                folder_path=(folder_path if config.make_single_file is False else None),
                output_name=output_name,
                compression=config.backprojection_compression,
                max_ram_gb=config.max_ram_gb,
                order=config.upsample_order,
                binary=config.make_backprojection_binary,
            )

            if error is not None:
                return error

            # Remove the original backprojected volume
            if config.make_single_file:
                os.remove(folder_path + ".tif")
            else:
                shutil.rmtree(folder_path)

            # Rename the rescaled volume
            if config.make_single_file:
                os.rename(output_name + ".tif", folder_path + ".tif")
            else:
                os.rename(output_name, folder_path)

        # Update the pipeline input with the output file path
        pipeline_input.backprojected_folder_path = folder_path

        self.add_timing("export", time.perf_counter() - start)
        
        if config.make_single_file:        
            shutil.rmtree(folder_path)

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
    straightened_volume = tifffile.memmap(straightened_volume_path, mode="r")
    durations["memmap"] = [time.perf_counter() - start_total]

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

    if len(values) == 0:
        # No data to write from this chunk, so return as such.
        durations["total_process"] = [time.perf_counter() - start_total]
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
        }
        durations["split"] = [time.perf_counter() - start]

        # Gets slices off full array corresponding to each Z value.
        z_idx = [0] + list(np.where(z_vals[:-1] != z_vals[1:])[0] + 1) + [len(z_vals)]
        z_stack = z_vals[z_idx[:-1]]
        z_slices = [np.s_[z_idx[i]: z_idx[i + 1]] for i in range(len(z_idx) - 1)]

        durations["stack"] = [time.perf_counter() - start]
        start = time.perf_counter()

        file_path = Path(config.output_file_folder,
                         f"{config.output_file_name}_t_{'_'.join(map(str, full_bounding_box.get_min(np.uint32)))}")
        file_path.mkdir(exist_ok=True, parents=True)

        def write_z(i, z_slice):
            offset_z = z_stack[i] + offset[0]
            file_path.joinpath(f"i_{offset_z:05}").mkdir(exist_ok=True, parents=True)
            write_small_intermediate(file_path.joinpath(f"i_{offset_z:05}", f"{index}.tif"),
                                     np.fromiter(offset_dict.values(), dtype=np.uint32, count=4),
                                     yx_vals[z_slice], values[z_slice], weights[z_slice])

        with ThreadPool(12) as pool:
            pool.starmap(write_z, enumerate(z_slices))

        durations["write_intermediate"] = [time.perf_counter() - start]
    except BaseException as be:
        print(f"Error on BP: {be}")
        traceback.print_tb(be.__traceback__, file=sys.stderr)
        raise be

    durations["total_process"] = [time.perf_counter() - start_total]

    return durations, index, z_stack + offset[0]


def rescale_mip_volume(
    source_url: str,
    current_mip: int,
    target_mip: int,
    single_path: str = None,
    folder_path: str = None,
    output_name: str = "out",
    compression: str = None,
    max_ram_gb: int = 0,
    order: int = 2,
    binary: bool = False,
) -> str | None:
    """
    Rescale the volume to the mip level.

    Parameters
    ----------
    source_url : str
        The URL of the volume.
    current_mip : int
        The current mip level of the volume.
    target_mip : int
        The target mip level of the volume.
    single_path : str
        The path to the single tif file.
    folder_path : str
        The path to the folder containing the tif files.
    output_name : str
        The path to the output.
    compression : str, optional
        The compression to use for the resulting tif file.
        The default is None.
    max_ram_gb : int, optional
        The maximum amount of RAM to use in GB.
        The default is 0.
    order : int, optional
        The order of the interpolation.
        The default is 2.
    binary : bool, optional
        Whether to make the backprojected volume binary.

    Returns
    -------
    str | None
        Error message if an error occurred.
    """

    if single_path is None and folder_path is None:
        return "Either single_path or folder_path must be provided."

    if target_mip == current_mip:
        return None

    if single_path is not None:
        return rescale_single_tif(
            source_url,
            current_mip,
            target_mip,
            single_path,
            compression=compression,
            file_name=output_name + ".tif",
            max_ram_gb=max_ram_gb,
            order=order,
            binary=binary,
        )

    return rescale_folder_tif(
        source_url,
        current_mip,
        target_mip,
        folder_path,
        compression=compression,
        folder_name=output_name,
        max_ram_gb=max_ram_gb,
        order=order,
        binary=binary,
    )


def rescale_single_tif(
    source_url: str,
    current_mip: int,
    target_mip: int,
    single_path: str,
    file_name: str = "out.tif",
    compression: str = None,
    max_ram_gb: int = 0,
    order: int = 1,
    binary: bool = False,
) -> str | None:
    with tifffile.TiffFile(single_path) as tif:
        tif_shape = (len(tif.pages),) + tif.pages[0].shape

        scaling_factors, _ = calculate_scaling_factors(
            source_url, current_mip, target_mip, tif_shape
        )

        # Calculate the output tiff shape
        output_shape = tuple(
            int(tif_shape[i] * scaling_factors[i]) for i in range(len(tif_shape))
        )

        # Note: The chunk size is divided by the scaling factor to account for the
        # number of slices that need to be loaded to produce chunk_size slices in the output volume
        chunk_size = max(
            int(
                calculate_chunk_size(
                    output_shape, tif.pages[0].dtype, max_ram_gb=max_ram_gb
                )
                / scaling_factors[0]
            ),
            1,
        )

        with tifffile.TiffWriter(file_name) as output_volume:
            for i in range(0, tif_shape[0], chunk_size):
                # Stack the tif layers along the first axis (chunk_size)
                tif_layer = np.array(
                    [
                        tif.pages[j].asarray()
                        for j in range(i, min(i + chunk_size, tif_shape[0]))
                    ]
                )

                layers = scipy.ndimage.zoom(tif_layer, scaling_factors, order=order)

                if binary:
                    layers = make_volume_binary(layers)

                size = layers.shape[0]

                # Save the layers to the tif file
                for j in range(size):
                    output_volume.write(
                        layers[j],
                        contiguous=compression is None or compression == "none",
                        compression=compression,
                        software="ouroboros",
                    )

    return None


def rescale_folder_tif(
    source_url: str,
    current_mip: int,
    target_mip: int,
    folder_path: str,
    folder_name: str = "out",
    compression: str = None,
    max_ram_gb: int = 0,
    order: int = 1,
    binary: bool = False,
) -> str | None:
    # Create output folder if it doesn't exist
    output_folder = folder_name
    os.makedirs(output_folder, exist_ok=True)

    tifs = get_sorted_tif_files(folder_path)

    if len(tifs) == 0:
        return "No tif files found in the folder."

    sample_tif = tifffile.imread(join_path(folder_path, tifs[0]))

    # Determine the shape of the tif stack
    new_shape = (len(tifs), *sample_tif.shape)

    scaling_factors, resolution_factors = calculate_scaling_factors(
        source_url, current_mip, target_mip, new_shape
    )

    # Note: The chunk size is divided by the scaling factor to account for the
    # number of slices that need to be loaded to produce chunk_size slices in the output volume
    chunk_size = max(
        int(
            calculate_chunk_size(new_shape, sample_tif.dtype, max_ram_gb=max_ram_gb)
            / scaling_factors[0]
        ),
        1,
    )

    num_digits = num_digits_for_n_files(len(tifs))

    first_index = parse_tiff_name(tifs[0])

    output_index = int(first_index * resolution_factors[0])

    # Resize the volume
    for i in range(0, len(tifs), chunk_size):
        # Stack the tif layers along the first axis (chunk_size)
        tif = np.array(
            [
                tifffile.imread(join_path(folder_path, tifs[j]))
                for j in range(i, min(i + chunk_size, len(tifs)))
            ]
        )

        layers = scipy.ndimage.zoom(tif, scaling_factors, order=order)

        if binary:
            layers = make_volume_binary(layers)

        size = layers.shape[0]

        # Write the layers to new tif files
        for j in range(size):
            tifffile.imwrite(
                join_path(output_folder, format_tiff_name(output_index, num_digits)),
                layers[j],
                contiguous=True if compression is None else False,
                compression=compression,
                software="ouroboros",
            )
            output_index += 1

    return None


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

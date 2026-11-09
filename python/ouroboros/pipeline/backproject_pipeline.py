import concurrent.futures
from dataclasses import astuple, replace
from functools import partial
from math import ceil
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
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
from ouroboros.helpers.volume_cache import CloudVolumeInterface, update_writable_rects
from ouroboros.helpers.bounding_boxes import BoundingBox
from .pipeline import PipelineStep
from .pipeline_input import PipelineInput
from ouroboros.helpers.options import BackprojectOptions
from ouroboros.helpers.files import (
    format_backproject_resave_volume,
    get_sorted_tif_files,
    generate_tiff_write,
    StraightenedVolumeInfo,
    validate_straightened_volume_for_backprojection,
    write_conv_vol,
    write_raw_intermediate
)
from ouroboros.helpers.shapes import DataRange, DataShape, ImgSliceC, NGOrder, ImgSlice, Z


class BackprojectPipelineStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(
            inputs=(
                "backproject_options",
                "source_url",
                "slice_rects",
            )
        )

    def _process(self,
                 input_data: tuple[BackprojectOptions, str, np.ndarray, PipelineInput]
                 ) -> tuple[any, None] | tuple[None, any]:
        config, source_url, slice_rects, pipeline_input = input_data
        self.set_stage("preparing_input")

        # Verify that a config object is provided
        if not isinstance(config, BackprojectOptions):
            return "Input data must contain a BackprojectOptions object."

        # Verify that input_data is a string containing a path to a tif file
        if not isinstance(config.straightened_volume_path, str):
            return "Input data must contain a string containing a path to a tif file."

        if not isinstance(config.slice_options_path, str) or config.slice_options_path.strip() == "":
            return (
                "Missing slice config: BackprojectOptions.slice_options_path must point to "
                "the matching slicing configuration."
            )

        if pipeline_input.slice_options is None:
            return "Missing slice config: loaded SliceOptions are required for backprojection."

        # Verify that slice rects is given
        if not isinstance(slice_rects, np.ndarray):
            return "Input data must contain an array of slice rects."

        # Backprojection partitions the straightened input into its own chunks.
        # It only needs the slice envelope and source metadata, not a slice cache
        # or the download partitioner's candidate tree and planning pool.
        source = CloudVolumeInterface(source_url)
        pipeline_input.backprojection_source = source
        full_bounding_box = BoundingBox.from_rects(slice_rects)
        pipeline_input.backprojection_box = full_bounding_box

        straightened_volume_path = config.straightened_volume_path

        # Make sure the strai.tolist()ghtened volume exists
        if not os.path.exists(straightened_volume_path):
            return (f"The straightened volume does not exist at {straightened_volume_path}.")

        try:
            volume_info = validate_straightened_volume_for_backprojection(
                straightened_volume_path, slice_rects
            )
        except (TypeError, ValueError) as e:
            return str(e)

        owns_temporary_input = False
        FPShape = FrontProjStack(
            D=volume_info.shape.D,
            V=volume_info.shape.V,
            U=volume_info.shape.U,
        )
        cannot_memmap = False
        try:
            _ = tifffile.memmap(volume_info.sample_path, mode="r")
        except:     # noqa: E722
            # Check here is because memmap needs certain types of dataoffsets, which aren't always there
            # separate to compression.
            cannot_memmap = True

        if volume_info.is_compressed or cannot_memmap or volume_info.path.is_dir():
            print("Preparing a contiguous straightened TIFF for backprojection.")

            # Create a new path for the straightened volume
            new_straightened_volume_path = Path(
                config.output_file_folder,
                format_backproject_resave_volume(config.output_file_name),
            )

            # Save the straightened volume to a new tif file
            input_is_big_tiff = is_big_tiff(astuple(volume_info.shape), None, volume_info.channels, volume_info.dtype)
            # The output folder may be the input folder. Capture its members
            # before creating the temporary TIFF so it cannot read itself.
            images = get_sorted_tif_files(volume_info.path) if volume_info.path.is_dir() else None
            if new_straightened_volume_path.exists():
                overwrites_input = (
                    new_straightened_volume_path.samefile(volume_info.path) if images is None else
                    any(new_straightened_volume_path.samefile(volume_info.path / image) for image in images)
                )
                if overwrites_input:
                    return "Cannot prepare a temporary straightened TIFF over an input file."
            with tifffile.TiffWriter(new_straightened_volume_path, bigtiff=input_is_big_tiff) as tif:
                write_page = partial(tif.write, contiguous=True, compression=None,
                                     photometric=volume_info.photometric,
                                     **({"planarconfig": "contig"} if volume_info.channels > 1 else {}))
                if images is None:
                    # Read the single tif file
                    with tifffile.TiffFile(volume_info.path) as tif_in:
                        for i in range(len(tif_in.pages)):
                            write_page(tif_in.pages[i].asarray())
                else:
                    # Read the tif files from the folder
                    for image in images:
                        write_page(tifffile.imread(volume_info.path / image))

            volume_info = replace(volume_info, path=new_straightened_volume_path,
                                  sample_path=new_straightened_volume_path, is_compressed=False)
            owns_temporary_input = True

        slice_mip = pipeline_input.slice_options.output_mip_level

        output_origin = get_output_origin(source, slice_mip, config.output_mip_level,
                                          full_bounding_box.get_integer_origin())
        pipeline_input.output_file_path = (f"{config.output_file_name}_{'_'.join(map(str, output_origin))}")
        folder_path = Path(config.output_file_folder, pipeline_input.output_file_path)
        folder_path.mkdir(exist_ok=True, parents=True)

        write_shape = NGOrder(*np.flip(full_bounding_box.get_shape()), volume_info.channels)
        output_shape = get_output_shape(source, slice_mip, config.output_mip_level, write_shape)

        # Intermediate Path
        i_path = Path(config.output_file_folder, f"{os.getpid()}_{config.output_file_name}")

        output_dtype = np.uint8 if config.make_backprojection_binary else np.uint16
        page_is_big_tiff = is_big_tiff(output_shape.extract(ImgSliceC), output_dtype)
        volume_is_big_tiff = is_big_tiff(output_shape, output_dtype) if config.make_single_file else page_is_big_tiff

        # Generate image writing function
        # Combining compression with binary images can cause issues.
        bp_offset = pipeline_input.backprojection_offset if config.backproject_min_bounding_box else None
        tif_write = partial(generate_tiff_write,
                            compression=config.backprojection_compression,
                            micron_resolution=source.get_resolution_um(config.output_mip_level),
                            backprojection_offset=bp_offset,
                            photometric=volume_info.photometric,
                            **({"planarconfig": "contig"} if volume_info.channels > 1 else {}))

        # Allocate procs equally between BP math and writing; 1:2 if scaling.
        self.exec_config = {
            "backproject_processes": int(config.process_count // (3 if output_shape != write_shape else 2)),
            "writer_processes": int(config.process_count // (1.5 if output_shape != write_shape is not None else 2)),
            "intermediate_threads_per_process": 12,
        }

        # Process each bounding box in parallel, writing the results to the backprojected volume
        try:
            self.set_stage("backprojecting_and_writing", determinate=True)
            with (concurrent.futures.ProcessPoolExecutor(self.exec_config["backproject_processes"]) as executor,
                  concurrent.futures.ProcessPoolExecutor(self.exec_config["writer_processes"]) as write_executor):
                bp_futures = []
                write_futures = []

                chunk_range = DataRange(FPShape.make_with(0), FPShape, FPShape.make_with(config.chunk_size))
                chunk_iter = partial(BackProjectIter, shape=FPShape, slice_rects=np.array(slice_rects))
                processed = np.zeros(astuple(chunk_range.length))
                z_sources = np.zeros((write_shape[0], ) + astuple(chunk_range.length), dtype=bool)

                for chunk, _, chunk_rects, _, index in chunk_range.get_iter(chunk_iter):
                    bp_futures.append(executor.submit(
                        process_chunk,
                        intermediate_directory=i_path,
                        volume_info=volume_info,
                        chunk_rects=chunk_rects,
                        chunk=chunk,
                        index=index,
                        full_bounding_box=full_bounding_box,
                        intermediate_threads=self.exec_config["intermediate_threads_per_process"],
                    ))

                # Track what's written.
                min_dim = full_bounding_box.get_integer_origin()[2]
                num_pages = full_bounding_box.get_shape()[2]
                writeable = np.zeros(num_pages)
                pages_written = 0
                callback_errors = []

                def note_written(write_future):
                    nonlocal pages_written
                    try:
                        durations = write_future.result()
                        pages_written += 1
                        self.update_progress(
                            (np.sum(processed) / len(chunk_range))
                            * (self.exec_config["backproject_processes"] / config.process_count)
                            + (pages_written / num_pages)
                            * (self.exec_config["writer_processes"] / config.process_count)
                        )
                        for key, value in durations.items():
                            includes = tuple(name for name in durations if name != key) \
                                if key == "Total Chunk Merge" else ()
                            self.add_timing(key, value, includes=includes)
                    except BaseException as error:
                        callback_errors.append(error)

                for bp_future in concurrent.futures.as_completed(bp_futures):
                    start = time.perf_counter()
                    # Store the durations for each bounding box
                    durations, intermediate_bytes, index, z_stack = bp_future.result()
                    self.add_byte_count("intermediate_payload_bytes", intermediate_bytes)
                    for key, value in durations.items():
                        includes = tuple(name for name in durations if name != key) \
                            if key == "total_chunk_process" else ()
                        self.add_timing_list(key, value, includes=includes)

                    z_sources[(z_stack, ) + index] = True

                    # Update the progress bar
                    processed[index] = 1
                    self.update_progress(
                        (np.sum(processed) / len(chunk_range))
                        * (self.exec_config["backproject_processes"] / config.process_count)
                        + (pages_written / num_pages)
                        * (self.exec_config["writer_processes"] / config.process_count)
                    )

                    update_writable_rects(processed, slice_rects, min_dim, writeable, config.chunk_size)

                    if np.any(writeable == 1):
                        write = np.flatnonzero(writeable == 1)
                        # Single File needs to be in order
                        for index in write:
                            write_futures.append(write_executor.submit(
                                write_conv_vol,
                                writer=tif_write(partial(tifffile.imwrite, bigtiff=page_is_big_tiff or None)),
                                source_path=i_path.joinpath(f"i_{index:05}.dat"),
                                target_folder=folder_path,
                                shape=write_shape.extract(ImgSliceC()),
                                output_shape=output_shape.extract(ImgSliceC),
                                z_steps=round(output_shape.C / write_shape.C),
                                dtype=bool if config.make_backprojection_binary else np.uint16,
                                index=index,
                                interpolation=config.upsample_order,
                                discrete=config.make_backprojection_discrete
                            ))
                            write_futures[-1].add_done_callback(note_written)

                        writeable[write] = 2

                    self.add_timing("Process Backproject Future", time.perf_counter() - start, scope="coordinator")

                for write_future in concurrent.futures.as_completed(write_futures):
                    write_future.result()
            # Executor shutdown joins its callback thread before this check.
            if callback_errors:
                raise callback_errors[0]

        except Exception as e:
            traceback.print_tb(e.__traceback__, file=sys.stderr)
            return f"An error occurred while processing the bounding boxes: {e}"

        start = time.perf_counter()

        if config.make_single_file:
            self.set_stage("exporting_tiff")
            pipeline_input.output_file_path += ".tif"
            with tifffile.TiffWriter(folder_path.with_suffix(".tif"), bigtiff=volume_is_big_tiff) as output_tiff:
                writer = tif_write(output_tiff.write)
                for fname in get_sorted_tif_files(folder_path):
                    writer(tifffile.imread(folder_path.joinpath(fname)))

        # Update the pipeline input with the output file path
        pipeline_input.backprojected_folder_path = folder_path

        self.add_timing("export", time.perf_counter() - start, scope="coordinator")

        self.set_stage("cleaning_intermediates")
        if config.make_single_file:
            shutil.rmtree(folder_path)
        shutil.rmtree(i_path)
        # Workers have exited. Keep failed runs' inputs with their intermediates.
        if owns_temporary_input:
            volume_info.path.unlink()

        return None


def process_chunk(
    intermediate_directory: Path,
    volume_info: StraightenedVolumeInfo,
    chunk_rects: list[np.ndarray],
    chunk: tuple[slice],
    index: tuple[int],
    full_bounding_box: BoundingBox,
    intermediate_threads: int = 12,
) -> tuple[dict, int, tuple[int], np.ndarray | list]:
    durations = {}

    start_total = time.perf_counter()

    # Load the straightened volume
    try:
        # TIFF readers can omit singleton axes; validated metadata owns DVUC.
        straightened_volume = tifffile.memmap(volume_info.path, mode="r").reshape(
            (*astuple(volume_info.shape), volume_info.channels)
        )
        durations["memmap"] = [time.perf_counter() - start_total]
    except BaseException as be:
        print(f"Error loading Volume: {be} : {volume_info.path}")
        traceback.print_tb(be.__traceback__, file=sys.stderr)
        raise be

    # Preserve depth, height and width even in a one-sample tail chunk.
    start = time.perf_counter()
    slices = straightened_volume[chunk]
    if volume_info.channels == 1:
        slices = slices[..., 0]
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
    intermediate_bytes = int(lookup.nbytes + values.nbytes + weights.nbytes)

    if values.nbytes == 0:
        # No data to write from this chunk, so return as such.
        durations["total_chunk_process"] = [time.perf_counter() - start_total]
        return durations, intermediate_bytes, index, []

    # Save the data
    try:
        start = time.perf_counter()

        z_vals, yx_vals = np.divmod(lookup, np.prod(bounding_box.get_shape()[:2], dtype=np.uint32))
        offset = np.flip(bounding_box.get_integer_origin() - full_bounding_box.get_integer_origin()).astype(np.uint32)

        offset_dict = {
            # Columns are Y, Rows are X;  Offset is ZYX; Bounding Box Shapes are XYZ
            "source_rows": bounding_box.get_shape()[0],
            "target_rows": full_bounding_box.get_shape()[0],
            "offset_columns": offset[1],
            "offset_rows": offset[2],
            "channel_count": np.uint32(volume_info.channels),
        }
        type_ar = np.array([yx_vals.dtype.str, values.dtype.str, weights.dtype.str], dtype='S8')
        durations["split"] = [time.perf_counter() - start]

        start = time.perf_counter()
        # Gets slices off full array corresponding to each Z value.
        z_idx = [0] + list(np.where(z_vals[:-1] != z_vals[1:])[0] + 1) + [len(z_vals)]
        z_stack = z_vals[z_idx[:-1]]
        z_slices = [np.s_[z_idx[i]: z_idx[i + 1]] for i in range(len(z_idx) - 1)]

        durations["stack"] = [time.perf_counter() - start]
        start = time.perf_counter()

        intermediate_directory.mkdir(exist_ok=True, parents=True)

        def make_z(i, z_slice):
            offset_z = z_stack[i] + offset[0]
            z_path = intermediate_directory.joinpath(f"i_{offset_z:05}.dat")
            write_raw_intermediate(
                z_path,
                np.fromiter(offset_dict.values(), dtype=np.uint32, count=5).tobytes(),
                np.uint32(len(yx_vals[z_slice])).tobytes(),
                type_ar.tobytes(),
                np.ascontiguousarray(yx_vals[z_slice]),
                np.ascontiguousarray(values[..., z_slice]),
                np.ascontiguousarray(weights[z_slice]),
            )

        with ThreadPool(intermediate_threads) as pool:
            pool.starmap(make_z, enumerate(z_slices))

        durations["write_intermediate"] = [time.perf_counter() - start]
    except BaseException as be:
        print(f"Error on BP: {be}")
        traceback.print_tb(be.__traceback__, file=sys.stderr)
        raise be

    durations["total_chunk_process"] = [time.perf_counter() - start_total]

    return durations, intermediate_bytes, index, z_stack + offset[0]


def is_big_tiff(shape: DataShape, output_dtype: np.dtype):
    return calculate_gigabytes_from_dimensions(astuple(shape), output_dtype) > 4


def get_output_shape(source: CloudVolumeInterface, slice_mip: int, target_mip: int, write_shape: NGOrder) -> NGOrder:
    if target_mip > slice_mip:
        raise NotImplementedError(("Downsampling during backprojection not yet implemented;"
                                   f" Was requested from {slice_mip} to {target_mip}"))
    resolution_change = NGOrder(*(source.get_resolution_nm(slice_mip) // source.get_resolution_nm(target_mip)), 1)
    shape_change = NGOrder(*(source.get_volume_shape(target_mip) / source.get_volume_shape(slice_mip)), 1)

    # As we slice sequentially, we need a hard multiplier for Z, so we use resolution for that.
    # However, the mips map imperfectly so we use shape to situate properly in X/Y.
    output_shape = write_shape * resolution_change.extract(Z) * shape_change.extract(ImgSlice)
    return replace(output_shape, X=ceil(output_shape.X), Y=ceil(output_shape.Y))


def get_output_origin(source: CloudVolumeInterface, slice_mip: int, target_mip: int, origin: tuple[int]) -> tuple[int]:
    if target_mip > slice_mip:
        raise NotImplementedError(("Downsampling during backprojection not yet implemented;"
                                   f" Was requested from {slice_mip} to {target_mip}"))
    shape_change = source.get_volume_shape(target_mip) / source.get_volume_shape(slice_mip)
    output_origin = origin * shape_change
    return np.ceil(output_origin).astype(np.uint64)

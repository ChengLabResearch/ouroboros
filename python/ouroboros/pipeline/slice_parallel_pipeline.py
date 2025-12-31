from functools import partial
from pathlib import Path
import psutil
import secrets
import sys
import threading
import traceback

from ouroboros.helpers.slice import (
    coordinate_grid,
    slice_volume_from_grids
)
from ouroboros.helpers.mem import SharedNPManager
from ouroboros.helpers.volume_cache import VolumeCache, download_volume
from ouroboros.helpers.files import (
    format_slice_output_file,
    format_slice_output_multiple,
    format_tiff_name,
    join_path,
    num_digits_for_n_files,
    np_convert
)
from .pipeline import PipelineStep
from ouroboros.helpers.mem import SharedNPArray
from ouroboros.helpers.options import SliceOptions
import numpy as np
import concurrent.futures
from tifffile import imwrite, memmap
import multiprocessing
import time


class SliceParallelPipelineStep(PipelineStep):
    def __init__(
        self,
        threads=1,
        processes=multiprocessing.cpu_count(),
        delete_intermediate=False,
    ) -> None:
        super().__init__(inputs=("slice_options", "volume_cache", "slice_rects"))

        self.num_threads = threads
        self.num_processes = processes
        self.delete_intermediate = delete_intermediate

    def with_delete_intermediate(self) -> "SliceParallelPipelineStep":
        self.delete_intermediate = True
        return self

    def with_processes(self, processes: int) -> "SliceParallelPipelineStep":
        self.num_processes = processes
        return self

    def _process(self, input_data: tuple[any]) -> None | str:
        config, volume_cache, slice_rects, pipeline_input = input_data

        # Verify that a config object is provided
        if not isinstance(config, SliceOptions):
            return "Input data must contain a SliceOptions object."

        # Verify that a volume cache is given
        if not isinstance(volume_cache, VolumeCache):
            return "Input data must contain a VolumeCache object."

        # Verify that slice rects is given
        if not isinstance(slice_rects, np.ndarray):
            return "Input data must contain an array of slice rects."

        # Make sure slice rects is not empty
        if len(slice_rects) == 0:
            return "No slice rects were provided."

        # Makes output file folder
        Path(config.output_file_folder).mkdir(parents=True, exist_ok=True)

        # Create a folder with the same name as the output file
        if config.make_single_file:
            output_file_path = join_path(config.output_file_folder, format_slice_output_file(config.output_file_name))
        else:
            output_folder = Path(config.output_file_folder, format_slice_output_multiple(config.output_file_name))
            output_folder.mkdir(parents=True, exist_ok=True)

        temp_file_path = Path(
            config.output_file_folder, f"{config.output_file_name}_temp"
        ).with_suffix(".tif")

        # Start setting up metadata
        # Volume cache resolution is in voxel size, but .tiff XY resolution is in voxels per unit, so we invert.
        resolution = [1.0 / voxel_size for voxel_size in volume_cache.get_resolution_um()[:2] * 0.0001]
        resolutionunit = "CENTIMETER"
        # However, Z Resolution doesn't have an inbuilt property or strong convention, so going with this.
        metadata = {
            "spacing": volume_cache.get_resolution_um()[2],
            "unit": "um"
        }

        # Determine the dimensions of the image
        has_color_channels = volume_cache.has_color_channels()
        num_color_channels = (
            volume_cache.get_num_channels() if has_color_channels else None
        )

        tiff_metadata = {
            "software": "ouroboros",
            "resolution": resolution[:2] + [resolutionunit],     # XY Resolution
            "photometric": ("rgb" if has_color_channels and num_color_channels > 1 else "minisblack"),
            "metadata": metadata
        }

        # Create temporary memmap (single tif file with the same dimensions as the slices)
        temp_shape = (
            slice_rects.shape[0],
            config.slice_width,
            config.slice_height,
        ) + ((int(num_color_channels),) if has_color_channels else ())

        temp_file = memmap(temp_file_path, shape=temp_shape, dtype=np.float32, **tiff_metadata)

        # Calculate the number of digits needed to store the number of slices
        num_digits = num_digits_for_n_files(len(slice_rects))

        # Processing completion marker.
        all_work_done = threading.Event()

        # Minimum and maximum boundaries.
        boundaries = np.zeros(2, dtype=np.float32)

        # Set an SharedMemoryManager key so we can pass it around later.
        authkey = secrets.token_bytes(32)

        # Start the download volumes process and process downloaded volumes as they become available in the queue
        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_processes // 4
            ) as download_executor, concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_processes * 3 // 4
            ) as process_executor, SharedNPManager(authkey=authkey) as (shm_host, ):
                download_futures = []
                process_futures = []
                volume_cache.connect_shm(shm_host.address, authkey)

                vol_range = list(reversed(range(len(volume_cache.volumes))))

                # Process the data in a separate process
                # Note: If the maximum number of processes is reached, this will enqueue the arguments
                # and wait for a process to become available
                partial_slice_executor = partial(
                                process_worker_save_parallel,
                                config=config,
                                slice_rects=slice_rects,
                                temporary_path=temp_file_path,
                                shared=volume_cache.use_shared)

                partial_dl_executor = partial(download_volume,
                                              cv=volume_cache.cv,
                                              mip=volume_cache.mip,
                                              parallel=False,
                                              use_shared=volume_cache.use_shared,
                                              shm_address=shm_host.address,
                                              shm_authkey=authkey)

                def dl_completed(future):
                    volume, bounding_box, download_time, index = future.result()
                    self.add_timing("Download Time", download_time)
                    self.add_timing("Free Memory", psutil.virtual_memory().available)
                    process_futures.append(process_executor.submit(partial_slice_executor,
                                                                   volume=volume,
                                                                   bounding_box=bounding_box,
                                                                   slice_indices=volume_cache.get_slice_indices(index),
                                                                   volume_index=index
                                                                   ))
                    process_futures[-1].add_done_callback(processor_completed)
                    self.update_progress(
                        np.sum([future.done() for future in download_futures]) / (len(volume_cache.volumes) * 4) +
                        np.sum([future.done() for future in process_futures]) / (len(volume_cache.volumes) * 4 // 3)
                    )
                    if volume_cache.use_shared or volume_cache.cache_volume:
                        volume_cache.volumes[index] = volume

                def processor_completed(future):
                    volume_index, durations, min_val, max_val = future.result()
                    boundaries[0] = min(boundaries[0], min_val)
                    boundaries[1] = max(boundaries[1], max_val)
                    self.add_timing("Free Memory", psutil.virtual_memory().available)
                    if volume_cache.use_shared:
                        volume_cache.remove_volume(volume_index, destroy_shared=True)
                    for key, value in durations.items():
                        self.add_timing_list(key, value)

                    self.update_progress(
                        np.sum([future.done() for future in download_futures]) / (len(volume_cache.volumes) * 4) +
                        np.sum([future.done() for future in process_futures]) / (len(volume_cache.volumes) * 4 // 3)
                    )
                    if len(vol_range) > 0:
                        index = vol_range.pop()
                        download_futures.append(
                            download_executor.submit(partial_dl_executor,
                                                     bounding_box=volume_cache.bounding_boxes[index],
                                                     volume_index=index))
                        download_futures[-1].add_done_callback(dl_completed)
                    if self.progress >= 1.0:
                        all_work_done.set()

                # Download all volumes in parallel, and add the callback to process them as they finish.
                for _ in range(self.num_processes * 3 // 4 + 1):
                    index = vol_range.pop()
                    download_futures.append(
                        download_executor.submit(partial_dl_executor,
                                                 bounding_box=volume_cache.bounding_boxes[index],
                                                 volume_index=index))
                    download_futures[-1].add_done_callback(dl_completed)

                all_work_done.wait()

                with multiprocessing.pool.ThreadPool(self.num_processes) as pool:
                    start_time = time.perf_counter()
                    convert_func = partial(np_convert,
                                           target_dtype=volume_cache.get_volume_dtype(),
                                           normalize=config.normalize_output,
                                           zero_guard=config.zeroguard_output,
                                           preset_min=boundaries[0],
                                           preset_max=boundaries[1])
                    if config.make_single_file:
                        target_file = memmap(output_file_path, shape=temp_shape, dtype=volume_cache.get_volume_dtype(),
                                             **tiff_metadata)
                        pool.starmap(memmap_normalized, [(temp_file, target_file, convert_func, i)
                                                         for i in range(len(temp_file))])
                    else:
                        pool.starmap(write_normalized,
                                     [(temp_file,
                                       partial(imwrite, output_folder.joinpath(format_tiff_name(i, num_digits)), **tiff_metadata),  # noqa: E501
                                       convert_func,
                                       i) for i in range(len(temp_file))])
                    del temp_file
                    temp_file_path.unlink()
                    self.add_timing("Rewrite Temp", time.perf_counter() - start_time)
        except BaseException as e:
            traceback.print_tb(e.__traceback__, file=sys.stderr)
            return f"Error downloading data: {e}"

        # Update the pipeline input with the output file path
        if config.make_single_file:
            pipeline_input.output_file_path = output_file_path
        else:
            pipeline_input.output_file_path = str(output_folder)

        return None


def process_worker_save_parallel(
    config: SliceOptions,
    volume: np.ndarray | SharedNPArray,
    bounding_box: np.ndarray,
    slice_indices: np.ndarray,
    volume_index: int,
    slice_rects: np.ndarray,
    temporary_path: str = None,
    shared: bool = False
) -> tuple[int, dict[str, list[float]]]:
    start_total = time.perf_counter()

    if shared:
        volume_data = volume.array()
    else:
        volume_data = volume

    durations = {
        "initial_load": [],
        "generate_grid": [],
        "slice_volume": [],
        "memmap_write": [],
        "total_process": []
    }

    durations["initial_load"].append(time.perf_counter() - start_total)

    # Generate a grid for each slice and stack them along the first axis
    start = time.perf_counter()

    grid_call = partial(coordinate_grid, shape=(config.slice_height, config.slice_width), floor=bounding_box.get_min())
    grids = np.array(list(map(grid_call, slice_rects[slice_indices])))
    durations["generate_grid"].append(time.perf_counter() - start)

    # Slice the volume using the grids
    start = time.perf_counter()
    slices = slice_volume_from_grids(
        volume_data, bounding_box, grids, config.slice_width, config.slice_height
    )
    durations["slice_volume"].append(time.perf_counter() - start)

    start = time.perf_counter()
    # Save the slices to a previously created tiff file
    mmap = memmap(temporary_path)
    mmap[slice_indices] = slices
    mmap.flush()
    del mmap

    durations["memmap_write"].append(time.perf_counter() - start)
    durations["total_process"].append(time.perf_counter() - start_total)

    return volume_index, durations, np.min(slices), np.max(slices)


def memmap_normalized(source: np.memmap, target: np.memmap, convert: callable, index: int):
    target[index] = convert(source=source[index, :])


def write_normalized(source: np.memmap, writer: callable, convert: callable, index: int):
    writer(data=convert(source=source[index]))

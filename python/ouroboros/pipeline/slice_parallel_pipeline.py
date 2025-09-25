from functools import partial
import sys
import threading
import traceback

from ouroboros.helpers.slice import (
    coordinate_grid,
    slice_volume_from_grids
)
from ouroboros.helpers.volume_cache import VolumeCache
from ouroboros.helpers.files import (
    format_slice_output_file,
    format_slice_output_multiple,
    format_tiff_name,
    join_path,
    num_digits_for_n_files,
)
from .pipeline import PipelineStep
from ouroboros.helpers.mem import SharedNPArray
from ouroboros.helpers.options import SliceOptions
import numpy as np
import concurrent.futures
from tifffile import imwrite, memmap
import os
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

        # Create a folder with the same name as the output file
        folder_name = join_path(
            config.output_file_folder,
            format_slice_output_multiple(config.output_file_name),
        )

        if config.make_single_file:
            os.makedirs(folder_name, exist_ok=True)

        output_file_path = join_path(
            config.output_file_folder, format_slice_output_file(config.output_file_name)
        )

        # Create an empty tiff to store the slices
        if config.make_single_file:
            # Make sure slice rects is not empty
            if len(slice_rects) == 0:
                return "No slice rects were provided."

            try:
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

                # Create a single tif file with the same dimensions as the slices
                temp_shape = (
                    slice_rects.shape[0],
                    config.slice_width,
                    config.slice_height,
                ) + ((num_color_channels,) if has_color_channels else ())
                temp_data = np.zeros(temp_shape, dtype=volume_cache.get_volume_dtype())

                imwrite(
                    output_file_path,
                    temp_data,
                    software="ouroboros",
                    resolution=resolution[:2],     # XY Resolution
                    resolutionunit=resolutionunit,
                    photometric=(
                        "rgb"
                        if has_color_channels and num_color_channels > 1
                        else "minisblack"
                    ),
                    metadata=metadata,
                )
            except BaseException as e:
                return f"Error creating single tif file: {e}"

        # Calculate the number of digits needed to store the number of slices
        num_digits = num_digits_for_n_files(len(slice_rects))

        # Processing completion marker.
        all_work_done = threading.Event()

        # Start the download volumes process and process downloaded volumes as they become available in the queue
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(self.num_threads, 4)
            ) as download_executor, concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_processes
            ) as process_executor:
                download_futures = []
                process_futures = []

                vol_range = list(reversed(range(len(volume_cache.volumes))))

                # Process the data in a separate process
                # Note: If the maximum number of processes is reached, this will enqueue the arguments
                # and wait for a process to become available
                partial_slice_executor = partial(
                                process_worker_save_parallel,
                                config=config,
                                folder_name=folder_name,
                                slice_rects=slice_rects,
                                num_threads=self.num_threads,
                                num_digits=num_digits,
                                single_output_path=output_file_path if config.make_single_file else None,
                                shared=volume_cache.use_shared)

                partial_dl_executor = partial(dl_worker,
                                              volume_cache=volume_cache,
                                              parallel_fetch=(self.num_threads == 1))

                def dl_completed(future):
                    process_futures.append(process_executor.submit(partial_slice_executor,
                                                                   processing_data=future.result()))
                    process_futures[-1].add_done_callback(processor_completed)

                def processor_completed(future):
                    volume_index, durations = future.result()
                    if volume_cache.use_shared:
                        volume_cache.remove_volume(volume_index, destroy_shared=True)
                    for key, value in durations.items():
                        self.add_timing_list(key, value)

                    # Update the progress bar
                    # 1/3 DL, 2/3 Process
                    self.update_progress(
                        np.sum([future.done() for future in download_futures]) / (len(volume_cache.volumes) * 3) +
                        np.sum([future.done() for future in process_futures]) / (len(volume_cache.volumes) * 3 / 2)
                    )
                    if len(vol_range) > 0:
                        download_futures.append(download_executor.submit(partial_dl_executor, volume=vol_range.pop()))
                        download_futures[-1].add_done_callback(dl_completed)
                    if self.progress >= 1.0:
                        all_work_done.set()

                # Download all volumes in parallel, and add the callback to process them as they finish.
                for _ in range(np.min([self.num_processes + 4, len(vol_range)])):
                    download_futures.append(download_executor.submit(partial_dl_executor, volume=vol_range.pop()))
                    download_futures[-1].add_done_callback(dl_completed)

                all_work_done.wait()

        except BaseException as e:
            traceback.print_tb(e.__traceback__, file=sys.stderr)
            return f"Error downloading data: {e}"

        # Update the pipeline input with the output file path
        pipeline_input.output_file_path = output_file_path

        return None


def dl_worker(volume_cache: VolumeCache, volume: int, parallel_fetch: bool = False):
    packet = volume_cache.create_processing_data(volume, parallel=parallel_fetch)

    # Remove the volume from the cache after the packet is created
    volume_cache.remove_volume(volume)

    return packet


def process_worker_save_parallel(
    config: SliceOptions,
    folder_name: str,
    processing_data: tuple[np.ndarray | SharedNPArray, np.ndarray, np.ndarray, int],
    slice_rects: np.ndarray,
    num_threads: int,
    num_digits: int,
    single_output_path: str = None,
    shared: bool = False
) -> tuple[int, dict[str, list[float]]]:
    volume, bounding_box, slice_indices, volume_index = processing_data
    if shared:
        volume_data = volume.array()
    else:
        volume_data = volume

    durations = {
        "generate_grid": [],
        "slice_volume": [],
        "save": [],
        "total_process": [],
    }

    start_total = time.perf_counter()

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

    if single_output_path is None:
        # Using a ThreadPoolExecutor within the process for saving slices
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as thread_executor:
            futures = []

            for i, slice_i in zip(slice_indices, slices):
                start = time.perf_counter()
                filename = join_path(folder_name, format_tiff_name(i, num_digits))
                futures.append(thread_executor.submit(save_thread, filename, slice_i))
                durations["save"].append(time.perf_counter() - start)

            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        # Save the slices to a previously created tiff file
        mmap = memmap(single_output_path)
        mmap[slice_indices] = slices
        mmap.flush()
        del mmap

    durations["total_process"].append(time.perf_counter() - start_total)

    return volume_index, durations


def save_thread(filename: str, data: np.ndarray):
    imwrite(filename, data, software="ouroboros")

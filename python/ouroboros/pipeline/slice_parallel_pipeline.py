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
from ouroboros.helpers.options import SliceOptions
import numpy as np
import concurrent.futures
from tifffile import imwrite, memmap
import os
import multiprocessing
import time
from multiprocessing import Queue
from typing import Iterable


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

        # Create a queue to hold downloaded data for processing
        data_queue = multiprocessing.Queue()

        # Start the download volumes process and process downloaded volumes as they become available in the queue
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_threads
            ) as download_executor, concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_processes
            ) as process_executor:
                download_futures = []

                ranges = np.array_split(
                    np.arange(len(volume_cache.volumes)), self.num_threads
                )

                # Download all volumes in parallel
                for volumes_range in ranges:
                    download_futures.append(
                        download_executor.submit(
                            thread_worker_iterative,
                            volume_cache,
                            volumes_range,
                            data_queue,
                            self.num_threads == 1,
                        )
                    )

                processing_futures = []

                # Check if all downloads are done
                def downloads_done():
                    return all([future.done() for future in download_futures])

                # Process downloaded data as it becomes available
                while True:
                    try:
                        data = data_queue.get(timeout=1)

                        # Process the data in a separate process
                        # Note: If the maximum number of processes is reached, this will enqueue the arguments
                        # and wait for a process to become available
                        # TODO: Avoid passing in all of slice rects, rather pass in either a smaller version
                        # or use shared memory
                        processing_futures.append(
                            process_executor.submit(
                                process_worker_save_parallel,
                                config,
                                folder_name,
                                data,
                                slice_rects,
                                self.num_threads,
                                num_digits,
                                single_output_path=(
                                    output_file_path
                                    if config.make_single_file
                                    else None
                                ),
                            )
                        )

                        # Update progress
                        self.update_progress(
                            len(
                                [
                                    future
                                    for future in processing_futures
                                    if future.done()
                                ]
                            )
                            / len(volume_cache.volumes)
                        )
                    except multiprocessing.queues.Empty:
                        if downloads_done() and data_queue.empty():
                            break
                    except BaseException as e:
                        download_executor.shutdown(wait=False, cancel_futures=True)
                        process_executor.shutdown(wait=False, cancel_futures=True)
                        return f"Error processing data: {e}"

                # Track the number of completed futures
                completed = 0
                total_futures = len(processing_futures)

                for future in concurrent.futures.as_completed(processing_futures):
                    _, durations = future.result()
                    for key, value in durations.items():
                        self.add_timing_list(key, value)

                    # Update the progress bar
                    completed += 1
                    self.update_progress(
                        max(completed / total_futures, self.get_progress())
                    )
        except BaseException as e:
            return f"Error downloading data: {e}"

        # Update the pipeline input with the output file path
        pipeline_input.output_file_path = output_file_path

        return None


def thread_worker_iterative(
    volume_cache: VolumeCache, volumes_range: Iterable[int], data_queue: Queue, parallel_fetch: bool = False
):
    for i in volumes_range:
        # Create a packet of data to process - Make the threading check make more sense.
        data = volume_cache.create_processing_data(i, parallel=parallel_fetch)

        data_queue.put(data)

        # Remove the volume from the cache after the packet is created
        # TODO: Change this if the data the data is shared not copied
        volume_cache.remove_volume(i)


def process_worker_save_parallel(
    config: SliceOptions,
    folder_name: str,
    processing_data: tuple[np.ndarray, np.ndarray, np.ndarray, int],
    slice_rects: np.ndarray,
    num_threads: int,
    num_digits: int,
    single_output_path: str = None,
) -> tuple[int, dict[str, list[float]]]:
    volume, bounding_box, slice_indices, volume_index = processing_data

    durations = {
        "generate_grid": [],
        "slice_volume": [],
        "save": [],
        "total_process": [],
    }

    start_total = time.perf_counter()

    # Generate a grid for each slice and stack them along the first axis
    start = time.perf_counter()
    grids = np.array(
        [
            coordinate_grid(
                slice_rects[i], (config.slice_height, config.slice_width)
            )
            for i in slice_indices
        ]
    )
    durations["generate_grid"].append(time.perf_counter() - start)

    # Slice the volume using the grids
    start = time.perf_counter()
    slices = slice_volume_from_grids(
        volume, bounding_box, grids, config.slice_width, config.slice_height
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

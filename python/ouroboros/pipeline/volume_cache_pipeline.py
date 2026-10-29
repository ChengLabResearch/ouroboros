from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np

from ouroboros.helpers.bounding_boxes import calculate_bounding_boxes_bsp_link_rects
from ouroboros.helpers.volume_cache import CloudVolumeInterface, VolumeCache
from ouroboros.helpers.options import SliceOptions
from ouroboros.helpers.slice import SliceTaskPayloadEstimator
from .pipeline import PipelineStep


# Startup-inclusive recorded-path measurements support these conservative
# cutoffs. Other start methods keep serial planning until measured separately.
_MIN_PARALLEL_SLICES = {"fork": 512, "spawn": 8192}


class VolumeCachePipelineStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(inputs=("slice_options", "slice_rects", "source_url"))

    def _process(self, input_data: tuple[any]) -> None | str:
        config, slice_rects, source_url, pipeline_input = input_data

        # Verify that a config object is provided
        if not isinstance(config, SliceOptions):
            return "Input data must contain a SliceOptions object."

        # Verify that slice rects is given
        if not isinstance(slice_rects, np.ndarray):
            return "Input data must contain an array of slice rects."

        cloud_volume_interface = CloudVolumeInterface(source_url)

        task_payload = SliceTaskPayloadEstimator(
            slice_shape=(config.slice_height, config.slice_width),
            source_dtype=cloud_volume_interface.dtype,
            channels=cloud_volume_interface.num_channels,
        )
        context = multiprocessing.get_context()
        cutoff = _MIN_PARALLEL_SLICES.get(context.get_start_method())
        self.execution_resources = {"planner_processes": 0}

        def planner_executor():
            # Invoked only after the tree owner finds a splittable root. This
            # owner supplies the executor allocation and the statistics record.
            self.execution_resources.update(
                planner_processes=min(1, config.process_count),
                planner_start_method=context.get_start_method(),
            )
            return ProcessPoolExecutor(
                max_workers=self.execution_resources["planner_processes"],
                mp_context=context,
            )

        parallel = (
            cutoff is not None and len(slice_rects) >= cutoff
            and config.bounding_box_params.max_depth >= 2
            and config.process_count >= 1
        )

        bounding_boxes, link_rects = calculate_bounding_boxes_bsp_link_rects(
            slice_rects,
            target_slices_per_box=config.bounding_box_params.target_slices_per_box,
            max_depth=config.bounding_box_params.max_depth,
            task_payload=task_payload,
            executor_factory=planner_executor if parallel else None,
        )

        self.update_progress(0.5)

        volume_cache = VolumeCache(
            bounding_boxes,
            link_rects,
            cloud_volume_interface,
            flush_cache=config.flush_cache,
            mip=config.output_mip_level,
            use_shared=True
        )

        # Update the pipeline input with the volume cache
        pipeline_input.volume_cache = volume_cache

        return None

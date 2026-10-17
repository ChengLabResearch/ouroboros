"""One-way JSON adapters for the volumes already owned by a pipeline run.

No pixel data is loaded here. Geometry comes from BoundingBox/VolumeCache and
effective stack dimensions come from the same header inspector as backprojection.
The returned dictionaries are reporting snapshots, never pipeline inputs.
"""

import json
from math import prod
from pathlib import Path

import numpy as np
import tifffile

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.files import get_sorted_tif_files, inspect_straightened_volume
from ouroboros.helpers.volume_cache import download_volume_geometry


def _unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


def _volume(shape, axes, dtype, channels) -> dict:
    dimensions = [int(value) for value in shape]
    dtype = np.dtype(dtype)
    channels = int(channels)
    return {
        "dimensions": dimensions,
        "axis_order": axes,
        "channels": channels,
        "dtype": dtype.name,
        "logical_uncompressed_bytes": prod(dimensions) * channels * dtype.itemsize,
    }


def _source_at_mip(cache, mip) -> dict:
    if mip not in cache.cv.available_mips:
        return _unavailable(f"Source metadata does not provide mip {mip}.")
    result = _volume(
        cache.cv.get_volume_shape(mip), "XYZ", cache.get_volume_dtype(),
        cache.get_num_channels(),
    )
    result.update({
        "mip": int(mip),
        "voxel_spacing_um": [float(value) for value in cache.cv.get_resolution_um(mip)],
    })
    return result


def _tiff_spacing(sample_path) -> dict:
    """Read the writer's existing TIFF tags; do not infer spacing from config."""
    with tifffile.TiffFile(sample_path) as tif:
        page = tif.pages[0]
        tags = page.tags
        unit_tag = tags.get("ResolutionUnit")
        unit = unit_tag.value if unit_tag is not None else None
        unit_um = {tifffile.RESUNIT.CENTIMETER: 10000, tifffile.RESUNIT.INCH: 25400}.get(unit)
        xy = []
        for key in ("XResolution", "YResolution"):
            tag = tags.get(key)
            if unit_um is None or tag is None or tag.value[0] == 0:
                xy.append(None)
            else:
                numerator, denominator = tag.value
                xy.append(float(unit_um * denominator / numerator))
        description = page.description
        metadata = {}
        if description:
            try:
                parsed = json.loads(description)
            except json.JSONDecodeError:
                pass  # Other TIFF writers need not use JSON image descriptions.
            else:
                if isinstance(parsed, dict):
                    metadata = parsed
        z = metadata.get("spacing") if metadata.get("unit") == "um" else None
        if not isinstance(z, (float, int)):
            z = None
        return {
            "values": [*xy, float(z) if z is not None else None],
        }


def _tiff_volume(path, axes, data_kind) -> dict:
    if path is None:
        return _unavailable("No volume path was produced by this run.")
    path = Path(path)
    try:
        info = inspect_straightened_volume(str(path))
        spacing = _tiff_spacing(info.sample_path)
        files = [path.joinpath(name) for name in get_sorted_tif_files(str(path))] if path.is_dir() else [path]
        physical_bytes = sum(file.stat().st_size for file in files)
    except (OSError, ValueError, tifffile.TiffFileError) as error:
        return {**_unavailable(f"Cannot inspect TIFF headers: {error}"), "path": str(path)}

    result = _volume((info.shape.D, info.shape.V, info.shape.U), axes, info.dtype, info.channels)
    result.update({
        "path": str(path),
        "layout": "directory_of_planes" if path.is_dir() else "tiff_stack",
        "data_kind": data_kind,
        "physical_file_bytes": physical_bytes,
        "header_spacing": spacing,
    })
    if axes == "DVU":
        result["header_spacing"]["axis_order"] = "UVD"
    else:
        result["header_spacing"]["axis_order"] = "XYZ"
        if path.is_dir():
            try:
                indices = sorted(int(file.stem) for file in files)
            except ValueError:
                result["plane_index_extent"] = _unavailable("TIFF filenames do not all encode numeric Z indices.")
            else:
                result["plane_index_extent"] = {
                    "minimum": indices[0], "maximum": indices[-1],
                }
    return result


def _bounds(box) -> dict:
    return {"minimum": box.minpt.tolist(), "maximum": box.maxpt.tolist()}


def _geometry(cache, operation) -> dict:
    if not cache.bounding_boxes:
        return _unavailable("The pipeline has not produced bounding boxes.")

    # These are the exact owning methods used by backprojection and downloads.
    envelope = BoundingBox.bound_boxes(cache.bounding_boxes)
    result = _volume(envelope.get_shape(), "XYZ", cache.get_volume_dtype(), cache.get_num_channels())
    result.update({
        "bounds": _bounds(envelope.to_cloudvolume_bbox()),
        "mip": int(cache.get_volume_mip()),
        "voxel_spacing_um": [float(value) for value in cache.get_resolution_um()],
    })

    if operation == "slice":
        allocated_bytes = clipped_bytes = 0
        itemsize = np.dtype(cache.get_volume_dtype()).itemsize
        for box in cache.bounding_boxes:
            _, clipped, allocated = download_volume_geometry(cache.cv, box)
            # Logical source payload per request, including repeated overlap, not RAM usage.
            allocated_bytes += prod(
                int(value) for value in (allocated.X, allocated.Y, allocated.Z, allocated.C)
            ) * itemsize
            clipped_bytes += prod(int(value) for value in clipped.size3()) * int(allocated.C) * itemsize
        result["download_partitions"] = {
            "count": len(cache.bounding_boxes),
            "allocated_logical_bytes": allocated_bytes,
            "source_clipped_logical_bytes": clipped_bytes,
            "clipping_bounds_mip": int(cache.cv.cv.mip),
        }
    return result


def _reduction(straightened, reference, reference_name) -> dict:
    numerator = straightened.get("logical_uncompressed_bytes")
    denominator = reference.get("logical_uncompressed_bytes")
    if numerator is None or denominator is None or denominator <= 0:
        return _unavailable("Logical straightened bytes and a positive reference byte count are required.")
    return {
        "percentage": 100 * (1 - numerator / denominator),
        "numerator_bytes": numerator,
        "denominator_bytes": denominator,
        "reference": reference_name,
    }


def collect_volume_statistics(input_data, operation: str) -> dict:
    """Serialize effective native volume metadata without changing pipeline state."""
    if operation not in ("slice", "backproject"):
        raise ValueError(f"Unknown operation: {operation}")
    if input_data is None:
        return _unavailable("The pipeline has not been initialized.")
    cache = input_data.volume_cache
    if cache is None:
        full_source = used_source = _unavailable("The pipeline has not loaded source metadata.")
        geometry = _unavailable("The pipeline has not generated a volume cache.")
        source_kind = "unknown"
    else:
        full_source = _source_at_mip(cache, 0)
        used_source = _source_at_mip(cache, cache.get_volume_mip())
        geometry = _geometry(cache, operation)
        kind = cache.cv.cv.info.get("type")
        source_kind = kind if kind in ("image", "segmentation") else "unknown"

    if operation == "backproject":
        config = input_data.backproject_options
        straightened = _tiff_volume(config.straightened_volume_path if config else None, "DVU", "unknown")
        output_path = input_data.output_file_path
        if output_path is not None and not Path(output_path).is_absolute() and config is not None:
            output_path = str(Path(config.output_file_folder, output_path))
        output = _tiff_volume(output_path, "ZYX", "unknown")
        output["final_nonzero_extent"] = None
        if config is not None:
            output["requested_mip"] = config.output_mip_level
    else:
        straightened = _tiff_volume(input_data.output_file_path, "DVU", source_kind)

    return {
        "source": {"url": input_data.source_url, "data_kind": source_kind,
                   "full_resolution": full_source, "used_resolution": used_source},
        "straightened": straightened,
        "bounding_box": geometry,
        **({"backprojected_output": output} if operation == "backproject" else {}),
        "storage_reduction": {
            "source_full_resolution": _reduction(
                straightened, full_source, "source.full_resolution.logical_uncompressed_bytes"
            ),
            "source_used_resolution": _reduction(
                straightened, used_source, "source.used_resolution.logical_uncompressed_bytes"
            ),
            "bounding_box_relative": _reduction(straightened, geometry, "bounding_box.logical_uncompressed_bytes"),
        },
    }

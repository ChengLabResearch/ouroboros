from functools import partial
from multiprocessing.pool import ThreadPool
import os

import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import cv2
from tifffile import TiffWriter, TiffFile
import time

from .shapes import DataShape


def get_sorted_tif_files(directory: str) -> list[str]:
    """
    Get all .tif files in a directory and sort them numerically.

    Assumes that the files are named with a number at the beginning of the file name.
    E.g. 0001.tif, 0002.tif, 0003.tif, etc.

    Parameters
    ----------
    directory : str
        The directory to search for .tif files.

    Returns
    -------
    list[str]
        The sorted list of .tif files in the directory.
    """

    # Get all files in the directory
    files = os.listdir(directory)

    # Filter to include only .tif files and sort them numerically
    tif_files = sorted((file for file in files if file.endswith((".tif", ".tiff"))))

    return tif_files


def join_path(*args) -> str:
    return str(Path(*args))


def combine_unknown_folder(directory_path: str, filename: str) -> str:
    """
    Combine a directory path and a filename into a single path.

    Automatically determines the correct path separator to use based on the directory path.

    Parameters
    ----------
    directory_path : str
        The directory path.
    filename : str
        The filename.

    Returns
    -------
    str
        The combined path.
    """

    if not directory_path.endswith("/") and not directory_path.endswith("\\"):
        # Attempt to determine the correct path separator
        if "/" in directory_path:
            directory_path += "/"
        else:
            directory_path += "\\"

    return directory_path + filename


def format_slice_output_file(output_name: str) -> str:
    return output_name + ".tif"


def format_slice_output_multiple(output_name: str) -> str:
    return output_name + "-slices"


def format_slice_output_config_file(output_name: str) -> str:
    return output_name + "-configuration.json"


def format_backproject_output_file(output_name: str, offset: tuple[int] | None = None) -> str:
    if offset is not None:
        offset_str = "-".join(map(str, offset))
        return output_name + f"-backprojected-{offset_str}.tif"

    return output_name + "-backprojected.tif"


def format_backproject_output_multiple(
    output_name: str, offset: tuple[int] | None = None
) -> str:
    if offset is not None:
        offset_str = "-".join(map(str, offset))
        return output_name + f"-backprojected-{offset_str}"

    return output_name + "-backprojected"


def format_backproject_tempvolumes(output_name: str) -> str:
    return output_name + "-tempvolumes"


def format_backproject_resave_volume(output_name: str) -> str:
    return output_name + "-temp-straightened.tif"


def format_tiff_name(i: int, num_digits: int) -> str:
    return f"{str(i).zfill(num_digits)}.tif"


def parse_tiff_name(tiff_name: str) -> int:
    return int(tiff_name.split(".")[0])


def num_digits_for_n_files(n: int) -> int:
    return len(str(n - 1))


def np_convert(target_dtype: np.dtype, source: ArrayLike, normalize=True, safe_bool=False):
    """ TODO: Fix for Negative Values """
    if safe_bool and target_dtype == bool:
        return source.astype(target_dtype).astype(np.uint8)
    elif np.issubdtype(target_dtype, np.integer) and normalize:
        dtype_range = np.iinfo(target_dtype).max - np.iinfo(target_dtype).min
        source_range = np.max(source) - np.min(source)

        # Avoid divide by 0, esp. as numpy segfaults when you do.
        if source_range == 0.0:
            source_range = 1.0

        return (source * max(int(dtype_range / source_range), 1)).astype(target_dtype)
    elif np.issubdtype(target_dtype, np.floating) and normalize:
        source_range = np.max(source) - np.min(source)

        # Avoid divide by 0, esp. as numpy segfaults when you do.
        if source_range == 0.0:
            source_range = 1.0

        return (source / source_range).astype(target_dtype)
    else:
        return source.astype(target_dtype)


def generate_tiff_write(write_func: callable, compression: str | None, micron_resolution: np.ndarray[float],
                        backprojection_offset: np.ndarray, **kwargs):
    # Volume cache resolution is in voxel size, but .tiff XY resolution is in voxels per unit, so we invert.
    resolution = [1.0 / voxel_size for voxel_size in micron_resolution[:2] * 0.0001]
    resolutionunit = "CENTIMETER"
    # However, Z Resolution doesn't have an inbuilt property or strong convention, so going with this atm.
    metadata = {
        "spacing": micron_resolution[2],
        "unit": "um"
    }

    if backprojection_offset is not None:
        metadata["backprojection_offset_min_xyz"] = backprojection_offset

    return partial(write_func,
                   contiguous=compression is None or compression == "none",
                   compression=compression,
                   metadata=metadata,
                   resolution=resolution,
                   resolutionunit=resolutionunit,
                   software="ouroboros",
                   **kwargs)


def write_small_intermediate(file_path: os.PathLike, *series):
    with TiffWriter(file_path, append=True) as tif:
        for entry in series:
            tif.write(entry, dtype=entry.dtype)


def ravel_map_2d(index, source_rows, target_rows, offset):
    return np.add.reduce(np.add(np.divmod(index, source_rows), offset) * ((target_rows, ), (np.uint32(1), )))


def load_z_intermediate(path: Path, offset: int = 0):
    with TiffFile(path) as tif:
        meta = tif.series[offset].asarray()
        source_rows, target_rows, offset_rows, offset_columns = meta
        return (ravel_map_2d(tif.series[offset + 1].asarray(),
                             source_rows, target_rows,
                             ((offset_rows, ), (offset_columns, ))),
                tif.series[offset + 2].asarray(),
                tif.series[offset + 3].asarray())


def increment_volume(path: Path, vol: np.ndarray, offset: int = 0, cleanup=False):
    indicies, values, weights = load_z_intermediate(path, offset)
    for i in range(0, vol.shape[0] - 1):
        np.add.at(vol[i], indicies, np.atleast_2d(values)[i])
    np.add.at(vol[-1], indicies, weights)

    if cleanup:
        path.unlink()


def volume_from_intermediates(path: Path, shape: DataShape, thread_count: int = 4):
    vol = np.zeros((1 + shape.C, np.prod((shape.Y, shape.X))), dtype=np.float32)
    with ThreadPool(thread_count) as pool:
        if not path.exists():
            # We don't have any intermediate(s) for this value, so return empty.
            return vol[0]
        elif path.is_dir():
            pool.starmap(increment_volume, [(i, vol, 0, False) for i in path.glob("**/*.tif*")])
        else:
            with TiffFile(path) as tif:
                offset_set = range(0, len(tif.series), 4)
            pool.starmap(increment_volume, [(path, vol, i, False) for i in offset_set])

    nz = np.flatnonzero(vol[-1])
    vol[:-1, nz] /= vol[-1, nz]
    return vol[:-1]


def write_conv_vol(writer: callable, source_path, shape, dtype, scaling, target_folder, index, interpolation):
    perf = {}
    start = time.perf_counter()
    vol = volume_from_intermediates(source_path, shape)
    perf["Merge Volume"] = time.perf_counter() - start
    if scaling is not None:
        start = time.perf_counter()
        # CV2 is only 2D but we're resizing from the 1D image anyway at the moment.
        new_volume = cv2.resize(
                        np_convert(dtype, vol.T.reshape(shape.Y, shape.X, shape.C), normalize=False, safe_bool=True),
                        None, fx=scaling[1], fy=scaling[2], interpolation=interpolation)
        perf["Zoom"] = time.perf_counter() - start
        start = time.perf_counter()
        for i in range(round(scaling[0])):
            writer(target_folder.joinpath(f"{(index * round(scaling[0]) + i):05}.tif"), data=new_volume)
        perf["Write Merged"] = time.perf_counter() - start
    else:
        start = time.perf_counter()
        writer(target_folder.joinpath(f"{index}.tif"),
               data=np_convert(dtype, vol.T.reshape(shape.Y, shape.X, shape.C), normalize=False, safe_bool=True))
        perf["Write Merged"] = time.perf_counter() - start
    return perf

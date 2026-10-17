import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
from ouroboros.common.volume_server_interface import (
    get_volume_path,
    clear_plugin_folder,
    copy_to_host,
    copy_to_volume,
)
from ouroboros.helpers.files import (
    combine_unknown_folder,
    format_backproject_output_file,
    format_backproject_output_multiple,
    format_slice_output_file,
    format_slice_output_multiple,
)
from ouroboros.helpers.options import BackprojectOptions, SliceOptions


STATISTICS_FOLDER = "main-statistics"


def write_statistics_file(path: Path, formatted_statistics: str) -> None:
    """Publish a complete JSON document; never leave a truncated final report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(formatted_statistics)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_statistics(formatted_statistics: str, filename: str, output_folder: str,
                       docker: bool = False) -> dict:
    """Use the output transfer seam, retaining undelivered reports outside scientific scratch."""
    local_path = Path(get_volume_path(STATISTICS_FOLDER) if docker else output_folder, filename)
    destination = combine_unknown_folder(output_folder, filename) if docker else str(local_path)
    delivery = {"status": "failed", "path": destination, "recovery_path": None, "error": None,
                "presentation_error": None, "presentation_interrupted": False}
    try:
        write_statistics_file(local_path, formatted_statistics)
        if docker:
            delivery["recovery_path"] = str(local_path)
            success, error = copy_to_host(
                [{"sourcePath": destination, "targetPath": "./"}],
                plugin_folder_name=STATISTICS_FOLDER,
            )
            if not success:
                delivery["error"] = error or "Statistics transfer failed."
                return delivery
            local_path.unlink()
        delivery.update(status="published", recovery_path=None)
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            delivery["status"] = "interrupted"
        delivery["error"] = str(error) or type(error).__name__
        if local_path.exists():
            delivery["recovery_path"] = str(local_path)
    return delivery


def get_path_name(path: str):
    return Path(path.replace("\\", os.sep)).name


def load_options_for_backproject(options_path: str) -> BackprojectOptions | str:
    """
    Loads the options for backprojecting a volume.

    Parameters
    ----------
    options_path : str
        The path to the options file.

    Returns
    -------
    BackprojectOptions | Exception
        The options for backprojecting the volume, or a string of the exception if it could not be loaded.
    """

    options = BackprojectOptions.load_from_json(options_path)

    return options


def load_options_for_backproject_docker(
    options_path: str, target_path: str = "./", *, on_options: Callable | None = None
) -> tuple[BackprojectOptions, SliceOptions, str, str | None, str] | str:
    """
    Loads the options for backprojecting a volume and copies the necessary files to the docker volume.

    Parameters
    ----------
    options_path : str
        The path to the options file.
    target_path : str, optional
        The path to the target folder in the docker volume, by default "./"

    Returns
    -------
    tuple[BackprojectOptions, SliceOptions, str, str | None, str] | str
        The options for backprojecting the volume, the options for slicing the volume, the host path to the output file,
        the host path to the slices folder if the output is not a single file, and the host output folder.
    """

    # Copy the file to the docker volume
    files = [
        {"sourcePath": options_path, "targetPath": target_path},
    ]
    success, error = copy_to_volume(files)

    if not success:
        return error

    # Define the path to the copied file in the docker volume
    options_path = get_volume_path() + get_path_name(options_path)
    options = load_options_for_backproject(options_path)

    if isinstance(options, str):
        return options

    if on_options is not None:
        on_options(options)

    # Copy the straightened volume and config files to the docker volume
    files = [
        {"sourcePath": options.straightened_volume_path, "targetPath": target_path},
    ]

    success, error = copy_to_volume(files)

    if not success:
        return error

    # Define the output file paths
    host_output_folder = options.output_file_folder
    host_output_file = combine_unknown_folder(
        host_output_folder, format_backproject_output_file(options.output_file_name)
    )
    host_output_slices = (
        combine_unknown_folder(
            host_output_folder,
            format_backproject_output_multiple(options.output_file_name),
        )
        if options.make_single_file is False
        else None
    )

    # Define the path to the copied straightened volume and config files in the docker volume
    options.straightened_volume_path = get_volume_path() + get_path_name(
        options.straightened_volume_path
    )

    # Modify the output file folder to be in the docker volume
    options.output_file_folder = get_volume_path()

    # Load the options for slicing the volume,
    # which contains necessary information for backprojecting the volume
    slice_load_result = load_options_for_slice_docker(
        options.slice_options_path, target_path, on_options=on_options
    )

    if isinstance(slice_load_result, str):
        return slice_load_result

    slice_options = slice_load_result[0]

    return (
        options,
        slice_options,
        host_output_file,
        host_output_slices,
        host_output_folder,
    )


def save_output_for_backproject_docker(
    host_output_file: str,
    host_output_slices: str = None,
    target_path: str = "./",
    *, cleanup: bool = True,
) -> None | str:
    """
    Saves the output files for backprojecting a volume to the host.

    Parameters
    ----------
    host_output_file : str
        The path to the output file on the host.
    host_output_slices : str, optional
        The path to the slices folder on the host, by default None
    target_path : str, optional
        The path to the target folder in the docker volume, by default "./"

    Returns
    -------
    None | str
        None if the operation was successful, an error message otherwise.
    """

    # Copy the output files to the host
    files = [
        {
            "sourcePath": (
                host_output_slices
                if host_output_slices is not None
                else host_output_file
            ),
            "targetPath": target_path,
        }
    ]

    success, error = copy_to_host(files)

    if not success:
        return error

    # Clear the plugin folder
    if cleanup:
        success, error = clear_plugin_folder()
        if not success:
            return error or "Output cleanup failed."


def load_options_for_slice(options_path: str) -> SliceOptions | str:
    """
    Loads the options for slicing a volume.

    Parameters
    ----------
    options_path : str
        The path to the options file.

    Returns
    -------
    SliceOptions | str
        The options for slicing the volume, or a string of the exception if the options could not be loaded.
    """

    slice_options = SliceOptions.load_from_json(options_path)

    return slice_options


def load_options_for_slice_docker(
    options_path: str, target_path: str = "./", *, on_options: Callable | None = None
) -> tuple[SliceOptions, str, str | None] | str:
    """
    Loads the options for slicing a volume and copies the necessary files to the docker volume.

    Parameters
    ----------
    options_path : str
        The path to the options file.
    target_path : str, optional
        The path to the target folder in the docker volume, by default "./"

    Returns
    -------
    tuple[SliceOptions, str, str | None] | str
        The options for slicing the volume, the host path to the output file, and the host path to output slices.
    """

    # Copy the file to the docker volume
    files = [{"sourcePath": options_path, "targetPath": target_path}]
    success, error = copy_to_volume(files)

    if not success:
        return error

    # Define the path to the copied file in the docker volume
    options_path = get_volume_path() + get_path_name(options_path)

    slice_options = load_options_for_slice(options_path)

    if isinstance(slice_options, str):
        return slice_options

    if on_options is not None:
        on_options(slice_options)

    host_output_folder = slice_options.output_file_folder
    host_output_file = combine_unknown_folder(
        host_output_folder, format_slice_output_file(slice_options.output_file_name)
    )
    host_output_slices = (
        combine_unknown_folder(
            host_output_folder,
            format_slice_output_multiple(slice_options.output_file_name),
        )
        if slice_options.make_single_file is False
        else None
    )

    # Modify the output file folder to be in the docker volume
    slice_options.output_file_folder = get_volume_path()

    # Copy the neuroglancer json file to the docker volume
    files = [
        {
            "sourcePath": slice_options.neuroglancer_json,
            "targetPath": target_path,
        }
    ]

    success, error = copy_to_volume(files)

    if not success:
        return error

    # Define the path to the copied neuroglancer json file in the docker volume
    slice_options.neuroglancer_json = get_volume_path() + get_path_name(
        slice_options.neuroglancer_json
    )

    return slice_options, host_output_file, host_output_slices


def save_output_for_slice_docker(
    host_output_file: str,
    host_output_slices: str = None,
    target_path: str = "./",
    *, cleanup: bool = True,
) -> None | str:
    """
    Saves the output files for slicing a volume to the host.

    Parameters
    ----------
    host_output_file : str
        The path to the output file on the host.
    host_output_slices : str, optional
        The path to the slices folder on the host, by default None
    target_path : str, optional
        The path to the target folder in the docker volume, by default "./"

    Returns
    -------
    None | str
        None if the operation was successful, an error message otherwise.
    """

    # Copy the output files to the host
    files = [
        {
            "sourcePath": (
                host_output_slices
                if host_output_slices is not None
                else host_output_file
            ),
            "targetPath": target_path,
        },
    ]
    success, error = copy_to_host(files)

    if not success:
        return error

    # Clear the plugin folder
    if cleanup:
        success, error = clear_plugin_folder()
        if not success:
            return error or "Output cleanup failed."

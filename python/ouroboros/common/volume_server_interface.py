import requests

VOLUME_SERVER_URL = "http://host.docker.internal:3001"
PLUGIN_NAME = "main"


def get_volume_path() -> str:
    """
    Get the path to the main server's volume folder.
    """
    return f"/volume/{PLUGIN_NAME}/"


def copy_to_volume(files: list[dict]) -> tuple[bool, str]:
    """
    Copy host files to the main server's volume folder.

    Parameters
    ----------
    files : list[dict]
        A list of dictionaries containing the source and target paths for the files to copy.

    Returns
    -------
    tuple[bool, str]
        A tuple containing a boolean indicating if the operation was successful and an error message if it
        was not.
    """

    data = {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": "main",
        "files": files,
    }
    print(f"data to volume: {data}")

    return request_volume_server("copy-to-volume", data)


def copy_to_host(files: list[dict]) -> tuple[bool, str]:
    """
    Copy files from the main server's volume folder to the host.

    Parameters
    ----------
    files : list[dict]
        A list of dictionaries containing the source and target paths for the files to copy.

    Returns
    -------
    tuple[bool, str]
        A tuple containing a boolean indicating if the operation was successful and an error message if it
        was not.
    """

    data = {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": "main",
        "files": files,
    }
    print(f"data to host: {data}")

    return request_volume_server("copy-to-host", data)


def clear_plugin_folder() -> tuple[bool, str]:
    """
    Clear the main server's plugin folder.

    Returns
    -------
    tuple[bool, str]
        A tuple containing a boolean indicating if the operation was successful and an error message if it
        was not.
    """

    data = {
        "volumeName": "ouroboros-volume",
        "pluginFolderName": "main",
    }

    return request_volume_server("clear-plugin-folder", data)


def request_volume_server(path: str, data: dict) -> tuple[bool, str]:
    """
    Send a request to the main server's volume server.

    Parameters
    ----------
    path : str
        The path to the endpoint on the volume server.
    data : dict
        The data to send in the request.

    Returns
    -------
    tuple[bool, str]
        A tuple containing a boolean indicating if the operation was successful and an error message if it
        was not.
    """

    url = f"{VOLUME_SERVER_URL}/{path}"
    try:
        result = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=data,
        )
        print(f"result: {result.text}")
        if not result.ok:
            return False, result.text
        else:
            return True, ""
    except Exception as error:
        print(f"request error: {error}")
        return False, str(error)

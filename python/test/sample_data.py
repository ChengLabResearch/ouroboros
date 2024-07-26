import json
from pathlib import Path
import numpy as np


def generate_sample_curve_helix(
    start_z=-10, end_z=10, radius=1, num_points=100
) -> np.ndarray:
    """
    Generates a sample helix curve for testing purposes.

    Parameters
    ----------
    start_z : float
        The starting z value.
    end_z : float
        The ending z value.
    radius : float
        The radius of the helix.
    num_points : int
        The number of points to generate.

    Returns
    -------
    np.ndarray
        The sample helix curve (num_points, 3).
    """
    t = np.linspace(start_z, end_z, num_points)
    x = np.cos(t) * radius
    y = np.sin(t) * radius
    z = t

    return np.vstack((x, y, z)).T


def generate_sample_neuroglancer_json(
    tmp_folder: Path, file_name: str = "sample_data.json"
) -> str:
    """
    Generates a sample neuroglancer JSON file for testing purposes.

    Parameters
    ----------
    tmp_folder : Path
        The temporary folder to save the file.
    file_name : str, optional
        The name of the file.

    Returns
    -------
    str
        The path to the generated file.
    """

    sample_curve = generate_sample_curve_helix()

    points = [{"point": point.tolist(), "type": "point"} for point in sample_curve]

    data = {
        "layers": [
            {
                "type": "annotation",
                "name": "annotations",
                "annotations": points,
            },
            {
                "type": "image",
                "source": "precomputed://http://sourcewebsite.com/image",
                "name": "image_layer",
            },
            {
                "type": "image",
                "source": {"url": "precomputed://http://sourcewebsite.com/image2"},
                "name": "other_image_layer",
            },
        ]
    }

    path = tmp_folder / file_name

    path.write_text(json.dumps(data), encoding="utf-8")

    return str(path)

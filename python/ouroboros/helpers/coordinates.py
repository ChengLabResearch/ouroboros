import numpy as np


def convert_axes(data: np.ndarray, current_axes: str, target_axes: str) -> np.ndarray:
    """
    Convert the axes of a numpy array from one format to another.

    Parameters
    ----------
        data : np.ndarray
            The data to convert.
        current_axes : str
            The current axes format (e.g. XYZ, XYZC).
        target_axes : str
            The target axes format.

    Returns
    -------
        np.ndarray
            The data with the axes converted.
    """

    # Ensure that the axes are valid
    if not set(current_axes) == set(target_axes):
        raise ValueError(
            "The current and target axes must contain the same characters."
        )

    axis_to_index = {axis: i for i, axis in enumerate(current_axes)}

    # Create a list of the new axis order
    new_order = [axis_to_index[ax] for ax in target_axes]

    # Transpose the data to the new order
    return data.transpose(new_order)


def convert_points_between_volumes(
    points: np.ndarray, source_shape: tuple, target_shape: tuple
) -> np.ndarray:
    """
    Convert points from one volume shape to another.

    Parameters
    ----------
        points : np.ndarray
            The points to convert.
        source_shape : tuple
            The shape of the volume that the points are currently in.
        target_shape : tuple
            The shape of the volume that the points will be converted to.

    Returns
    -------
        np.ndarray
            The points converted into the space of the target shape.
    """

    # Calculate the scaling factors for each axis
    scale_factors = np.array(source_shape) / np.array(target_shape)

    # Scale the points
    scaled_points = points / scale_factors

    return scaled_points

import numpy as np

from scipy.ndimage import map_coordinates

from cloudvolume import VolumeCutout
from .bounding_boxes import BoundingBox

from .spline import Spline

def calculate_slice_rects(times: np.ndarray, spline: Spline, width, height, spline_points=None) -> np.ndarray:
    """
    Calculate the slice rectangles for a spline at a set of time points.

    Parameters:
    ----------
        times (numpy.ndarray): The time points at which to calculate the slice rectangles.
        spline (Spline): The spline object.
        width (float): The width of the slice rectangles.
        height (float): The height of the slice rectangles.
        spline_points (numpy.ndarray): The points on the spline at the given time points (3, n).

    Returns:
    -------
        numpy.ndarray: The slice rectangles at the given time points (n, 4, 3).
    """

    # Calculate the tangent, normal, and binormal vectors
    tangent_vectors, normal_vectors, binormal_vectors = spline.calculate_rotation_minimizing_vectors(times)

    # Transpose the vectors for vectpr-by-vector indexing (3, n) -> (n, 3)
    tangent_vectors = tangent_vectors.T
    normal_vectors = normal_vectors.T
    binormal_vectors = binormal_vectors.T

    if spline_points is None:
        spline_points = spline(times)

    # (3, n) -> (n, 3)
    spline_points = spline_points.T

    rects = []

    for i in range(len(times)):
        point = spline_points[i]

        localx = normal_vectors[i]
        localy = binormal_vectors[i]

        width_vec = localx * width
        height_vec = localy * height

        top_left = point - width_vec + height_vec
        top_right = point + width_vec + height_vec
        bottom_right = point + width_vec - height_vec
        bottom_left = point - width_vec - height_vec

        rects.append(np.array([top_left, top_right, bottom_right, bottom_left]))

    # Output the rects in the form (n, 4, 3)
    return np.array(rects)

def generate_coordinate_grid_for_rect(rect: np.ndarray, width, height) -> np.ndarray:
    """
    Generate a coordinate grid for a rectangle.

    Parameters:
    ----------
        rect (numpy.ndarray): The corners of the rectangle as a list of 3D coordinates.
        width (int): The width of the grid.
        height (int): The height of the grid.

    Returns:
    -------
        numpy.ndarray: The grid of coordinates (width, height, 3).
    """

    top_left, top_right, bottom_right, bottom_left = rect

    # Generate a grid of (u, v) coordinates
    u = np.linspace(0, 1, width)
    v = np.linspace(0, 1, height)
    u, v = np.meshgrid(u, v) # TODO: need 'ij'?

    # Interpolate the 3D coordinates
    points = (1 - u)[:, :, np.newaxis] * (1 - v)[:, :, np.newaxis] * top_left + \
            u[:, :, np.newaxis] * (1 - v)[:, :, np.newaxis] * top_right + \
            (1 - u)[:, :, np.newaxis] * v[:, :, np.newaxis] * bottom_left + \
            u[:, :, np.newaxis] * v[:, :, np.newaxis] * bottom_right

    return points

# TODO: When slicing volume, need to know offset of slice in volume
# TODO: Consider if i and j need to be swapped because of map_coordinates behavior

def slice_volume_from_grid(volume: VolumeCutout, bounding_box: BoundingBox, grid: np.ndarray, width, height) -> np.ndarray:
    """
    Slice a volume based on a grid of coordinates.

    Parameters:
    ----------
        volume (VolumeCutout): The volume of shape (x, y, z, 1) to slice.
        bounding_box (BoundingBox): The bounding box of the volume.
        grid (numpy.ndarray): The grid of coordinates to slice the volume.
        width (int): The width of the grid.
        height (int): The height of the grid.

    Returns:
    -------
        numpy.ndarray: The slice of the volume as a 2D array.
    """

    # Remove the last dimension from the volume
    # TODO: Figure out how to modify this to support multiple channels (maybe add an axis to points?)
    # Include support for choosing tiff color mode based on this
    squeezed_volume = np.squeeze(volume, axis=-1)

    # Normalize grid coordinates based on bounding box (since volume coordinates are truncated)
    bounding_box_min = np.array([bounding_box.x_min, bounding_box.y_min, bounding_box.z_min])

    # Subtract the bounding box min from the grid (width, height, 3)
    normalized_grid = grid - bounding_box_min

    # Reshape the grid to be (3, width * height)
    normalized_grid = normalized_grid.reshape(-1, 3).T

    # Map the grid coordinates to the volume
    slice_points = map_coordinates(squeezed_volume, normalized_grid, mode='nearest')

    return slice_points.reshape(height, width)
    
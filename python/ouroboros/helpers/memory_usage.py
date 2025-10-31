import numpy as np

GIGABYTE = 1024**3


def calculate_gigabytes_from_dimensions(shape: tuple[int], dtype: np.dtype) -> float:
    """
    Calculate the number of gigabytes in an array with the given shape and data type.

    Parameters:
    ----------
        shape (tuple): The shape of the array.
        dtype (numpy.dtype): The data type of the array.

    Returns:
    -------
        float: The number of gigabytes in the array.
    """

    # Get the size of the data type in bytes
    dtype_size = np.dtype(dtype).itemsize

    # Calculate the total number of elements in the array
    num_elements = np.prod(shape)

    # Calculate the total number of bytes
    num_bytes = np.multiply(num_elements, dtype_size, dtype=np.uint64, casting='unsafe')

    return num_bytes / GIGABYTE

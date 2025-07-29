import numpy as np

GIGABYTE = 1024**3


def calculate_gigabytes_in_array(array: np.ndarray) -> float:
    """
    Calculate the number of gigabytes in a numpy array.

    Parameters:
    ----------
        array (numpy.ndarray): The numpy array.

    Returns:
    -------
        float: The number of gigabytes in the numpy array.
    """

    return array.nbytes / GIGABYTE


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


def calculate_chunk_size(shape: tuple, dtype: np.dtype, max_ram_gb: int = 0, axis=0) -> int:
    """
    Calculate the chunk size based on the shape and dtype of the volume.

    Parameters
    ----------
    shape : tuple
        The shape of the volume.
    dtype : numpy.dtype
        The dtype of the volume.
    max_ram_gb : int
        The maximum amount of RAM to use in GB.
    axis : int, optional
        The axis along which to calculate the chunk size.
        The default is 0.

    Returns
    -------
    int
        The chunk size.
    """
    # Calculate the memory usage of the volume
    total_gb = calculate_gigabytes_from_dimensions(shape, dtype)

    # Determine the length of the axis
    axis_length = shape[axis]

    # Calculate the memory usage along the axis
    axis_gb = total_gb / axis_length

    # Calculate the chunk size
    # Note: The max function is used to ensure that the chunk size is at least 1
    return max(int(max_ram_gb / axis_gb), 1)

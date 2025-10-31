import numpy as np

from ouroboros.helpers.memory_usage import calculate_gigabytes_from_dimensions


def test_calculate_gigabytes_from_dimensions():
    # Define the shape and data type
    shape = (1024, 1024, 1024)
    dtype = np.float64

    # Calculate the expected result
    expected_result = 8 * 1

    # Call the function and check the result
    assert calculate_gigabytes_from_dimensions(shape, dtype) == expected_result

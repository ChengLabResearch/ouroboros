from ouroboros.helpers.coordinates import convert_axes, convert_points_between_volumes

import numpy as np


def test_convert_axes():
    test_data = np.random.randint(0, 255, (400, 256, 4), "uint8")

    source = "XYC"
    target = "YXC"

    # Calculate the expected result
    expected_result = test_data.transpose((1, 0, 2)).shape

    # Call the function and check the result
    assert convert_axes(test_data, source, target).shape == expected_result


def test_convert_axes_invalid_axes():
    test_data = np.random.randint(0, 255, (400, 256, 4), "uint8")

    source = "XYC"
    target = "YXZ"

    # Call the function and check the result
    try:
        convert_axes(test_data, source, target)
    except ValueError as e:
        assert str(e) == "The current and target axes must contain the same characters."


def test_convert_axes_truncate():
    test_data = np.random.randint(0, 255, (256, 4), "uint8")

    source = "XYC"
    target = "YXC"

    # Call the function and check the result
    try:
        convert_axes(test_data, source, target)
    except ValueError as e:
        assert (
            str(e)
            == "The data must have the same number of dimensions as the current axes."
        )


def test_convert_points_between_volumes():
    test_data = np.random.randint(0, 255, (100, 3), "uint8")

    source_shape = (1000, 1000, 1000)
    target_shape = (500, 500, 500)

    # Calculate the expected result
    expected_result = test_data / np.array((2, 2, 2))

    # Call the function and check the result
    assert np.array_equal(
        convert_points_between_volumes(test_data, source_shape, target_shape),
        expected_result,
    )

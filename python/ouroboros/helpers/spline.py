import numpy as np
from scipy.interpolate import splprep, splev


class Spline:
    def __init__(self, sample_points: np.ndarray, degree: int = 3) -> None:
        # Guarantee that the degree is at least 2
        degree = max(degree, 2)

        self.tck, self.u = self.fit_spline(sample_points, degree=degree)

    def __call__(self, times: np.ndarray, derivative: int = 0):
        return self.evaluate_spline(self.tck, times, derivative=derivative)

    @staticmethod
    def fit_spline(sample_points: np.ndarray, degree: int = 3):
        """
        Fit a B-spline to a set of sample points.

        Parameters:
        ----------
            sample_points (numpy.ndarray): A 2D array of shape (n, 3) containing the sample points.
            degree (int): The degree of the B-spline.

        Returns:
        -------
            tuple: A tuple containing the knot vector and the coefficients of the B-spline.
        """

        x, y, z = sample_points.T

        # Fit a B-spline to the sample points
        # t = knots, c = coefficients, k = degree
        tck, u = splprep([x, y, z], k=degree)

        return tck, u

    @staticmethod
    def evaluate_spline(tck: tuple, times: np.ndarray, derivative: int = 0) -> np.ndarray:
        """
        Evaluate a B-spline at a set of time points.

        Parameters:
        ----------
            tck (tuple): A tuple containing the knot vector and the coefficients of the B-spline.
            times (numpy.ndarray): The time points at which to evaluate the B-spline.
            derivative (int): The order of the derivative to evaluate (default is 0).

        Returns:
        -------
            numpy.ndarray: The points on the B-spline at the given time points (3, n).
        """

        return np.array(splev(times, tck, der=derivative))

    def calculate_vectors(self, times: np.ndarray) -> tuple:
        """
        Calculate the tangent, normal, and binormal vectors of the spline at a set of time points.

        Parameters:
        ----------
            times (numpy.ndarray): The time points at which to calculate the vectors.

        Returns:
        -------
            tuple: A tuple containing the tangent, normal, and binormal vectors at the given time points.
                   Each has shape (3, n).
        """

        # Handle the case where times is empty
        if len(times) == 0:
            return np.array([]), np.array([]), np.array([])

        # Calculate the first derivative of the spline
        first_derivatives = self.evaluate_spline(self.tck, times, derivative=1)

        # Calculate the second derivative of the spline
        second_derivatives = self.evaluate_spline(self.tck, times, derivative=2)

        # Calculate tangent vectors
        tangent_vectors = first_derivatives / np.linalg.norm(first_derivatives, axis=0)

        # Calculate normal vectors
        # https://tex.stackexchange.com/questions/643915/tangent-normal-and-binormal-vectors
        normals = np.cross(
            first_derivatives,
            np.cross(second_derivatives, first_derivatives, axis=0),
            axis=0,
        )
        normal_vectors = normals / np.linalg.norm(normals, axis=0)

        # Calculate binormal vectors
        binormal_vectors = np.cross(tangent_vectors, normal_vectors, axis=0)

        return tangent_vectors, normal_vectors, binormal_vectors

    def calculate_rotation_minimizing_vectors(self, times: np.ndarray) -> tuple:
        """
        Calculate the rotation minimizing frames of the spline at a set of time points.

        The frames calculated with `calculate_vectors` (using normal and binormal as coordinate frames)
        are not rotation minimizing. That means that as the spline changes direction, the normal and binormal
        may flip, causing the frames to rotate more than necessary.

        This function calculates rotation minimizing frames by rotating the previous frame to the current frame
        using Rodrigues' rotation formula. This ensures that the rotation angle is minimized.

        Parameters:
        ----------
            times (numpy.ndarray): The time points at which to calculate the rotation minimizing frames.

        Returns:
        -------
            tuple: A tuple containing the tangent, normal, and binormal vectors at the given time points.
                   Each has shape (3, n).
        """

        # Handle the case where times is empty
        if len(times) == 0:
            return np.array([]), np.array([]), np.array([])

        # Calculate the first derivative of the spline
        first_derivatives = self.evaluate_spline(self.tck, times, derivative=1)

        # Calculate tangent vectors
        tangent_vectors = first_derivatives / np.linalg.norm(first_derivatives, axis=0)
        tangent_vectors = tangent_vectors.T

        # Calculate initial frame
        initial_tangent = tangent_vectors[0]

        # Choose an arbitrary vector that is not parallel to the tangent
        if np.abs(initial_tangent[0]) < 1e-6 and np.abs(initial_tangent[1]) < 1e-6:
            initial_normal = np.array([0.0, 1.0, 0.0])
        else:
            initial_normal = np.array([-initial_tangent[1], initial_tangent[0], 0.0])

        # Normalize the normal vector
        initial_normal /= np.linalg.norm(initial_normal)

        # Compute the binormal vector as the cross product of T0 and N0
        initial_binormal = np.cross(initial_tangent, initial_normal)

        # Recompute the normal vector as the cross product of B0 and T0
        initial_normal = np.cross(initial_binormal, initial_tangent)

        tangents = [initial_tangent]
        normals = [initial_normal]
        binormals = [initial_binormal]

        for i in range(1, len(times)):
            previous_tangent, previous_normal, previous_binormal = (
                tangents[-1],
                normals[-1],
                binormals[-1],
            )
            current_tangent = tangent_vectors[i]

            # Calculate the rotation axis
            rotation_axis = np.cross(previous_tangent, current_tangent)
            if np.linalg.norm(rotation_axis) < 1e-6:
                # If the rotation axis is too small, keep the previous frame
                tangents.append(previous_tangent)
                normals.append(previous_normal)
                binormals.append(previous_binormal)
                continue

            # Normalize the rotation axis
            rotation_axis /= np.linalg.norm(rotation_axis)

            # Calculate the rotation angle
            dot_product = np.dot(previous_tangent, current_tangent)
            rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            # Create the rotation matrix using Rodrigues' rotation formula
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            rotation_matrix = (
                np.eye(3)
                + np.sin(rotation_angle) * K
                + (1 - np.cos(rotation_angle)) * np.dot(K, K)
            )

            # Update the normal and binormal vectors
            current_normal = np.dot(rotation_matrix, previous_normal)
            current_binormal = np.dot(rotation_matrix, previous_binormal)

            tangents.append(current_tangent)
            normals.append(current_normal)
            binormals.append(current_binormal)

        tangent_vectors = np.array(tangents).T
        normal_vectors = np.array(normals).T
        binormal_vectors = np.array(binormals).T

        # Make sure that the normal and binormal vectors are normalized
        normal_vectors /= np.linalg.norm(normal_vectors, axis=0)
        binormal_vectors /= np.linalg.norm(binormal_vectors, axis=0)

        return tangent_vectors, normal_vectors, binormal_vectors

    def calculate_equidistant_parameters(self, distance_between_points: float) -> np.ndarray:
        """
        Calculate the parameter values that correspond to equidistant points along the spline.

        Parameters:
        ----------
            distance_between_points (float): The distance between consecutive points.

        Returns:
        -------
            numpy.ndarray: The parameter values that correspond to equidistant points along the spline.
        """

        if distance_between_points <= 0:
            raise ValueError(
                "The distance between points must be positive and non-zero."
            )

        arc_length = calculate_arc_length(self, self.u)
        total_length = arc_length[-1]

        # Determine the number of points n based on the desired distance d
        n = int(np.floor(total_length / distance_between_points)) + 1

        # Interpolate distances to find equidistant parameters
        desired_arc_lengths = np.linspace(0, arc_length[-1], n)
        equidistant_params = np.interp(desired_arc_lengths, arc_length, self.u)

        return equidistant_params

    def calculate_adaptive_parameters(
        self,
        distance_between_points: float,
        ratio: float = 0.5,
        calculation_params_ratio: int = 10,
    ) -> np.ndarray:
        """
        Sample spline parameters based on both curvature and arc length.

        Based on: https://doi.org/10.1016/j.cagd.2017.11.004

        Parameters:
        ----------
            distance_between_points (float): The distance between consecutive points.
            ratio (float): The ratio between the curvature and arc length components.
                1 means equal weight, 0.5 means arc length is twice as important as curvature,
                and 2 means curvature is twice as important as arc length.
            calculation_params_ratio (int): The ratio between the number of calculation parameters
                                            and the number of points.

        Returns:
        -------
            numpy.ndarray:
        """

        if distance_between_points <= 0:
            raise ValueError(
                "The distance between points must be positive and non-zero."
            )

        # Estimate the total arc length
        arc_length = calculate_arc_length(self, self.u)
        total_arc_length = arc_length[-1]

        n_sample_params = int(total_arc_length / distance_between_points) + 1

        # Determine the number of calculation parameters to use
        n_calculation_params = calculation_params_ratio * n_sample_params

        sample_params = np.linspace(0, 1, n_sample_params)
        calculation_params = np.linspace(0, 1, n_calculation_params)

        # Sample the spline parameters based on curvature and arc length
        parameters = adaptive_curvature_parameterization(
            self, sample_params, calculation_params, ratio=ratio
        )

        return parameters


def calculate_spline_curvature(spline: Spline, t: np.ndarray) -> np.ndarray:
    """
    Calculate the curvature of a spline at a given set of points.

    Parameters
    ----------
    spline : Spline
        The spline to evaluate.
    t : np.ndarray
        The points at which to evaluate the curvature.

    Returns
    -------
    np.ndarray
        The curvature at each point.
    """
    # Calculate the first and second derivatives
    first_derivative = spline(t, derivative=1).T
    second_derivative = spline(t, derivative=2).T

    # Calculate the curvature
    numerator = np.linalg.norm(np.cross(first_derivative, second_derivative), axis=1)
    denominator = np.linalg.norm(first_derivative) ** 3
    curvature = numerator / denominator

    return curvature


def calculate_curvature_parameterization(spline: Spline, t: np.ndarray) -> np.ndarray:
    """
    Calculate the curvature parameterization of a spline at a given set of points.

    Parameters
    ----------
    spline : Spline
        The spline to evaluate.
    t : np.ndarray
        The points at which to evaluate the curvature parameterization.

    Returns
    -------
    np.ndarray
        The curvature parameterization at each point.
    """
    # Calculate the curvature
    curvature = calculate_spline_curvature(spline, t)

    # Calculate the first derivative of the curvature
    first_derivative = spline(t, derivative=1).T
    len_term = np.linalg.norm(first_derivative)

    # Calculate the cumulative sum of the curvature
    curvature_sum = np.cumsum(curvature * len_term)

    return curvature_sum


def calculate_arc_length(spline: Spline, t: np.ndarray) -> np.ndarray:
    """
    Calculate the cumulative arc length of a spline at a given set of points.

    Parameters
    ----------
    spline : Spline
        The spline to evaluate.
    t : np.ndarray
        The points at which to evaluate the arc length.

    Returns
    -------
    np.ndarray
        The arc length at each point.
    """

    # Evaluate the B-spline derivative
    dx, dy, dz = spline(t, derivative=1)

    # Compute the cumulative arc length using the derivatives
    dists = np.linalg.norm([dx, dy, dz], axis=0)
    arc_length = np.cumsum(dists * np.diff(t, prepend=0))

    return arc_length


def adaptive_curvature_parameterization(
    spline: Spline, sample_params: np.ndarray, calculation_params: np.ndarray, ratio: float = 1
) -> np.ndarray:
    """
    Sample spline parameters based on both curvature and arc length.

    Based on: https://doi.org/10.1016/j.cagd.2017.11.004

    Parameters
    ----------
    spline : Spline
        The spline to sample.
    sample_params : np.ndarray
        The points at which to sample the spline.
    calculation_params : np.ndarray
        The points at which to calculate the curvature and arc length.
    ratio : float
        The ratio between the curvature and arc length components.

    Returns
    -------
    np.ndarray
        The sampled parameters.
    """

    # Convert ratio to weights for the components
    curvature_weight = ratio / (1 + ratio)
    arc_length_weight = 1 - curvature_weight

    arc_length = calculate_arc_length(spline, calculation_params)
    curvature = calculate_curvature_parameterization(spline, calculation_params)

    arc_length_component = arc_length / arc_length[-1]
    curvature_component = curvature / curvature[-1]

    hybrid_parameter = (
        arc_length_weight * arc_length_component
        + curvature_weight * curvature_component
    )

    # Invert the parameterization axes
    # Note: This is done because we want more points where the curvature is high
    # and less points where the curvature is low. This is the opposite of the
    # original parameterization.
    return np.interp(sample_params, hybrid_parameter, calculation_params)

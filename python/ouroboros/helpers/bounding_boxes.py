import numpy as np

from cloudvolume import Bbox
from dataclasses import dataclass

DEFAULT_SPLIT_THRESHOLD = 0.9
DEFAULT_MAX_DEPTH = 10
DEFAULT_TARGET_SLICES_PER_BOX = 128


@dataclass
class BoundingBoxParams:
    max_depth: int = DEFAULT_MAX_DEPTH
    target_slices_per_box: int = DEFAULT_TARGET_SLICES_PER_BOX

    def to_dict(self):
        return {
            "max_depth": self.max_depth,
            "target_slices_per_box": self.target_slices_per_box,
        }

    @staticmethod
    def from_dict(data: dict) -> "BoundingBoxParams":
        max_depth = data.get("max_depth", DEFAULT_MAX_DEPTH)
        target_slices_per_box = data.get(
            "target_slices_per_box", DEFAULT_TARGET_SLICES_PER_BOX
        )

        return BoundingBoxParams(max_depth, target_slices_per_box)


class BoundingBox:
    def __init__(self, initial_rect: tuple) -> dict:
        x_min, x_max, y_min, y_max, z_min, z_max = BoundingBox.get_rect_bounds(
            initial_rect
        )

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max

        self.approx_bounds_memo = None

    def vol_slice(self) -> tuple[slice]:
        return np.s_[self.x_min: self.x_max, self.y_min: self.y_max, self.z_min: self.z_max]

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }

    def from_dict(data: dict):
        x_min = data["x_min"]
        x_max = data["x_max"]
        y_min = data["y_min"]
        y_max = data["y_max"]
        z_min = data["z_min"]
        z_max = data["z_max"]

        return BoundingBox(
            BoundingBox.bounds_to_rect(x_min, x_max, y_min, y_max, z_min, z_max)
        )

    @staticmethod
    def bound_rects(rects: np.ndarray) -> np.ndarray:
        """
        Get the minimum bounding box of a set of slices.

        Parameters:
        ----------
            rects (numpy.ndarray): A 3D array of shape (n, 4, 3) containing the slices to bound.

        Returns:
        -------
            (numpy.ndarray): A new rect that bounds all the slices.
        """

        # Isolate the x, y, and z coordinates
        x_coords = rects[:, :, 0]
        y_coords = rects[:, :, 1]
        z_coords = rects[:, :, 2]

        # Find the minimum and maximum values for x, y, and z
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)

        return BoundingBox.bounds_to_rect(x_min, x_max, y_min, y_max, z_min, z_max)

    @staticmethod
    def bound_boxes(bounding_boxes: list["BoundingBox"], use_approx_bounds: bool = True) -> "BoundingBox":
        """
        Get the minimum bounding box of a set of bounding boxes.

        Parameters:
        ----------
            bounding_boxes (list[BoundingBox]): A list of bounding boxes to bound.
            use_approx_bounds (bool): Whether to use the approximate bounds of the bounding boxes.

        Returns:
        -------
            (BoundingBox): A new bounding box that bounds all the bounding boxes.
        """

        if use_approx_bounds:
            x_mins = [bb.approx_bounds()[0] for bb in bounding_boxes]
            x_maxs = [bb.approx_bounds()[1] for bb in bounding_boxes]
            y_mins = [bb.approx_bounds()[2] for bb in bounding_boxes]
            y_maxs = [bb.approx_bounds()[3] for bb in bounding_boxes]
            z_mins = [bb.approx_bounds()[4] for bb in bounding_boxes]
            z_maxs = [bb.approx_bounds()[5] for bb in bounding_boxes]
        else:
            x_mins = [bb.x_min for bb in bounding_boxes]
            x_maxs = [bb.x_max for bb in bounding_boxes]
            y_mins = [bb.y_min for bb in bounding_boxes]
            y_maxs = [bb.y_max for bb in bounding_boxes]
            z_mins = [bb.z_min for bb in bounding_boxes]
            z_maxs = [bb.z_max for bb in bounding_boxes]

        x_min = min(x_mins)
        x_max = max(x_maxs)
        y_min = min(y_mins)
        y_max = max(y_maxs)
        z_min = min(z_mins)
        z_max = max(z_maxs)

        return BoundingBox(
            BoundingBox.bounds_to_rect(x_min, x_max, y_min, y_max, z_min, z_max)
        )

    @staticmethod
    def from_rects(rects: np.ndarray) -> "BoundingBox":
        """
        Create a bounding box from a set of slices.

        Parameters:
        ----------
            rects (numpy.ndarray): A 3D array of shape (n, 4, 3) containing the slices to bound.

        Returns:
        -------
            (BoundingBox): The bounding box object that contains the slices.
        """

        return BoundingBox(BoundingBox.bound_rects(rects))

    @staticmethod
    def bounds_to_rect(x_min, x_max, y_min, y_max, z_min, z_max) -> np.ndarray:
        return np.array(
            [
                [x_min, y_max, z_max],
                [x_max, y_max, z_min],
                [x_max, y_min, z_min],
                [x_min, y_min, z_max],
            ]
        )

    @staticmethod
    def get_rect_bounds(rect: np.ndarray) -> tuple:
        x, y, z = rect.T
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        z_min = min(z)
        z_max = max(z)

        return x_min, x_max, y_min, y_max, z_min, z_max

    def to_cloudvolume_bbox(self) -> Bbox:
        return Bbox(
            (self.x_min, self.y_min, self.z_min), (self.x_max, self.y_max, self.z_max)
        )

    def approx_bounds(self) -> tuple:
        if self.approx_bounds_memo is not None:
            return self.approx_bounds_memo

        x_min = int(np.floor(self.x_min))
        x_max = int(np.ceil(self.x_max))
        y_min = int(np.floor(self.y_min))
        y_max = int(np.ceil(self.y_max))
        z_min = int(np.floor(self.z_min))
        z_max = int(np.ceil(self.z_max))

        self.approx_bounds_memo = (x_min, x_max, y_min, y_max, z_min, z_max)

        return self.approx_bounds_memo

    def get_shape(self) -> tuple:
        x_min, x_max, y_min, y_max, z_min, z_max = self.approx_bounds()
        return (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)

    def intersects(self, other) -> bool:
        x_min, x_max, y_min, y_max, z_min, z_max = self.approx_bounds()
        other_x_min, other_x_max, other_y_min, other_y_max, other_z_min, other_z_max = (
            other.approx_bounds()
        )

        return (
            (x_min <= other_x_max and x_max >= other_x_min)
            and (y_min <= other_y_max and y_max >= other_y_min)
            and (z_min <= other_z_max and z_max >= other_z_min)
        )

    def intersection(self, other) -> "BoundingBox":
        """
        Get the intersection of two bounding boxes.

        Note: This function assumes that the bounding boxes intersect.
        Note 2: The intersection is calculated based on approximate bounds.
        """

        x_min, x_max, y_min, y_max, z_min, z_max = self.approx_bounds()
        other_x_min, other_x_max, other_y_min, other_y_max, other_z_min, other_z_max = (
            other.approx_bounds()
        )

        x_min = max(x_min, other_x_min)
        x_max = min(x_max, other_x_max)
        y_min = max(y_min, other_y_min)
        y_max = min(y_max, other_y_max)
        z_min = max(z_min, other_z_min)
        z_max = min(z_max, other_z_max)

        return BoundingBox(
            BoundingBox.bounds_to_rect(x_min, x_max, y_min, y_max, z_min, z_max)
        )

    def to_empty_volume(self, dtype=np.float32, num_channels=None) -> np.ndarray:
        # Add an extra dimension for the number of channels
        shape = (
            self.get_shape() + (num_channels,)
            if num_channels is not None
            else self.get_shape()
        )

        return np.zeros(shape, dtype=dtype)

    def calculate_volume(self) -> float:
        return np.prod(self.get_shape())

    def should_be_divided(self, utilized_volume: float) -> bool:
        return utilized_volume < (1 - DEFAULT_SPLIT_THRESHOLD) * self.calculate_volume()

    def longest_dimension(self):
        x_range = self.x_max - self.x_min
        y_range = self.y_max - self.y_min
        z_range = self.z_max - self.z_min

        return np.argmax([x_range, y_range, z_range])

    def get_min(self, dtype: type = None):
        min_val = np.array([self.x_min, self.y_min, self.z_min])
        return min_val.astype(dtype) if dtype is not None else min_val

    def get_max(self, dtype: type = None):
        max_val = np.array([self.x_max, self.y_max, self.z_max])
        return max_val.astype(dtype) if dtype is not None else max_val

    def to_prism(self) -> np.ndarray:
        vertices = np.array(
            [
                [self.x_min, self.y_max, self.z_max],
                [self.x_max, self.y_max, self.z_max],
                [self.x_max, self.y_min, self.z_max],
                [self.x_min, self.y_min, self.z_max],
                [self.x_min, self.y_max, self.z_min],
                [self.x_max, self.y_max, self.z_min],
                [self.x_max, self.y_min, self.z_min],
                [self.x_min, self.y_min, self.z_min],
            ]
        )

        faces = np.array(
            [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # Top
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # Bottom
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
            ]
        )

        return faces


def boxes_dim_range(boxes: list[BoundingBox], dim="z"):
    """
    Calculate the range of values in a specified dimension for a set of bounding boxes.

    Paramters:
    ----------
        boxes (list[BoundingBox]): A list of the bounding boxes making up the scope.
        dim (str): A string ('x', 'y', 'z')
    """
    if dim not in ["x", "y", "z"]:
        raise ValueError("Dim parameter must be one of 'x', 'y' or 'z'")
    if len(boxes):
        return np.arange(np.min(np.floor([getattr(box, f"{dim}_min") for box in boxes]).astype(int)),
                         np.max(np.floor([getattr(box, f"{dim}_max") for box in boxes]).astype(int)) + 2,
                         dtype=int)
    else:
        return np.arange(0)


def calculate_bounding_boxes_bsp_link_rects(
    rects: np.ndarray,
    target_slices_per_box: int = DEFAULT_TARGET_SLICES_PER_BOX,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> tuple[list[BoundingBox], list[int]]:
    """
    Use binary space partitioning to calculate the bounding boxes of the slices,
    starting with a bounding box that bounds all of the rects.

    Parameters:
    ----------
        rects (numpy.ndarray): A 3D array of shape (n, 4, 3) containing the slices to bound.

    Returns:
    -------
        (list, list): A list of bounding boxes that closely fit the slices and a list
                      mapping each rect index to its bounding box.
    """

    # TODO: heuristic to avoid overlap

    if len(rects) == 0:
        return [], []

    # Calculate the initial bounding box that encompasses all rects
    initial_bounding_box = BoundingBox.from_rects(rects)

    # Stack for iterative BSP: each item is a tuple (rect indices subset, bounding box, repeat entry)
    indices = np.arange(len(rects))
    stack = [(indices, initial_bounding_box, False, max_depth)]
    bounding_boxes = []
    rect_to_box_map = [None] * len(rects)

    while stack:
        current_indices, current_bounding_box, repeat, depth = stack.pop()
        current_rects = rects[current_indices]

        # Determine if further division is necessary or efficient
        if len(current_rects) <= target_slices_per_box or depth == 0:
            bounding_boxes.append(current_bounding_box)
            for idx in current_indices:
                rect_to_box_map[idx] = len(bounding_boxes) - 1
            continue

        # Determine the longest dimension to split along
        longest_dim = current_bounding_box.longest_dimension()

        # Split the current set of rects based on the median of the longest dimension
        median = np.median(current_rects[:, :, longest_dim])
        left_mask = current_rects[:, :, longest_dim].mean(axis=1) < median
        right_mask = ~left_mask

        left_partition_indices = current_indices[left_mask]
        right_partition_indices = current_indices[right_mask]

        # Handle any unsplittable boxes
        if repeat and (
            len(left_partition_indices) == 0 or len(right_partition_indices) == 0
        ):
            bounding_boxes.append(current_bounding_box)
            for idx in current_indices:
                rect_to_box_map[idx] = len(bounding_boxes) - 1
            continue

        # Calculate bounding boxes for the partitions
        if len(left_partition_indices) > 0:
            left_bounding_box = BoundingBox.from_rects(rects[left_partition_indices])
            stack.append(
                (
                    left_partition_indices,
                    left_bounding_box,
                    len(right_partition_indices) == 0,
                    depth - 1,
                )
            )
        if len(right_partition_indices) > 0:
            right_bounding_box = BoundingBox.from_rects(rects[right_partition_indices])
            stack.append(
                (
                    right_partition_indices,
                    right_bounding_box,
                    len(left_partition_indices) == 0,
                    depth - 1,
                )
            )

    return bounding_boxes, rect_to_box_map

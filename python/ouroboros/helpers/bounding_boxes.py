import heapq
from bisect import bisect_left
from collections.abc import Callable

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

    def get_shape(self, tight=False) -> tuple:
        x_min, x_max, y_min, y_max, z_min, z_max = self.approx_bounds()
        return (x_max - x_min + 1 - tight * 2, y_max - y_min + 1 - tight * 2, z_max - z_min + 1 - tight * 2)

    def get_integer_origin(self) -> np.ndarray:
        """World coordinate of voxel zero in the inclusive integer extent."""
        return np.array(self.approx_bounds()[::2], dtype=np.int64)

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
        envelope = BoundingBox.bound_boxes(boxes)
        axis = "xyz".index(dim)
        origin = envelope.get_integer_origin()[axis]
        return np.arange(origin, origin + envelope.get_shape()[axis], dtype=int)
    else:
        return np.arange(0)


def _calculate_bounding_boxes_count(
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


def _slice_index_discontinuities(indices: np.ndarray) -> np.ndarray:
    return np.diff(indices) != 1


def slice_indices_are_contiguous(indices: np.ndarray) -> bool:
    """Whether indices identify a nonempty, consecutive output interval."""
    return bool(
        len(indices) > 0
        and indices[0] >= 0
        and not np.any(_slice_index_discontinuities(indices))
    )


@dataclass
class _PartitionNode:
    box: BoundingBox
    indices: np.ndarray
    payload: int
    children: tuple[int, int] | None = None


_TaskPayload = Callable[[BoundingBox, int, bool], int]


def _box_from_extents(lower: np.ndarray, upper: np.ndarray) -> BoundingBox:
    # Extrema are temporary scan results. BoundingBox remains responsible for
    # converting these floating corner bounds into inclusive integer extents.
    return BoundingBox(BoundingBox.bounds_to_rect(
        lower[0], upper[0], lower[1], upper[1], lower[2], upper[2]
    ))


def _spatially_smaller(child: BoundingBox, parent_volume: int) -> bool:
    # Require a five-percent reduction in each child, using exact integer
    # arithmetic at the inclusive voxel boundary.
    return 20 * int(child.calculate_volume()) <= 19 * parent_volume


def _path_partition(
    indices: np.ndarray,
    current_rects: np.ndarray,
    parent_volume: int,
    task_payload: _TaskPayload,
) -> tuple[_PartitionNode, _PartitionNode] | None:
    """Choose the minimax path cut within the spatially admissible interval."""
    lower = np.min(current_rects, axis=1)
    upper = np.max(current_rects, axis=1)
    prefix_lower = np.minimum.accumulate(lower, axis=0)
    prefix_upper = np.maximum.accumulate(upper, axis=0)
    suffix_lower = np.minimum.accumulate(lower[::-1], axis=0)[::-1]
    suffix_upper = np.maximum.accumulate(upper[::-1], axis=0)[::-1]
    gaps = np.concatenate(([0], np.cumsum(_slice_index_discontinuities(indices))))
    # Prefix boxes only grow and suffix boxes only shrink. Locate the interval
    # passing the canonical volume rule before constructing/scoring every cut.
    # Payloads need not be monotone; all admissible cuts retain their old order.
    stop = 1 + bisect_left(range(1, len(indices)), True, key=lambda cut: not _spatially_smaller(
        _box_from_extents(prefix_lower[cut - 1], prefix_upper[cut - 1]), parent_volume,
    ))
    start = 1 + bisect_left(range(1, stop), True, key=lambda cut: _spatially_smaller(
        _box_from_extents(suffix_lower[cut], suffix_upper[cut]), parent_volume,
    ))
    best = None
    best_score = None
    for cut in range(start, stop):
        left = _box_from_extents(prefix_lower[cut - 1], prefix_upper[cut - 1])
        right = _box_from_extents(suffix_lower[cut], suffix_upper[cut])
        left_payload = int(task_payload(left, cut, bool(gaps[cut - 1] == 0)))
        right_payload = int(task_payload(
            right, len(indices) - cut, bool(gaps[-1] == gaps[cut])
        ))
        score = (
            max(left_payload, right_payload),
            int(left.calculate_volume()) + int(right.calculate_volume()),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = (
                _PartitionNode(left, indices[:cut], left_payload),
                _PartitionNode(right, indices[cut:], right_payload),
            )
    return best


def _best_partition(
    node: _PartitionNode, rects: np.ndarray, task_payload: _TaskPayload
) -> tuple[_PartitionNode, _PartitionNode] | None:
    indices = node.indices
    current_rects = rects[indices]
    centers = current_rects.mean(axis=1)
    parent_volume = int(node.box.calculate_volume())
    candidates = []
    # Fixed order resolves exact ties: axis medians, axis gaps, index gap,
    # then the best valid path cut. Every child retains sorted slice indices.
    for axis in range(3):
        mask = centers[:, axis] < np.median(centers[:, axis])
        candidates.append((indices[mask], indices[~mask]))
    for axis in range(3):
        order = np.argsort(centers[:, axis], kind="stable")
        gaps = np.diff(centers[order, axis])
        cut = int(np.argmax(gaps)) + 1
        if gaps[cut - 1] > 0:
            candidates.append((np.sort(indices[order[:cut]]), np.sort(indices[order[cut:]])))
    index_gaps = np.diff(indices)
    if np.any(index_gaps > 1):
        cut = int(np.argmax(index_gaps)) + 1
        candidates.append((indices[:cut], indices[cut:]))

    best = None
    best_score = None
    for left_indices, right_indices in candidates:
        if len(left_indices) == 0 or len(right_indices) == 0:
            continue
        left = BoundingBox.from_rects(rects[left_indices])
        right = BoundingBox.from_rects(rects[right_indices])
        if not (_spatially_smaller(left, parent_volume) and _spatially_smaller(right, parent_volume)):
            continue
        left_payload = int(task_payload(
            left, len(left_indices), slice_indices_are_contiguous(left_indices)
        ))
        right_payload = int(task_payload(
            right, len(right_indices), slice_indices_are_contiguous(right_indices)
        ))
        score = (
            max(left_payload, right_payload),
            int(left.calculate_volume()) + int(right.calculate_volume()),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = (
                _PartitionNode(left, left_indices, left_payload),
                _PartitionNode(right, right_indices, right_payload),
            )
    path = _path_partition(indices, current_rects, parent_volume, task_payload)
    if path is not None:
        score = (
            max(child.payload for child in path),
            sum(int(child.box.calculate_volume()) for child in path),
        )
        if best_score is None or score < best_score:
            best = path
    return best


def _partition_tree(
    rects: np.ndarray, max_depth: int, task_payload: _TaskPayload
) -> list[_PartitionNode]:
    root_box = BoundingBox.from_rects(rects)
    nodes = [_PartitionNode(
        root_box, np.arange(len(rects)), int(task_payload(root_box, len(rects), True))
    )]
    stack = [(0, max_depth)]
    while stack:
        index, depth = stack.pop()
        node = nodes[index]
        if depth == 0 or len(node.indices) <= 1:
            continue
        children = _best_partition(node, rects, task_payload)
        if children is not None:
            node.children = (len(nodes), len(nodes) + 1)
            nodes.extend(children)
            stack.extend((child, depth - 1) for child in node.children)
    return nodes


def _select_partition_threshold(
    nodes: list[_PartitionNode], reference_peak: int, reference_volume: int
) -> int | None:
    """Evaluate every distinct tree threshold and the fully expanded frontier.

    Descending thresholds only expand leaves. Cumulative requested volume can
    increase or decrease, so every resulting frontier is checked. The heaps
    track expansion events and actual peak payload in O(nodes log nodes).
    """
    active = np.zeros(len(nodes), dtype=bool)
    active[0] = True
    expandable = [(-nodes[0].payload, 0)] if nodes[0].children is not None else []
    payloads = [(-nodes[0].payload, 0)]
    volume = int(nodes[0].box.calculate_volume())
    count = 1
    best_score = None
    best_threshold = None
    thresholds = sorted({node.payload for node in nodes}, reverse=True)
    # A pure payload callback need not be monotone under subdivision. If an
    # expandable node has the smallest score, reaching that score still keeps
    # it intact. Cross the final boundary to consider its children as well.
    thresholds.append(thresholds[-1] - 1)
    for threshold in thresholds:
        while expandable and -expandable[0][0] > threshold:
            _, index = heapq.heappop(expandable)
            node = nodes[index]
            active[index] = False
            volume -= int(node.box.calculate_volume())
            count -= 1
            for child_index in node.children:
                child = nodes[child_index]
                active[child_index] = True
                volume += int(child.box.calculate_volume())
                count += 1
                heapq.heappush(payloads, (-child.payload, child_index))
                if child.children is not None:
                    heapq.heappush(expandable, (-child.payload, child_index))
        while not active[payloads[0][1]]:
            heapq.heappop(payloads)
        peak = -payloads[0][0]
        score = (peak, volume, count)
        if peak < reference_peak and volume <= reference_volume:
            if best_score is None or score < best_score:
                best_score = score
                best_threshold = threshold
    return best_threshold


def calculate_bounding_boxes_bsp_link_rects(
    rects: np.ndarray,
    target_slices_per_box: int = DEFAULT_TARGET_SLICES_PER_BOX,
    max_depth: int = DEFAULT_MAX_DEPTH,
    *,
    task_payload: _TaskPayload | None = None,
) -> tuple[list[BoundingBox], list[int]]:
    """Partition slice rectangles with a bounded, optional task-payload model.

    Existing calls retain the count-based partition. With ``task_payload``, a
    deterministic candidate tree is evaluated at all of its payload thresholds.
    The callback receives the canonical bounding box, slice count, and whether
    the assigned indices form one contiguous run. It must be a pure function.

    A candidate is returned only if its actual largest task payload is smaller
    than the count-based reference and its total inclusive source volume does
    not increase. Otherwise the reference is returned unchanged. The target
    count defines that reference; candidate leaves may fall on either side of
    it. Depth and the five-percent spatial-shrink rule remain safety limits,
    so a threshold is a selection parameter, not a hard memory guarantee.
    """
    reference = _calculate_bounding_boxes_count(rects, target_slices_per_box, max_depth)
    if task_payload is None or len(rects) == 0:
        return reference
    reference_boxes, reference_mapping = reference
    groups = [[] for _ in reference_boxes]
    for index, box_index in enumerate(reference_mapping):
        groups[box_index].append(index)
    reference_peak = max(
        int(task_payload(box, len(indices), slice_indices_are_contiguous(np.asarray(indices))))
        for box, indices in zip(reference_boxes, groups)
    )
    reference_volume = sum(int(box.calculate_volume()) for box in reference_boxes)
    nodes = _partition_tree(rects, max_depth, task_payload)
    threshold = _select_partition_threshold(nodes, reference_peak, reference_volume)
    if threshold is None:
        return reference
    boxes = []
    mapping = [None] * len(rects)
    stack = [0]
    while stack:
        node = nodes[stack.pop()]
        if node.payload > threshold and node.children is not None:
            stack.extend(node.children)
            continue
        for index in node.indices:
            mapping[index] = len(boxes)
        boxes.append(node.box)
    return boxes, mapping

from types import SimpleNamespace

import numpy as np
import tifffile
from cloudvolume import Bbox
from scipy.ndimage import map_coordinates

from ouroboros.helpers.bounding_boxes import calculate_bounding_boxes_bsp_link_rects
from ouroboros.helpers.options import SliceOptions
from ouroboros.helpers.slice import calculate_slice_rects, estimate_slice_task_payload
from ouroboros.helpers.spline import Spline
from ouroboros.helpers.volume_cache import download_volume
from ouroboros.pipeline import slice_parallel_pipeline as slice_module
from test.sample_data import generate_sample_curve_helix


def test_payload_partition_preserves_world_samples_and_tiff_slice_assignment(tmp_path, monkeypatch):
    # Use the production path-to-rectangle calculation. The known source field
    # supplies a numerical oracle; specimen geometry/performance is tested
    # separately. These settings produce different count and payload partitions.
    height, width = 33, 31
    points = generate_sample_curve_helix(start_z=0, end_z=6, radius=12, num_points=80) + 1040
    spline = Spline(points, degree=3)
    parameters = spline.calculate_equidistant_parameters(1)
    rects = calculate_slice_rects(parameters, spline, width, height)
    original_rects = rects.copy()
    options = SliceOptions(
        output_file_folder=str(tmp_path), output_file_name="sample", neuroglancer_json="local.json",
        slice_width=width, slice_height=height, process_count=4,
    )

    world_origin = np.full(3, 976)
    x, y, z = np.ogrid[:128, :128, :128]
    world = (128 + 0.5 * x + 0.25 * y + 0.125 * z).astype(np.float32)
    source_bounds = Bbox(world_origin, world_origin + world.shape)

    def download(bbox, **kwargs):
        assert source_bounds.contains_bbox(bbox)
        start, stop = bbox.minpt - world_origin, bbox.maxpt - world_origin
        return world[tuple(slice(int(low), int(high)) for low, high in zip(start, stop))][..., None]

    source = SimpleNamespace(
        cv=SimpleNamespace(bounds=source_bounds, num_channels=1, download=download), dtype=world.dtype,
    )

    def task_payload(box, count, contiguous):
        return estimate_slice_task_payload(
            box, count, contiguous, slice_shape=(height, width), source_dtype=world.dtype, channels=1,
        ).peak_bytes

    partitions = [
        calculate_bounding_boxes_bsp_link_rects(rects, target_slices_per_box=16, max_depth=10),
        calculate_bounding_boxes_bsp_link_rects(
            rects, target_slices_per_box=16, max_depth=10, task_payload=task_payload,
        ),
    ]
    # Compare memberships, not arbitrary box labels, so this cannot pass merely
    # because an otherwise identical partition was returned in another order.
    groups = [
        sorted(tuple(np.flatnonzero(np.asarray(links) == index)) for index in range(len(boxes)))
        for boxes, links in partitions
    ]
    assert groups[0] != groups[1]
    np.testing.assert_array_equal(rects, original_rects)

    # Independent float64 evaluation of each original slice, without the
    # production coordinate helper or either partition's origin arithmetic.
    horizontal = np.linspace(0, 1, width)[None, None, :, None]
    vertical = np.linspace(0, 1, height)[None, :, None, None]
    exact_world_grid = (
        rects[:, 0, None, None] + (rects[:, 1] - rects[:, 0])[:, None, None] * horizontal
        + (rects[:, 3] - rects[:, 0])[:, None, None] * vertical - world_origin
    )
    exact_reference = map_coordinates(
        world, exact_world_grid.reshape(-1, 3).T, output=np.float32,
    ).reshape(len(rects), height, width)
    interpolate = slice_module.slice_volume_from_grids
    results, interiors = [], []

    for partition_index, (boxes, links) in enumerate(partitions):
        path = tmp_path / f"partition-{partition_index}.tif"
        tifffile.imwrite(
            path, np.full((len(rects), height, width, 1), -999, dtype=np.float32), photometric="minisblack",
        )
        actual_world_grid = np.full(exact_world_grid.shape, np.nan)
        interior = np.zeros((len(rects), height, width), dtype=bool)

        def observe_interpolation(volume, box, grids, width, height, *, output=None):
            assert np.all(grids >= 0)
            assert np.all(grids <= np.asarray(volume.shape[:3]) - 1)
            actual_world_grid[slice_indices] = grids + box.get_integer_origin() - world_origin
            distance = np.minimum(grids, np.asarray(box.get_shape()) - 1 - grids).min(axis=-1)
            interior[slice_indices] = distance > 8
            return interpolate(volume, box, grids, width, height, output=output)

        monkeypatch.setattr(slice_module, "slice_volume_from_grids", observe_interpolation)
        for box_index, box in enumerate(boxes):
            slice_indices = np.flatnonzero(np.asarray(links) == box_index).tolist()
            volume, returned_box, _ = download_volume(source, box, mip=0)
            assert returned_box is box
            assert volume.shape[:3] == box.get_shape()
            slice_module.process_worker_save_parallel(
                options, volume, box, slice_indices, box_index, rects, str(path),
            )

        sampled = tifffile.imread(path).reshape(len(rects), height, width)
        assert np.all(sampled > 0), "Every original slice must be written without constant-fill zero edges"
        assert np.all(np.isfinite(actual_world_grid))
        coordinate_tolerance = 3 * np.spacing(np.float32(rects.max()))
        np.testing.assert_allclose(actual_world_grid, exact_world_grid, rtol=0, atol=coordinate_tolerance)

        # Match test_statistics' established smooth-linear-field check: the
        # >8-voxel interior limits this fixture's filtering residual to one
        # intensity ULP. This is not a cutoff for arbitrary specimen data.
        assert interior.sum() > 100
        coordinate_reference = map_coordinates(
            world, actual_world_grid.reshape(-1, 3).T, output=np.float32,
        ).reshape(len(rects), height, width)
        np.testing.assert_array_max_ulp(sampled[interior], coordinate_reference[interior], maxulp=1)
        # Separately allow the coordinate quantization already present in the
        # worker; do not attribute that difference to partition assignment.
        np.testing.assert_array_max_ulp(sampled[interior], exact_reference[interior], maxulp=32)
        results.append(sampled)
        interiors.append(interior)

    common_interior = interiors[0] & interiors[1]
    assert common_interior.sum() > 100
    np.testing.assert_array_max_ulp(results[0][common_interior], results[1][common_interior], maxulp=32)

"""A small, offline source → straightened TIFF → source-grid regression."""

from shutil import copyfile

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import tifffile

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.files import (
    generate_tiff_write,
    np_convert,
    validate_straightened_volume_for_backprojection,
    volume_from_intermediates,
)
from ouroboros.helpers.options import SliceOptions
from ouroboros.helpers.shapes import ImgSliceC
from ouroboros.helpers.slice import coordinate_grid, slice_volume_from_grids
from ouroboros.metrics import round_trip_accuracy
from ouroboros.pipeline.backproject_pipeline import process_chunk
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput
from ouroboros.pipeline.slices_geom_pipeline import SlicesGeometryPipelineStep
from test.harness import write_page


def _round_trip(folder, source, rects, slice_shape, box, *, normalize=True, binary=False):
    """Reconstruct supported intensity and mask TIFFs through the same production path."""
    folder.mkdir()
    sliced = slice_volume_from_grids(
        source, lambda index: coordinate_grid(rects[index], slice_shape),
        output=np.empty((len(rects), *slice_shape), dtype=np.float32),
        slice_indices=range(len(rects)),
    )
    intermediates = folder / "intermediates"
    x, y, z = box.get_shape()
    # Use the slicing pipeline's global bounds and default zero guard for
    # both paths. Disabling normalization keeps the uint8 label units; the
    # zero-guard shift and integer truncation still affect mask fidelity.
    converted = np_convert(source.dtype, sliced, preset_min=sliced.min(), preset_max=sliced.max(),
                           normalize=normalize)
    straightened = folder / "straightened.tif"
    tifffile.imwrite(straightened, converted, photometric="minisblack")
    info = validate_straightened_volume_for_backprojection(str(straightened), rects)
    # Several chunks contribute to the same output pages, exercising the
    # production merge rather than a test copy of weighted division.
    for index, start in enumerate(range(0, len(rects), 17)):
        stop = min(start + 17, len(rects))
        process_chunk(intermediates, info, rects[start:stop],
                      (slice(start, stop), slice(None), slice(None)),
                      (index, 0, 0), box, intermediate_threads=1)

    target = folder / "backprojected"
    target.mkdir()
    writer = generate_tiff_write(tifffile.imwrite, None, np.ones(3),
                                 box.get_integer_origin().tolist(), photometric="minisblack")
    page_shape = ImgSliceC(Y=y, X=x, C=1)
    averaged = np.zeros((x, y, z), dtype=np.float32)
    weights = np.zeros_like(averaged)
    written = np.zeros((x, y, z), dtype=np.uint8 if binary else source.dtype)
    for index in range(z):
        path = intermediates / f"i_{index:05}.dat"
        # Both canonical consumers delete intermediates. Copy their boundary
        # representation to observe weights and the actual TIFF writer once.
        writer_path = folder / "writer.dat"
        if path.exists():
            copyfile(path, writer_path)
        accumulator = np.zeros((2, x * y), dtype=np.float32)
        values = volume_from_intermediates(path, page_shape, accumulator=accumulator)
        averaged[:, :, index] = values.reshape(y, x).T
        weights[:, :, index] = accumulator[-1].reshape(y, x).T
        write_page(writer, writer_path, target, page_shape,
                   dtype=bool if binary else source.dtype, index=index)
        written[:, :, index] = tifffile.imread(target / f"{index:05}.tif").reshape(y, x).T
    return averaged, weights, written


@pytest.mark.needs("backprojection")
def test_sinusoidal_tube_round_trip_accuracy(tmp_path):
    """Track interpolation loss and binary support growth at default spacing."""
    shape = (48, 48, 72)
    rng = np.random.default_rng(254)
    texture = gaussian_filter(rng.standard_normal(shape), sigma=1.5)
    source = np_convert(np.uint16, texture)

    def centerline(z):
        return 24 + 4 * np.sin(z / 8), 24 + 3 * np.cos(z / 10)

    x, y, z = np.indices(shape)
    center_x, center_y = centerline(z)
    radius = 5  # Known radius in the source XY sections along the sinusoidal trace.
    mask = (((x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2)
            & (z >= 10) & (z <= 61)).astype(np.uint8)
    trace_z = np.linspace(10, 61, 40)
    trace_x, trace_y = centerline(trace_z)
    options = SliceOptions(output_file_folder=str(tmp_path), output_file_name="tube",
                           neuroglancer_json="unused.json", slice_width=20, slice_height=20)
    data = PipelineInput(slice_options=options,
                         sample_points=np.column_stack((trace_x, trace_y, trace_z)),
                         source_url="memory://synthetic-tube")
    result, error = Pipeline([SlicesGeometryPipelineStep()]).process(data)
    assert error is None, error
    rects = result.slice_rects
    box = BoundingBox.from_rects(rects)
    slice_shape = (options.slice_height, options.slice_width)
    _, weights, reconstructed = _round_trip(tmp_path / "intensity", source, rects, slice_shape, box)
    weighted_mask, mask_weights, binary_mask = _round_trip(
        tmp_path / "mask", mask, rects, slice_shape, box, normalize=False, binary=True,
    )
    # Both validated TIFFs now use exactly the same geometry and chunk path.
    np.testing.assert_array_equal(weights, mask_weights)
    np.testing.assert_array_equal(
        binary_mask, np_convert(bool, weighted_mask, normalize=False, safe_bool=True),
    )

    origin = box.get_integer_origin()
    crop = tuple(slice(start, start + size) for start, size in zip(origin, box.get_shape()))
    measurements = round_trip_accuracy(
        source[crop], reconstructed, weights, source_mask=mask[crop],
        backprojected_mask=weighted_mask, slice_rects=rects - origin,
        slice_shape=slice_shape,
    )

    # Thresholds are set after measuring this seeded fixture, with room for
    # numerical differences across supported NumPy/SciPy versions.
    assert measurements["coverage"]["covered_fraction"] > 0.3
    assert measurements["coverage"]["interior_fraction"] > 0.055
    intensity = measurements["intensity"]
    assert intensity["interior"]["pearson"] > 0.99
    assert intensity["interior"]["nrmse"] < 0.012
    assert intensity["interior"]["psnr"] > 22
    assert intensity["interior"]["fitted_psnr"] > 37
    assert intensity["covered"]["pearson"] > 0.97
    assert intensity["covered"]["nrmse"] < 0.03
    assert intensity["boundary"]["nrmse"] > intensity["interior"]["nrmse"]
    masks = measurements["masks"]
    assert masks["weighted"]["dice"] > 0.95
    assert masks["weighted"]["iou"] > 0.9
    assert masks["weighted"]["precision"] > 0.98
    assert masks["weighted"]["recall"] > 0.91
    assert masks["weighted"]["hd95"] <= 1
    assert masks["weighted"]["centroid_distance"] < 0.08
    assert masks["binary"]["dice"] > 0.84
    assert masks["binary"]["iou"] > 0.73
    assert masks["binary"]["recall"] > 0.98
    assert masks["binary"]["precision"] > 0.73
    assert masks["binary"]["hd95"] < 2
    # Calibrated on adaptive geometry using per-point spline speed.
    assert masks["binary"]["centroid_distance"] < 0.15
    assert masks["binary"]["precision"] < masks["weighted"]["precision"]
    assert masks["binary"]["dice"] < masks["weighted"]["dice"]
    for name, mode in masks.items():
        distances = mode["per_slice_centroid_distance"]
        assert len(distances) == len(rects)
        assert all(distance is not None for distance in distances)
        assert max(distances) < (0.4 if name == "weighted" else 0.65)
    assert measurements["geometry"]["slice_shape"] == list(slice_shape)
    np.testing.assert_array_equal(measurements["geometry"]["slice_rects"], rects - origin)

"""Offline accuracy measurements on aligned source-grid volumes."""

import numpy as np
from scipy.ndimage import binary_erosion, center_of_mass, distance_transform_edt

from ouroboros.helpers.files import np_convert
from ouroboros.helpers.slice import coordinate_grid, slice_volume_from_grids


def _volume(value, name, shape=None):
    value = np.asarray(value)
    if value.ndim != 3 or not value.size or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} must be a nonempty XYZ volume with the same shape as source")
    if value.dtype.kind not in "buif" or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real values")
    return value


def _intensity(source, reconstructed, region):
    x = source[region].astype(np.float64)
    y = reconstructed[region].astype(np.float64)
    result = dict.fromkeys(("pearson", "nrmse", "psnr", "fitted_psnr", "fit_scale",
                           "fit_offset", "rmse", "fitted_rmse"))
    result["count"] = int(x.size)
    if not x.size:
        return result

    xc, yc = x - x.mean(), y - y.mean()
    xx, yy, xy = np.dot(xc, xc), np.dot(yc, yc), np.dot(xc, yc)
    scale = xy / yy if yy > 0 else 0.0
    offset = x.mean() - scale * y.mean()
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    fitted_rmse = float(np.sqrt(np.mean((xc - scale * yc) ** 2)))
    data_range = float(np.ptp(x))
    result.update(fit_scale=float(scale), fit_offset=float(offset), rmse=rmse,
                  fitted_rmse=fitted_rmse)
    if xx > 0 and yy > 0:
        result["pearson"] = float(np.clip(xy / np.sqrt(xx * yy), -1, 1))
    if data_range > 0:
        result["nrmse"] = fitted_rmse / data_range
        for key, error in (("psnr", rmse), ("fitted_psnr", fitted_rmse)):
            result[key] = float(20 * np.log10(data_range / error)) if error else float("inf")
    return result


def _offset(reference, reconstructed):
    if reference is None or reconstructed is None:
        return None, None
    delta = np.asarray(reconstructed) - reference
    return delta.tolist(), float(np.linalg.norm(delta))


def _centroids_on_rects(mask, rects, shape):
    # Use exactly the production raster coordinates and cubic sampler. This is
    # a diagnostic re-slice of the two masks, not a second geometry convention.
    planes = slice_volume_from_grids(
        mask, lambda i: coordinate_grid(rects[i], shape),
        output=np.empty((len(rects), *shape), dtype=np.float32),
        slice_indices=range(len(rects)),
    )
    centroids = []
    for rect, plane in zip(rects, planes):
        foreground = plane >= 0.5
        centroids.append(coordinate_grid(rect, shape)[foreground].mean(axis=0, dtype=np.float64)
                         if foreground.any() else None)
    return centroids


def _mask_metrics(source, reconstructed, rects, shape, source_centroids):
    expected, actual = int(source.sum()), int(reconstructed.sum())
    intersection = int(np.count_nonzero(source & reconstructed))
    union = expected + actual - intersection
    delta, distance = _offset(center_of_mass(source) if expected else None,
                              center_of_mass(reconstructed) if actual else None)
    if expected and actual:
        # Six-connected foreground boundary; outside the bounding box is zero.
        reference_surface = source & ~binary_erosion(source, border_value=0)
        actual_surface = reconstructed & ~binary_erosion(reconstructed, border_value=0)
        to_reference = distance_transform_edt(~reference_surface)[actual_surface]
        to_actual = distance_transform_edt(~actual_surface)[reference_surface]
        hd95 = float(max(np.percentile(to_reference, 95), np.percentile(to_actual, 95)))
    else:
        hd95 = 0.0 if expected == actual else float("inf")
    result = {
        "dice": 2 * intersection / (expected + actual) if expected + actual else 1.0,
        "iou": intersection / union if union else 1.0,
        "precision": intersection / actual if actual else None,
        "recall": intersection / expected if expected else None,
        "hd95": hd95,
        "centroid_offset": delta,
        "centroid_distance": distance,
        "per_slice_centroid_offset": None,
        "per_slice_centroid_distance": None,
    }
    if rects is not None:
        measured = _centroids_on_rects(reconstructed, rects, shape)
        offsets = [_offset(a, b) for a, b in zip(source_centroids, measured)]
        result["per_slice_centroid_offset"] = [item[0] for item in offsets]
        result["per_slice_centroid_distance"] = [item[1] for item in offsets]
    return result


def round_trip_accuracy(
    source: np.ndarray,
    backprojected: np.ndarray,
    weights: np.ndarray,
    *,
    source_mask: np.ndarray | None = None,
    backprojected_mask: np.ndarray | None = None,
    slice_rects: np.ndarray | None = None,
    slice_shape: tuple[int, int] | None = None,
) -> dict:
    """Measure a source → straightened → backprojected round trip.

    All volumes must be finite, scalar XYZ arrays on the same source-voxel
    grid, cropped to the backprojection's bounding box. ``backprojected`` is
    already divided by accumulated ``weights``; pass raw weights, not a
    binary coverage mask. TIFF ZYX arrays must be transposed by the caller.
    Evaluate channels separately. The function does not register, renormalize,
    or modify the supplied volumes.

    Coverage uses weight > 0 and weight >= 1. Intensity results describe
    covered, interior (>= 1), and boundary (0 < weight < 1) regions. Each
    region fits source = scale * backprojected + offset independently.
    NRMSE is fitted RMSE / source peak-to-peak range in that region. PSNR
    uses the same range and raw RMSE; fitted_psnr uses fitted RMSE. Empty
    regions and undefined correlations/range-normalized scores return None;
    zero error with a nonzero range gives infinite PSNR.

    Supply both mask arguments to measure a separate binary-source round
    trip. ``source_mask`` must be 0/1; ``backprojected_mask`` is its averaged
    reconstruction, before thresholding. Weighted mode uses >= 0.5; binary
    mode uses the production writer's nonzero conversion, including negative
    values. Scores use the entire bounding box, including uncovered voxels.
    HD95 is the maximum of the two directed 95th-percentile Euclidean surface
    distances on six-connected boundaries, in voxels. Both empty masks have
    Dice/IoU 1 and HD95 0; one empty mask gives infinite HD95. Undefined
    precision, recall, or centroid offsets are None.

    Optional ``slice_rects`` (n, 4, 3), in bounding-box-local XYZ coordinates,
    and ``slice_shape`` (height, width) record the sampling geometry. Supply
    both together. With masks, their per-slice centroid offsets are measured
    by re-slicing each final mask with the production cubic sampler and
    thresholding at 0.5. Offsets are reconstructed minus source in XYZ voxel
    units, ordered along the supplied trace; empty planes return None.

    This offline function allocates working arrays and does not stream large
    volumes. Results contain Python scalars/lists (including None/infinity).
    See the round-trip accuracy reference for usage and expected losses.
    """
    source = _volume(source, "source")
    backprojected = _volume(backprojected, "backprojected", source.shape)
    weights = _volume(weights, "weights", source.shape)
    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative accumulated weights")
    if (source_mask is None) != (backprojected_mask is None):
        raise ValueError("source_mask and backprojected_mask must be supplied together")
    if (slice_rects is None) != (slice_shape is None):
        raise ValueError("slice_rects and slice_shape must be supplied together")
    if slice_rects is not None:
        slice_rects = np.asarray(slice_rects, dtype=np.float64)
        if (slice_rects.ndim != 3 or slice_rects.shape[1:] != (4, 3)
                or not len(slice_rects) or not np.isfinite(slice_rects).all()):
            raise ValueError("slice_rects must have finite shape (n, 4, 3), n > 0")
        if (len(slice_shape) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1
                                         for n in slice_shape)):
            raise ValueError("slice_shape must contain positive integer height and width")
        slice_shape = tuple(int(n) for n in slice_shape)

    covered, interior = weights > 0, weights >= 1
    result = {
        "coverage": {
            "total_voxels": int(source.size),
            "covered_voxels": int(covered.sum()),
            "interior_voxels": int(interior.sum()),
            "covered_fraction": float(covered.mean()),
            "interior_fraction": float(interior.mean()),
        },
        "intensity": {name: _intensity(source, backprojected, region) for name, region in (
            ("covered", covered), ("interior", interior), ("boundary", covered & ~interior))},
        "masks": None,
        "geometry": None if slice_rects is None else {
            "slice_rects": slice_rects.tolist(), "slice_shape": list(slice_shape),
        },
    }
    if source_mask is not None:
        source_mask = _volume(source_mask, "source_mask", source.shape)
        if not np.all((source_mask == 0) | (source_mask == 1)):
            raise ValueError("source_mask must contain only 0 and 1")
        source_mask = source_mask.astype(bool)
        backprojected_mask = _volume(backprojected_mask, "backprojected_mask", source.shape)
        source_centroids = (None if slice_rects is None else
                            _centroids_on_rects(source_mask, slice_rects, slice_shape))
        binary = np_convert(bool, backprojected_mask, normalize=False, safe_bool=True).astype(bool)
        result["masks"] = {
            name: _mask_metrics(source_mask, mask, slice_rects, slice_shape, source_centroids)
            for name, mask in (("weighted", backprojected_mask >= 0.5), ("binary", binary))
        }
    return result

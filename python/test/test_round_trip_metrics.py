"""Known-value checks for the public scientific metric definitions."""

import numpy as np
import pytest

from ouroboros.metrics import round_trip_accuracy


def test_coverage_and_affine_fit_use_only_their_weight_regions():
    source = np.arange(24).reshape(2, 3, 4)
    reconstructed = 3 * source + 7
    weights = np.zeros(source.shape)
    weights[0] = 0.5
    weights[1, :2] = 1
    reconstructed[weights == 0] = 100000
    result = round_trip_accuracy(source, reconstructed, weights)
    assert result["coverage"] == {
        "total_voxels": 24, "covered_voxels": 20, "interior_voxels": 8,
        "covered_fraction": 20 / 24, "interior_fraction": 8 / 24,
    }
    for name, count in (("covered", 20), ("interior", 8), ("boundary", 12)):
        score = result["intensity"][name]
        assert score["count"] == count
        assert score["pearson"] == pytest.approx(1)
        assert score["fit_scale"] == pytest.approx(1 / 3)
        assert score["fit_offset"] == pytest.approx(-7 / 3)
        assert score["nrmse"] == pytest.approx(0, abs=1e-14)
        assert score["psnr"] < 0  # Raw intensities still differ.
    assert result["masks"] is None
    assert result["geometry"] is None


def test_raw_psnr_and_identity():
    source = np.arange(8).reshape(2, 2, 2)
    result = round_trip_accuracy(source, source + 1, np.ones_like(source))
    assert result["intensity"]["covered"]["psnr"] == pytest.approx(20 * np.log10(7))
    identity = round_trip_accuracy(source, source, np.ones_like(source))
    assert identity["intensity"]["covered"]["psnr"] == float("inf")
    assert identity["intensity"]["covered"]["fitted_psnr"] == float("inf")


def test_mask_translation_has_known_overlap_surface_distance_and_xyz_offsets():
    source = np.zeros((7, 6, 5), dtype=bool)
    source[1:4, 1:4, 1:4] = True
    reconstructed = np.zeros_like(source)
    reconstructed[2:5, 1:4, 1:4] = True
    rects = np.array([[[0, 0, z], [6, 0, z], [6, 5, z], [0, 5, z]]
                      for z in (0, 1, 2, 3, 4)])
    result = round_trip_accuracy(
        source, reconstructed, np.ones(source.shape), source_mask=source,
        backprojected_mask=reconstructed, slice_rects=rects, slice_shape=(6, 7),
    )
    for score in result["masks"].values():
        assert score["dice"] == pytest.approx(2 / 3)
        assert score["iou"] == pytest.approx(1 / 2)
        assert score["precision"] == pytest.approx(2 / 3)
        assert score["recall"] == pytest.approx(2 / 3)
        assert score["hd95"] == 1
        assert score["centroid_offset"] == [1, 0, 0]
        assert score["centroid_distance"] == 1
        assert score["per_slice_centroid_offset"] == [None, [1, 0, 0], [1, 0, 0], [1, 0, 0], None]
        assert score["per_slice_centroid_distance"] == [None, 1, 1, 1, None]
    assert result["geometry"] == {"slice_rects": rects.tolist(), "slice_shape": [6, 7]}


def test_binary_uses_nonzero_including_negative_ringing_and_counts_uncovered_misses():
    source = np.zeros((3, 3, 3))
    source[1, 1, 1] = source[2, 2, 2] = 1
    reconstructed = np.zeros_like(source)
    reconstructed[1, 1, 1] = 0.5
    reconstructed[0, 0, 0] = -0.01
    weights = np.zeros_like(source)
    weights[1, 1, 1] = weights[0, 0, 0] = 0.5
    result = round_trip_accuracy(source, reconstructed, weights,
                                 source_mask=source, backprojected_mask=reconstructed)
    weighted, binary = result["masks"]["weighted"], result["masks"]["binary"]
    assert weighted["precision"] == 1
    assert weighted["recall"] == 0.5
    assert binary["precision"] == binary["recall"] == 0.5


def test_empty_and_constant_regions_are_explicit_without_runtime_warnings():
    empty = np.zeros((3, 3, 3))
    with np.errstate(all="raise"):
        result = round_trip_accuracy(empty, empty, empty,
                                     source_mask=empty, backprojected_mask=empty)
        constant = round_trip_accuracy(empty, np.ones_like(empty), np.ones_like(empty))
        missing = round_trip_accuracy(empty, empty, empty,
                                      source_mask=empty, backprojected_mask=np.ones_like(empty))
    for region in result["intensity"].values():
        assert region["count"] == 0
        assert all(value is None for key, value in region.items() if key != "count")
    score = constant["intensity"]["covered"]
    assert score["pearson"] is score["nrmse"] is score["psnr"] is score["fitted_psnr"] is None
    assert score["rmse"] == 1
    assert score["fitted_rmse"] == 0
    for score in result["masks"].values():
        assert score["dice"] == score["iou"] == 1
        assert score["hd95"] == 0
        assert score["precision"] is score["recall"] is score["centroid_offset"] is None
    assert missing["masks"]["weighted"]["hd95"] == float("inf")


@pytest.mark.parametrize("argument,value,message", [
    ("backprojected", np.zeros((2, 2, 3)), "same shape"),
    ("weights", np.full((2, 2, 2), -1), "nonnegative"),
    ("source", np.full((2, 2, 2), np.nan), "finite"),
    ("source_mask", np.zeros((2, 2, 2)), "supplied together"),
    ("slice_shape", (2, 2), "supplied together"),
])
def test_invalid_inputs_cannot_silently_produce_scores(argument, value, message):
    arguments = dict(source=np.zeros((2, 2, 2)), backprojected=np.zeros((2, 2, 2)),
                     weights=np.ones((2, 2, 2)))
    arguments[argument] = value
    with pytest.raises(ValueError, match=message):
        round_trip_accuracy(**arguments)

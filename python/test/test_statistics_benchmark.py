"""Mode scheduling and comparisons supplement actual CLI benchmark validation."""

import json
import sys

import numpy as np
import pytest
import tifffile

from benchmarks import benchmark_run_statistics as benchmark
from ouroboros.common.logging_mode import LoggingMode


@pytest.mark.parametrize("names", [None, ["detailed", "standard", "off"]])
def test_cli_defaults_to_standard_and_preserves_selected_mode_order(monkeypatch, names):
    argv = ["benchmark"] + (["--modes", *names] if names else [])
    captured = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(benchmark, "benchmark", captured.append)
    benchmark.main()
    assert captured[0].modes == [LoggingMode(name) for name in (names or ["standard"])]
    assert all(isinstance(mode, LoggingMode) for mode in captured[0].modes)


def test_cli_rejects_duplicate_modes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--modes", "standard", "standard"])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


@pytest.mark.parametrize("modes", [list(LoggingMode), [LoggingMode.STANDARD],
                                  [LoggingMode.DETAILED, LoggingMode.STANDARD]])
def test_every_round_runs_each_mode_once_and_rotates_first_mode(modes):
    rounds = [benchmark.modes_for_round(modes, index) for index in range(len(modes) * 2)]
    assert all(len(order) == len(modes) and set(order) == set(modes) for order in rounds)
    assert [order[0] for order in rounds] == modes * 2


def make_runs(modes):
    # Known ratios let the summary test verify comparisons despite rotating order.
    times = {LoggingMode.OFF: 100, LoggingMode.STANDARD: 101, LoggingMode.DETAILED: 120}
    return [{"mode": mode.value, "round": round_index + 1,
             "handler_elapsed_seconds": times[mode] * (round_index + 1),
             "output": {"shape": [2, 3, 4], "dtype": "uint16", "pixel_sha256": "same-pixels"}}
            for round_index in range(3)
            for mode in benchmark.modes_for_round(modes, round_index)]


def test_standard_only_summary_has_distribution_without_overhead():
    result = benchmark.summary(make_runs([LoggingMode.STANDARD]))
    assert result["modes"]["standard"] == {
        "median_seconds": 202, "min_seconds": 101, "max_seconds": 303,
        "median_absolute_deviation_seconds": 101, "samples_seconds": [101, 202, 303],
    }
    assert result["baseline_mode"] is None
    assert result["comparisons"] == {}
    assert result["pixel_equivalence"] is True
    assert "measured_median_at_most_one_percent" not in json.dumps(result)


def test_all_modes_compare_each_round_to_off_regardless_of_selection_order():
    result = benchmark.summary(make_runs([LoggingMode.DETAILED, LoggingMode.OFF, LoggingMode.STANDARD]))
    assert result["baseline_mode"] == "off"
    assert set(result["comparisons"]) == {"standard", "detailed"}
    for mode, expected in (("standard", 1), ("detailed", 20)):
        comparison = result["comparisons"][mode]
        assert comparison["median_elapsed_overhead_percent"] == pytest.approx(expected)
        assert [row["round"] for row in comparison["per_round_overhead_percent"]] == [1, 2, 3]
        assert [row["overhead_percent"] for row in comparison["per_round_overhead_percent"]] == pytest.approx([expected] * 3)
        assert "measured_median_at_most_one_percent" in comparison
    assert result["comparisons"]["detailed"]["measured_median_at_most_one_percent"] is False


def test_standard_baseline_omits_off_only_threshold():
    result = benchmark.summary(make_runs([LoggingMode.DETAILED, LoggingMode.STANDARD]))
    assert result["baseline_mode"] == "standard"
    assert result["comparisons"]["detailed"]["median_elapsed_overhead_percent"] == pytest.approx((120 / 101 - 1) * 100)
    assert "measured_median_at_most_one_percent" not in json.dumps(result)


def test_summary_checks_pixels_and_shape_in_every_mode():
    runs = make_runs(list(LoggingMode))
    runs[-1]["output"]["pixel_sha256"] = "different-pixels"
    assert benchmark.summary(runs)["pixel_equivalence"] is False
    runs[-1]["output"]["pixel_sha256"] = "same-pixels"
    runs[-1]["output"]["shape"] = [2, 4, 3]
    assert benchmark.summary(runs)["pixel_equivalence"] is False


def test_summary_rejects_missing_or_duplicate_trials():
    runs = make_runs(list(LoggingMode))
    with pytest.raises(ValueError, match="Every round"):
        benchmark.summary(runs[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.summary(runs + [runs[0]])


def test_pagewise_pixel_evidence_matches_single_file_and_directory_outputs(tmp_path):
    pixels = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    single = tmp_path / "volume.tif"
    directory = tmp_path / "planes"
    directory.mkdir()
    tifffile.imwrite(single, pixels, photometric="minisblack")
    # Write out of order so hashing must reuse production TIFF ordering.
    for index in (2, 0, 1):
        tifffile.imwrite(directory / f"{index:04}.tif", pixels[index], photometric="minisblack")
    single_evidence = benchmark.tiff_evidence(single)
    directory_evidence = benchmark.tiff_evidence(directory)
    for key in ("shape", "dtype", "pages", "pixel_bytes", "pixel_sha256"):
        assert single_evidence[key] == directory_evidence[key]
    assert directory_evidence["files"] == 3
    assert directory_evidence["storage_bytes"] == sum(file.stat().st_size for file in directory.iterdir())

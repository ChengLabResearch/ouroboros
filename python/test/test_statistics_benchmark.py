"""Mode scheduling and comparisons supplement actual CLI benchmark validation."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from benchmarks import benchmark_run_statistics as benchmark
from ouroboros.common.logging_mode import LoggingMode
from ouroboros.common.memory_statistics import unavailable


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


@pytest.fixture
def child_outputs(monkeypatch):
    """Supply completed CLI artifacts while exercising the benchmark's readers."""
    # Both benchmark entry points set these for their subprocesses.
    for variable in ("CLOUD_VOLUME_DIR", "CLOUD_VOLUME_CACHE_DIR", "CLOUD_FILES_DIR"):
        monkeypatch.delenv(variable, raising=False)
    state = {
        "calls": [],
        "memory": {
            LoggingMode.DETAILED: unavailable("No complete RAM sample"),
            LoggingMode.STANDARD: unavailable("RAM monitoring disabled", status="disabled"),
        },
    }
    pixels = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)

    def completed_child(command, *, cwd, stdout, stderr, check, timeout):
        def argument(name):
            return command[command.index(name) + 1]

        operation = argument("--operation")
        mode = LoggingMode(argument("--logging-mode"))
        options = json.loads(Path(argument("--options")).read_text())
        scope = argument("--scope")
        state["calls"].append({"operation": operation, "mode": mode,
                               "scope": scope, "options": options})
        if operation == "backproject":
            # The orchestration must retain the matching slicing output until
            # backprojection consumes it, including when RAM was unavailable.
            np.testing.assert_array_equal(tifffile.imread(options["straightened_volume_path"]), pixels)
            assert Path(options["slice_options_path"]).is_file()
        output_folder = Path(options["output_file_folder"])
        tifffile.imwrite(output_folder / f"{operation}.tif", pixels, photometric="minisblack")
        measurement = {"handler_elapsed_seconds": 1.0}
        if scope == "pipeline":
            measurement.update(pipeline_elapsed_seconds=0.8,
                               data_operation_elapsed_seconds=0.6)
        benchmark.write_json(Path(argument("--measurement")), measurement)
        if mode.reporting_enabled:
            report_path = output_folder / f"{operation}-statistics.json"
            benchmark.write_json(report_path, {
                "outcome": "succeeded", "error": None, "collection_errors": [],
                "logging_mode": mode.value,
                "steps": [{"custom_times": {"data": 0.6}}],
                "memory": state["memory"][mode],
            })
            stdout.write(report_path.read_text())

    monkeypatch.setattr(benchmark, "subprocess", SimpleNamespace(run=completed_child))
    return state


@pytest.mark.parametrize("scope", ["handler", "pipeline"])
def test_detailed_run_requires_ram_sample_by_default(tmp_path, child_outputs, scope):
    with pytest.raises(RuntimeError, match="Memory sampling unavailable"):
        benchmark.run_once(tmp_path, "slice", {}, LoggingMode.DETAILED,
                           4, "detailed", 30, scope)


@pytest.mark.parametrize("scope", ["handler", "pipeline"])
def test_tolerant_detailed_run_retains_timings_memory_and_pixels(tmp_path, child_outputs, scope):
    result, output = benchmark.run_once(tmp_path, "slice", {}, LoggingMode.DETAILED,
                                       4, "detailed", 30, scope, require_memory=False)
    assert result["handler_elapsed_seconds"] == 1.0
    if scope == "pipeline":
        assert result["pipeline_elapsed_seconds"] == 0.8
        assert result["data_operation_elapsed_seconds"] == 0.6
    assert result["memory"] == child_outputs["memory"][LoggingMode.DETAILED]
    assert result["memory"]["sampled_peak_bytes"] is None
    assert result["output"] == benchmark.tiff_evidence(output)
    assert json.loads((tmp_path / "detailed" / "measurement.json").read_text()) == result


@pytest.mark.parametrize("memory", [
    unavailable("No complete RAM sample"),
    {"status": "sampled", "sampled_peak_bytes": 1024},
    {"status": "disabled", "sampled_peak_bytes": 1024},
])
def test_tolerance_does_not_relax_standard_disabled_memory_contract(tmp_path, child_outputs, memory):
    child_outputs["memory"][LoggingMode.STANDARD] = memory
    with pytest.raises(RuntimeError, match="Memory was not disabled"):
        benchmark.run_once(tmp_path, "slice", {}, LoggingMode.STANDARD,
                           4, "standard", 30, "pipeline", require_memory=False)


@pytest.mark.parametrize("scope", ["handler", "pipeline"])
def test_fixture_benchmark_requires_detailed_ram_samples(tmp_path, child_outputs, scope):
    fixture_folder = tmp_path / "fixture"
    fixture_folder.mkdir()
    benchmark.write_json(fixture_folder / "results.json", {
        "fixture": {"slice_options": {"make_single_file": True}},
    })
    tifffile.imwrite(fixture_folder / "segmentation.tif", np.zeros((3, 8, 8), dtype=np.uint8),
                     photometric="minisblack")
    args = SimpleNamespace(work_dir=tmp_path / "run", reuse_fixture=fixture_folder,
                           workers=4, scope=scope, pairs=2, timeout=30,
                           modes=[LoggingMode.DETAILED])
    with pytest.raises(RuntimeError, match="Memory sampling unavailable"):
        benchmark.benchmark(args)
    assert len(child_outputs["calls"]) == 1


def test_real_data_keeps_all_modes_and_backprojection_when_detailed_ram_is_unavailable(
        tmp_path, child_outputs):
    annotations = tmp_path / "source-annotations.json"
    benchmark.write_json(annotations, {"layers": [{
        "type": "annotation", "name": "path", "annotations": [{"point": [1, 2, 3]}],
    }]})
    slice_options = tmp_path / "slice-options.json"
    backproject_options = tmp_path / "backproject-options.json"
    benchmark.write_json(slice_options, {"neuroglancer_json": str(annotations)})
    benchmark.write_json(backproject_options, {})
    args = SimpleNamespace(work_dir=tmp_path / "run", resume_real_data=False,
                           real_slice_options=slice_options,
                           real_backproject_options=backproject_options,
                           annotation_layer=None, cache_dir=None, workers=4,
                           pairs=2, timeout=30,
                           modes=[LoggingMode.DETAILED, LoggingMode.OFF, LoggingMode.STANDARD])

    benchmark.benchmark_real_data(args)

    result = json.loads((args.work_dir / "results.json").read_text())
    path = result["paths"]["path"]
    assert result["production_code_unchanged"] is True
    assert path["slicing_warmup"]["memory"] == child_outputs["memory"][LoggingMode.DETAILED]
    assert len(child_outputs["calls"]) == 13  # One warmup, two rounds of six operations.
    assert all(call["scope"] == "pipeline" for call in child_outputs["calls"])
    assert len(path["runs"]) == 12
    assert all("status" not in run for run in path["runs"])
    for run in path["runs"]:
        if run["mode"] == "detailed":
            assert run["memory"] == child_outputs["memory"][LoggingMode.DETAILED]
    for summaries in (path["operations"], path["pipeline_operations"]):
        assert set(summaries) == {"slice", "backproject"}
        for measured in summaries.values():
            assert set(measured["modes"]) == {"off", "standard", "detailed"}
            assert measured["baseline_mode"] == "off"
            assert set(measured["comparisons"]) == {"standard", "detailed"}
            assert measured["pixel_equivalence"] is True
            assert len(measured["runs"]) == 6
    assert not list(args.work_dir.rglob("*.tif"))

#!/usr/bin/env python3
"""Measure real local CLI runs with and without run-statistics reporting.

Run from an environment with the project's dependencies installed. By default,
generate local CloudVolume chunks; --real-slice-options uses an existing dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time


PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def child_run(args: argparse.Namespace) -> None:
    # Set these before importing numpy/scipy. The same limits apply to both
    # treatments, including the worker processes created by the pipeline.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ.setdefault("CLOUD_VOLUME_DIR", str(args.options.parent / "cloudvolume"))
    os.environ.setdefault("CLOUD_VOLUME_CACHE_DIR", str(args.options.parent / "cloudvolume-cache"))
    os.environ.setdefault("CLOUD_FILES_DIR", str(args.options.parent / "cloudfiles"))

    import ouroboros.cli as cli
    from ouroboros.pipeline.pipeline import Pipeline

    pipeline_times = []
    step_times = []
    if args.scope == "pipeline":
        process = Pipeline.process

        def timed_process(pipeline, input_data):
            started = time.perf_counter()
            try:
                return process(pipeline, input_data)
            finally:
                pipeline_times.append(time.perf_counter() - started)
                # Basic step timers run in both treatments. Read their canonical
                # durations after the external timer, without another clock.
                step_times.append({step.step_name: step.get_duration()
                                   for step in pipeline.steps})

        Pipeline.process = timed_process

    sys.argv = ["ouroboros-cli", args.operation, str(args.options)]
    if args.enabled:
        sys.argv.append("--verbose")
    started = time.perf_counter()
    cli.main()
    sys.stdout.flush()
    sys.stderr.flush()
    elapsed = time.perf_counter() - started
    measurement = {"handler_elapsed_seconds": elapsed}
    if args.scope == "pipeline":
        if len(pipeline_times) != 1:
            raise RuntimeError(f"Expected one pipeline call, measured {len(pipeline_times)}")
        measurement["pipeline_elapsed_seconds"] = pipeline_times[0]
        measurement["step_elapsed_seconds"] = step_times[0]
        data_step = ("SliceParallelPipelineStep" if args.operation == "slice"
                     else "BackprojectPipelineStep")
        measurement["data_operation_elapsed_seconds"] = step_times[0][data_step]
    write_json(args.measurement, measurement)


def make_fixture(folder: Path, size: int, workers: int) -> dict:
    import numpy as np
    from cloudvolume import CloudVolume

    source = folder / "source"
    url = "file://" + str(source)
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type="uint16",
        encoding="raw",
        resolution=[1000, 1000, 1000],
        voxel_offset=[0, 0, 0],
        chunk_size=[64, 64, 64],
        volume_size=[size, size, size],
    )
    cv = CloudVolume(url, info=info, cache=False, compress=False, progress=False)
    cv.commit_info()
    rng = np.random.default_rng(138)
    voxels = rng.integers(0, 65536, size=(size, size, size), dtype=np.uint16)
    cv[:, :, :] = voxels[..., None]
    source_digest = hashlib.sha256(voxels.tobytes()).hexdigest()
    del voxels

    t = np.linspace(0, 2 * np.pi, 32)
    points = np.column_stack(
        (size * (0.5 + 0.07 * np.sin(t)),
         size * (0.5 + 0.07 * np.cos(t)),
         np.linspace(size * 0.2, size * 0.8, len(t)))
    )
    annotation_path = folder / "annotations.json"
    write_json(annotation_path, {"layers": [
        {"type": "annotation", "name": "path", "annotations": [
            {"type": "point", "point": point.tolist()} for point in points
        ]},
        {"type": "image", "name": "local", "source": url},
    ]})
    slice_options = {
        "slice_width": size // 2,
        "slice_height": size // 2,
        "neuroglancer_json": str(annotation_path),
        "neuroglancer_image_layer": "local",
        "neuroglancer_annotation_layer": "path",
        "slicing_params": {"dist_between_slices": 0.5, "use_adaptive_slicing": False},
        "bounding_box_params": {"target_slices_per_box": 32, "max_depth": 10},
        "normalize_output": False,
        "zeroguard_output": False,
        "make_single_file": True,
    }
    return {
        "volume_shape_xyz": [size, size, size],
        "source_dtype": "uint16",
        "source_voxel_bytes": size ** 3 * 2,
        "source_pixel_sha256": source_digest,
        "source_storage_bytes": sum(path.stat().st_size for path in source.rglob("*") if path.is_file()),
        "seed": 138,
        "workers": workers,
        "slice_options": slice_options,
    }


def tiff_evidence(path: Path) -> dict:
    import numpy as np
    import tifffile

    digest = hashlib.sha256()
    with tifffile.TiffFile(path) as tif:
        shape = (len(tif.pages), *tif.pages[0].shape)
        dtype = str(tif.pages[0].dtype)
        pixel_bytes = 0
        for page in tif.pages:
            pixels = np.ascontiguousarray(page.asarray())
            digest.update(pixels.tobytes())
            pixel_bytes += pixels.nbytes
        pages = len(tif.pages)
    return {
        "shape": shape,
        "dtype": dtype,
        "pages": pages,
        "pixel_bytes": pixel_bytes,
        "storage_bytes": path.stat().st_size,
        "pixel_sha256": digest.hexdigest(),
    }


def run_once(folder: Path, operation: str, options: dict, enabled: bool,
             workers: int, label: str, timeout: int, scope: str = "handler",
             require_memory: bool = True) -> tuple[dict, Path]:
    run_folder = folder / label
    run_folder.mkdir()
    output_folder = run_folder / "output"
    output_folder.mkdir()
    options = dict(options, output_file_folder=str(output_folder), output_file_name=operation,
                   process_count=workers)
    options_path = run_folder / "options.json"
    write_json(options_path, options)
    timing_path = run_folder / "measurement.json"
    command = [sys.executable, str(Path(__file__).resolve()), "--child",
               "--operation", operation, "--options", str(options_path),
               "--measurement", str(timing_path), "--scope", scope]
    if enabled:
        command.append("--enabled")
    started = time.perf_counter()
    with (run_folder / "stdout.txt").open("w") as stdout, (run_folder / "stderr.txt").open("w") as stderr:
        subprocess.run(command, cwd=run_folder, stdout=stdout, stderr=stderr,
                       check=True, timeout=timeout)
    process_elapsed = time.perf_counter() - started
    outputs = list(output_folder.glob("*.tif"))
    if not options.get("make_single_file", True):
        outputs = [path for path in output_folder.iterdir()
                   if path.is_dir() and any(path.glob("*.tif"))]
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one finished TIFF in {output_folder}; found {outputs}. See run logs.")
    output = outputs[0]
    result = dict(json.loads(timing_path.read_text()), enabled=enabled,
                  process_elapsed_seconds=process_elapsed, label=label,
                  statistics_files=[str(path.relative_to(folder))
                                    for path in output_folder.glob("*-statistics.json")])
    if scope == "handler":
        result["output"] = tiff_evidence(output)
    else:
        # Inspect headers only, outside the operation timer; do not read pixels.
        import tifffile
        files = sorted(output.glob("*.tif")) if output.is_dir() else [output]
        with tifffile.TiffFile(files[0]) as tif:
            result["output_header"] = {
                "files": len(files), "pages_in_first_file": len(tif.pages),
                "page_shape": list(tif.pages[0].shape), "dtype": str(tif.pages[0].dtype),
                "storage_bytes": sum(path.stat().st_size for path in files),
            }
    if enabled and not result["statistics_files"]:
        raise RuntimeError(f"Verbose run did not deliver a statistics file: {run_folder}")
    if not enabled and result["statistics_files"]:
        raise RuntimeError(f"Non-verbose run unexpectedly wrote statistics: {run_folder}")
    for report_path in result["statistics_files"]:
        formatted = (folder / report_path).read_text()
        record = json.loads(formatted)
        if record["outcome"] != "succeeded" or record["collection_errors"]:
            raise RuntimeError(f"Reporting failed in {report_path}: {record['collection_errors']}")
        if formatted.strip() not in (run_folder / "stdout.txt").read_text():
            raise RuntimeError(f"Saved statistics differ from stdout in {run_folder}")
        memory = record.get("memory")
        if require_memory and (not memory or memory["sampled_peak_bytes"] is None):
            raise RuntimeError(f"Memory sampling unavailable in {report_path}: {memory}")
        result["memory"] = memory
    write_json(timing_path, result)
    return result, output


def summary(runs: list[dict], elapsed_field: str = "handler_elapsed_seconds") -> dict:
    disabled = [run[elapsed_field] for run in runs if not run["enabled"]]
    enabled = [run[elapsed_field] for run in runs if run["enabled"]]

    def distribution(values: list[float]) -> dict:
        median = statistics.median(values)
        return {"median_seconds": median, "min_seconds": min(values), "max_seconds": max(values),
                "median_absolute_deviation_seconds": statistics.median(abs(value - median) for value in values),
                "samples_seconds": values}

    overhead = (statistics.median(enabled) / statistics.median(disabled) - 1) * 100
    pairs = []
    for index in range(0, len(runs), 2):
        pair = {run["enabled"]: run[elapsed_field] for run in runs[index:index + 2]}
        pairs.append((pair[True] / pair[False] - 1) * 100)
    return {
        "disabled": distribution(disabled),
        "enabled": distribution(enabled),
        "median_elapsed_overhead_percent": overhead,
        "paired_overhead_percent": pairs,
        "measured_median_at_most_one_percent": overhead <= 1,
        "pixel_equivalence": (len({run["output"]["pixel_sha256"] for run in runs}) == 1
                              if all("output" in run for run in runs) else None),
        "runs": runs,
    }


def production_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((PYTHON_ROOT / "ouroboros").rglob("*.py")):
        digest.update(path.relative_to(PYTHON_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def benchmark(args: argparse.Namespace) -> None:
    import numpy as np
    import tifffile

    folder = args.work_dir.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    fixture_folder = args.reuse_fixture.resolve() if args.reuse_fixture else folder
    os.environ["CLOUD_VOLUME_DIR"] = str(fixture_folder / "cloudvolume")
    os.environ["CLOUD_VOLUME_CACHE_DIR"] = str(fixture_folder / "cloudvolume-cache")
    os.environ["CLOUD_FILES_DIR"] = str(fixture_folder / "cloudfiles")
    if args.reuse_fixture:
        fixture = json.loads((fixture_folder / "results.json").read_text())["fixture"]
        if not (fixture_folder / "segmentation.tif").is_file():
            raise ValueError("Reused fixture has no prepared segmentation.tif")
    else:
        fixture = make_fixture(folder, args.size, args.workers)
    elapsed_field = "pipeline_elapsed_seconds" if args.scope == "pipeline" else "handler_elapsed_seconds"
    cpu_model = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu_model = next((line.split(":", 1)[1].strip()
                          for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    result = {
        "schema_version": 1,
        "scope": ("External timer around Pipeline.process only: includes scientific preparation, data operations, "
                  "worker measurement transport/aggregation, concurrent RAM sampling and scientific output; excludes options loading, "
                  "sampler startup/shutdown, "
                  "volume-statistics collection, report construction/printing/publication. Basic duration/progress "
                  "timers remain enabled in both treatments. No pixel hashing."
                  if args.scope == "pipeline" else
                  "Native CLI handler including options load, pipeline, report formatting, stdout flush, and report write; no Docker transfer."),
        "elapsed_field": elapsed_field,
        "cache_policy": ("Warm filesystem and CloudVolume caches; one disabled warmup per operation."
                         if args.scope == "pipeline" else
                         "Warm filesystem and CloudVolume caches; one unmeasured run of each treatment per operation."),
        "hardware": {"platform": platform.platform(), "processor": cpu_model,
                     "logical_cpus": os.cpu_count(), "python": platform.python_version(),
                     "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None},
        "production_python_sha256": production_digest(),
        "benchmark_python_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "tifffile", "cloud-volume", "psutil")},
        "fixture": fixture,
        "pairs": args.pairs,
        "operations": {},
    }
    options = fixture["slice_options"]
    for operation in ("slice", "backproject"):
        if operation == "backproject":
            options = {
                "straightened_volume_path": str(fixture_folder / "segmentation.tif"),
                "slice_options_path": str(fixture_folder / "slice-reference-options.json"),
                "make_single_file": True,
                "make_backprojection_binary": True,
                "backprojection_compression": "zlib",
                "process_count": args.workers,
                "chunk_size": 64,
            }
        warmups = []
        for enabled in ((False,) if args.scope == "pipeline" else (False, True)):
            print(f"{operation}: warmup enabled={enabled}", flush=True)
            warmup, output = run_once(folder, operation, options, enabled, args.workers,
                                      f"{operation}-warmup-{int(enabled)}", args.timeout, args.scope)
            warmups.append(warmup)
            if operation == "slice" and not enabled and not args.reuse_fixture:
                # A fixed segmentation derived once from actual slicing is shared
                # by every backprojection treatment, never regenerated per pair.
                pixels = tifffile.imread(output)
                segmentation = np.squeeze(pixels > 32767).astype(np.uint8)
                tifffile.imwrite(folder / "segmentation.tif", segmentation,
                                 photometric="minisblack", metadata={"axes": "ZYX"})
                reference_options = dict(options, output_file_folder=str(folder), output_file_name="reference")
                write_json(folder / "slice-reference-options.json", reference_options)
                result["fixture"]["segmentation"] = tiff_evidence(folder / "segmentation.tif")
            output.unlink()
        runs = []
        for pair in range(args.pairs):
            for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
                print(f"{operation}: pair {pair + 1}/{args.pairs} enabled={enabled}", flush=True)
                measurement, output = run_once(folder, operation, options, enabled, args.workers,
                                               f"{operation}-{pair:02}-{int(enabled)}", args.timeout, args.scope)
                runs.append(measurement)
                output.unlink()
        result["operations"][operation] = summary(runs, elapsed_field)
        result["operations"][operation]["warmups"] = warmups
        write_json(folder / "results.json", result)
        measured = result["operations"][operation]
        print(f"{operation}: median overhead {measured['median_elapsed_overhead_percent']:.3f}%; "
              f"pixel equivalent={measured['pixel_equivalence']}", flush=True)
        if measured["pixel_equivalence"] is False:
            raise RuntimeError(f"{operation} output pixels differed; inspect {folder / 'results.json'}")
    result["production_code_unchanged"] = production_digest() == result["production_python_sha256"]
    write_json(folder / "results.json", result)
    if not result["production_code_unchanged"]:
        raise RuntimeError("Production source changed during the benchmark; repeat with a stable checkout.")
    print(f"Full results and per-run logs: {folder / 'results.json'}")


def benchmark_real_data(args: argparse.Namespace) -> None:
    """Pair real slicing with backprojection of that exact run's intensity TIFF."""
    folder = args.work_dir.resolve()
    folder.mkdir(parents=True, exist_ok=args.resume_real_data)
    slice_options = json.loads(args.real_slice_options.read_text())
    backproject_options = json.loads(args.real_backproject_options.read_text())
    annotation_source = Path(slice_options["neuroglancer_json"])
    annotations = json.loads(annotation_source.read_text())
    annotation_copy = folder / "annotations.json"
    if args.resume_real_data:
        if annotation_source.read_bytes() != annotation_copy.read_bytes():
            raise ValueError("Cannot resume with changed annotations")
    else:
        shutil.copyfile(annotation_source, annotation_copy)
    slice_options["neuroglancer_json"] = str(annotation_copy)
    layers = [layer for layer in annotations["layers"]
              if layer.get("type") == "annotation" and layer.get("annotations")]
    if args.annotation_layer:
        requested = set(args.annotation_layer)
        missing = requested - {layer["name"] for layer in layers}
        if missing:
            raise ValueError(f"Requested annotation paths are missing or empty: {sorted(missing)}")
        layers = [layer for layer in layers if layer["name"] in requested]
    for variable, name in (("CLOUD_VOLUME_DIR", "cloudvolume"),
                           ("CLOUD_VOLUME_CACHE_DIR", "cloudvolume-cache"),
                           ("CLOUD_FILES_DIR", "cloudfiles")):
        os.environ[variable] = str(folder / name)
    if args.cache_dir:
        os.environ["CLOUD_VOLUME_CACHE_DIR"] = str(args.cache_dir.resolve())
    result = {
        "schema_version": 1,
        "scope": "Existing data-processing step duration; excludes preceding parsing, geometry and source/cache setup. External Pipeline.process timer retained separately. Both exclude sampler startup/shutdown and final reporting.",
        "cache_policy": "One disabled slicing warmup per path; shared CloudVolume cache across all trials. Remote metadata requests may remain.",
        "cloudvolume_cache_directory": os.environ["CLOUD_VOLUME_CACHE_DIR"],
        "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                     "logical_cpus": os.cpu_count(), "python": platform.python_version()},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "scipy", "tifffile", "cloud-volume", "psutil")},
        "production_python_sha256": production_digest(),
        "benchmark_python_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256(annotation_copy.read_bytes()).hexdigest(),
        "annotation_layers": sorted(args.annotation_layer) if args.annotation_layer else None,
        "slice_options_template": slice_options,
        "backproject_options_template": backproject_options,
        "workers": args.workers, "pairs": args.pairs, "paths": {},
    }
    if args.resume_real_data:
        previous = json.loads((folder / "results.json").read_text())
        for key in ("production_python_sha256", "annotation_sha256", "annotation_layers", "workers", "pairs",
                    "slice_options_template", "backproject_options_template", "versions"):
            if previous.get(key) != result[key]:
                raise ValueError(f"Cannot resume with changed {key}")
        previous["resumed_benchmark_python_sha256"] = result["benchmark_python_sha256"]
        previous["resume_note"] = "Retain completed trials, record failed existing trials without retry, and finish remaining scheduled trials using the same run_once timing boundary."
        result = previous

    def remove_output(path):
        shutil.rmtree(path) if path.is_dir() else path.unlink()

    def failed_trial(label, operation, enabled, error):
        run_folder = folder / label
        stderr = run_folder / "stderr.txt"
        record = {"label": label, "operation": operation, "enabled": enabled,
                  "status": "failed_trial", "error": str(error),
                  "stderr": stderr.read_text(errors="replace")[-12000:] if stderr.exists() else None}
        reports = list((run_folder / "output").glob("*-statistics.json"))
        if reports:
            report = json.loads(reports[0].read_text())
            record["reported_outcome"] = report["outcome"]
            record["reported_error"] = report["error"]
            record["memory"] = report.get("memory")
        return record, None

    def attempt(operation, options, enabled, label):
        if (folder / label).exists():
            return failed_trial(label, operation, enabled, "Prior attempt did not complete; retained without retry")
        try:
            return run_once(folder, operation, options, enabled, args.workers,
                            label, args.timeout, "pipeline", require_memory=False)
        except Exception as error:
            print(f"{label}: failed: {error}", flush=True)
            return failed_trial(label, operation, enabled, error)

    # Short paths first; preserve the original annotation names and coordinates.
    for index, layer in enumerate(sorted(layers, key=lambda item: len(item["annotations"]))):
        name = layer["name"]
        options = dict(slice_options, neuroglancer_annotation_layer=name)
        path_result = result["paths"].setdefault(name, {"annotation_points": len(layer["annotations"]), "runs": []})
        if "slicing_warmup" not in path_result:
            print(f"{name}: slicing cache warmup", flush=True)
            warmup, output = attempt("slice", options, False, f"path-{index}-warmup")
            path_result["slicing_warmup"] = warmup
            if output is not None:
                remove_output(output)
        write_json(folder / "results.json", result)
        for pair in range(args.pairs):
            for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
                label = f"path-{index}-pair-{pair + 1}-{int(enabled)}"
                if any(run["label"] == label + "-backproject" for run in path_result["runs"]):
                    continue
                print(f"{name}: pair {pair + 1}/{args.pairs}, enabled={enabled}, slicing", flush=True)
                sliced = next((run for run in path_result["runs"] if run["label"] == label + "-slice"), None)
                if sliced is None:
                    sliced, sliced_output = attempt("slice", options, enabled, label + "-slice")
                    path_result["runs"].append(dict(sliced, operation="slice"))
                else:
                    outputs = list((folder / (label + "-slice") / "output").glob("*.tif"))
                    sliced_output = outputs[0] if len(outputs) == 1 and "status" not in sliced else None
                write_json(folder / "results.json", result)
                if sliced_output is None:
                    path_result["runs"].append({"operation": "backproject", "label": label + "-backproject",
                        "enabled": enabled, "status": "skipped", "error": "Matching slicing trial did not complete"})
                    write_json(folder / "results.json", result)
                    continue
                bp_options = dict(backproject_options,
                    straightened_volume_path=str(sliced_output),
                    slice_options_path=str(folder / (label + "-slice") / "options.json"),
                    process_count=args.workers,
                    make_backprojection_binary=False, make_backprojection_discrete=False)
                print(f"{name}: pair {pair + 1}/{args.pairs}, enabled={enabled}, backprojection", flush=True)
                projected, projected_output = attempt("backproject", bp_options, enabled, label + "-backproject")
                path_result["runs"].append(dict(projected, operation="backproject"))
                write_json(folder / "results.json", result)
                # Keep logs, reports and headers; release large scientific outputs
                # only after the matching backprojection has finished.
                if projected_output is not None:
                    remove_output(projected_output)
                    remove_output(sliced_output)
        operations = {operation: [run for run in path_result["runs"] if run["operation"] == operation]
                      for operation in ("slice", "backproject")}
        def measured_or_incomplete(runs, field):
            if any(field not in run for run in runs):
                return {"status": "incomplete_pairs", "runs": runs}
            return summary(runs, field)
        path_result["operations"] = {operation: measured_or_incomplete(runs, "data_operation_elapsed_seconds")
                                     for operation, runs in operations.items()}
        path_result["pipeline_operations"] = {operation: measured_or_incomplete(runs, "pipeline_elapsed_seconds")
                                              for operation, runs in operations.items()}
        for operation, measured in path_result["operations"].items():
            print(f"{name}: {operation}: {measured.get('median_elapsed_overhead_percent', 'incomplete pairs')}", flush=True)
        write_json(folder / "results.json", result)
    result["production_code_unchanged"] = production_digest() == result["production_python_sha256"]
    write_json(folder / "results.json", result)
    if not result["production_code_unchanged"]:
        raise RuntimeError("Production source changed during the benchmark.")
    print(f"Full results and per-run logs: {folder / 'results.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ouroboros-statistics-benchmark"))
    parser.add_argument("--reuse-fixture", type=Path, help="Reuse a previous benchmark's prepared local data.")
    parser.add_argument("--real-slice-options", type=Path,
                        help="Existing real-data slice options; run every annotation path, paired with backprojection.")
    parser.add_argument("--real-backproject-options", type=Path,
                        help="Existing backprojection settings for real-data mode.")
    parser.add_argument("--annotation-layer", action="append",
                        help="Select an exact annotation path in real-data mode; repeat for multiple paths.")
    parser.add_argument("--cache-dir", type=Path,
                        help="Reuse this CloudVolume chunk cache in real-data mode.")
    parser.add_argument("--resume-real-data", action="store_true",
                        help="Finish remaining real-data trials, preserving completed and failed attempts without retry.")
    parser.add_argument("--scope", choices=("handler", "pipeline"), default="handler",
                        help="Timing boundary; pipeline excludes all final reporting and skips pixel hashing.")
    parser.add_argument("--size", type=int, default=384, help="Cubic source edge in voxels (default: 384).")
    parser.add_argument("--workers", type=int, default=4, help="Multiple of four (default: 4).")
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=600, help="Maximum seconds per child operation.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=("slice", "backproject"), help=argparse.SUPPRESS)
    parser.add_argument("--options", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--measurement", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--enabled", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 4 or args.workers % 4:
        parser.error("--workers must be a positive multiple of four")
    if args.pairs < 2:
        parser.error("--pairs must be at least two")
    if bool(args.real_slice_options) != bool(args.real_backproject_options):
        parser.error("Real-data mode requires both slice and backproject options")
    if args.resume_real_data and not args.real_slice_options:
        parser.error("--resume-real-data requires real-data options")
    if args.annotation_layer and not args.real_slice_options:
        parser.error("--annotation-layer requires real-data options")
    if args.size < 64 or args.size % 64:
        parser.error("--size must be a positive multiple of 64, at least 64")
    if args.child:
        child_run(args)
    elif args.real_slice_options:
        benchmark_real_data(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()

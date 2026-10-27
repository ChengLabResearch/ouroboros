#!/usr/bin/env python3
"""Compare partitions of recorded specimen annotations without fetching voxels.

Run from the Python environment, for example:
    python benchmarks/benchmark_partitioning.py --output /tmp/partitioning.json

The committed reduced Neuroglancer states regenerate the original full paths via
geometry_pipeline. Payloads come only from the production estimator. Timings are
partitioner wall time; this benchmark does not measure slicing runtime or RAM.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from math import prod
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np


PYTHON_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_ROOT))
DEFAULT_FIXTURES = PYTHON_ROOT / "test" / "fixtures" / "partitioning"


def load_specimen(name: str, fixture_dir: Path = DEFAULT_FIXTURES):
    """Regenerate real recorded geometry through the existing pipeline, offline."""
    from ouroboros.common.pipelines import geometry_pipeline
    from ouroboros.helpers.options import SliceOptions

    manifest = json.loads((fixture_dir / "manifest.json").read_text())
    specimen = manifest["specimens"][name]
    config = dict(specimen["slice_options"])
    if config["annotation_mip_level"] != config["output_mip_level"]:
        raise ValueError("Offline fixture regeneration requires equal annotation/output mip levels")
    state_path = fixture_dir / specimen["annotations_file"]
    recorded_hash = specimen["provenance"]["reduced_state_sha256"]
    if hashlib.sha256(state_path.read_bytes()).hexdigest() != recorded_hash:
        raise ValueError(f"Recorded annotation fixture changed: {state_path}")
    config["neuroglancer_json"] = str(state_path)
    # Required option fields are unused by geometry_pipeline, which writes no output.
    config.update(output_file_folder=".", output_file_name=name)
    options = SliceOptions(**config)
    pipeline, data = geometry_pipeline(options)
    data, error = pipeline.process(data)
    if error is not None:
        raise RuntimeError(error)
    return data.slice_rects, specimen


def _chunk_coverage(boxes, specimen):
    """Report source-clipped chunk references through CloudVolume's grid adapter."""
    from cloudvolume import Bbox
    from cloudvolume.datasource.precomputed.image.common import gridpoints
    from ouroboros.helpers.volume_cache import download_volume_geometry

    scale = specimen["source_scale"]
    chunk = np.asarray(scale["chunk_sizes"][0], dtype=np.int64)
    offset = np.asarray(scale["voxel_offset"], dtype=np.int64)
    bounds = Bbox(offset, offset + np.asarray(scale["size"], dtype=np.int64))
    # A metadata-only input adapter for the canonical download-extent conversion.
    # No CloudVolume instance is constructed, and no request is sent.
    metadata_view = SimpleNamespace(cv=SimpleNamespace(
        bounds=bounds, num_channels=specimen["channels"],
    ))
    item_bytes = np.dtype(specimen["source_dtype"]).itemsize * specimen["channels"]
    unique = set()
    references = decoded_total = clipped_total = max_references = max_decoded = 0

    def decoded_bytes(points):
        return sum(prod(int(v) for v in (
            np.minimum(bounds.minpt + (np.asarray(point) + 1) * chunk, bounds.maxpt)
            - (bounds.minpt + np.asarray(point) * chunk)
        )) * item_bytes for point in points)

    for box in boxes:
        _, clipped, _ = download_volume_geometry(metadata_view, box)
        clipped_total += prod(max(0, int(v)) for v in clipped.size3()) * item_bytes
        if any(v <= 0 for v in clipped.size3()):
            continue
        expanded = Bbox.clamp(clipped.expand_to_chunk_size(chunk, offset=offset), bounds)
        points = {tuple(int(v) for v in point) for point in gridpoints(expanded, bounds, chunk)}
        size = decoded_bytes(points)
        references += len(points)
        decoded_total += size
        max_references = max(max_references, len(points))
        max_decoded = max(max_decoded, size)
        unique.update(points)
    return {
        "basis": "Source-clipped chunk geometry; repeated references are counted before cache reuse. "
                 "Decoded logical bytes are not measured network traffic or RAM.",
        "chunk_shape": chunk.tolist(),
        "source_clipped_requested_bytes": clipped_total,
        "chunk_references": references,
        "unique_chunks": len(unique),
        "repeat_references": references - len(unique),
        "decoded_logical_bytes_with_repeats": decoded_total,
        "unique_decoded_logical_bytes": decoded_bytes(unique),
        "max_chunks_per_box": max_references,
        "max_decoded_logical_bytes_per_box": max_decoded,
    }


def partition_report(rects, boxes, mapping, specimen, *, include_chunks=True):
    """Serialize one-way evidence from production boxes and payload estimates."""
    from ouroboros.helpers.bounding_boxes import BoundingBox, slice_indices_are_contiguous
    from ouroboros.helpers.slice import estimate_slice_task_payload

    mapping = np.asarray(mapping)
    if mapping.shape != (len(rects),) or not np.issubdtype(mapping.dtype, np.integer):
        raise AssertionError("Each rectangle must have exactly one integer box assignment")
    if len(rects) and (mapping.min() < 0 or mapping.max() >= len(boxes)):
        raise AssertionError("Rectangle assignment is outside the returned boxes")
    rows = []
    config = specimen["slice_options"]
    shape = (config["slice_height"], config["slice_width"])
    for number, box in enumerate(boxes):
        indices = np.flatnonzero(mapping == number)
        if not len(indices):
            raise AssertionError("A returned box has no assigned rectangles")
        expected = BoundingBox.from_rects(rects[indices])
        if box.to_dict() != expected.to_dict():
            raise AssertionError("Returned bounds differ from assigned rectangle corner bounds")
        contiguous = slice_indices_are_contiguous(indices)
        run_count = 1 + int(np.count_nonzero(np.diff(indices) != 1))
        if contiguous != (run_count == 1):
            raise AssertionError("Report runs disagree with the production dispatch predicate")
        payload = estimate_slice_task_payload(
            box, len(indices), contiguous, slice_shape=shape,
            source_dtype=specimen["source_dtype"], channels=specimen["channels"],
        )
        rows.append({
            "box": number, "slice_count": len(indices), "runs": run_count,
            "first_slice": int(indices[0]), "last_slice": int(indices[-1]),
            "shape": list(box.get_shape()), "bounds": box.to_dict(),
            "payload_components_bytes": asdict(payload), "peak_bytes": payload.peak_bytes,
        })
    counts = [row["slice_count"] for row in rows]
    target = config["bounding_box_params"]["target_slices_per_box"]
    result = {
        "boxes": len(rows), "slices": len(rects),
        "slice_count_distribution": {
            "min": min(counts, default=0), "median": statistics.median(counts) if counts else 0,
            "p95": float(np.percentile(counts, 95)) if counts else 0,
            "max": max(counts, default=0), "boxes_over_target": sum(n > target for n in counts),
            "histogram": dict(sorted(Counter(counts).items())),
        },
        "max_payload_bytes": max((row["peak_bytes"] for row in rows), default=0),
        "cumulative_requested_source_bytes": sum(row["payload_components_bytes"]["source_bytes"] for row in rows),
        "component_maxima_bytes": {
            name: max(row["payload_components_bytes"][name] for row in rows)
            for name in rows[0]["payload_components_bytes"]
        } if rows else {},
        "largest_payload_box": max(rows, key=lambda row: row["peak_bytes"], default=None),
        "noncontiguous_boxes": sum(row["runs"] > 1 for row in rows),
        "maximum_runs_per_box": max((row["runs"] for row in rows), default=0),
        "total_runs": sum(row["runs"] for row in rows),
        "exactly_once_assignment_and_tight_corner_bounds": True,
        "mapping_sha256": hashlib.sha256(mapping.astype("<i8").tobytes()).hexdigest(),
        "partitions": rows,
    }
    if include_chunks:
        result["chunk_coverage"] = _chunk_coverage(boxes, specimen)
    return result


def benchmark_specimen(name, fixture_dir=DEFAULT_FIXTURES, repeats=1):
    from ouroboros.helpers.bounding_boxes import (
        _calculate_bounding_boxes_count, calculate_bounding_boxes_bsp_link_rects,
    )
    from ouroboros.helpers.slice import estimate_slice_task_payload

    rects, specimen = load_specimen(name, fixture_dir)
    original = rects.copy()
    config = specimen["slice_options"]
    params = config["bounding_box_params"]

    def task_payload(box, count, contiguous):
        return estimate_slice_task_payload(
            box, count, contiguous,
            slice_shape=(config["slice_height"], config["slice_width"]),
            source_dtype=specimen["source_dtype"], channels=specimen["channels"],
        ).peak_bytes

    results = {}
    for label, function, keywords in (
        ("count_reference", _calculate_bounding_boxes_count, {}),
        ("payload_partition", calculate_bounding_boxes_bsp_link_rects, {"task_payload": task_payload}),
    ):
        elapsed = []
        first_mapping = first_bounds = None
        for _ in range(repeats):
            started = time.perf_counter()
            boxes, mapping = function(rects, **params, **keywords)
            elapsed.append(time.perf_counter() - started)
            bounds = [box.to_dict() for box in boxes]
            if first_mapping is not None and (mapping != first_mapping or bounds != first_bounds):
                raise AssertionError("Repeated partitioning changed the assignments or bounds")
            first_mapping, first_bounds = mapping, bounds
            np.testing.assert_array_equal(rects, original)
        results[label] = partition_report(rects, boxes, mapping, specimen)
        results[label]["partition_elapsed_seconds"] = {
            "samples": elapsed, "median": statistics.median(elapsed),
            "scope": "Partitioner call only; excludes geometry generation and report construction. "
                     "Includes reference construction inside the payload partitioner.",
        }
    reference, selected = results["count_reference"], results["payload_partition"]
    geometry_hash = hashlib.sha256(rects.tobytes()).hexdigest()
    return {
        "specimen": name, "provenance": specimen,
        "generated_rectangles_sha256": geometry_hash,
        "matches_recorded_rectangles": geometry_hash == specimen["provenance"]["saved_rectangles_sha256"],
        "cumulative_requested_byte_cap_ratio": 1.0,
        "requested_byte_ratio": selected["cumulative_requested_source_bytes"] / reference["cumulative_requested_source_bytes"],
        "max_payload_ratio": selected["max_payload_bytes"] / reference["max_payload_bytes"],
        "input_rectangles_unchanged": True, "results": results,
    }


def main():
    from ouroboros.common.run_statistics import runtime_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--specimen", action="append", help="Recorded specimen name; repeat to select several.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each partitioner; also check deterministic output.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    names = args.specimen or list(manifest["specimens"])
    for name in names:
        if name not in manifest["specimens"]:
            parser.error(f"Unknown specimen: {name}")
    report = {
        "schema_version": 1,
        "scope": "Geometry-only evidence from real recorded annotations. No voxel downloads, interpolation, "
                 "slicing throughput, or measured worker/process-tree RAM. Payload excludes mappings, "
                 "CloudVolume internals, Python overhead, and concurrency.",
        "reference": "Unchanged count-based partitioner evaluated with the current inclusive integer download extent.",
        "payload_estimator": "ouroboros.helpers.slice.estimate_slice_task_payload",
        "requested_byte_basis": "Allocated inclusive source extent including padding and repeated box overlap; not source-clipped bytes.",
        "runtime": runtime_metadata(),
        "implementation_sha256": {
            str(path.relative_to(PYTHON_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                PYTHON_ROOT / "ouroboros/helpers/bounding_boxes.py",
                PYTHON_ROOT / "ouroboros/helpers/slice.py",
                PYTHON_ROOT / "ouroboros/pipeline/volume_cache_pipeline.py",
                PYTHON_ROOT / "ouroboros/pipeline/slice_parallel_pipeline.py",
            )
        },
        "versions": {name: metadata.version(name) for name in ("numpy", "scipy", "cloud-volume")},
        "fixture_manifest_sha256": hashlib.sha256((args.fixtures / "manifest.json").read_bytes()).hexdigest(),
        "specimens": [benchmark_specimen(name, args.fixtures, args.repeats) for name in names],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for record in report["specimens"]:
        print(f"{record['specimen']}: max payload {record['max_payload_ratio']:.3f} × reference; "
              f"requested bytes {record['requested_byte_ratio']:.3f} × reference")
    print(args.output)


if __name__ == "__main__":
    main()

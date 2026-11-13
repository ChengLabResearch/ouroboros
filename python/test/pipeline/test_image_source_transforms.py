"""Selected image placement reaches the shared geometry conversion once."""

from copy import deepcopy
import json

import numpy as np
import pytest

from ouroboros.common.pipelines import backproject_pipeline, geometry_pipeline
from ouroboros.helpers.options import BackprojectOptions
from ouroboros.helpers.coordinates import ImageSourceTransform
from ouroboros.pipeline.pipeline_input import PipelineInput


def _geometry(document, options, path):
    path.write_text(json.dumps(document), encoding="utf-8")
    return geometry_pipeline(options.model_copy(update={"neuroglancer_json": str(path)}))


def _dimensions(resolution_nm):
    return {axis: [scale / 1e9, "m"] for axis, scale in zip("xyz", resolution_nm)}


def _set_points(document, points):
    document["layers"][0]["annotations"] = [
        {"type": "point", "point": point.tolist()} for point in points
    ]


@pytest.mark.needs("slice-geometry")
@pytest.mark.parametrize("output_mip", [0, 1])
@pytest.mark.parametrize("linear,input_resolution,translation", [
    pytest.param(np.eye(3), None, (-5, 7, -3), id="translation-native-input-units"),
    pytest.param(np.diag((2, 0.5, 1.5)), None, (-5, 7, -3), id="scale-native-input-units"),
    pytest.param(np.eye(3) * 1e-5, None, (0, 0, 0), id="small-global-scale-retains-distinct-points"),
    pytest.param(np.array(((1, 0.25, 0), (0, 1.5, 0.2), (0.1, 0, 0.75))),
                 (150, 250, 450), (-5, 7, -3), id="mixed-axes-explicit-input-units"),
])
def test_image_placement_matches_geometry_after_rewriting_annotations(
    geometry_input, annotation_document, source_metadata, tmp_path,
    output_mip, linear, input_resolution, translation,
):
    native_resolution = np.array((100, 200, 300))
    annotation_resolution = np.array((700, 1100, 1300))
    output_dimensions = np.array((400, 500, 600))
    translation = np.asarray(translation)
    source_points = geometry_input.sample_points.copy()
    input_units = native_resolution if input_resolution is None else np.array(input_resolution)
    # Forward-place a known source path. The production inverse must recover it.
    global_points = ((source_points * input_units) @ linear.T
                     + translation * output_dimensions) / annotation_resolution
    _set_points(annotation_document, global_points)
    transform = {"matrix": np.column_stack((linear, translation)).tolist(),
                 "outputDimensions": _dimensions(output_dimensions)}
    if input_resolution is not None:
        transform["inputDimensions"] = _dimensions(input_resolution)
    annotation_document["layers"][1]["source"] = {"url": "memory://source", "transform": transform}
    options = geometry_input.slice_options
    options.output_mip_level = output_mip
    options.annotation_mip_level = 9  # Explicit units remain authoritative.
    options.slicing_params.use_adaptive_slicing = output_mip == 1

    rewritten = deepcopy(annotation_document)
    rewritten["layers"][1]["source"] = "memory://source"
    _set_points(rewritten, source_points * native_resolution / annotation_resolution)
    reference, reference_data = _geometry(rewritten, options, tmp_path / "rewritten.json")
    expected, error = reference.process(reference_data)
    assert error is None
    source_metadata.opened.clear()
    source_metadata.resolutions.clear()
    source_metadata.shape_requests.clear()

    pipeline, data = _geometry(annotation_document, options, tmp_path / "transformed.json")
    imported, normalized = [], []
    pipeline.steps[0].listen_for_progress(lambda progress: imported.append(
        (data.sample_points.copy(), data.sample_points_resolution_nm, data.image_source_transform))
        if progress == 1 else None)
    pipeline.steps[1].listen_for_progress(lambda progress: normalized.append(
        (data.sample_points.copy(), data.sample_points_resolution_nm, data.image_source_transform))
        if progress == 0.5 else None)

    actual, error = pipeline.process(data)

    assert error is None
    assert len(imported) == len(normalized) == 1
    np.testing.assert_array_equal(imported[0][0], global_points)
    np.testing.assert_allclose(imported[0][1], annotation_resolution)
    placement = imported[0][2]
    np.testing.assert_array_equal(placement.linear, linear)
    np.testing.assert_allclose(placement.translation_nm, translation * output_dimensions)
    if input_resolution is None:
        assert placement.input_resolution_nm is None
    else:
        np.testing.assert_allclose(placement.input_resolution_nm, input_resolution)
    np.testing.assert_allclose(normalized[0][0], source_points / 2 ** output_mip, rtol=1e-12)
    assert normalized[0][1:] == (None, None), "Published source voxels must carry no unconsumed placement"
    assert source_metadata.opened == ["memory://source"]
    assert set(source_metadata.resolutions) == ({output_mip} if input_resolution is None else {0, output_mip})
    assert len(source_metadata.resolutions) == len(set(source_metadata.resolutions)), "Read each required MIP once"
    assert source_metadata.shape_requests == [], "Physical placement must not use rounded volume sizes"
    np.testing.assert_allclose(actual.slice_rects, expected.slice_rects, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.annotation_points, expected.annotation_points, rtol=1e-10, atol=1e-10)
    assert actual.sample_points is actual.sample_points_resolution_nm is actual.image_source_transform is None


@pytest.mark.parametrize("input_dimensions", [False, True], ids=["translation", "input-units-only"])
def test_image_placement_without_annotation_units_fails_before_source_access(
    geometry_input, annotation_document, source_metadata, tmp_path, input_dimensions,
):
    annotation_document["layers"][0].pop("source")
    transform = {"outputDimensions": _dimensions((700, 700, 700))}
    if input_dimensions:
        transform["inputDimensions"] = _dimensions((150, 250, 450))
    else:
        transform["matrix"] = [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]]
    annotation_document["layers"][1]["source"] = {"url": "memory://source", "transform": transform}
    pipeline, data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "legacy.json")

    result, error = pipeline.process(data)

    assert result is None
    assert error == "Image source transforms require explicit annotation coordinate scales."
    assert source_metadata.opened == []
    np.testing.assert_array_equal(data.sample_points, geometry_input.sample_points)
    assert data.sample_points_resolution_nm is None
    assert data.image_source_transform is not None
    assert data.slice_rects is data.annotation_points is None
    assert pipeline.steps[0].timing["status"] == "succeeded"
    assert pipeline.steps[1].timing["status"] == "failed"


@pytest.mark.needs("slice-geometry")
@pytest.mark.parametrize("output_mip", [0, 1])
def test_identity_image_placement_preserves_legacy_mip_conversion(
    geometry_input, annotation_document, source_metadata, tmp_path, output_mip,
):
    annotation_document["layers"][0].pop("source")
    source_metadata.mip_sizes.update({0: np.array((21, 31, 41)), 1: np.array((11, 16, 21))})
    geometry_input.slice_options.output_mip_level = output_mip
    reference, reference_data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "plain.json")
    expected, error = reference.process(reference_data)
    assert error is None
    source_metadata.opened.clear()
    source_metadata.shape_requests.clear()
    annotation_document["layers"][1]["source"] = {"url": "memory://source", "transform": {
        "matrix": np.eye(3, 4).tolist(), "outputDimensions": _dimensions((700, 1100, 1300)),
    }}
    pipeline, data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "identity.json")
    imported = []
    pipeline.steps[0].listen_for_progress(lambda progress: imported.append(data.image_source_transform)
                                         if progress == 1 else None)

    actual, error = pipeline.process(data)

    assert error is None
    assert imported == [None]
    assert source_metadata.opened == ([] if output_mip == 0 else ["memory://source"])
    assert source_metadata.shape_requests == ([] if output_mip == 0 else [0, 1])
    assert source_metadata.resolutions == []
    np.testing.assert_array_equal(actual.slice_rects, expected.slice_rects)
    np.testing.assert_array_equal(actual.annotation_points, expected.annotation_points)


def test_native_resolution_failure_preserves_annotation_frame_and_image_placement(
    geometry_input, annotation_document, source_metadata, tmp_path,
):
    annotation_document["layers"][1]["source"] = {"url": "memory://source", "transform": {
        "outputDimensions": _dimensions((700, 700, 700)),
        "inputDimensions": _dimensions((150, 250, 450)),
    }}
    geometry_input.slice_options.output_mip_level = 1
    original_resolution = source_metadata.native.mip_resolution

    def unreadable_native_resolution(mip):
        if mip == 0:
            raise OSError("native mip 0 resolution unavailable")
        return original_resolution(mip)

    source_metadata.native.mip_resolution = unreadable_native_resolution
    pipeline, data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "metadata-failure.json")
    imported = []
    pipeline.steps[0].listen_for_progress(lambda progress: imported.append(data.image_source_transform)
                                         if progress == 1 else None)

    result, error = pipeline.process(data)

    assert result is None
    assert error == "Failed to get the voxel resolution from the volume: native mip 0 resolution unavailable"
    assert source_metadata.opened == ["memory://source"]
    np.testing.assert_array_equal(data.sample_points, geometry_input.sample_points)
    np.testing.assert_allclose(data.sample_points_resolution_nm, (700, 1100, 1300))
    assert len(imported) == 1 and data.image_source_transform is imported[0]
    assert data.slice_rects is data.annotation_points is None
    assert pipeline.steps[1].get_progress() == 0


def test_image_placement_survives_pipeline_input_json_round_trip(
    geometry_input, annotation_document, source_metadata, tmp_path,
):
    annotation_document["layers"][1]["source"] = {"url": "memory://source", "transform": {
        "matrix": [[1, 0.25, 0, -5], [0, 1, 0, 7], [0, 0, 2, -3]],
        "outputDimensions": _dimensions((400, 500, 600)),
        "inputDimensions": _dimensions((150, 250, 450)),
    }}
    pipeline, data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "placement.json")
    _, error = pipeline.steps[0].process(data)
    assert error is None

    restored = PipelineInput.from_json(data.to_json())

    assert isinstance(restored, PipelineInput)
    assert isinstance(restored.image_source_transform, ImageSourceTransform)
    assert restored.image_source_transform == data.image_source_transform
    assert set(json.loads(data.to_json())["image_source_transform"]) == {
        "linear", "translation_nm", "input_resolution_nm",
    }
    expected, error = pipeline.steps[1].process(data)
    assert error is None
    actual, error = pipeline.steps[1].process(restored)
    assert error is None
    np.testing.assert_array_equal(actual.slice_rects, expected.slice_rects)
    np.testing.assert_array_equal(actual.annotation_points, expected.annotation_points)
    assert actual.image_source_transform is actual.sample_points_resolution_nm is actual.sample_points is None


@pytest.mark.needs("slice-geometry")
@pytest.mark.parametrize("output_mip", [0, 1])
@pytest.mark.parametrize("voxel_offset", [(0, 0, 0), (100, 200, 300)], ids=["zero-offset", "nonzero-offset"])
def test_issue250_translation_uses_absolute_voxels_from_local_precomputed_source(
    geometry_input, annotation_document, tmp_path, output_mip, voxel_offset,
):
    source_folder = tmp_path / "volume"
    source_folder.mkdir()
    scales = [{"key": str(mip), "size": list(size), "resolution": [700 * 2 ** mip] * 3,
               "voxel_offset": [offset // 2 ** mip for offset in voxel_offset],
               "chunk_sizes": [[64, 64, 64]], "encoding": "raw"}
              for mip, size in enumerate(((15601, 15601, 9535), (7801, 7801, 4768)))]
    (source_folder / "info").write_text(json.dumps({
        "data_type": "uint8", "num_channels": 1, "type": "image", "scales": scales,
    }), encoding="utf-8")
    source_url = f"precomputed://file://{source_folder}"
    # Only the first point and translation published in issue 250 are used.
    first_point = np.array((3284.27, 11473.20, 8164.84))
    points = first_point + geometry_input.sample_points - geometry_input.sample_points[0]
    translation = np.array((-1699, -1700, -366))
    _set_points(annotation_document, points)
    annotation_document["layers"][0]["source"]["transform"]["outputDimensions"] = _dimensions((700, 700, 700))
    annotation_document["layers"][1]["source"] = {"url": source_url, "transform": {
        "matrix": np.column_stack((np.eye(3), translation)).tolist(),
        "outputDimensions": _dimensions((700, 700, 700)),
    }}
    geometry_input.slice_options.output_mip_level = output_mip
    rewritten = deepcopy(annotation_document)
    rewritten["layers"][1]["source"] = source_url
    _set_points(rewritten, points - translation)
    reference, reference_data = _geometry(rewritten, geometry_input.slice_options, tmp_path / "rewritten.json")
    expected, error = reference.process(reference_data)
    assert error is None
    pipeline, data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "translated.json")
    normalized = []
    pipeline.steps[1].listen_for_progress(lambda progress: normalized.append(data.sample_points.copy())
                                         if progress == 0.5 else None)

    actual, error = pipeline.process(data)

    assert error is None
    assert len(normalized) == 1
    np.testing.assert_allclose(normalized[0][0], np.array((4983.27, 13173.20, 8530.84)) / 2 ** output_mip,
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(actual.slice_rects, expected.slice_rects, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(actual.annotation_points, expected.annotation_points, rtol=1e-10, atol=1e-8)


@pytest.mark.needs("slice-geometry")
def test_backprojection_regenerates_the_same_placed_image_geometry(
    geometry_input, annotation_document, source_metadata, tmp_path, monkeypatch,
):
    annotation_document["layers"][1]["source"] = {"url": "memory://source", "transform": {
        "matrix": [[1, 0, 0, -5], [0, 1, 0, 7], [0, 0, 1, -3]],
        "outputDimensions": _dimensions((400, 500, 600)),
    }}
    geometry_input.slice_options.output_mip_level = 1
    geometry, geometry_data = _geometry(annotation_document, geometry_input.slice_options, tmp_path / "placed.json")
    expected, error = geometry.process(geometry_data)
    assert error is None
    backprojection, data = backproject_pipeline(BackprojectOptions(
        output_file_folder=str(tmp_path), output_file_name="backproject",
        straightened_volume_path=str(tmp_path / "unused.tif"), slice_options_path=str(tmp_path / "unused.json"),
    ), geometry_data.slice_options)
    consumed = []

    def consume_geometry(input_data):
        pipeline_input = tuple(input_data)[-1]
        consumed.append(pipeline_input.slice_rects.copy())
        assert pipeline_input.image_source_transform is pipeline_input.sample_points_resolution_nm is None

    # Keep the real parse and geometry steps; replace only the output consumer.
    monkeypatch.setattr(backprojection.steps[-1], "_process", consume_geometry)
    actual, error = backprojection.process(data)

    assert error is None
    assert len(consumed) == 1
    np.testing.assert_array_equal(consumed[0], expected.slice_rects)
    np.testing.assert_array_equal(actual.annotation_points, expected.annotation_points)

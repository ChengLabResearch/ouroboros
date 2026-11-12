"""Annotation MIP coordinates must reach real spline geometry in the output MIP."""

import json

import numpy as np
import pytest

from ouroboros.common.pipelines import geometry_pipeline
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.slices_geom_pipeline import SlicesGeometryPipelineStep


@pytest.mark.canary("slice-geometry")
def test_same_mip_geometry_does_not_open_the_source(geometry_input, source_metadata):
    result, error = Pipeline([SlicesGeometryPipelineStep()]).process(geometry_input)

    assert error is None, f"Same-MIP geometry failed: {error}"
    assert result is geometry_input
    assert source_metadata.opened == [], "Same-MIP geometry needs no source pyramid lookup"
    assert result.sample_points is None, "Completed geometry must release its input points"
    assert result.slice_rects.shape[1:] == (4, 3)
    assert len(result.slice_rects) > 1
    assert result.annotation_points.shape == (16, 3)
    assert np.isfinite(result.annotation_points).all()
    np.testing.assert_allclose(np.linalg.norm(result.slice_rects[:, 1] - result.slice_rects[:, 0], axis=1),
                               5, err_msg="Slice widths must remain in output-MIP voxels")
    np.testing.assert_allclose(np.linalg.norm(result.slice_rects[:, 3] - result.slice_rects[:, 0], axis=1),
                               7, err_msg="Slice heights must remain in output-MIP voxels")


@pytest.mark.needs("slice-geometry")
@pytest.mark.canary("annotation-mip-rescaling", case="coarser-output-equidistant")
@pytest.mark.parametrize("adaptive", [False, True], ids=["equidistant", "adaptive"])
@pytest.mark.parametrize("annotation_mip,output_mip,annotation_scale", [
    pytest.param(0, 1, (21 / 11, 31 / 16, 41 / 21), id="coarser-output"),
    pytest.param(1, 0, (11 / 21, 16 / 31, 21 / 41), id="finer-output"),
])
def test_annotation_mip_rescaling_matches_geometry_already_in_output_coordinates(
    geometry_input, source_metadata, adaptive, annotation_mip, output_mip, annotation_scale,
):
    # Odd pyramid dimensions give each axis a distinct ratio. This catches a
    # reversed conversion, axis swap, or assumed power-of-two scale.
    source_metadata.mip_sizes.update({0: np.array((21, 31, 41)), 1: np.array((11, 16, 21))})
    options = geometry_input.slice_options
    options.annotation_mip_level = options.output_mip_level = output_mip
    options.slicing_params.use_adaptive_slicing = adaptive
    output_points = geometry_input.sample_points.copy()
    annotations = geometry_input.model_copy(deep=True)
    annotations.slice_options.annotation_mip_level = annotation_mip
    annotations.sample_points *= annotation_scale
    original_annotations = annotations.sample_points.copy()

    expected, error = Pipeline([SlicesGeometryPipelineStep()]).process(geometry_input)
    assert error is None, f"Output-MIP reference geometry failed: {error}"
    assert source_metadata.opened == [], "The reference must not take the rescaling path"

    step = SlicesGeometryPipelineStep()
    published_points = []
    step.listen_for_progress(lambda progress: published_points.append(annotations.sample_points.copy())
                             if progress == 0.5 else None)
    actual, error = Pipeline([step]).process(annotations)

    assert error is None, f"Annotation MIP {annotation_mip} to output MIP {output_mip}: {error}"
    assert source_metadata.opened == ["memory://source"]
    assert source_metadata.shape_requests == [0, 1], "Rescaling must read the source pyramid sizes"
    assert len(published_points) == 1, "Rescaled points must be published before geometry completes"
    np.testing.assert_allclose(published_points[0], output_points, rtol=1e-12, atol=1e-12,
                               err_msg="Pipeline state must expose output-MIP coordinates")
    np.testing.assert_allclose(actual.slice_rects, expected.slice_rects, rtol=1e-10, atol=1e-10,
                               err_msg="Slice placement must use the rescaled annotations")
    np.testing.assert_allclose(actual.annotation_points, expected.annotation_points, rtol=1e-10, atol=1e-10,
                               err_msg="Straightened annotation positions must use the output-MIP spline")
    assert actual.sample_points is None, "Completed rescaling must preserve the input-release lifecycle"
    assert not np.allclose(original_annotations, output_points), "The fixture must require rescaling"


@pytest.mark.parametrize("missing,error", [
    pytest.param("slice_options", "Input data must contain a SliceOptions object.", id="missing-options"),
    pytest.param("sample_points", "Input data must contain an array of sample points.", id="missing-points"),
])
def test_geometry_reports_missing_inputs_before_source_access(geometry_input, source_metadata, missing, error):
    setattr(geometry_input, missing, None)
    result, actual_error = Pipeline([SlicesGeometryPipelineStep()]).process(geometry_input)

    assert result is None
    assert actual_error == error
    assert source_metadata.opened == [], "Invalid inputs must fail before opening a source"
    assert geometry_input.slice_rects is None
    assert geometry_input.annotation_points is None


@pytest.mark.parametrize("available_mips,fail_read,error", [
    pytest.param([], False, "Failed to get the mip sizes from the volume.", id="empty-pyramid"),
    pytest.param([0, 1], True, "Failed to get the mip sizes from the volume.", id="metadata-read-failure"),
    pytest.param([1], False, "The specified mip levels are not present in the volume.", id="missing-annotation-mip"),
    pytest.param([0], False, "The specified mip levels are not present in the volume.", id="missing-output-mip"),
])
def test_rescaling_reports_unusable_source_pyramids_without_publishing_geometry(
    geometry_input, source_metadata, available_mips, fail_read, error,
):
    geometry_input.slice_options.output_mip_level = 1
    source_metadata.native.available_mips = available_mips
    if fail_read:
        def unreadable_size(mip):
            raise OSError("Source pyramid metadata is unavailable")

        source_metadata.native.mip_volume_size = unreadable_size
    original_points = geometry_input.sample_points.copy()
    step = SlicesGeometryPipelineStep()
    result, actual_error = Pipeline([step]).process(geometry_input)

    assert result is None
    assert actual_error == error
    assert source_metadata.opened == ["memory://source"]
    assert geometry_input.slice_rects is None, "Failed rescaling must not publish slice rectangles"
    assert geometry_input.annotation_points is None, "Failed rescaling must not publish annotations"
    np.testing.assert_array_equal(geometry_input.sample_points, original_points,
                                  err_msg="Failed rescaling must retain the original annotation coordinates")
    assert step.timing["status"] == "failed"
    assert step.get_progress() == 0


@pytest.fixture
def annotation_document(geometry_input):
    return {
        "layers": [
            {"type": "annotation", "name": "annotations",
             "source": {"url": "local://annotations", "transform": {
                 "outputDimensions": {axis: [scale, "m"] for axis, scale in
                                      zip("xyz", (7e-7, 1.1e-6, 1.3e-6))}}},
             "annotations": [{"type": "point", "point": point.tolist()}
                             for point in geometry_input.sample_points]},
            {"type": "image", "name": "image", "source": "memory://source"},
        ],
    }


def _import_geometry(geometry_input, annotation_document, tmp_path):
    path = tmp_path / "annotation-coordinates.json"
    path.write_text(json.dumps(annotation_document), encoding="utf-8")
    geometry_input.slice_options.neuroglancer_json = str(path)
    return geometry_pipeline(geometry_input.slice_options)


@pytest.mark.needs("slice-geometry")
@pytest.mark.parametrize("output_mip,annotation_mip,resolution_nm,adaptive", [
    pytest.param(0, 0, (700, 1100, 1300), False, id="same-mip-physical-units"),
    pytest.param(1, 0, (700, 1100, 1300), True, id="coarser-output"),
    pytest.param(0, 1, (700, 1100, 1300), True, id="ignore-legacy-mip"),
    pytest.param(1, 9, (700, 1100, 1300), False, id="ignore-unavailable-legacy-mip"),
    pytest.param(0, 0, (100, 200, 300), False, id="already-source-units"),
])
def test_physical_annotation_import_matches_output_voxel_geometry(
    geometry_input, annotation_document, source_metadata, tmp_path,
    output_mip, annotation_mip, resolution_nm, adaptive,
):
    original_points = geometry_input.sample_points.copy()
    source_resolution = np.array((100, 200, 300)) * 2 ** output_mip
    expected_points = original_points * np.asarray(resolution_nm) / source_resolution
    reference = geometry_input.model_copy(deep=True)
    reference.sample_points = expected_points
    reference.slice_options.annotation_mip_level = reference.slice_options.output_mip_level = output_mip
    reference.slice_options.slicing_params.use_adaptive_slicing = adaptive
    expected, error = Pipeline([SlicesGeometryPipelineStep()]).process(reference)
    assert error is None
    assert source_metadata.opened == []

    geometry_input.slice_options.output_mip_level = output_mip
    geometry_input.slice_options.annotation_mip_level = annotation_mip
    geometry_input.slice_options.slicing_params.use_adaptive_slicing = adaptive
    annotation_document["layers"][0]["source"]["transform"]["outputDimensions"] = {
        axis: [scale / 1e9, "m"] for axis, scale in zip("xyz", resolution_nm)
    }
    pipeline, data = _import_geometry(geometry_input, annotation_document, tmp_path)
    imported, normalized = [], []
    pipeline.steps[0].listen_for_progress(lambda progress: imported.append(
        (data.sample_points.copy(), data.sample_points_resolution_nm)) if progress == 1 else None)
    pipeline.steps[1].listen_for_progress(lambda progress: normalized.append(
        (data.sample_points.copy(), data.sample_points_resolution_nm)) if progress == 0.5 else None)

    actual, error = pipeline.process(data)

    assert error is None
    assert len(imported) == len(normalized) == 1
    np.testing.assert_array_equal(imported[0][0], original_points)
    np.testing.assert_allclose(imported[0][1], resolution_nm, rtol=1e-15)
    np.testing.assert_allclose(normalized[0][0] * source_resolution,
                               original_points * resolution_nm, rtol=1e-12)
    assert normalized[0][1] is None, "Normalized points must no longer carry annotation units"
    assert source_metadata.opened == ["memory://source"]
    assert source_metadata.resolutions == [output_mip]
    assert source_metadata.shape_requests == [], "Physical coordinates must not use rounded volume sizes"
    np.testing.assert_allclose(actual.slice_rects, expected.slice_rects, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.annotation_points, expected.annotation_points, rtol=1e-10, atol=1e-10)
    assert actual.sample_points is actual.sample_points_resolution_nm is None


@pytest.mark.needs("slice-geometry")
@pytest.mark.parametrize("annotation_source", [None, {"url": "local://annotations", "transform": {}}])
def test_global_viewer_dimensions_do_not_change_legacy_import(
    geometry_input, annotation_document, source_metadata, tmp_path, annotation_source,
):
    annotation_document["layers"][0]["source"] = annotation_source
    annotation_document["dimensions"] = {axis: [7e-7, "m"] for axis in "xyz"}
    reference = geometry_input.model_copy(deep=True)
    expected, error = Pipeline([SlicesGeometryPipelineStep()]).process(reference)
    assert error is None
    pipeline, data = _import_geometry(geometry_input, annotation_document, tmp_path)

    actual, error = pipeline.process(data)

    assert error is None
    assert source_metadata.opened == []
    np.testing.assert_array_equal(actual.slice_rects, expected.slice_rects)
    np.testing.assert_array_equal(actual.annotation_points, expected.annotation_points)


@pytest.mark.parametrize("failure", ["missing-output-mip", "unreadable-resolution"])
def test_physical_import_keeps_original_coordinates_when_source_metadata_fails(
    geometry_input, annotation_document, source_metadata, tmp_path, failure,
):
    original_points = geometry_input.sample_points.copy()
    pipeline, data = _import_geometry(geometry_input, annotation_document, tmp_path)
    if failure == "missing-output-mip":
        source_metadata.native.available_mips = [1]
        result, error = pipeline.process(data)
        assert result is None
        assert error == "The specified output mip level is not present in the volume."
    else:
        def unreadable_resolution(mip):
            raise OSError("Source resolution metadata is unavailable")

        source_metadata.native.mip_resolution = unreadable_resolution
        with pytest.raises(OSError, match="Source resolution metadata is unavailable"):
            pipeline.process(data)
    assert source_metadata.opened == ["memory://source"]
    np.testing.assert_array_equal(data.sample_points, original_points)
    np.testing.assert_allclose(data.sample_points_resolution_nm, (700, 1100, 1300), rtol=1e-15)
    assert data.slice_rects is data.annotation_points is None
    assert pipeline.steps[1].timing["status"] == "failed"


@pytest.mark.parametrize("unsupported,error", [
    pytest.param("annotation-matrix", "Only identity annotation transform matrices", id="translated-annotation"),
    pytest.param("annotation-input-units", "Different annotation inputDimensions", id="different-input-units"),
    pytest.param("annotation-source", "Only local XYZ", id="nonlocal-annotation"),
    pytest.param("image-transform", "Image source transforms are not supported", id="transformed-image"),
    pytest.param("image-layer-transform", "Image layer transforms are not supported", id="transformed-image-layer"),
])
def test_unsupported_coordinate_frames_stop_before_source_access(
    geometry_input, annotation_document, source_metadata, tmp_path, unsupported, error,
):
    # Ignoring these inputs would apply otherwise valid scales in the wrong frame.
    annotation = annotation_document["layers"][0]["source"]
    if unsupported == "annotation-matrix":
        annotation["transform"]["matrix"] = [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]]
    elif unsupported == "annotation-input-units":
        annotation["transform"]["inputDimensions"] = {axis: [1e-6, "m"] for axis in "xyz"}
    elif unsupported == "annotation-source":
        annotation["url"] = "precomputed://remote-annotations"
    elif unsupported == "image-layer-transform":
        annotation_document["layers"][1]["transform"] = {"translation": [10, 0, 0]}
    else:
        annotation_document["layers"][1]["source"] = {
            "url": "memory://source", "transform": {"matrix": [[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]]},
        }
    pipeline, data = _import_geometry(geometry_input, annotation_document, tmp_path)

    result, actual_error = pipeline.process(data)

    assert result is None
    assert error in actual_error
    assert source_metadata.opened == []
    assert data.sample_points is data.sample_points_resolution_nm is None
    assert data.slice_rects is data.annotation_points is None

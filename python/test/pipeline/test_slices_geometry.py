"""Annotation MIP coordinates must reach real spline geometry in the output MIP."""

import numpy as np
import pytest

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

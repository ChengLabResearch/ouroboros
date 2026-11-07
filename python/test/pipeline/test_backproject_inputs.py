"""Backprojection TIFF pages retain their independent Z identity."""

import numpy as np
import pytest
import tifffile

from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline.backproject_pipeline import BackprojectPipelineStep
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput
from test.pipeline.test_backproject_geometry import source_metadata  # noqa: F401


def _run_backprojection(folder, shape, channels, photometric, *, chunk_size=5):
    folder.mkdir()
    depth, height, width = shape
    # Every source plane identifies its own Z, independently of export order.
    plane_values = ((np.arange(depth)[:, None] + 1)
                    * 10 ** np.arange(channels)[None, :]).astype(np.uint16)
    source_pixels = np.broadcast_to(plane_values[:, None, None, :],
                                   (*shape, channels)).copy()
    if channels == 1:
        source_pixels = source_pixels[..., 0]
    options = dict(photometric=photometric)
    if channels > 1:
        options["planarconfig"] = "contig"
    source = folder / "straightened.tif"
    tifffile.imwrite(source, source_pixels, **options)

    rects = np.array([
        [[0, 0, z], [width, 0, z], [width, height, z], [0, height, z]]
        for z in range(depth)
    ], dtype=float)
    slicing = SliceOptions(
        output_file_folder=str(folder), output_file_name="sample",
        neuroglancer_json="unused.json", slice_width=width, slice_height=height,
        output_mip_level=0, process_count=4,
    )
    backprojecting = BackprojectOptions(
        output_file_folder=str(folder), output_file_name="sample",
        straightened_volume_path=str(source), slice_options_path="unused.json",
        make_single_file=True, backprojection_compression="none",
        chunk_size=chunk_size, output_mip_level=0, process_count=4,
    )
    data = PipelineInput(
        slice_options=slicing, backproject_options=backprojecting,
        source_url="memory://source", slice_rects=rects,
    )
    result, error = Pipeline([BackprojectPipelineStep()]).process(data)
    assert error is None, error
    assert result is data

    output = folder / data.output_file_path
    pages = []
    with tifffile.TiffFile(output) as tif:
        assert len(tif.pages) == depth
        for z, page in enumerate(tif.pages):
            assert page.photometric == tifffile.PHOTOMETRIC[photometric.upper()]
            assert page.samplesperpixel == channels
            assert page.dtype == np.uint16
            actual = page.asarray().reshape(height + 1, width + 1, channels)
            expected = np.zeros((height + 1, width + 1, channels), dtype=np.uint16)
            # A 3-sample axis spans 0, 1.5, 3 and covers all four output voxels.
            expected[:] = plane_values[z]
            np.testing.assert_array_equal(actual, expected, err_msg=f"Output Z page {z}")
            pages.append(actual)
    return np.stack(pages)


@pytest.mark.parametrize("channels,photometric", [
    pytest.param(1, "minisblack", id="grayscale"),
    pytest.param(3, "rgb", id="rgb"),
    pytest.param(2, "minisblack", id="two-channel-minisblack"),
])
def test_file_backprojection_preserves_z_positions(
    tmp_path, source_metadata, channels, photometric,
):
    _run_backprojection(tmp_path / "file", (12, 3, 3),
                        channels, photometric, chunk_size=5)

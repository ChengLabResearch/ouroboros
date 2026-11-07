"""Backprojection accepts TIFF layouts without changing channel or Z identity."""

import numpy as np
import pytest
import tifffile

from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline.backproject_pipeline import BackprojectPipelineStep
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput
from test.pipeline.test_backproject_geometry import source_metadata  # noqa: F401


def _run_backprojection(folder, shape, channels, photometric, layout, *,
                        chunk_size=2, compression=None, input_is_output=False):
    folder.mkdir()
    depth, height, width = shape
    # Every source plane identifies its own Z, independently of export order.
    plane_values = ((np.arange(depth)[:, None] + 1)
                    * 10 ** np.arange(channels)[None, :]).astype(np.uint16)
    source_pixels = np.broadcast_to(plane_values[:, None, None, :],
                                   (*shape, channels)).copy()
    if channels == 1:
        source_pixels = source_pixels[..., 0]
    options = dict(photometric=photometric, compression=compression)
    if channels > 1:
        options["planarconfig"] = "contig"
    if layout == "folder":
        source = folder if input_is_output else folder / "straightened"
        if not input_is_output:
            source.mkdir()
        # Unpadded names and reversed creation order exercise numeric input order.
        for z in reversed(range(depth)):
            tifffile.imwrite(source / f"{z}.tif", source_pixels[z], **options)
    else:
        source = folder / "straightened.tif"
        pixels = source_pixels[0] if depth == 1 else source_pixels
        tifffile.imwrite(source, pixels, **options)
        if depth == 1 and compression is None:
            mapped = tifffile.memmap(source, mode="r")
            assert mapped.shape == source_pixels.shape[1:]
            del mapped

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
            # A 3-sample axis spans 0, 1.5, 3 and covers all four output voxels;
            # a singleton samples only the first endpoint of its two-voxel box.
            covered_height = 1 if height == 1 else height + 1
            covered_width = 1 if width == 1 else width + 1
            expected[:covered_height, :covered_width] = plane_values[z]
            np.testing.assert_array_equal(actual, expected, err_msg=f"Output Z page {z}")
            pages.append(actual)
    return np.stack(pages)


@pytest.mark.parametrize("channels,photometric", [
    pytest.param(1, "minisblack", id="grayscale"),
    pytest.param(3, "rgb", id="rgb"),
    pytest.param(2, "minisblack", id="two-channel-minisblack"),
])
def test_file_and_numeric_folder_backprojection_preserve_z_positions(
    tmp_path, source_metadata, channels, photometric,
):
    file_output = _run_backprojection(tmp_path / "file", (12, 3, 3),
                                     channels, photometric, "file", chunk_size=5)
    folder_output = _run_backprojection(tmp_path / "folder", (12, 3, 3),
                                       channels, photometric, "folder", chunk_size=5)
    np.testing.assert_array_equal(folder_output, file_output)


@pytest.mark.parametrize("shape,channels,photometric", [
    pytest.param((3, 3, 3), 1, "minisblack", id="grayscale-singleton-chunk-tails"),
    pytest.param((3, 3, 3), 3, "rgb", id="rgb-singleton-chunk-tails"),
    pytest.param((3, 3, 3), 2, "minisblack", id="two-channel-singleton-chunk-tails"),
    pytest.param((1, 3, 3), 1, "minisblack", id="grayscale-omitted-depth"),
    pytest.param((1, 3, 3), 3, "rgb", id="rgb-omitted-depth"),
    pytest.param((3, 1, 3), 2, "minisblack", id="two-channel-singleton-height"),
    pytest.param((3, 3, 1), 3, "rgb", id="rgb-singleton-width"),
    pytest.param((1, 1, 1), 1, "minisblack", id="single-voxel"),
])
def test_backprojection_preserves_singleton_input_and_chunk_axes(
    tmp_path, source_metadata, shape, channels, photometric,
):
    _run_backprojection(tmp_path / "run", shape, channels, photometric, "file")


@pytest.mark.parametrize("layout,channels,photometric", [
    pytest.param("file", 3, "rgb", id="compressed-rgb-file"),
    pytest.param("folder", 2, "minisblack", id="compressed-two-channel-folder"),
])
def test_compressed_backprojection_input_keeps_shape_and_photometric(
    tmp_path, source_metadata, layout, channels, photometric,
):
    _run_backprojection(tmp_path / "run", (3, 3, 3), channels, photometric,
                        layout, compression="zlib")


def test_backprojection_input_folder_can_also_hold_its_outputs(tmp_path, source_metadata):
    _run_backprojection(tmp_path / "run", (3, 3, 3), 3, "rgb", "folder",
                        input_is_output=True)

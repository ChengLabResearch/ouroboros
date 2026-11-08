"""Backprojection accepts TIFF layouts without changing channel or Z identity."""

from contextlib import ExitStack

import numpy as np
import pytest
import tifffile

from ouroboros.helpers.files import format_backproject_resave_volume
from ouroboros.helpers.options import BackprojectOptions, SliceOptions
from ouroboros.pipeline.backproject_pipeline import BackprojectPipelineStep
from ouroboros.pipeline.pipeline import Pipeline
from ouroboros.pipeline.pipeline_input import PipelineInput
from test.pipeline.test_backproject_geometry import source_metadata  # noqa: F401


def _run_backprojection(folder, shape, channels, photometric, layout, *,
                        chunk_size=2, compression=None, input_is_output=False,
                        input_file_name=None, single_file=True, expected_error=None):
    folder.mkdir()
    depth, height, width = shape
    # Every source plane identifies its own Z, independently of export order.
    plane_values = ((np.arange(depth)[:, None] + 1) * 10 ** np.arange(channels)[None, :]).astype(np.uint16)
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
            name = input_file_name if z == 0 and input_file_name else f"{z}.tif"
            tifffile.imwrite(source / name, source_pixels[z], **options)
    else:
        source = folder / (input_file_name or "straightened.tif")
        pixels = source_pixels[0] if depth == 1 else source_pixels
        tifffile.imwrite(source, pixels, **options)
        if depth == 1 and compression is None:
            mapped = tifffile.memmap(source, mode="r")
            assert mapped.shape == source_pixels.shape[1:]
            del mapped
    input_bytes = {path: path.read_bytes()
                   for path in (source.glob("*.tif") if source.is_dir() else [source])}

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
        make_single_file=single_file, backprojection_compression="none",
        chunk_size=chunk_size, output_mip_level=0, process_count=4,
    )
    data = PipelineInput(
        slice_options=slicing, backproject_options=backprojecting,
        source_url="memory://source", slice_rects=rects,
    )
    try:
        result, error = Pipeline([BackprojectPipelineStep()]).process(data)
    finally:
        for path, original in input_bytes.items():
            assert path.read_bytes() == original, f"Input changed: {path}"
    if expected_error is not None:
        assert result is None
        assert error == expected_error
        return
    assert error is None, error
    assert result is data
    if layout == "folder" or compression is not None:
        assert not (folder / format_backproject_resave_volume("sample")).exists()

    output = folder / data.output_file_path
    pages = []
    paths = [output] if single_file else [output / f"{z}.tif" for z in range(depth)]
    with ExitStack() as stack:
        tiffs = [stack.enter_context(tifffile.TiffFile(path)) for path in paths]
        output_pages = [page for tif in tiffs for page in tif.pages]
        assert len(output_pages) == depth
        for z, page in enumerate(output_pages):
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


@pytest.mark.parametrize("layout,compression", [("folder", None), ("file", "zlib")])
def test_backprojection_removes_temporary_input_with_per_slice_output(
    tmp_path, source_metadata, layout, compression,
):
    _run_backprojection(tmp_path / "run", (3, 3, 3), 3, "rgb", layout,
                        compression=compression, single_file=False)


def test_backprojection_retains_user_input_named_like_temporary_file(tmp_path, source_metadata):
    _run_backprojection(tmp_path / "run", (3, 3, 3), 3, "rgb", "file",
                        input_file_name=format_backproject_resave_volume("sample"))


@pytest.mark.parametrize("layout,compression", [("file", "zlib"), ("folder", None)])
def test_backprojection_rejects_temporary_path_over_input(
    tmp_path, source_metadata, layout, compression,
):
    _run_backprojection(
        tmp_path / "run", (3, 3, 3), 3, "rgb", layout,
        compression=compression, input_is_output=True,
        input_file_name=format_backproject_resave_volume("sample"),
        expected_error="Cannot prepare a temporary straightened TIFF over an input file.",
    )


def test_failed_export_retains_temporary_input(tmp_path, source_metadata, monkeypatch):
    set_stage = BackprojectPipelineStep.set_stage

    def fail_export(step, stage, *args, **kwargs):
        if stage == "exporting_tiff":
            raise OSError("injected export failure")
        return set_stage(step, stage, *args, **kwargs)

    monkeypatch.setattr(BackprojectPipelineStep, "set_stage", fail_export)
    folder = tmp_path / "run"
    with pytest.raises(OSError, match="injected export failure"):
        _run_backprojection(folder, (3, 3, 3), 3, "rgb", "file", compression="zlib")
    temporary = folder / format_backproject_resave_volume("sample")
    with tifffile.TiffFile(temporary) as tif:
        assert len(tif.pages) == 3

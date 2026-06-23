import json
import sys
import types

import numpy as np
import tifffile


def install_cloudvolume_import_stub():
    if "cloudvolume" in sys.modules:
        return

    cloudvolume = types.ModuleType("cloudvolume")
    cloudvolume.CloudVolume = object
    cloudvolume.VolumeCutout = object
    cloudvolume.Bbox = object
    sys.modules["cloudvolume"] = cloudvolume


install_cloudvolume_import_stub()

from ouroboros.pipeline.slice_parallel_pipeline import build_straightened_tiff_metadata  # noqa: E402


class FakeVolumeCache:
    def get_resolution_um(self):
        return np.array([0.5, 0.25, 1.5], dtype=np.float32)


ANNOTATION_POINTS_XYZ = np.array(
    [
        [4.5, 8.25, 0.0],
        [6.75, 9.5, 2.0],
    ],
    dtype=np.float32,
)


def straightened_metadata_kwargs():
    return build_straightened_tiff_metadata(
        volume_cache=FakeVolumeCache(),
        has_color_channels=False,
        num_color_channels=None,
        annotation_points=ANNOTATION_POINTS_XYZ,
    )


def read_first_page_metadata(path):
    with tifffile.TiffFile(path) as tif:
        return json.loads(tif.pages[0].tags["ImageDescription"].value)


def assert_annotation_points_metadata(metadata):
    assert metadata["spacing"] == 1.5
    assert metadata["unit"] == "um"
    assert metadata["annotation_points"] == ANNOTATION_POINTS_XYZ.tolist()

    first_x, first_y, first_z = metadata["annotation_points"][0]
    assert first_x == 4.5
    assert first_y == 8.25
    assert first_z == 0.0

    assert all(len(row) == 3 for row in metadata["annotation_points"])


def write_single_stack_tiff(tmp_path):
    stack_path = tmp_path / "straightened.tif"
    stack = tifffile.memmap(
        stack_path,
        shape=(3, 4, 5),
        dtype=np.uint8,
        **straightened_metadata_kwargs(),
    )
    stack[:] = np.arange(stack.size, dtype=np.uint8).reshape(stack.shape)
    stack.flush()
    del stack

    return stack_path


def write_directory_tiff(tmp_path):
    frame_dir = tmp_path / "straightened"
    frame_dir.mkdir()
    metadata_kwargs = straightened_metadata_kwargs()

    for frame_index in range(3):
        frame = np.full((4, 5), frame_index, dtype=np.uint8)
        tifffile.imwrite(
            frame_dir / f"{frame_index:02}.tif",
            frame,
            **metadata_kwargs,
        )

    return frame_dir


def test_single_stack_tiff_preserves_annotation_points_metadata(tmp_path):
    stack_path = write_single_stack_tiff(tmp_path)
    metadata = read_first_page_metadata(stack_path)

    assert_annotation_points_metadata(metadata)


def test_directory_tiff_output_preserves_annotation_points_metadata(tmp_path):
    frame_dir = write_directory_tiff(tmp_path)

    for frame_path in sorted(frame_dir.glob("*.tif")):
        metadata = read_first_page_metadata(frame_path)
        assert_annotation_points_metadata(metadata)

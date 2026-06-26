import sys
import types

import numpy as np


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


def test_straightened_tiff_metadata_includes_annotation_points_in_xyz_order():
    metadata = straightened_metadata_kwargs()["metadata"]

    assert metadata["spacing"] == np.float32(1.5)
    assert metadata["unit"] == "um"
    assert metadata["annotation_points"] == ANNOTATION_POINTS_XYZ.tolist()

    first_x, first_y, first_z = metadata["annotation_points"][0]
    assert first_x == 4.5
    assert first_y == 8.25
    assert first_z == 0.0

    assert all(len(row) == 3 for row in metadata["annotation_points"])


def test_straightened_tiff_metadata_preserves_existing_tiff_kwargs_shape():
    kwargs = straightened_metadata_kwargs()

    assert kwargs["software"] == "ouroboros"
    assert kwargs["photometric"] == "minisblack"
    assert kwargs["resolution"][-1] == "CENTIMETER"
    assert len(kwargs["resolution"]) == 3
    assert "resolutionunit" not in kwargs

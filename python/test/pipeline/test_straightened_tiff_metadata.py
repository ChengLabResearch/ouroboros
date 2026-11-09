"""Straightened TIFF metadata retains annotation XYZ order and physical units."""

import numpy as np
import pytest

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.volume_cache import CloudVolumeInterface, VolumeCache
from ouroboros.pipeline.slice_parallel_pipeline import build_straightened_tiff_metadata
from test.harness import make_rects


def test_straightened_tiff_metadata_preserves_annotations_and_physical_resolution(source_metadata):
    cache = VolumeCache([BoundingBox.from_rects(make_rects())], [0],
                        CloudVolumeInterface("memory://source"), mip=1)
    points = np.array([[4.5, 8.25, 0.0], [6.75, 9.5, 2.0]], dtype=np.float32)
    kwargs = build_straightened_tiff_metadata(
        volume_cache=cache, has_color_channels=False, num_color_channels=None, annotation_points=points,
    )
    metadata = kwargs["metadata"]

    assert metadata["spacing"] == pytest.approx(0.6), "Z spacing must remain in micrometers"
    assert metadata["unit"] == "um"
    assert metadata["annotation_points"] == [[4.5, 8.25, 0.0], [6.75, 9.5, 2.0]]
    # MIP 1 has 200 nm X and 400 nm Y voxels: 50,000 and 25,000 voxels/cm.
    assert kwargs["resolution"][:2] == pytest.approx([50_000, 25_000]), (
        "TIFF X/Y resolution must invert voxel spacing and convert micrometers to centimeters"
    )
    assert kwargs["resolution"][-1] == "CENTIMETER"
    assert len(kwargs["resolution"]) == 3
    assert "resolutionunit" not in kwargs
    assert kwargs["software"] == "ouroboros"
    assert kwargs["photometric"] == "minisblack"
    assert source_metadata.resolutions and set(source_metadata.resolutions) == {1}

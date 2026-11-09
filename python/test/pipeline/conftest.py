"""Fixtures shared by pipeline tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.helpers import volume_cache as cache_module
from ouroboros.pipeline.pipeline_input import PipelineInput
from test.harness import slice_options
from test.sample_data import generate_sample_curve_helix


@pytest.fixture
def source_metadata(monkeypatch):
    """Replace only cloud I/O; production source and cache adapters stay real."""
    opened, resolutions, shape_requests = [], [], []
    mip_sizes = {0: np.array((20, 30, 40)), 1: np.array((10, 15, 20))}

    def resolution(mip):
        resolutions.append(mip)
        return np.array((100, 200, 300)) * 2 ** mip

    def volume_size(mip):
        shape_requests.append(mip)
        return mip_sizes[mip]

    native = SimpleNamespace(
        shape=(20, 30, 40, 1), dtype=np.dtype("uint16"),
        available_mips=[0, 1], info={"type": "image"},
        cache=SimpleNamespace(enabled=True),
        mip_volume_size=volume_size,
        mip_resolution=resolution,
    )

    def open_source(url, **kwargs):
        opened.append(url)
        return native

    monkeypatch.setattr(cache_module, "CloudVolume", open_source)
    return SimpleNamespace(opened=opened, resolutions=resolutions, native=native,
                           mip_sizes=mip_sizes, shape_requests=shape_requests)


@pytest.fixture
def geometry_input(tmp_path):
    """A smooth, nondegenerate path for real spline fitting and slice geometry."""
    options = slice_options(tmp_path).model_copy(update={"slice_width": 5, "slice_height": 7})
    points = generate_sample_curve_helix(start_z=-1, end_z=1, radius=2, num_points=16) + 10
    return PipelineInput(slice_options=options, sample_points=points, source_url="memory://source")

"""Fixtures shared by pipeline tests."""

from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.helpers import volume_cache as cache_module


@pytest.fixture
def source_metadata(monkeypatch):
    opened, resolutions = [], []

    def resolution(mip):
        resolutions.append(mip)
        return np.array((100, 200, 300)) * 2 ** mip

    native = SimpleNamespace(
        shape=(20, 30, 40, 1), dtype=np.dtype("uint16"),
        available_mips=[0, 1], info={"type": "image"},
        cache=SimpleNamespace(enabled=True),
        mip_volume_size=lambda mip: np.array((20, 30, 40)) // 2 ** mip,
        mip_resolution=resolution,
    )

    def open_source(url, **kwargs):
        opened.append(url)
        return native

    monkeypatch.setattr(cache_module, "CloudVolume", open_source)
    return SimpleNamespace(opened=opened, resolutions=resolutions)

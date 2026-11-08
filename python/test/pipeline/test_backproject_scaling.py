"""Backprojection scales per source axis and sizes its TIFFs from the scaled output."""

import cv2
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from ouroboros.pipeline import backproject_pipeline as backproject_module
from ouroboros.pipeline.backproject_pipeline import calculate_scaling_factors, is_big_tiff

SOURCE = "memory://source"
# 4 GiB of single-byte samples: the largest output that still fits a classic TIFF.
AT_LIMIT = (2 ** 12, 2 ** 10, 2 ** 10)
HALF_LIMIT_UINT16 = (2 ** 10, 2 ** 10, 2 ** 10)


def _scaling(monkeypatch, sizes, current, target):
    def mip_sizes(url):
        assert url == SOURCE
        return sizes

    monkeypatch.setattr(backproject_module, "get_mip_volume_sizes", mip_sizes)
    return calculate_scaling_factors(SOURCE, current, target, (1, 1, 1))

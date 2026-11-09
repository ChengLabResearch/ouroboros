"""Test doubles and call adapters shared across test modules.

Each adapter is the one place that tracks a production signature, so changing
that signature means changing one test line.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ouroboros.helpers.files import write_conv_vol, write_raw_intermediate
from ouroboros.helpers.options import SliceOptions


class ContextExecutor:
    """Context-manager surface of an executor; subclasses supply submit."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def clock(values):
    return SimpleNamespace(perf_counter=iter(values).__next__)


def make_rects():
    return np.array([[[0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1]]], dtype=float)


def slice_options(tmp_path):
    return SliceOptions(output_file_folder=str(tmp_path), output_file_name="sample", neuroglancer_json="local.json",
                        slice_width=2, slice_height=2, process_count=4)


def write_intermediate_file(path: Path, indexes, values, weights, source_rows=2, target_rows=2):
    indexes = np.asarray(indexes, dtype=np.uint32)
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    meta = np.array(
        [source_rows, target_rows, 0, 0, np.atleast_2d(values).shape[0], len(indexes)],
        dtype=np.uint32,
    )
    type_ar = np.array([indexes.dtype.str, values.dtype.str, weights.dtype.str], dtype="S8")

    write_raw_intermediate(
        path,
        meta.tobytes(),
        type_ar.tobytes(),
        indexes.tobytes(),
        values.tobytes(),
        weights.tobytes(),
    )


def write_page(writer, source, target_folder, shape, *, output_shape=None, z_steps=1, dtype=np.uint16,
               index=0, interpolation=0, discrete=False):
    return write_conv_vol(writer, source, target_folder, shape, output_shape or shape, z_steps, dtype,
                          index, interpolation, discrete)

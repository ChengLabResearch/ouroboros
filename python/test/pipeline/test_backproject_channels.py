"""Multichannel TIFFs retain every channel through chunk records and TIFF output."""

import numpy as np
import pytest
import tifffile

from ouroboros.helpers.bounding_boxes import BoundingBox
from ouroboros.helpers.files import (
    generate_tiff_write,
    load_raw_file_intermediate,
    validate_straightened_volume_for_backprojection,
    write_conv_vol,
)
from ouroboros.helpers.shapes import ImgSliceC
from ouroboros.pipeline.backproject_pipeline import process_chunk


def _backproject_tiff(folder, data, rects, photometric, *, discrete, binary, scaling):
    folder.mkdir()
    source = folder / "straightened.tif"
    channels = data.shape[-1] if data.ndim == 4 else 1
    tifffile.imwrite(source, data, photometric=photometric,
                     **({"planarconfig": "contig"} if channels > 1 else {}))
    info = validate_straightened_volume_for_backprojection(str(source), rects)
    assert info.channels == channels
    intermediates = folder / "intermediates"
    box = BoundingBox.from_rects(rects)
    chunk_pages = []
    # Both chunks contain several source planes and contribute to common Z pages.
    for chunk_index, start in enumerate((0, 2)):
        _, _, _, pages = process_chunk(
            intermediates, info, rects[start:start + 2],
            (slice(start, start + 2), slice(None), slice(None)),
            (chunk_index, 0, 0), box, intermediate_threads=1,
        )
        chunk_pages.append(set(pages))
    assert chunk_pages[0] & chunk_pages[1]

    records = sorted(intermediates.glob("i_*.dat"))
    assert len(records) > 1
    appended = False
    for path in records:
        record_count = 0
        with path.open("rb") as handle:
            while handle.tell() < path.stat().st_size:
                indices, values, weights = load_raw_file_intermediate(handle)
                assert values.shape == (channels, len(indices))
                assert weights.shape == indices.shape
                record_count += 1
        appended |= record_count > 1
    assert appended

    target = folder / "output"
    target.mkdir()
    writer = generate_tiff_write(
        tifffile.imwrite, None, np.ones(3), None,
        photometric="rgb" if channels > 1 else "minisblack",
    )
    x, y, _ = box.get_shape()
    for path in records:
        write_conv_vol(
            writer, path, ImgSliceC(Y=y, X=x, C=channels),
            bool if binary else np.uint16, scaling, target,
            int(path.stem.removeprefix("i_")), 0, discrete, merge_threads=1,
        )
    assert not list(intermediates.glob("*.dat"))
    return {path.name: tifffile.imread(path) for path in target.glob("*.tif")}


@pytest.mark.parametrize(
    "photometric,discrete,binary,scaling",
    [
        pytest.param("rgb", False, False, None, id="rgb"),
        pytest.param("minisblack", False, False, None, id="minisblack"),
        pytest.param("rgb", True, False, None, id="discrete"),
        pytest.param("rgb", False, True, None, id="binary"),
        pytest.param("rgb", False, False, (2, 2, 2), id="scaled"),
    ],
)
def test_multichannel_backprojection_matches_scalar_tiffs(
    tmp_path, photometric, discrete, binary, scaling,
):
    depth, row, column = np.indices((4, 3, 4))
    data = np.stack((np.zeros_like(depth), 1 + 4 * depth + 2 * row + column,
                     20 + 3 * depth + 5 * row + 2 * column), axis=-1).astype(np.uint16)
    rects = np.array([
        [[0.25, 0.25, z], [4.25, 0.25, z],
         [4.25, 3.25, z], [0.25, 3.25, z]]
        for z in (0.25, 1.25, 1.75, 2.25)
    ])
    options = dict(discrete=discrete, binary=binary, scaling=scaling)
    actual = _backproject_tiff(tmp_path / "channels", data, rects, photometric, **options)
    expected = [
        _backproject_tiff(tmp_path / f"scalar-{channel}", data[..., channel],
                          rects, "minisblack", **options)
        for channel in range(3)
    ]
    assert actual.keys() == expected[0].keys() == expected[1].keys() == expected[2].keys()
    assert len(actual) > 1
    for name, page in actual.items():
        assert page.shape[-1] == 3
        assert page.dtype == (np.uint8 if binary else np.uint16)
        assert not np.any(page[..., 0])
        for channel in range(3):
            scalar_page = expected[channel][name]
            # The scalar writer retains a singleton sample axis before scaling.
            if scalar_page.ndim == 3:
                scalar_page = scalar_page[..., 0]
            np.testing.assert_array_equal(page[..., channel], scalar_page)
    assert any(np.any(page[..., 1]) for page in actual.values())
    assert any(np.any(page[..., 2]) for page in actual.values())

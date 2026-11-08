from contextlib import contextmanager
import errno
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ouroboros.helpers import files as files_mod
from ouroboros.helpers.files import (
    format_backproject_output_file,
    format_backproject_output_multiple,
    format_backproject_resave_volume,
    format_backproject_tempvolumes,
    format_slice_output_config_file,
    format_slice_output_file,
    format_slice_output_multiple,
    format_tiff_name,
    get_sorted_tif_files,
    join_path,
    combine_unknown_folder,
    num_digits_for_n_files,
    parse_tiff_name,
    np_convert,
    generate_tiff_write,
    expected_straightened_volume_shape,
    inspect_straightened_volume,
    ravel_map_2d,
    increment_volume,
    load_raw_file_intermediate,
    validate_straightened_volume_for_backprojection,
    volume_from_intermediates,
    write_conv_vol,
    write_raw_intermediate
)
from ouroboros.helpers.shapes import ImgSlice, ImgSliceC


def test_get_sorted_tif_files(tmp_path):
    import random

    # Generate some sample tiff files
    filenames = [Path(tmp_path, f"{i:03d}.tif") for i in range(101)]

    # Ensure both .tif and .tiff files are handled.
    for i in [random.randrange(101) for j in range(35)]:
        filenames[i] = filenames[i].with_suffix('.tiff')

    original_filenames = [path.name for path in filenames]

    # Randomize the order
    random.shuffle(filenames)

    # Create the files
    for path in filenames:
        path.touch()

    sorted_files = get_sorted_tif_files(str(tmp_path))
    assert sorted_files == original_filenames


def test_get_sorted_tif_files_prefix(tmp_path):
    import random

    # Generate some sample tiff files
    filenames = [Path(tmp_path, f"slice_{i:03d}.tif") for i in range(101)]

    # Ensure both .tif and .tiff files are handled.
    for i in [random.randrange(101) for j in range(35)]:
        filenames[i] = filenames[i].with_suffix('.tiff')

    original_filenames = [path.name for path in filenames]

    # Randomize the order
    random.shuffle(filenames)

    # Create the files
    for path in filenames:
        path.touch()

    sorted_files = get_sorted_tif_files(str(tmp_path))
    assert sorted_files == original_filenames


def test_get_sorted_tif_files_unpadded_indices(tmp_path):
    expected = [f"{index}.tif" for index in (0, 1, 2, 10, 100, 99999, 100000)]
    for name in reversed(expected):
        (tmp_path / name).touch()

    assert get_sorted_tif_files(str(tmp_path)) == expected


def test_get_sorted_tif_files_equal_indices_use_filename_order(tmp_path):
    expected = ["00001.tif", "1.tif", "1.tiff", "2.tif"]
    for name in reversed(expected):
        (tmp_path / name).touch()

    assert get_sorted_tif_files(str(tmp_path)) == expected


@pytest.mark.parametrize("names", [
    ["slice_2.tif", "slice_10.tif", "slice_1.tiff"],
    ["2.tif", "10.tif", "slice_1.tiff"],
])
def test_get_sorted_tif_files_preserves_legacy_nonnumeric_order(tmp_path, names):
    for name in names:
        (tmp_path / name).touch()

    assert get_sorted_tif_files(str(tmp_path)) == sorted(names)


def test_join_path():
    path = join_path("folder", "subfolder", "file.txt")
    assert path == os.path.join("folder", "subfolder", "file.txt")


def test_combine_unknown_folder_posix_no_slash():
    directory = "/path/to/directory"
    filename = "file.txt"
    combined = combine_unknown_folder(directory, filename)
    assert combined == "/path/to/directory/file.txt"


def test_combine_unknown_folder_posix_slash():
    directory = "/path/to/directory/"
    filename = "file.txt"
    combined = combine_unknown_folder(directory, filename)
    assert combined == "/path/to/directory/file.txt"


def test_combine_unknown_folder_win_no_slash():
    directory = "C:\\path\\to\\directory"
    filename = "file.txt"
    combined = combine_unknown_folder(directory, filename)
    assert combined == "C:\\path\\to\\directory\\file.txt"


def test_combine_unknown_folder_win_slash():
    directory = "C:\\path\\to\\directory\\"
    filename = "file.txt"
    combined = combine_unknown_folder(directory, filename)
    assert combined == "C:\\path\\to\\directory\\file.txt"


def test_format_slice_output_file():
    result = format_slice_output_file("test")
    assert isinstance(result, str)


def test_format_slice_output_multiple():
    result = format_slice_output_multiple("test")
    assert isinstance(result, str)


def test_format_slice_output_config_file():
    result = format_slice_output_config_file("test")
    assert isinstance(result, str)


def test_format_backproject_output_file():
    result = format_backproject_output_file("test")
    assert isinstance(result, str)
    assert result == "test-backprojected.tif"

    result = format_backproject_output_file("test", offset=(1, 2, 3))
    assert isinstance(result, str)
    assert result == "test-backprojected-1-2-3.tif"


def test_format_backproject_output_multiple():
    result = format_backproject_output_multiple("test")
    assert isinstance(result, str)
    assert result == "test-backprojected"

    result = format_backproject_output_multiple("test", offset=(1, 2, 3))
    assert isinstance(result, str)
    assert result == "test-backprojected-1-2-3"


def test_format_backproject_tempvolumes():
    result = format_backproject_tempvolumes("test")
    assert isinstance(result, str)


def test_format_backproject_resave_volume():
    result = format_backproject_resave_volume("test")
    assert isinstance(result, str)


def test_format_tiff_name():
    result = format_tiff_name(1, 3)
    assert isinstance(result, str)
    assert result == f"{str(1).zfill(3)}.tif"


def test_parse_tiff_name():
    result = parse_tiff_name("001.tif")
    assert isinstance(result, int)
    assert result == 1


def test_num_digits_for_n_files():
    result = num_digits_for_n_files(100)
    assert isinstance(result, int)
    assert result == 2


def test_generate_tiff_write(tmp_path):
    micron_resolution = np.array([0.7, 0.7, 0.7])
    backprojection_offset = (55, 44, 77)
    compression = "zlib"
    temp_file = tmp_path.joinpath("foot.tiff")

    import tifffile as tf
    import json
    write_func = tf.imwrite

    tiff_write = generate_tiff_write(write_func, compression, micron_resolution, backprojection_offset)
    tiff_write(temp_file, np.arange(0, 64).reshape(8, 8))

    with tf.TiffFile(temp_file) as tif:
        print(tif.pages[0].tags)
        assert tif.pages[0].tags["ResolutionUnit"].value == tf.RESUNIT.CENTIMETER
        assert tif.pages[0].tags["Software"].value == "ouroboros"
        assert tif.pages[0].tags["Compression"].value == tf.COMPRESSION.ADOBE_DEFLATE
        assert tif.pages[0].tags["XResolution"].value == (4294967295, 300648)
        assert tif.pages[0].tags["YResolution"].value == (4294967295, 300648)
        json_metadata = json.loads(tif.pages[0].tags["ImageDescription"].value)
        assert json_metadata["spacing"] == 0.7
        assert json_metadata["unit"] == "um"
        assert json_metadata["backprojection_offset_min_xyz"] == [55, 44, 77]


def _backproject_slice_rects(width=4, height=5, count=3):
    return np.array(
        [
            [[0, 0, z], [width, 0, z], [width, height, z], [0, height, z]]
            for z in range(count)
        ],
        dtype=np.float32,
    )


def _write_test_tiff_stack(path: Path, shape=(3, 5, 4), dtype=np.uint8):
    import tifffile as tf

    with tf.TiffWriter(path) as tif:
        for z in range(shape[0]):
            tif.write(np.full(shape[1:], z, dtype=dtype), compression=None)


def test_expected_straightened_volume_shape_uses_duv_from_slice_rects():
    expected = expected_straightened_volume_shape(_backproject_slice_rects())

    assert expected.D == 3
    assert expected.V == 5
    assert expected.U == 4


def test_expected_straightened_volume_shape_requires_slice_config():
    with pytest.raises(ValueError, match="Missing slice config"):
        expected_straightened_volume_shape(np.empty((0, 4, 3)))


def test_validate_straightened_volume_accepts_plugin_mask_stack(tmp_path):
    straightened_volume_path = tmp_path.joinpath("plugin-output.tif")
    _write_test_tiff_stack(straightened_volume_path)

    info = validate_straightened_volume_for_backprojection(
        str(straightened_volume_path), _backproject_slice_rects()
    )

    assert info.shape.D == 3
    assert info.shape.V == 5
    assert info.shape.U == 4
    assert info.dtype == np.dtype(np.uint8)
    assert info.channels == 1
    assert not info.is_compressed


def test_validate_straightened_volume_accepts_directory_of_plugin_masks(tmp_path):
    import tifffile as tf

    straightened_volume_folder = tmp_path.joinpath("plugin-output")
    straightened_volume_folder.mkdir()
    for z in range(3):
        tf.imwrite(
            straightened_volume_folder.joinpath(f"{z:02}.tif"),
            np.full((5, 4), z, dtype=np.uint8),
            compression=None,
        )

    info = validate_straightened_volume_for_backprojection(
        str(straightened_volume_folder), _backproject_slice_rects()
    )

    assert info.sample_path.name == "00.tif"
    assert info.shape.D == 3
    assert info.dtype == np.dtype(np.uint8)


def test_validate_straightened_volume_distinguishes_shape_mismatch(tmp_path):
    straightened_volume_path = tmp_path.joinpath("plugin-output.tif")
    _write_test_tiff_stack(straightened_volume_path, shape=(2, 5, 4))

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_straightened_volume_for_backprojection(
            str(straightened_volume_path), _backproject_slice_rects()
        )


def test_validate_straightened_volume_distinguishes_dtype_mismatch(tmp_path):
    straightened_volume_path = tmp_path.joinpath("plugin-output.tif")
    _write_test_tiff_stack(straightened_volume_path, dtype=np.float32)

    with pytest.raises(TypeError, match="dtype mismatch"):
        validate_straightened_volume_for_backprojection(
            str(straightened_volume_path), _backproject_slice_rects()
        )


def test_inspect_straightened_volume_distinguishes_empty_directory(tmp_path):
    straightened_volume_folder = tmp_path.joinpath("empty")
    straightened_volume_folder.mkdir()

    with pytest.raises(ValueError, match="contains no TIFF files"):
        inspect_straightened_volume(str(straightened_volume_folder))


def test_inspect_straightened_volume_distinguishes_bad_tiff_pages(tmp_path, monkeypatch):
    class FakeTiff:
        def __init__(self, pages):
            self.pages = pages

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(files_mod.tifffile, "TiffFile", lambda _path: FakeTiff([]))
    with pytest.raises(ValueError, match="contains no pages"):
        inspect_straightened_volume(str(tmp_path.joinpath("empty-pages.tif")))

    one_dim_page = SimpleNamespace(
        shape=(4,),
        dtype=np.dtype(np.uint8),
        compression=files_mod.tifffile.COMPRESSION.NONE,
    )
    monkeypatch.setattr(files_mod.tifffile, "TiffFile", lambda _path: FakeTiff([one_dim_page]))
    with pytest.raises(ValueError, match="at least two-dimensional"):
        inspect_straightened_volume(str(tmp_path.joinpath("one-dimensional.tif")))


def test_ravel_map_2d():
    offset = ((60, ), (40, ))
    source_rows = 20
    target_rows = 60
    source_coords = np.random.randint(0, 20, 200).reshape(2, 100)
    mapped_coords = source_coords + ((60, ), (40, ))

    raveled_source = np.ravel_multi_index(source_coords, (20, 20))
    raveled_mapped = np.ravel_multi_index(mapped_coords, (80, 60))
    result = ravel_map_2d(raveled_source, source_rows, target_rows, offset)

    assert np.all(result == raveled_mapped)


def test_write_intermediate(tmp_path):
    sample_path = Path(tmp_path, "inter.tif")
    offset = ((np.uint32(60), ), (np.uint32(40), ))
    source_coords = np.random.randint(0, 20, 200).reshape(2, 100)
    mapped_coords = source_coords + offset
    raveled_source = np.ravel_multi_index(source_coords, (20, 20)).astype(np.uint32)
    raveled_mapped = np.ravel_multi_index(mapped_coords, (80, 60)).astype(np.uint32)
    source_values = np.random.rand(100).astype(np.float32)
    source_weights = np.random.rand(100).astype(np.float32)

    offset_dict = {
        "source_rows": np.uint32(20),
        "target_rows": np.uint32(60),
        "offset_columns": np.uint32(60),
        "offset_rows": np.uint32(40),
        "channels": np.uint32(1)
    }
    type_ar = np.array([raveled_source.dtype.str, source_values.dtype.str, source_weights.dtype.str], dtype='S8')

    write_raw_intermediate(sample_path,
                           np.fromiter(offset_dict.values(), dtype=np.uint32, count=5).tobytes(),
                           np.uint32(len(raveled_source)).tobytes(),
                           type_ar.tobytes(),
                           raveled_source.tobytes(),
                           source_values.tobytes(),
                           source_weights.tobytes())

    indicies, values, weights = load_raw_file_intermediate(open(sample_path, "rb"))

    assert len(indicies) == 100
    assert values.shape == (1, 100)
    assert values.dtype == np.float32
    assert len(weights) == 100
    assert weights.dtype == np.float32
    assert np.all(values[0] == source_values)
    assert np.all(weights == source_weights)
    assert np.all(indicies == raveled_mapped)


def test_increment_volume(tmp_path):
    sample_path = Path(tmp_path, "inter.tif")
    offset = ((np.uint32(60), ), (np.uint32(40), ))
    source_coords = np.array([[5, 5, 5], [12, 7, 12]], dtype=np.uint32)

    raveled_source = np.ravel_multi_index(source_coords, (20, 20)).astype(np.uint32)
    mapped_source = ravel_map_2d(raveled_source, 20, 60, offset)
    source_values = np.random.rand(3).astype(np.float32)
    source_weights = np.random.rand(3).astype(np.float32)

    offset_dict = {
        "source_rows": np.uint32(20),
        "target_rows": np.uint32(60),
        "offset_columns": np.uint32(60),
        "offset_rows": np.uint32(40),
        "channels": np.uint32(1)
    }
    type_ar = np.array([raveled_source.dtype.str, source_values.dtype.str, source_weights.dtype.str], dtype='S8')

    write_raw_intermediate(sample_path,
                           np.fromiter(offset_dict.values(), dtype=np.uint32, count=5).tobytes(),
                           np.uint32(len(raveled_source)).tobytes(),
                           type_ar.tobytes(),
                           raveled_source.tobytes(),
                           source_values.tobytes(),
                           source_weights.tobytes())

    volume = np.zeros((2, 80 * 60))
    increment_volume(sample_path, volume[:], cleanup=True)

    assert volume[0, mapped_source[1]] == source_values[1]
    assert np.allclose(volume[0, mapped_source[0]], np.sum(source_values[[0, 2]]))
    assert volume[1, mapped_source[1]] == source_weights[1]
    assert np.allclose(volume[1, mapped_source[0]], np.sum(source_weights[[0, 2]]))
    assert np.all(np.nonzero(volume)[0] == np.array([0, 0, 1, 1]))
    assert np.all(np.nonzero(volume)[1] == np.array([3947, 3952, 3947, 3952]))

    assert not sample_path.exists()


def test_np_convert_from_int():
    base = np.random.randint(0, 10, 6400).reshape(80, 80)

    # Direct Conversion
    assert np.all(np_convert(np.float32, base, normalize=False) == base.astype(np.float32))

    # Normalized Conversion
    np.testing.assert_allclose(
        np_convert(np.float32, base),
        base.astype(np.float32) / (np.max(base) - np.min(base)),
    )

    # Safe Bool
    safe_bool = np_convert(bool, base, safe_bool=True)
    assert safe_bool.dtype == np.uint8
    assert np.all(safe_bool == (base > 0))

    # Unsafe Bool - Raw bool datatype
    safe_bool = np_convert(bool, base, safe_bool=False)
    assert safe_bool.dtype == bool
    assert np.all(safe_bool == (base > 0))


def test_np_convert_from_float():
    base = np.random.randint(0, 16, 6400).reshape(80, 80) * np.random.rand(80, 80)
    float_data = np.linspace(0, 1, 16)

    # Direct Conversion
    assert np.all(np_convert(np.uint16, float_data, normalize=False) == [0] * 15 + [1])

    # Safe Bool
    safe_bool = np_convert(bool, base, safe_bool=True)
    assert safe_bool.dtype == np.uint8
    assert np.all(safe_bool == (base > 0))

    # Unsafe Bool - Raw bool datatype
    safe_bool = np_convert(bool, base, safe_bool=False)
    assert safe_bool.dtype == bool
    assert np.all(safe_bool == (base > 0))

    # Normalized Conversion; slight inaccuracy introduced for type safety
    target = np.arange(0, np.iinfo(np.uint16).max + 1, (np.iinfo(np.uint16).max) // 15)
    target[1:] = target[1:] - 1
    assert np.all(np_convert(np.uint16, float_data) == target)


def test_np_convert_preset_and_zero_range_branches():
    constant = np.array([5.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(np_convert(np.float32, constant), np.array([0.0, 0.0], dtype=np.float32))

    preset = np_convert(
        np.float32,
        np.array([10.0, 15.0], dtype=np.float32),
        preset_min=10.0,
        preset_max=20.0,
    )
    np.testing.assert_allclose(preset, np.array([0.0, 0.5], dtype=np.float32))

    shifted = np_convert(
        np.uint8,
        np.array([-2, 0, 2], dtype=np.int16),
        normalize=False,
        preset_min=-2,
    )
    np.testing.assert_array_equal(shifted, np.array([0, 2, 4], dtype=np.uint8))


def _write_intermediate_file(path: Path, indexes, values, weights, source_rows=2, target_rows=2):
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX vectored append")
def test_raw_append_preserves_record_bytes_in_one_unlocked_writev(tmp_path, monkeypatch):
    import fcntl

    path = tmp_path / "i_00000.dat"
    indexes = np.arange(3, dtype=np.uint32)
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    weights = np.ones(3, dtype=np.float32)
    segments = (
        np.array([3, 3, 0, 0, 2], dtype=np.uint32).tobytes(),
        np.uint32(3).tobytes(),
        np.array([indexes.dtype.str, values.dtype.str, weights.dtype.str], dtype="S8").tobytes(),
        indexes, values, weights,
    )
    legacy_record = b"".join(
        entry.tobytes() if isinstance(entry, np.ndarray) else entry for entry in segments
    )
    path.write_bytes(legacy_record)
    calls = []
    original_writev = os.writev

    def record_writev(fd, buffers):
        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND
        assert buffers[4].obj is values
        calls.append(fd)
        return original_writev(fd, buffers)

    def unexpected_lock(*args, **kwargs):
        pytest.fail("POSIX record append must not acquire a file lock")

    monkeypatch.setattr(files_mod.os, "writev", record_writev)
    monkeypatch.setattr(files_mod, "FileLock", unexpected_lock)
    assert write_raw_intermediate(path, *segments) == len(legacy_record)
    assert len(calls) == 1
    assert path.read_bytes() == legacy_record * 2
    assert list(tmp_path.glob("*.lock")) == []
    with path.open("rb") as handle:
        for _ in range(2):
            actual_indexes, actual_values, actual_weights = load_raw_file_intermediate(handle)
            np.testing.assert_array_equal(actual_indexes, indexes)
            np.testing.assert_array_equal(actual_values, values)
            np.testing.assert_array_equal(actual_weights, weights)


@pytest.mark.skipif(os.name == "nt", reason="POSIX vectored append")
def test_short_raw_append_closes_descriptor_without_retry(tmp_path, monkeypatch):
    path = tmp_path / "i_00000.dat"
    calls = []

    def short_write(fd, buffers):
        calls.append(fd)
        return os.write(fd, b"x")

    monkeypatch.setattr(files_mod.os, "writev", short_write)
    with pytest.raises(OSError, match="short append.*1 of 10 bytes"):
        write_raw_intermediate(path, b"header", np.array([1], dtype=np.uint32))
    assert len(calls) == 1
    with pytest.raises(OSError) as error:
        os.fstat(calls[0])
    assert error.value.errno == errno.EBADF
    assert path.read_bytes() == b"x"


def test_windows_raw_append_holds_lock_for_one_binary_write(tmp_path, monkeypatch):
    path = tmp_path / "i_00000.dat"
    values = np.arange(6, dtype=np.uint16).reshape(2, 3)
    record = b"header" + values.tobytes()
    events = []

    @contextmanager
    def lock(lock_path):
        events.append(("lock", lock_path))
        yield
        events.append(("unlock", lock_path))

    def open_fd(target, flags, mode):
        events.append(("open", target, flags, mode))
        return 123

    def write(fd, data):
        events.append(("write", fd, data))
        return len(data)

    binary_flag = 0x8000
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | binary_flag
    monkeypatch.setattr(files_mod, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(files_mod, "FileLock", lock)
    monkeypatch.setattr(files_mod, "os", SimpleNamespace(
        O_WRONLY=os.O_WRONLY, O_APPEND=os.O_APPEND, O_CREAT=os.O_CREAT, O_BINARY=binary_flag,
        open=open_fd, write=write, close=lambda fd: events.append(("close", fd)),
    ))
    assert write_raw_intermediate(path, b"header", values) == len(record)
    assert events == [
        ("lock", path.with_suffix(".lock")),
        ("open", path, flags, 0o666),
        ("write", 123, record),
        ("close", 123),
        ("unlock", path.with_suffix(".lock")),
    ]


def _append_intermediate_records(path, worker, start):
    indexes = np.arange(4096, dtype=np.uint32)
    weights = np.ones(len(indexes), dtype=np.float32)
    start.wait(timeout=15)
    for record in range(worker * 8, (worker + 1) * 8):
        _write_intermediate_file(
            path, indexes, np.full(len(indexes), record, dtype=np.float32), weights,
            source_rows=64, target_rows=64,
        )


def test_concurrent_raw_appends_remain_complete_reader_records(tmp_path):
    path = tmp_path / "i_00000.dat"
    context = multiprocessing.get_context("spawn")
    start = context.Barrier(3)
    processes = [context.Process(target=_append_intermediate_records, args=(path, worker, start))
                 for worker in range(3)]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    records = []
    with path.open("rb") as handle:
        while handle.tell() < path.stat().st_size:
            indexes, values, weights = load_raw_file_intermediate(handle)
            np.testing.assert_array_equal(indexes, np.arange(4096))
            np.testing.assert_array_equal(weights, np.ones(4096, dtype=np.float32))
            np.testing.assert_array_equal(values, np.full((1, 4096), values[0, 0], dtype=np.float32))
            records.append(int(values[0, 0]))
    assert sorted(records) == list(range(24))
    if os.name != "nt":
        assert list(tmp_path.glob("*.lock")) == []


def test_volume_from_intermediates(tmp_path):
    shape = ImgSlice(Y=2, X=3)
    single_file = tmp_path.joinpath("single.tif")
    _write_intermediate_file(single_file, [0, 1], [4, 6], [2, 3])

    merged = volume_from_intermediates(single_file, shape)

    assert merged.shape == (6,)
    np.testing.assert_allclose(merged[:3], np.array([2.0, 2.0, 0.0], dtype=np.float32))
    assert not single_file.exists()

    appended = tmp_path / "appended.dat"
    _write_intermediate_file(appended, [0], [3], [2])
    _write_intermediate_file(appended, [1], [4], [2])

    discrete = volume_from_intermediates(appended, shape, discrete=True)

    np.testing.assert_allclose(discrete[:3], np.array([0.0, 2.0, 0.0], dtype=np.float32))
    assert not appended.exists()
    assert not np.shares_memory(merged, discrete)
    np.testing.assert_array_equal(merged, [2, 2, 0, 0, 0, 0])


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("dimensions", [(1, 1), (2, 3)])
def test_multichannel_merge_uses_shared_weights(tmp_path, discrete, dimensions):
    shape = ImgSliceC(*dimensions, C=3)
    path = tmp_path / "channels.dat"
    # Appended contributions share weights even when channel zero is empty.
    _write_intermediate_file(path, [0], [[0], [3], [4]], [2])
    _write_intermediate_file(path, [0], [[0], [4], [8]], [2])

    merged = volume_from_intermediates(path, shape, discrete=discrete)

    expected = np.zeros((3, shape.Y * shape.X), dtype=np.float32)
    expected[:, 0] = [0, 0 if discrete else 1.75, 3]
    assert merged.shape == expected.shape
    np.testing.assert_array_equal(merged, expected)
    assert not path.exists()


def test_write_conv_vol(tmp_path):
    shape = ImgSliceC(Y=2, X=2, C=1)
    target_folder = tmp_path.joinpath("output")
    target_folder.mkdir()
    writes = []

    source = tmp_path / "page.dat"
    _write_intermediate_file(source, [0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1])

    def writer(path, data):
        writes.append((path.name, np.array(data)))

    perf = write_conv_vol(
        writer,
        source,
        shape,
        np.uint8,
        None,
        target_folder,
        7,
        0,
        False,
    )

    assert writes[0][0] == "7.tif"
    assert writes[0][1].shape == (2, 2, 1)
    np.testing.assert_array_equal(writes[0][1][..., 0], [[0, 1], [2, 3]])
    assert {"Merge Volume", "Write Merged", "Total Chunk Merge"} <= perf.keys()

    writes.clear()
    _write_intermediate_file(source, [0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1])
    perf = write_conv_vol(
        writer,
        source,
        shape,
        np.uint8,
        (2, 1, 1),
        target_folder,
        3,
        0,
        False,
    )

    assert [name for name, _data in writes] == ["00006.tif", "00007.tif"]
    for _, data in writes:
        np.testing.assert_array_equal(data, [[0, 1], [2, 3]])
    assert {"Merge Volume", "Zoom", "Write Merged", "Total Chunk Merge"} <= perf.keys()


def test_writer_reuses_storage_and_clears_disjoint_empty_and_missing_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "_writer_page_buffers", None)
    shape = ImgSliceC(2, 3, 2)
    first, second, empty, missing = [tmp_path / f"{name}.dat" for name in ("first", "second", "empty", "missing")]
    _write_intermediate_file(first, [0, 2], [[0, 8], [6, 10]], [2, 2], target_rows=3, source_rows=3)
    _write_intermediate_file(second, [5], [[14], [18]], [2], target_rows=3, source_rows=3)
    empty.touch()
    accumulators, outputs, pages = [], [], []

    def writer(path, data):
        outputs.append(data)
        pages.append(data.copy())

    for index, source in enumerate((first, second, empty, missing)):
        write_conv_vol(writer, source, shape, np.uint16, None, tmp_path, index, 0, False)
        accumulator, output = files_mod._writer_page_buffers
        accumulators.append(accumulator)
        assert output is outputs[-1]
        assert not np.any(accumulator)
        assert not np.any(output)
        assert not source.exists()

    assert all(array is accumulators[0] for array in accumulators)
    assert all(array is outputs[0] for array in outputs)
    expected = np.zeros((4, 2, 3, 2), dtype=np.uint16)
    expected[0, 0, 0] = [0, 3]
    expected[0, 0, 2] = [4, 5]
    expected[1, 1, 2] = [7, 9]
    np.testing.assert_array_equal(pages, expected)


def test_writer_adapts_storage_to_geometry_channels_and_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "_writer_page_buffers", None)
    for shape, dtype in (
        (ImgSliceC(2, 3, 1), np.uint16), (ImgSliceC(3, 2, 1), np.uint16),
        (ImgSliceC(3, 2, 2), np.uint16), (ImgSliceC(3, 2, 2), bool),
        (ImgSliceC(3, 2, 2), np.bool_), (ImgSliceC(2, 3, 1), np.uint16),
    ):
        source = tmp_path / "page.dat"
        _write_intermediate_file(source, [0], [[4]] * shape.C, [2])
        pages = []
        write_conv_vol(lambda path, data: pages.append(data.copy()), source, shape, dtype,
                       None, tmp_path, 0, 0, False)
        accumulator, output = files_mod._writer_page_buffers
        assert accumulator.shape == (shape.C + 1, shape.Y * shape.X)
        assert output.shape == (shape.Y, shape.X, shape.C)
        assert pages[0].shape == output.shape
        assert pages[0].dtype == (np.uint8 if dtype is bool else dtype)
        expected = np.zeros(output.shape, dtype=pages[0].dtype)
        expected[0, 0] = [1 if dtype in (bool, np.bool_) else 2] * shape.C
        np.testing.assert_array_equal(pages[0], expected)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("dtype, output_dtype", [(bool, np.uint8), (np.bool_, np.bool_), (np.dtype(bool), np.uint8)])
def test_sparse_binary_conversion_preserves_dtype_truthiness_and_discrete_rounding(tmp_path, discrete,
                                                                                   dtype, output_dtype):
    source = tmp_path / "page.dat"
    _write_intermediate_file(source, [0, 1, 2, 3], [-2, 0, 1.5, 2.999], [1, 1, 1, 1])
    pages = []
    write_conv_vol(lambda path, data: pages.append(data.copy()), source, ImgSliceC(2, 2, 1),
                   dtype, None, tmp_path, 0, 0, discrete)
    assert pages[0].dtype == output_dtype
    np.testing.assert_array_equal(pages[0].ravel(), [1, 0, 0 if discrete else 1, 1])


def test_writer_failure_discards_storage_before_next_page(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "_writer_page_buffers", None)
    source = tmp_path / "page.dat"
    shape = ImgSliceC(2, 2, 1)
    write_conv_vol(lambda path, data: None, source, shape, np.uint16, None, tmp_path, 0, 0, False)
    failed_accumulator, failed_output = files_mod._writer_page_buffers
    _write_intermediate_file(source, [0], [8], [2])

    def fail_write(path, data):
        assert data is failed_output
        assert data[0, 0, 0] == 4
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_conv_vol(fail_write, source, shape, np.uint16, None, tmp_path, 0, 0, False)
    assert files_mod._writer_page_buffers is None
    _write_intermediate_file(source, [3], [6], [2])
    pages = []
    write_conv_vol(lambda path, data: pages.append(data.copy()), source, shape, np.uint16,
                   None, tmp_path, 1, 0, False)
    accumulator, output = files_mod._writer_page_buffers
    assert accumulator is not failed_accumulator
    assert output is not failed_output
    np.testing.assert_array_equal(pages[0].ravel(), [0, 0, 0, 3])


def test_partial_accumulation_failure_discards_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(files_mod, "_writer_page_buffers", None)
    source = tmp_path / "page.dat"
    shape = ImgSliceC(2, 2, 2)
    write_conv_vol(lambda path, data: None, source, shape, np.uint16, None, tmp_path, 0, 0, False)
    failed_accumulator, failed_output = files_mod._writer_page_buffers
    # A one-channel record can update channel zero before failing on channel
    # one, leaving values at a pixel whose weights were never accumulated.
    _write_intermediate_file(source, [0], [8], [2])
    with pytest.raises(IndexError):
        write_conv_vol(lambda path, data: pytest.fail("invalid record reached writer"),
                       source, shape, np.uint16, None, tmp_path, 0, 0, False)
    assert failed_accumulator[0, 0] == 8
    assert not np.any(failed_accumulator[-1])
    assert files_mod._writer_page_buffers is None
    source.unlink()
    _write_intermediate_file(source, [3], [[6], [10]], [2])
    pages = []
    write_conv_vol(lambda path, data: pages.append(data.copy()), source, shape, np.uint16,
                   None, tmp_path, 1, 0, False)
    accumulator, output = files_mod._writer_page_buffers
    assert accumulator is not failed_accumulator
    assert output is not failed_output
    np.testing.assert_array_equal(pages[0].reshape(-1, 2), [[0, 0], [0, 0], [0, 0], [3, 5]])

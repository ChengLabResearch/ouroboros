import os
from pathlib import Path

import numpy as np
from tifffile import imwrite, TiffFile

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
    ravel_map_2d,
    load_z_intermediate,
    increment_volume,
    write_small_intermediate
)


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


def test_np_convert():
    float_data = np.linspace(0, 1, 16)
    int_data = np_convert(np.uint16, float_data)

    assert np.all(int_data == np.arange(0, np.iinfo(np.uint16).max + 1, np.iinfo(np.uint16).max // 15))
    assert np.all(np_convert(np.float32, int_data) == int_data.astype(np.float32))


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
    }

    write_small_intermediate(sample_path,
                             np.fromiter(offset_dict.values(), dtype=np.uint32, count=4),
                             raveled_source,
                             source_values,
                             source_weights)

    indicies, values, weights = load_z_intermediate(sample_path)

    assert len(indicies) == 100
    assert np.all(indicies == raveled_mapped)
    assert len(values) == 100
    assert values.dtype == np.float32
    assert np.all(values == source_values)
    assert len(weights) == 100
    assert np.all(weights == source_weights)
    assert weights.dtype == np.float32


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
    }

    write_small_intermediate(sample_path,
                             np.fromiter(offset_dict.values(), dtype=np.uint32, count=4),
                             raveled_source,
                             source_values,
                             source_weights)

    volume = np.zeros((2, 80 * 60))
    increment_volume(sample_path, volume[:], cleanup=True)

    assert volume[0, mapped_source[1]] == source_values[1]
    assert np.allclose(volume[0, mapped_source[0]], np.sum(source_values[[0, 2]]))
    assert volume[1, mapped_source[1]] == source_weights[1]
    assert np.allclose(volume[1, mapped_source[0]], np.sum(source_weights[[0, 2]]))
    assert np.all(np.nonzero(volume)[0] == np.array([0, 0, 1, 1]))
    assert np.all(np.nonzero(volume)[1] == np.array([3947, 3952, 3947, 3952]))
    
    assert not sample_path.exists()


def test_volume_from_intermediates():
    pass


def test_write_conv_vol():
    pass

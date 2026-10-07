from pathlib import Path
import shutil

import numpy as np

from precitec_data_parser import PrecitecData


def make_data() -> PrecitecData:
    data = object.__new__(PrecitecData)
    data.altitude_path = Path("measurement_altitude.csv")
    data.intensity_path = Path("measurement_intensity.csv")
    data.metadata_altitude = {"Signal": "altitude"}
    data.metadata_intensity = {"Signal": "intensity"}
    data.altitude = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    data.intensity = np.array([[10.0, 40.0, 50.0], [60.0, 20.0, 70.0]])
    data.nonmeasured = np.array(
        [[False, False, False], [False, True, False]], dtype=bool
    )
    data.xstep = 2.0
    data.ystep = 5.0
    data.x = np.array([0.0, 2.0, 4.0])
    data.y = np.array([0.0, 5.0])
    return data


def test_nonmeasured_points_are_nan_when_loaded() -> None:
    sample_dir = Path(__file__).parent / "sample_data"
    data = PrecitecData(
        sample_dir / "dummy_Altitude_Peak_Processed.csv",
        sample_dir / "dummy_Intensity_Peak_Processed.csv",
    )

    assert np.isnan(data.altitude[data.nonmeasured]).all()
    assert np.isnan(data.intensity[data.nonmeasured]).all()


def test_txt_loading_masks_zero_height_points(tmp_path: Path) -> None:
    txt_path = tmp_path / "measurement.txt"
    txt_path.write_text(
        "Precitec export\nX\tY\tZ\n0\t0\t0\n1\t0\t1\n0\t1\t2\n1\t1\t3\n",
        encoding="utf-16",
    )

    data = PrecitecData(txt_path, None)

    assert np.isnan(data.altitude[0, 0])
    assert np.isnan(data.intensity[0, 0])
    assert data.nonmeasured[0, 0]


def test_from_folder_loads_pairs_for_requested_filetype(tmp_path: Path) -> None:
    sample_dir = Path(__file__).parent / "sample_data"
    for path in sample_dir.glob("*.csv"):
        shutil.copy2(path, tmp_path / path.name)
    (tmp_path / "dummy_Altitude_Peak_Processed.bcrf").touch()
    (tmp_path / "dummy_Intensity_Peak_Processed.bcrf").touch()

    data = PrecitecData.from_folder(tmp_path, "csv")

    assert len(data) == 1
    assert data[0].altitude_path.name == "dummy_Altitude_Peak_Processed.csv"
    assert data[0].intensity_path is not None
    assert data[0].intensity_path.name == "dummy_Intensity_Peak_Processed.csv"


def test_find_file_pairs_reads_metadata_without_parsing_arrays(monkeypatch) -> None:
    sample_dir = Path(__file__).parent / "sample_data"
    altitude = sample_dir / "dummy_Altitude_Peak_Processed.csv"
    intensity = sample_dir / "dummy_Intensity_Peak_Processed.csv"

    def fail_parse(cls, path: Path) -> None:
        raise AssertionError("pair discovery should not parse measurement arrays")

    monkeypatch.setattr(PrecitecData, "_parse", classmethod(fail_parse))

    pairs = PrecitecData.find_file_pairs([altitude, intensity])

    assert pairs == [(altitude, intensity)]


def test_parse_bcrf_reads_binary_surface_into_owned_array(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    path = tmp_path / "measurement.bcrf"
    path.write_bytes(b"\0" * 16 + expected.tobytes())
    fields = {
        "headersize": 8,
        "xpixels": 2,
        "ypixels": 2,
        "intelmode": 1,
        "xlength": 0.002,
        "ylength": 0.004,
        "xunit": "mm",
        "yunit": "mm",
    }
    monkeypatch.setattr(
        PrecitecData,
        "_read_bcrf_header",
        staticmethod(lambda _: fields.copy()),
    )

    metadata, actual, xstep, ystep = PrecitecData._parse_bcrf(path)

    assert metadata == fields
    np.testing.assert_array_equal(actual, expected)
    assert isinstance(actual.base, np.ndarray)
    assert actual.base.flags.owndata
    assert xstep == 1.0
    assert ystep == 2.0


def test_threshold_intensity_returns_independent_data_by_default() -> None:
    original = make_data()

    thresholded = original.threshold_intensity(30.0)

    expected_mask = np.array([[True, False, False], [False, True, False]], dtype=bool)
    np.testing.assert_array_equal(thresholded.nonmeasured, expected_mask)
    assert np.isnan(thresholded.altitude[0, 0])
    assert np.isnan(thresholded.intensity[0, 0])

    np.testing.assert_array_equal(
        original.altitude,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    np.testing.assert_array_equal(
        original.intensity,
        np.array([[10.0, 40.0, 50.0], [60.0, 20.0, 70.0]]),
    )
    np.testing.assert_array_equal(
        original.nonmeasured,
        np.array([[False, False, False], [False, True, False]], dtype=bool),
    )
    assert thresholded.metadata_altitude is not original.metadata_altitude
    assert thresholded.metadata_intensity is not original.metadata_intensity


def test_threshold_intensity_supports_inplace_and_copy_modes() -> None:
    original = make_data()
    result = original.threshold_intensity(30.0, inplace=True)

    assert result is None
    np.testing.assert_array_equal(
        original.nonmeasured,
        np.array([[True, False, False], [False, True, False]], dtype=bool),
    )

    original = make_data()
    thresholded = original.threshold_intensity(30.0)

    assert thresholded is not None
    assert thresholded is not original
    assert thresholded.altitude is not original.altitude
    np.testing.assert_array_equal(
        thresholded.nonmeasured,
        np.array([[True, False, False], [False, True, False]], dtype=bool),
    )
    np.testing.assert_array_equal(
        original.nonmeasured,
        np.array([[False, False, False], [False, True, False]], dtype=bool),
    )


def test_reorient_swaps_geometry_without_mutating_original() -> None:
    original = make_data()

    transposed = original.reorient(swap_axes=True)

    np.testing.assert_array_equal(transposed.altitude, original.altitude.T)
    np.testing.assert_array_equal(transposed.intensity, original.intensity.T)
    np.testing.assert_array_equal(transposed.nonmeasured, original.nonmeasured.T)
    np.testing.assert_array_equal(transposed.x, original.y)
    np.testing.assert_array_equal(transposed.y, original.x)
    assert transposed.xstep == original.ystep
    assert transposed.ystep == original.xstep

    assert original.altitude.shape == (2, 3)
    np.testing.assert_array_equal(original.x, np.array([0.0, 2.0, 4.0]))
    np.testing.assert_array_equal(original.y, np.array([0.0, 5.0]))
    assert original.xstep == 2.0
    assert original.ystep == 5.0


def test_reorient_supports_inplace_and_copy_modes() -> None:
    original = make_data()
    result = original.reorient(swap_axes=True, inplace=True)

    assert result is None
    np.testing.assert_array_equal(original.altitude, make_data().altitude.T)
    np.testing.assert_array_equal(original.x, np.array([0.0, 5.0]))
    np.testing.assert_array_equal(original.y, np.array([0.0, 2.0, 4.0]))
    assert original.xstep == 5.0
    assert original.ystep == 2.0

    original = make_data()
    transposed = original.reorient(swap_axes=True)

    assert transposed is not None
    assert transposed is not original
    np.testing.assert_array_equal(transposed.altitude, original.altitude.T)
    np.testing.assert_array_equal(original.altitude, make_data().altitude)


def test_reorient_flip_x_mirrors_arrays_and_keeps_coordinates_ascending() -> None:
    original = make_data()

    flipped = original.reorient(flip_x=True)

    np.testing.assert_array_equal(flipped.altitude, original.altitude[:, ::-1])
    np.testing.assert_array_equal(flipped.intensity, original.intensity[:, ::-1])
    np.testing.assert_array_equal(flipped.nonmeasured, original.nonmeasured[:, ::-1])
    np.testing.assert_array_equal(flipped.x, original.x)
    np.testing.assert_array_equal(original.altitude, make_data().altitude)


def test_reorient_flip_x_supports_inplace_and_copy_modes() -> None:
    original = make_data()
    result = original.reorient(flip_x=True, inplace=True)

    assert result is None
    np.testing.assert_array_equal(original.altitude, make_data().altitude[:, ::-1])
    np.testing.assert_array_equal(original.x, make_data().x)

    original = make_data()
    flipped = original.reorient(flip_x=True)

    assert flipped is not None
    assert flipped is not original
    np.testing.assert_array_equal(flipped.altitude, original.altitude[:, ::-1])
    np.testing.assert_array_equal(flipped.x, original.x)
    np.testing.assert_array_equal(original.altitude, make_data().altitude)


def test_reorient_can_swap_and_flip_both_axes_in_one_copy() -> None:
    original = make_data()

    transformed = original.reorient(swap_axes=True, flip_x=True, flip_y=True)

    np.testing.assert_array_equal(transformed.altitude, original.altitude.T[::-1, ::-1])
    np.testing.assert_array_equal(transformed.x, original.y)
    np.testing.assert_array_equal(transformed.y, original.x)
    np.testing.assert_array_equal(original.altitude, make_data().altitude)


def test_get_signal_data_rejects_unknown_signal() -> None:
    with np.testing.assert_raises(ValueError):
        make_data().get_signal_data("unknown")  # type: ignore[arg-type]

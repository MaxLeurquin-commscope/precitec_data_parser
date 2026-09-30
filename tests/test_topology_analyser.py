from unittest.mock import Mock

import numpy as np
from scipy import ndimage
from surfalize import Surface

from precitec_data_parser import PrecitecSurfaceAnalyzer


def test_plot_3d_uses_signal_data_api() -> None:
    height_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = Mock()
    data.x = np.array([0.0, 1.0])
    data.y = np.array([0.0, 1.0])
    data.signal_data.return_value = height_data

    analyzer = object.__new__(PrecitecSurfaceAnalyzer)
    analyzer.data = data
    analyzer.signal = "altitude"

    figure = analyzer.plot_3d()

    data.signal_data.assert_called_once_with("altitude")
    np.testing.assert_array_equal(figure.data[0].z, height_data)


def test_oblique_profile_preserves_nonmeasured_mask_samples() -> None:
    height_data = np.arange(25.0).reshape(5, 5)
    nonmeasured = np.zeros((5, 5), dtype=bool)
    nonmeasured[2, 2] = True
    data = Mock()
    data.nonmeasured = nonmeasured
    data.to_surface.return_value = Surface(height_data, 1.0, 1.0)
    analyzer = PrecitecSurfaceAnalyzer(data, level=False)

    profile = analyzer.oblique_profile(0.0, 0.0, 4.0, 4.0)

    sample_coords = np.linspace(0.0, 4.0, profile.data.size)
    row_coords = 4.0 - sample_coords
    touched_nonmeasured = ndimage.map_coordinates(
        nonmeasured.astype(float), [row_coords, sample_coords], order=1
    )
    np.testing.assert_array_equal(np.isnan(profile.data), touched_nonmeasured > 0)

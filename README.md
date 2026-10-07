# Precitec Data Parser

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Parse and analyze height-map exports from Precitec CLS2 sensors (`.csv`, `.bcrf`, and `.txt`), with tools for 3D surface and profile analysis.

## ⚠️ Disclaimer

**This package is NOT officially supported by Precitec.** It is an independent project. Please do **not** contact Precitec for support or questions about this library. For issues, feature requests, or questions, please use the [GitHub Issues](https://github.com/yourusername/precitec-data-parser/issues) on this repository instead.

## Features

- **`PrecitecData`** - parses `.csv`, `.bcrf`, and `.txt` exports into height (`z`) and coordinate (`x`, `y`) arrays plus metadata, transparently handling multi-line tile stitching and reversed sweep lines.
  - `threshold_intensity()` returns an independently thresholded copy by default.
  - `reorient()` combines axis swapping and x/y flipping in one operation and returns an independent copy by default. Pass `inplace=True` to either method to mutate the object.
  - `get_signal_data()` returns the altitude or intensity array.
  - `to_surface()` - converts to a `surfalize.Surface`, filling non-measured points.
- **`PrecitecSurfaceAnalyzer`** - 3D surface and profile analysis on top of `PrecitecData`, backed by [`surfalize`](https://pypi.org/project/surfalize/).
  - `roughness_parameters()` / `height_parameters()` - ISO 25178 areal parameters (Sa, Sq, Sz, Sdr, ...).
  - `horizontal_profile()` / `vertical_profile()` / `oblique_profile()` - extract 1D profile cuts (with a bug fix for oblique profiles on anisotropic pixel grids) and their ISO 4287 parameters (Ra, Rq, Rz, ...).
  - `filter_profile()` - Gaussian low/high/bandpass smoothing or Hampel outlier removal on a profile.
  - `plot_3d()`, `plot_2d()`, `plot_profile()` - interactive (zoom/pan/hover) Plotly figures: shaded 3D rendering, top-down 2D map, and profile plots (with the cut line overlaid on the 2D map).

## Installation

Install from PyPI:

```bash
pip install precitec-data-parser
```

Or from source:

```bash
git clone https://github.com/yourusername/precitec-data-parser.git
cd precitec-data-parser
pip install -e .
```

Requires Python >=3.11.

To run the Streamlit viewer from a source checkout, install its optional dependency and start the app:

```bash
pip install -e ".[viewer]"
streamlit run app.py
```

## Quick Start

```python
from pathlib import Path
from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer

# Load and visualize a measurement
data = PrecitecData(
    Path("measurement_altitude.csv"),
    Path("measurement_intensity.csv"),
)

# Transform without changing the parsed source object
thresholded = data.threshold_intensity(30.0)
oriented = thresholded.reorient(swap_axes=True, flip_x=True)

# Analyze surface topology
analyzer = PrecitecSurfaceAnalyzer(oriented)
print(analyzer.height_parameters())
analyzer.plot_2d(show=True)

# Extract and filter profiles
profile = analyzer.horizontal_profile(y=data.y[len(data.y) // 2])
filtered = analyzer.filter_profile(profile, filter_args={"cutoff": 50})
analyzer.plot_profile(profile, filtered=filtered, show=True)
```

## Examples

See the [demo script](examples/demo.py) for profile extraction, filtering, and 3D visualization. Parser and topology tests are in [tests/test_data_parser.py](tests/test_data_parser.py) and [tests/test_topology_analyser.py](tests/test_topology_analyser.py).

## Supported Formats

- **`.csv`** - Precitec CLS2 CSV exports (text-based, supports multi-line stitching)
- **`.bcrf`** - Precitec binary height-map format (faster parsing)
- **`.txt`** - tab-separated XYZ exports (altitude only)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Maxime Leurquin (maxime.leurquin@gmail.com)

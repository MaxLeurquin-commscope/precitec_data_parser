# Precitec Data Parser

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Parse and analyze height-map exports from Precitec CLS2 sensors (`.csv` and `.bcrf`), with tools for 3D surface and profile analysis.

## ⚠️ Disclaimer

**This package is NOT officially supported by Precitec.** It is an independent project. Please do **not** contact Precitec for support or questions about this library. For issues, feature requests, or questions, please use the [GitHub Issues](https://github.com/yourusername/precitec-data-parser/issues) on this repository instead.

## Features

- **`PrecitecData`** - parses `.csv` and `.bcrf` exports into height (`z`) and coordinate (`x`, `y`) arrays plus metadata, transparently handling multi-line tile stitching and reversed sweep lines.
  - `plot_data()` - 2D height-map plot with a colorbar.
  - `to_surface()` - converts to a `surfalize.Surface`, filling non-measured points.
- **`PrecitecSurfaceAnalyzer`** - 3D surface and profile analysis on top of `PrecitecData`, backed by [`surfalize`](https://pypi.org/project/surfalize/).
  - `roughness_parameters()` / `height_parameters()` - ISO 25178 areal parameters (Sa, Sq, Sz, Sdr, ...).
  - `horizontal_profile()` / `vertical_profile()` / `oblique_profile()` - extract 1D profile cuts (with a bug fix for oblique profiles on anisotropic pixel grids) and their ISO 4287 parameters (Ra, Rq, Rz, ...).
  - `filter_profile()` - Gaussian low/high/bandpass smoothing or Hampel outlier removal on a profile.
  - `plot_3d()`, `plot_2d()`, `plot_profile()` - shaded 3D rendering, top-down 2D map, and profile plots (with the cut line overlaid on the 2D map).

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

Requires Python >=3.13.

## Quick Start

```python
from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer

# Load and visualize a measurement
with PrecitecData("measurement.csv") as data:
    data.plot_data(show=True)
    
    # Analyze surface topology
    analyzer = PrecitecSurfaceAnalyzer(data)
    print(analyzer.height_parameters())
    
    # Extract and filter profiles
    profile = analyzer.horizontal_profile(y=data.y[len(data.y) // 2])
    filtered = analyzer.filter_profile(profile, filter_args={"cutoff": 50})
    analyzer.plot_profile(profile, filtered=filtered, show=True)
```

## Examples

See [`tests/test_parser.py`](tests/test_parser.py) for a comprehensive demo of profile extraction, filtering, and 3D visualization.

## Supported Formats

- **`.csv`** - Precitec CLS2 CSV exports (text-based, supports multi-line stitching)
- **`.bcrf`** - Precitec binary height-map format (faster parsing)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Maxime Leurquin (maxime.leurquin@gmail.com)

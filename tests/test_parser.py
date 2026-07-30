"""Test suite for PrecitecData and PrecitecSurfaceAnalyzer."""
import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer


if __name__ == "__main__":
    my_data="my_precitec_measurement_file.csv" #.bcrf also works
   

    with PrecitecData(my_data) as data:
        print(data.metadata)
        data.plot_data(show=True)

        analyzer = PrecitecSurfaceAnalyzer(data)

        y_pos=data.y[len(data.y) // 2]
        print(f"{y_pos=}")
        profile_y = analyzer.horizontal_profile(y=y_pos)
        profile_x= analyzer.vertical_profile(x=data.x[len(data.x) // 2 +100])
        profile_oblique = analyzer.oblique_profile(
            x0=data.x[0], y0=data.y[0], x1=data.x[-1], y1=data.y[-1]
        )

        # plot_profile shows the profile alongside the top-down 2D map by
        # default, with a red line marking where the profile was cut from.
        # filter_profile demo: gaussian smooths the whole signal, hampel only
        # replaces individual outlier spikes.
        gaussian_x = analyzer.filter_profile(profile_x, method="gaussian", filter_args={"cutoff": 100})
        hampel_x = analyzer.filter_profile(profile_x, method="hampel", filter_args={"window_size": 5, "n_sigmas": 1.0})
        analyzer.plot_profile(profile_x, filtered=gaussian_x, show=True)
        analyzer.plot_profile(profile_x, filtered=hampel_x, show=True)

        filtered_x = analyzer.filter_profile(profile_x, filter_args={"cutoff": 50})
        filtered_oblique = analyzer.filter_profile(profile_oblique, filter_args={"cutoff": 50})
        analyzer.plot_profile(profile_x, show=True, filtered=filtered_x)
        analyzer.plot_profile(profile_oblique, show=True, filtered=filtered_oblique)

        # An oblique profile that does NOT start at (0, 0), to demonstrate
        # that it is still extracted correctly (surfalize's own
        # get_oblique_profile would silently get this one wrong).
        profile_oblique_offset = analyzer.oblique_profile(
            x0=data.x[len(data.x) // 4], y0=data.y[len(data.y) // 4],
            x1=data.x[3 * len(data.x) // 4], y1=data.y[3 * len(data.y) // 4],
        )
        analyzer.plot_profile(profile_oblique_offset, show=True)


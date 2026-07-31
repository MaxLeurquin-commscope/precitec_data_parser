"""
Demo script showing how to use PrecitecData and PrecitecSurfaceAnalyzer.

This script requires a .csv or .bcrf file to analyze. Update the `my_data` path
to point to your Precitec measurement file.

Run with: python examples/demo.py
"""
import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer


if __name__ == "__main__":
    # Update this path to your Precitec measurement file (.csv or .bcrf)
    my_data = "path/to/your/measurement.csv"
    
    if not Path(my_data).exists():
        print(f"Error: File not found: {my_data}")
        print("Please update the 'my_data' variable to point to your Precitec measurement file.")
        sys.exit(1)

    with PrecitecData(my_data) as data:
        print("=== Metadata ===")
        print(data.metadata)
        print()
        
        print("=== Plotting 2D height map ===")
        data.plot_data(show=True)

        analyzer = PrecitecSurfaceAnalyzer(data)

        print("=== Plotting 3D surface ===")
        analyzer.plot_3d(savepath="grooves_3d.png")
        print("Saved 3D plot to grooves_3d.png")

        # Extract profiles at different positions
        y_pos = data.y[len(data.y) // 2]
        print(f"\n=== Extracting profiles at y={y_pos:.2f} µm ===")
        profile_y = analyzer.horizontal_profile(y=y_pos)
        profile_x = analyzer.vertical_profile(x=data.x[len(data.x) // 2 + 100])
        profile_oblique = analyzer.oblique_profile(
            x0=data.x[0], y0=data.y[0], x1=data.x[-1], y1=data.y[-1]
        )

        # Apply filters
        print("\n=== Filtering profiles ===")
        gaussian_x = analyzer.filter_profile(
            profile_x, method="gaussian", filter_args={"cutoff": 100}
        )
        hampel_x = analyzer.filter_profile(
            profile_x, method="hampel", filter_args={"window_size": 5, "n_sigmas": 1.0}
        )

        print("Plotting Gaussian-filtered profile...")
        analyzer.plot_profile(profile_x, filtered=gaussian_x, show=True)
        
        print("Plotting Hampel-filtered profile...")
        analyzer.plot_profile(profile_x, filtered=hampel_x, show=True)

        # More filtering examples
        filtered_x = analyzer.filter_profile(profile_x, filter_args={"cutoff": 50})
        filtered_oblique = analyzer.filter_profile(profile_oblique, filter_args={"cutoff": 50})
        
        analyzer.plot_profile(profile_x, show=True, filtered=filtered_x)
        analyzer.plot_profile(profile_oblique, show=True, filtered=filtered_oblique)

        # Oblique profile from a non-origin point
        print("\n=== Oblique profile from offset point ===")
        profile_oblique_offset = analyzer.oblique_profile(
            x0=data.x[len(data.x) // 4],
            y0=data.y[len(data.y) // 4],
            x1=data.x[3 * len(data.x) // 4],
            y1=data.y[3 * len(data.y) // 4],
        )
        analyzer.plot_profile(profile_oblique_offset, show=True)
        
        print("\n✓ Demo complete!")

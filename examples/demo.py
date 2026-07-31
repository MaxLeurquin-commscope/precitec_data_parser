"""
Demo script showing how to use PrecitecData and PrecitecSurfaceAnalyzer.

This script requires a .csv or .bcrf file to analyze. Update the `my_data` path
to point to your Precitec measurement file.

Run with: python examples/demo.py
"""
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer


if __name__ == "__main__":
    # Update these paths to your Precitec measurement pair (.csv or .bcrf)
    altitude_path = rf"precitec_data_parser\tests\sample_data\dummy_Altitude_Peak_Processed.csv"
    intensity_path = rf"precitec_data_parser\tests\sample_data\dummy_Intensity_Peak_Processed.csv"
    

    data = PrecitecData(altitude_path, intensity_path)
    print("=== Altitude Metadata ===")
    print(data.metadata_altitude)
    
    print("\n=== Intensity Metadata ===")
    print(data.metadata_intensity)

    print(f"Available signals: {[k for k, v in data.signals.items() if v is not None]}")

    analyzer = PrecitecSurfaceAnalyzer(data, level=True)

    print("=== Plotting 2D altitude map ===")
    analyzer.plot_2d(show=True)

    print("=== Plotting 3D surface ===")
    analyzer.plot_3d(savepath="grooves_3d.png")
    print("Saved 3D plot to grooves_3d.png")

    # Extract profiles at different positions
    y_pos = data.y[len(data.y) // 2]
    print(f"\n=== Extracting profiles at y={y_pos:.2f} µm ===")
    profile_y = analyzer.horizontal_profile(y=y_pos)
    profile_x = analyzer.vertical_profile(x=data.x[len(data.x) // 2 ])
    profile_oblique = analyzer.oblique_profile(
        x0=data.x[0], y0=data.y[0], x1=data.x[-1], y1=data.y[-1]
    )

    print("\n=== Plotting extracted profiles ===")
    analyzer.plot_profile(profile_x, show=True)
    analyzer.plot_profile(profile_oblique, show=True)

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

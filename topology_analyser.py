"""
Author: Maxime Leurquin
Date: July 2026
Description: 3D surface and profile analysis for parsed `PrecitecData`, backed by `surfalize`.
"""

from typing import Any, cast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import ndimage
from types import SimpleNamespace
from surfalize import Profile
from pathlib import Path
from data_parser import PrecitecData

class PrecitecSurfaceAnalyzer:
    """3D surface and profile analysis for parsed `PrecitecData`, backed by `surfalize`.

    Wraps the height data in a `surfalize.Surface`, giving access to ISO 25178
    areal roughness/height parameters, 2D/3D surface plotting, and extraction of
    horizontal/vertical/oblique `surfalize.Profile` cuts (with ISO 4287 profile
    parameters such as Ra, Rq, Rz).

    Note on anisotropic pixel size (surfalize's "different pixel size in x and y"
    warning): `horizontal_profile`/`vertical_profile` are unaffected - they index
    a single row/column and use that axis's own step size, never mixing step_x
    and step_y. `oblique_profile` is reimplemented here (see its docstring) to
    fix a real bug in `surfalize.Surface.get_oblique_profile` and to compute the
    right number of samples for anisotropic pixels. Areal roughness/height
    parameters (Sa, Sq, Sdr, ...) are the ones most sensitive to the warning,
    since several definitions assume square pixels; they are out of scope here.

    Example:
        with PrecitecData("altitude.csv") as data:
            analyzer = PrecitecSurfaceAnalyzer(data)
            print(analyzer.height_parameters())
            profile = analyzer.horizontal_profile(y=data.y[len(data.y) // 2])
            print(profile.roughness_parameters())
            analyzer.plot_3d(savepath="surface_3d.png")

    Parameters
    ----------
    data : PrecitecData
        Parsed measurement to analyze.
    fill_nonmeasured : bool, default True
        Interpolate zero (no-measurement) samples before analysis; surfalize
        treats NaN as non-measured and would otherwise skew ISO parameters.
    """

    def __init__(self, data: PrecitecData, fill_nonmeasured: bool = True):
        self.data = data
        self.surface = data.to_surface(fill_nonmeasured)

    def horizontal_profile(self, y: float, **kwargs) -> Profile:
        """Extract a horizontal (constant-y) profile at position `y` (µm).

        Reads an existing row (rounded to the closest data point) - no
        interpolation is involved, unlike `oblique_profile`.
        """
        profile = self.surface.get_horizontal_profile(y, **kwargs)
        x0 = kwargs.get("start") or 0.0
        x1 = kwargs.get("end") or self.surface.width_um
        setattr(profile, "location", SimpleNamespace(kind="horizontal", x0=x0, y0=y, x1=x1, y1=y))
        return profile

    def vertical_profile(self, x: float, **kwargs) -> Profile:
        """Extract a vertical (constant-x) profile at position `x` (µm).

        Reads an existing column (rounded to the closest data point) - no
        interpolation is involved, unlike `oblique_profile`.
        """
        profile = self.surface.get_vertical_profile(x, **kwargs)
        y0 = kwargs.get("start") or 0.0
        y1 = kwargs.get("end") or self.surface.height_um
        setattr(profile, "location", SimpleNamespace(kind="vertical", x0=x, y0=y0, x1=x, y1=y1))
        return profile

    def oblique_profile(self, x0: float, y0: float, x1: float, y1: float) -> Profile:
        """Extract a profile along the line from (x0, y0) to (x1, y1) (µm).

        This does NOT delegate to `surfalize.Surface.get_oblique_profile`
        because that method has two problems (checked against surfalize
        0.18.0's source):

        1. A genuine bug: it samples along `y = (dy/dx) * x` in pixel-index
           space instead of the requested line, i.e. it drops the line's
           offset. It only returns correct results when (x0, y0) happens to
           be the pixel-grid origin (0, 0) - any other start point silently
           gives a profile shifted off the true line (verified with a
           reproducible test: line (10,5)-(25,18) on a 40x30 grid sampled
           rows ~8.7-21.7 instead of the correct 5-18).
        2. It picks the sample count via `int(np.hypot(dx, dy))` where dx/dy
           are raw pixel-index differences on potentially different physical
           scales (x pixels of size step_x, y pixels of size step_y) - the
           exact "different pixel size in x and y" issue the warning refers
           to; this over/under-samples the line and misreports its step size
           when step_x != step_y.

        Both are fixed below: the line is parametrized directly in physical
        (µm) coordinates, and the number of samples is derived from the true
        physical length divided by the finer of the two pixel resolutions.

        Interpolation note: like the original, this still uses
        `scipy.ndimage.map_coordinates` (cubic-spline interpolation) to read
        values, because an oblique line practically never lands exactly on
        grid points - unlike `horizontal_profile`/`vertical_profile`, which
        read existing rows/columns with no interpolation.
        """
        surface = self.surface
        if not (0 <= x0 <= surface.width_um and 0 <= x1 <= surface.width_um):
            raise ValueError("x0 and x1 must lie within [0, width_um].")
        if not (0 <= y0 <= surface.height_um and 0 <= y1 <= surface.height_um):
            raise ValueError("y0 and y1 must lie within [0, height_um].")

        ny, nx = surface.data.shape
        # Columns map directly to x; rows are measured from the top of the
        # array while y is measured from the bottom (consistent with
        # horizontal_profile/vertical_profile/plot_2d).
        x0px, x1px = x0 / surface.step_x, x1 / surface.step_x
        y0px, y1px = (ny - 1) - y0 / surface.step_y, (ny - 1) - y1 / surface.step_y

        length_um = float(np.hypot(x1 - x0, y1 - y0))
        finest_step = min(surface.step_x, surface.step_y)
        n_samples = max(2, int(round(length_um / finest_step)) + 1)

        xp = np.linspace(x0px, x1px, n_samples)
        yp = np.linspace(y0px, y1px, n_samples)
        data = ndimage.map_coordinates(surface.data, [yp, xp])

        step = length_um / (n_samples - 1)
        profile = Profile(data, step, length_um)
        setattr(profile, "location", SimpleNamespace(kind="oblique", x0=x0, y0=y0, x1=x1, y1=y1))
        return profile

    # Maps a `method` name to the `_<name>_filter` static method that implements
    # it. To add a new filter (e.g. "rolling_average", "hanning"), write a
    # `_<name>_filter(profile, **filter_args) -> Profile` static method below
    # and add an entry here - no other changes needed.
    _FILTERS: dict[str, str] = {
        "gaussian": "_gaussian_filter",
        "hampel": "_hampel_filter",
    }

    def filter_profile(
        self,
        profile: Profile,
        method: str = "gaussian",
        filter_args: dict[str, Any] | None = None,
    ) -> Profile:
        """Filter a profile to reduce noise.

        Parameters
        ----------
        method : str, default "gaussian"
            Which filter to apply - one of `PrecitecSurfaceAnalyzer._FILTERS`
            ("gaussian", "hampel").
        filter_args : dict, default None
            Keyword arguments forwarded to the chosen filter, e.g.
            `{"cutoff": 50, "filter_type": "lowpass"}` for "gaussian" (see
            `_gaussian_filter`) or `{"window_size": 5, "n_sigmas": 3.0}` for
            "hampel" (see `_hampel_filter`).
        """
        attr_name = self._FILTERS.get(method)
        if attr_name is None:
            raise ValueError(f'Unknown filter method "{method}", expected one of {sorted(self._FILTERS)}.')
        filter_func = getattr(self, attr_name)
        filtered = filter_func(profile, **(filter_args or {}))

        location = getattr(profile, "location", None)
        if location is not None:
            setattr(filtered, "location", location)
        return filtered

    @staticmethod
    def _gaussian_filter(profile: Profile, cutoff: float, filter_type: str = "lowpass", **kwargs) -> Profile:
        """Smooth the whole signal via `surfalize`'s Gaussian low/high/bandpass
        filter - every point is blended with its neighbors.

        `cutoff` (required) is the cutoff wavelength in µm; features shorter
        than this are removed for `filter_type="lowpass"` (kept for
        "highpass"). `filter_type` is one of "lowpass", "highpass", "bandpass".
        """
        if filter_type not in ("lowpass", "highpass", "bandpass"):
            raise ValueError(f'Unknown filter_type "{filter_type}", expected "lowpass", "highpass" or "bandpass".')
        return cast(Profile, profile.filter(filter_type, cutoff, **kwargs))

    @staticmethod
    def _hampel_filter(profile: Profile, window_size: int = 5, n_sigmas: float = 3.0) -> Profile:
        """Replace outlier points with the local median (Hampel identifier),
        leaving everything else untouched - unlike `_gaussian_filter`, which
        blends every point with its neighbors.

        A point is an outlier if it deviates more than `n_sigmas` scaled MADs
        from its local median, computed over a `2 * window_size + 1` window.
        """
        size = 2 * window_size + 1
        data = profile.data
        local_median = ndimage.median_filter(data, size=size, mode="reflect")
        deviation = np.abs(data - local_median)
        mad = ndimage.median_filter(deviation, size=size, mode="reflect")
        # 1.4826 scales the MAD to be a consistent estimator of the std. dev. for normally-distributed data
        threshold = 1.4826 * n_sigmas * mad
        cleaned = np.where(deviation > threshold, local_median, data)
        return Profile(cleaned, profile.step, profile.length_um)

    def roughness_parameters(self, parameters: list[str] | None = None) -> dict[str, float]:
        """ISO 25178 areal roughness parameters (Sa, Sq, Sz, Sdr, ... by default all)."""
        return self.surface.roughness_parameters(parameters)

    def height_parameters(self) -> dict[str, float]:
        """ISO 25178 areal height parameters (Sa, Sq, Sz, Sv, Sp, Ssk, Sku)."""
        return self.surface.height_parameters()

    def plot_3d(self, savepath: str | Path | None = None, **kwargs):
        """Render the surface as a 3D shaded plot, optionally saving it.

        Delegates to `surfalize.Surface.plot_3d`, which renders via pyvista
        and returns a static `PIL.Image` (no `ax` support - it isn't a
        matplotlib figure).
        """
        return self.surface.plot_3d(save_to=savepath, **kwargs)

    def plot_2d(
        self,
        ax: Axes | None = None,
        savepath: str | Path | None = None,
        show: bool = True,
        **kwargs,
    ):
        """Render the surface as a top-down color-mapped plot, optionally saving it."""
        created_fig = ax is None
        fig, ax = self.surface.plot_2d(ax=ax, save_to=savepath, **kwargs)
        if show and created_fig:
            plt.show()
        elif created_fig and not show:
            plt.close(fig)
        return fig, ax

    def plot_profile(
        self,
        profile: Profile,
        filtered: Profile | None = None,
        ax_profile: Axes | None = None,
        ax_2d: Axes | None = None,
        show_2d: bool = True,
        savepath: str | Path | None = None,
        show: bool = True,
        **plot_2d_kwargs,
    ):
        """Plot a `Profile` (e.g. from `horizontal_profile`).

        By default this also plots the top-down 2D surface map alongside it,
        with a line marking where the profile was extracted from - taken
        from `profile.location`, which `horizontal_profile`/`vertical_profile`/
        `oblique_profile` set automatically. Pass `show_2d=False` to only
        plot the profile curve.

        Parameters
        ----------
        filtered : Profile, default None
            A filtered version of `profile` (e.g. from `filter_profile`) to
            draw on top of the raw signal, for comparison.
        ax_profile, ax_2d : matplotlib axes, default None
            If both are given, draw into them instead of creating a new
            figure - use this to embed the plot into a larger, custom figure
            layout. If `show_2d` is False, only `ax_profile` is needed.
        """
        created_fig = ax_profile is None and ax_2d is None

        if show_2d:
            if created_fig:
                fig, (ax_2d, ax_profile) = plt.subplots(1, 2, figsize=(14, 6))
            elif ax_profile is None or ax_2d is None:
                raise ValueError("show_2d=True requires both ax_profile and ax_2d if either is given.")
            else:
                fig = cast(Figure, ax_profile.figure)

            self.plot_2d(ax=ax_2d, show=False, **plot_2d_kwargs)
            location = getattr(profile, "location", None)
            if location is not None:
                assert ax_2d is not None
                ax_2d.plot([location.x0, location.x1], [location.y0, location.y1], color="red", lw=2)
        else:
            if ax_profile is None:
                fig, ax_profile = plt.subplots(figsize=(10, 4))
            else:
                fig = cast(Figure, ax_profile.figure)

        assert ax_profile is not None
        profile.plot_2d(ax=ax_profile)
        if filtered is not None:
            ax_profile.plot(
                np.linspace(0, filtered.length_um, filtered.data.size),
                filtered.data, color="tab:orange", lw=1.5, label="filtered",
            )
            ax_profile.legend()
        ax_profile.grid(True)

        if savepath is not None:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")
        if show and created_fig:
            plt.show()
        elif created_fig and not show:
            plt.close(fig)
        return fig, ax_profile, ax_2d


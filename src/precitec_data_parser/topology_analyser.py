"""
Author: Maxime Leurquin
Date: July 2026
Description: 3D surface and profile analysis for parsed `PrecitecData`, backed by `surfalize`.
"""

from typing import Any, cast
import numpy as np
from scipy import ndimage
from types import SimpleNamespace
from surfalize import Profile, Surface
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .data_parser import PrecitecData


class PrecitecSurfaceAnalyzer:
    """3D surface and profile analysis for parsed `PrecitecData`, backed by `surfalize`.

    Wraps one signal of the measurement (altitude by default, or intensity) in a
    `surfalize.Surface`, giving access to ISO 25178 areal roughness/height
    parameters, 2D/3D surface plotting, and extraction of
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
        data = PrecitecData("altitude.csv", "intensity.csv")
        analyzer = PrecitecSurfaceAnalyzer(data)
        print(analyzer.height_parameters())
        profile = analyzer.horizontal_profile(y=data.y[len(data.y) // 2])
        print(profile.roughness_parameters())
        analyzer.plot_3d(savepath="surface_3d.png")

        # Analyze the intensity channel instead:
        PrecitecSurfaceAnalyzer(data, signal="intensity").plot_2d()

    Parameters
    ----------
    data : PrecitecData
        Parsed measurement to analyze.
    signal : str, default "altitude"
        Which signal to wrap - "altitude" (topology) or "intensity" (reflectance).
    level : bool, default True
        Subtract the least-squares tilt plane (fitted from measured points only)
        so a tilted sample does not bias roughness/height parameters or profiles.
    fill_nonmeasured : bool, default True
        Interpolate non-measured samples when computing areal ISO parameters;
        surfalize treats NaN as non-measured and would otherwise skew them. This
        never affects plotting or profiles - those always leave non-measured
        points blank. Non-measured points are shared across signals and defined
        by the altitude channel.
    """

    def __init__(
        self,
        data: PrecitecData,
        signal: str = "altitude",
        level: bool = True,
        fill_nonmeasured: bool = True,
    ):
        self.data = data
        self.signal = signal.strip().lower()
        self.fill_nonmeasured = fill_nonmeasured
        # Kept unfilled so non-measured points stay NaN and render blank in
        # plots/profiles instead of showing interpolated heights.
        self.surface = data.to_surface(fill_nonmeasured=False, signal=self.signal)
        if level:
            self.surface = self.surface.level()
        self._filled_surface: Surface | None = None

    @property
    def _filled(self) -> Surface:
        """Surface with non-measured points interpolated (built once, on demand).

        Derived from `self.surface` so it inherits its tilt correction.
        """
        if self._filled_surface is None:
            self._filled_surface = (
                self.surface.fill_nonmeasured()
                if self.surface.has_missing_points else self.surface
            )
        return self._filled_surface

    @property
    def analysis_surface(self) -> Surface:
        """Surface used for areal ISO parameters, with non-measured points filled
        when `fill_nonmeasured` is set (see class docstring)."""
        return self._filled if self.fill_nonmeasured else self.surface

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
        read existing rows/columns with no interpolation. Because that spline
        prefilter would smear NaN across the whole line, values are sampled
        from the non-measured-filled surface and the samples that fall on
        non-measured points are then blanked back to NaN.
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
        data = ndimage.map_coordinates(self._filled.data, [yp, xp])
        # Blank samples touching non-measured data (order=1 flags any contribution).
        touched_nonmeasured = ndimage.map_coordinates(
            self.data.nonmeasured.astype(float), [yp, xp], order=1
        )
        data[touched_nonmeasured > 0] = np.nan

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
        return self.analysis_surface.roughness_parameters(parameters)

    def height_parameters(self) -> dict[str, float]:
        """ISO 25178 areal height parameters (Sa, Sq, Sz, Sv, Sp, Ssk, Sku)."""
        return self.analysis_surface.height_parameters()

    def plot_3d(self, savepath: str | Path | None = None, show: bool = False, **kwargs) -> go.Figure:
        """Build an interactive 3D surface plot of the signal, optionally saving it to an HTML file."""
        height_data = self.data.signals[self.signal]
        fig = go.Figure(data=[go.Surface(x=self.data.x, y=self.data.y, z=height_data)])
        fig.update_layout(title=dict(text='Height data'))
        if savepath is not None:
            fig.write_html(savepath)
        if show:
            fig.show()
        return fig

    def _heatmap_trace(self, **kwargs) -> go.Heatmap:
        """Build the top-down color-mapped `go.Heatmap` trace for `self.surface`.

        Row 0 of `surface.data` is the top of the measurement (y = height_um),
        matching the orientation used by `oblique_profile`/`horizontal_profile`/
        `vertical_profile` and surfalize's own `imshow`-based `plot_2d`.
        """
        ny, nx = self.surface.data.shape
        x = np.arange(nx) * self.surface.step_x
        y = (ny - 1 - np.arange(ny)) * self.surface.step_y
        kwargs.setdefault("colorscale", "Jet")
        kwargs.setdefault("colorbar", dict(title=self.signal))
        return go.Heatmap(z=self.surface.data, x=x, y=y, hovertemplate="x: %{x:.2f} µm<br>y: %{y:.2f} µm<br>z: %{z:.3f}<extra></extra>", **kwargs)

    def plot_2d(self, savepath: str | Path | None = None, show: bool = False, **kwargs) -> go.Figure:
        """Render the surface as an interactive top-down color-mapped plot, optionally saving it to an HTML file."""
        fig = go.Figure(data=[self._heatmap_trace(**kwargs)])
        fig.update_layout(title=dict(text=f"{self.signal.capitalize()} (top-down)"), xaxis_title="X (µm)", yaxis_title="Y (µm)")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        if savepath is not None:
            fig.write_html(savepath)
        if show:
            fig.show()
        return fig

    def plot_profile(
        self,
        profile: Profile,
        filtered: Profile | None = None,
        show_2d: bool = True,
        savepath: str | Path | None = None,
        show: bool = False,
        **plot_2d_kwargs,
    ) -> go.Figure:
        """Plot a `Profile` (e.g. from `horizontal_profile`) as an interactive figure.

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
        """
        x_profile = np.linspace(0, profile.length_um, profile.data.size)

        if show_2d:
            fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{self.signal.capitalize()} (top-down)", "Profile"))
            fig.add_trace(self._heatmap_trace(**plot_2d_kwargs), row=1, col=1)
            location = getattr(profile, "location", None)
            if location is not None:
                fig.add_trace(
                    go.Scatter(
                        x=[location.x0, location.x1], y=[location.y0, location.y1],
                        mode="lines", line=dict(color="red", width=2), showlegend=False,
                    ),
                    row=1, col=1,
                )
            fig.update_xaxes(title_text="X (µm)", row=1, col=1)
            fig.update_yaxes(title_text="Y (µm)", scaleanchor="x", scaleratio=1, row=1, col=1)
            fig.update_xaxes(title_text="Distance (µm)", row=1, col=2)
            fig.update_yaxes(title_text=self.signal, row=1, col=2)
        else:
            fig = go.Figure()

        raw_trace = go.Scatter(x=x_profile, y=profile.data, mode="lines", line=dict(color="black", width=1), name="raw")
        if show_2d:
            fig.add_trace(raw_trace, row=1, col=2)
        else:
            fig.add_trace(raw_trace)
        if filtered is not None:
            filtered_trace = go.Scatter(
                x=np.linspace(0, filtered.length_um, filtered.data.size), y=filtered.data,
                mode="lines", line=dict(color="orange", width=1.5), name="filtered",
            )
            if show_2d:
                fig.add_trace(filtered_trace, row=1, col=2)
            else:
                fig.add_trace(filtered_trace)

        if not show_2d:
            fig.update_layout(xaxis_title="Distance (µm)", yaxis_title=self.signal)

        if savepath is not None:
            fig.write_html(savepath)
        if show:
            fig.show()
        return fig

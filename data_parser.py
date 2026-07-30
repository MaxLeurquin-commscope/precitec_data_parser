"""
Author: Maxime Leurquin
Date: July 2026
Description: parses Precitec CLS2 height-map exports (.csv or .bcrf) for analysis and plotting.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from surfalize import Surface

def _convert_value(value: str) -> Any:
    """Convert a raw metadata string to int, float or bool where possible."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value.replace(",", "."))
    except ValueError:
        pass
    return {"True": True, "False": False}.get(value, value)


#: BCRF header lengths/offsets are given in unit count, but the header text is
#: UTF-16 (2 bytes/char), and physical lengths are in mm regardless of xunit/yunit.
_UM_PER_MM = 1000.0


class PrecitecData:
    """Parses Precitec CLS2 height-map exports (.csv or .bcrf) for analysis and plotting.

    Both formats are confirmed to carry the same underlying height/intensity
    values (verified against matching .csv/.bcrf export pairs, max abs diff
    ~5e-5, i.e. float32 vs. decimal-text rounding). The associated MountainsMap
    .mnt project file is a full analysis session (OLE compound document with a
    preview image, processing operators and profile-study results) on top of the
    same surface data, but its surface matrix is stored in a proprietary,
    undocumented binary blob and isn't parsed here.

    Example:
        with PrecitecData("my_file.csv") as data:
            x_data = data.x
            y_data = data.y
            z_data = data.z
            print(data.metadata)
            data.plot_data(savepath)
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            self.metadata, z, xstep, ystep = self._parse_csv()
        elif suffix == ".bcrf":
            self.metadata, z, xstep, ystep = self._parse_bcrf()
        else:
            raise NotImplementedError(
                f"Only .csv and .bcrf exports are supported, got '{self.path.suffix}'. "
                "The MountainsMap .mnt format stores its surface data in an "
                "undocumented, proprietary binary blob."
            )

        # z[j, i] is the value at (x[i], y[j]); rows = channels, columns = samples
        self.z = z
        n_channels, n_samples = z.shape
        self.x = np.arange(n_samples) * xstep
        self.y = np.arange(n_channels) * ystep

    def _parse_csv(self) -> tuple[dict[str, Any], np.ndarray, float, float]:
        # the metadata block ends at the first blank line, followed by the
        # "Channels" column-index row (used by pandas as the data header)
        metadata_lines: list[str] = []
        with open(self.path, encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline().strip()
                if line == "":
                    break
                metadata_lines.append(line)

        metadata = {
            parts[0]: _convert_value(parts[1]) if len(parts) == 2
            else np.array([_convert_value(v) for v in parts[1:]])
            for parts in (line.split(";") for line in metadata_lines)
        }

        df = pd.read_csv(
            self.path,
            sep=";",
            decimal=",",
            skiprows=len(metadata_lines) + 1,  # metadata lines + blank line
            index_col=0,
            header=0,
            encoding="utf-8",
            encoding_errors="replace",
        )
        # a trailing ';' on each data row produces a spurious all-NaN column
        df = df.dropna(axis=1, how="all")

        if int(metadata.get("NumberOfLines", 1)) > 1:
            df = self._stitch_lines(df, metadata)

        xstep = float(metadata["XStep"])
        ystep = float(metadata["YStep"])
        return metadata, df.to_numpy().T, xstep, ystep

    @staticmethod
    def _stitch_lines(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        """Join separately-scanned channel tiles into one full-width surface.

        When `NumberOfLines` > 1, the sensor re-positioned between passes and each
        pass only covers `NumberOfChannels` channels; the CSV stores the passes
        (`NumberOfLines` of them, each `NumberOfSamplesPerLine` rows) stacked one
        after another in the row axis. Stitching them side-by-side as extra
        channels (columns) reconstructs the true, wider measurement - confirmed
        against the matching `.bcrf` export, whose pixel dimensions are
        `NumberOfSamplesPerLine` x `NumberOfLines * NumberOfChannels`. If a pass
        was captured on the return (reverse) sweep, `OddLineReturned` says its
        channel order needs flipping before stitching to keep the axis monotonic.
        """
        n_lines = int(metadata["NumberOfLines"])
        samples_per_line = int(metadata["NumberOfSamplesPerLine"])
        odd_line_returned = bool(metadata.get("OddLineReturned", False))

        tiles = []
        for i in range(n_lines):
            tile = df.iloc[i * samples_per_line : (i + 1) * samples_per_line]
            if odd_line_returned and i % 2 == 1:
                tile = tile.iloc[:, ::-1]
            tiles.append(tile.reset_index(drop=True))
        return pd.concat(tiles, axis=1, ignore_index=True)

    def _parse_bcrf(self) -> tuple[dict[str, Any], np.ndarray, float, float]:
        raw = self.path.read_bytes()

        # the text header is UTF-16 and '%'-padded; its true length (in bytes)
        # is 2x the declared "headersize" (which counts UTF-16 characters)
        preview = raw[:8192].decode("utf-16-le", errors="ignore")
        fields: dict[str, Any] = {}
        for line in preview.split("\r\n"):
            key, sep, value = line.partition("=")
            if not sep or key.strip("%") == "":
                break
            fields[key.strip()] = _convert_value(value.strip())

        header_bytes = int(fields["headersize"]) * 2
        xpixels = int(fields["xpixels"])
        ypixels = int(fields["ypixels"])
        dtype = "<f4" if fields.get("intelmode", 1) else ">f4"

        z = np.frombuffer(raw[header_bytes:], dtype=dtype).reshape(ypixels, xpixels)

        um_per_unit = {"mm": _UM_PER_MM, "um": 1.0, "nm": 1e-3}
        xstep = float(fields["xlength"]) * um_per_unit[fields["xunit"]] / xpixels
        ystep = float(fields["ylength"]) * um_per_unit[fields["yunit"]] / ypixels
        return fields, z, xstep, ystep

    def __enter__(self) -> "PrecitecData":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def z_masked(self) -> np.ndarray:
        """Height data with zero (no-measurement) samples replaced by NaN."""
        z = self.z.astype(float)
        z[z == 0] = np.nan
        return z

    def to_surface(self, fill_nonmeasured: bool = True) -> Surface:
        """Convert to a `surfalize.Surface` for areal/profile analysis.

        Parameters
        ----------
        fill_nonmeasured : bool, default True
            Interpolate zero (no-measurement) samples before conversion;
            surfalize treats NaN as non-measured and would otherwise skew
            ISO parameters.
        """
        xstep = float(self.x[1] - self.x[0])
        ystep = float(self.y[1] - self.y[0])
        z = self.z_masked() if fill_nonmeasured else self.z

        surface = Surface(z, xstep, ystep, metadata=self.metadata)
        if fill_nonmeasured and surface.has_missing_points:
            surface = surface.fill_nonmeasured()
        return surface

    def plot_data(
        self,
        ax: Axes | None = None,
        savepath: str | Path | None = None,
        show: bool = True,
        mask_zero: bool = True,
    ):
        """Plot the height map, optionally saving it to `savepath`.

        Parameters
        ----------
        ax : matplotlib axis, default None
            If given, draw on this axis instead of creating a new figure - use
            this to embed the plot into a larger, custom figure layout. When
            an axis is supplied, `show`/`savepath` still apply, but this
            function will not close the figure on your behalf.
        """
        created_fig = ax is None
        z = self.z_masked() if mask_zero else self.z
        nonzero = self.z[self.z != 0]
        vmin = float(np.nanmin(nonzero)) if nonzero.size else float(np.nanmin(self.z))
        vmax = float(np.nanmax(self.z))

        cmap = plt.get_cmap("jet").copy()
        cmap.set_bad(color="black")

        if created_fig:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = cast(Figure, ax.figure)
        im = ax.imshow(
            z,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]),
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, label="Height (µm)")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_title(self.path.name)

        if savepath is not None:
            fig.savefig(savepath, dpi=200, bbox_inches="tight")
        if show and created_fig:
            plt.show()
        elif created_fig and not show:
            plt.close(fig)
        return fig, ax
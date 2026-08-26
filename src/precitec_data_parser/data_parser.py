"""
Author: Maxime Leurquin.
Date: July 2026
Description: parse Precitec CLS2 exports (.csv or .bcrf) into altitude/intensity arrays.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from surfalize import Surface

# BCRF physical lengths are in mm regardless of the reported unit.
_UM_PER_MM = 1000.0

# Precitec signal identifier (CSV "IdSignal" / BCRF "signal_id") to signal name.
_SIGNAL_BY_ID: dict[int, str] = {16640: "altitude", 16641: "intensity"}


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

class PrecitecData:
    """Parse a Precitec measurement pair (altitude + intensity) into one object.

    Each measurement is exported as two files - one altitude signal and one
    intensity signal - sharing the same geometry. Both paths are supplied by the
    caller and validated to have identical shapes and axis steps. CSV and BCRF
    exports are both supported.
    """

    def __init__(self, altitude_path: str | Path, intensity_path: str | Path):
        self.altitude_path = Path(altitude_path)
        self.intensity_path = Path(intensity_path)

        self.metadata_altitude, self.altitude, xstep, ystep = self._parse(self.altitude_path)
        self.metadata_intensity, self.intensity, i_xstep, i_ystep = self._parse(self.intensity_path)

        if self.altitude.shape != self.intensity.shape:
            raise ValueError(
                f"Altitude/intensity shape mismatch: {self.altitude.shape} vs {self.intensity.shape}."
            )
        if not (np.isclose(xstep, i_xstep) and np.isclose(ystep, i_ystep)):
            raise ValueError(
                f"Altitude/intensity step mismatch: ({xstep}, {ystep}) vs ({i_xstep}, {i_ystep})."
            )

        self.xstep = xstep
        self.ystep = ystep
        self.signals = {"altitude": self.altitude, "intensity": self.intensity}

        # Non-measured points are where the sensor found no surface (altitude == 0);
        # this shared mask must drive both signals - intensity can legitimately read
        # 0 at a measured point, so it cannot define "non-measured" on its own.
        self.nonmeasured = self.altitude == 0.0

        # z[j, i] is the value at (x[i], y[j]); rows = channels, columns = samples.
        n_channels, n_samples = self.altitude.shape
        self.x = np.arange(n_samples) * xstep
        self.y = np.arange(n_channels) * ystep

    @staticmethod
    def _decode_signal(metadata: dict[str, Any]) -> Literal["altitude", "intensity"] | None:
        """Return 'altitude'/'intensity' from the metadata's signal id, else None."""
        signal_id = metadata.get("IdSignal") or metadata.get("signal_id")
        if signal_id is None:
            return None
        try:
            result = _SIGNAL_BY_ID.get(int(signal_id))
            return cast(Literal["altitude", "intensity"] | None, result)
        except (TypeError, ValueError):
            return None
        
    @classmethod
    def _parse(cls, path: Path) -> tuple[dict[str, Any], np.ndarray, float, float]:
        """Parse a single export into (metadata, z, xstep, ystep)."""
        suffix = path.suffix.lower()
        match suffix:
            case ".csv":
                metadata, z, xstep, ystep = cls._parse_csv(path)
            case ".bcrf":
                metadata, z, xstep, ystep = cls._parse_bcrf(path)
            case _:
                raise NotImplementedError(
                    f"Only .csv and .bcrf exports are supported, got '{path.suffix}'. "
                    "The MountainsMap .mnt format stores its surface data in an "
                    "undocumented, proprietary binary blob."
                )
        metadata["Signal"] = cls._decode_signal(metadata)
        return metadata, z, xstep, ystep


    @classmethod
    def find_file_pairs(cls,filepaths:list[Path]) -> list[tuple[Path, Path]]:
        """
        Given a list of filepaths identify the altitude/intensity file pairs and 
        create a list of (altitude_path, intensity_path) tuples from them.
        
        File pairs are identified by having the same base name when ignoring the altitude or intensity suffixes.
        For example:
            - "measurement1_altitude.csv" and "measurement1_intensity.csv".
            - "measurement2_100khz_altitude.bcrf" and "measurement2_intensity_100khz.bcrf".
        
        altitude or intensity suffixes are case-insensitive.
        Raises:
            - finds multiple altitude or intensity files for the same base name (happens if filepaths contains both .csv and .bcrf for the same measurement).
            - finds any altitude file without a corresponding intensity file, or vice versa.
            - finds a file named with altitude or intensity but the metadata indicates the opposite signal type.
        """
        altitude_files, intensity_files = {},{}
        
        files = [Path(f) for f in filepaths if Path(f).is_file()]
        for file in files:
            # Extract base name by removing altitude/intensity suffix (case-insensitive)
            stem = file.stem
            stem_lower = stem.lower()
            
            if "_altitude" in stem_lower:
                idx = stem_lower.index("_altitude")
                base_name = stem[:idx] + stem[idx + len("_altitude"):]
                signal_type = "altitude"
            elif "_intensity" in stem_lower:
                idx = stem_lower.index("_intensity")
                base_name = stem[:idx] + stem[idx + len("_intensity"):]
                signal_type = "intensity"
            else:
                raise ValueError(f"File '{file.name}' missing '_altitude' or '_intensity' in name.")
            
            # Verify metadata matches filename
            metadata, _, _, _ = cls._parse(file)
            actual_signal = metadata.get("Signal")
            if actual_signal != signal_type:
                raise ValueError(
                    f"File '{file.name}': filename indicates {signal_type} "
                    f"but metadata indicates {actual_signal}."
                )
            
            # Store in appropriate dict
            target_dict = altitude_files if signal_type == "altitude" else intensity_files
            if base_name in target_dict:
                raise ValueError(
                    f"Duplicate {signal_type} file for base name '{base_name}'. "
                    f"You probably provided both .csv and .bcrf for the same measurement."
                )
            target_dict[base_name] = file
        
        # Validate pairing
        altitude_keys = set(altitude_files.keys())
        intensity_keys = set(intensity_files.keys())
        
        missing_intensity = altitude_keys - intensity_keys
        missing_altitude = intensity_keys - altitude_keys
        
        if missing_intensity:
            raise ValueError(f"Missing intensity files for: {', '.join(sorted(missing_intensity))}.")
        if missing_altitude:
            raise ValueError(f"Missing altitude files for: {', '.join(sorted(missing_altitude))}.")
        return [(altitude_files[key], intensity_files[key]) for key in sorted(altitude_keys)]


    @classmethod
    def from_file_pairs(cls, file_pairs: list[tuple[Path,Path]]) -> list[PrecitecData]:
        """
        Given a list of (altitude_path, intensity_path) tuples, create a list of PrecitecData objects from them.
        """
        return [cls(altitude, intensity) for altitude, intensity in file_pairs]

    @classmethod
    def from_folder(cls, folder: Path, filetype: Literal["csv", "bcrf"]) -> list[PrecitecData]:
        """
        Given a folder, identify the file pairs of the given filetype in it.
        Then create a list of PrecitecData objects from them.
        """
        file_pairs= cls.find_file_pairs(folder, filetype)
        return [cls(altitude, intensity) for altitude, intensity in file_pairs]

    def __repr__(self) -> str:
        return (
            f"PrecitecData(altitude_path={self.altitude_path}, "
            f"intensity_path={self.intensity_path}, "
            f"xstep={self.xstep}, ystep={self.ystep}, "
            f"shape={self.altitude.shape})"
        )

    @staticmethod
    def _read_csv_metadata(path: Path) -> dict[str, Any]:
        """Read the leading ';'-delimited metadata block (ends at first blank line)."""
        lines: list[str] = []
        with open(path, encoding="utf-8", errors="replace") as f:
            while (line := f.readline().strip()) != "":
                lines.append(line)
        return {
            parts[0]: _convert_value(parts[1]) if len(parts) == 2
            else np.array([_convert_value(v) for v in parts[1:]])
            for parts in (line.split(";") for line in lines)
        }

    @classmethod
    def _parse_csv(cls, path: Path) -> tuple[dict[str, Any], np.ndarray, float, float]:
        metadata = cls._read_csv_metadata(path)
        # One metadata line per (unique) key, then a blank line and the "Channels" header row.
        n_metadata_lines = len(metadata)

        df = pd.read_csv(
            path,
            sep=";",
            decimal=",",
            skiprows=n_metadata_lines + 1,
            index_col=0,
            header=0,
            encoding="utf-8",
            encoding_errors="replace",
        )
        # A trailing ';' per row can create a spurious all-NaN column.
        df = df.dropna(axis=1, how="all")

        if int(metadata.get("NumberOfLines", 1)) > 1:
            df = cls._stitch_lines(df, metadata)

        return metadata, df.to_numpy().T, float(metadata["XStep"]), float(metadata["YStep"])

    @staticmethod
    def _stitch_lines(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        """Join separately-scanned channel tiles into one full-width surface."""
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

    @staticmethod
    def _read_bcrf_header(path: Path) -> dict[str, Any]:
        # Header text is UTF-16 (2 bytes/char); 8 KiB covers it comfortably.
        with open(path, "rb") as f:
            preview = f.read(8192).decode("utf-16-le", errors="ignore")
        fields: dict[str, Any] = {}
        for line in preview.split("\r\n"):
            key, sep, value = line.partition("=")
            if not sep or key.strip("%") == "":
                break
            fields[key.strip()] = _convert_value(value.strip())
        return fields

    @classmethod
    def _parse_bcrf(cls, path: Path) -> tuple[dict[str, Any], np.ndarray, float, float]:
        fields = cls._read_bcrf_header(path)
        header_bytes = int(fields["headersize"]) * 2  # unit count -> bytes (UTF-16)
        xpixels = int(fields["xpixels"])
        ypixels = int(fields["ypixels"])
        dtype = "<f4" if fields.get("intelmode", 1) else ">f4"

        z = np.frombuffer(path.read_bytes()[header_bytes:], dtype=dtype).reshape(ypixels, xpixels)

        um_per_unit = {"mm": _UM_PER_MM, "um": 1.0, "nm": 1e-3}
        xstep = float(fields["xlength"]) * um_per_unit[fields["xunit"]] / xpixels
        ystep = float(fields["ylength"]) * um_per_unit[fields["yunit"]] / ypixels
        return fields, z, xstep, ystep

    def signal_data(self, signal: Literal["altitude", "intensity"]) -> np.ndarray:
        """Return the altitude or intensity matrix."""
        try:
            return self.signals[signal.strip().lower()]
        except KeyError:
            raise KeyError("Signal must be 'altitude' or 'intensity'.") from None

    def to_surface(self, signal:Literal["altitude", "intensity"],fill_nonmeasured: bool = True, ) -> Surface:
        """Convert the altitude or intensity data to a surfalize Surface.

        signal: whether to use the altitude or intensity surface

        Non-measured points are always marked as NaN (surfalize renders them
        blank and excludes them from ISO parameters). `fill_nonmeasured` only
        controls whether they are then interpolated from their neighbours -
        enable it for areal roughness/height parameters, leave it off to keep
        the holes empty when plotting.
        """
        z = self.signal_data(signal).astype(float)
        z[self.nonmeasured] = np.nan
        metadata = self.metadata_altitude if signal == "altitude" else self.metadata_intensity

        surface = Surface(z, self.xstep, self.ystep, metadata=metadata)
        if fill_nonmeasured and surface.has_missing_points:
            surface = surface.fill_nonmeasured()
        return surface

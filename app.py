"""
Streamlit viewer for Precitec CLS2 measurements (altitude + intensity).

Usage:
    streamlit run app.py
"""
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

from precitec_data_parser import PrecitecData, PrecitecSurfaceAnalyzer


@st.cache_resource(show_spinner="Parsing measurement files …")
def load_data(altitude_bytes: bytes, altitude_name: str, intensity_bytes: bytes, intensity_name: str) -> PrecitecData:
    altitude_path = _save_upload_bytes(altitude_bytes, altitude_name)
    intensity_path = _save_upload_bytes(intensity_bytes, intensity_name)
    return PrecitecData(altitude_path, intensity_path)


def _save_upload_bytes(data: bytes, name: str) -> Path:
    suffix = Path(name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


@st.cache_resource(show_spinner="Applying threshold …")
def threshold_copy(
    _original: PrecitecData,
    files_hash: str,
    thresh: float,
    transpose: bool,
    flip_x: bool,
) -> PrecitecData:
    """Thresholded deep copy, built once per (files_hash, thresh) and reused (never mutated) across reruns."""
    data = _original.copy()
    data.reorient(swap_axes=transpose, flip_x=flip_x, inplace=True)
    data.threshold_intensity(thresh, inplace=True)
    return data


@st.cache_resource(show_spinner="Building surface analyzers …")
def get_analyzers(
    _data: PrecitecData,
    files_hash: str,
    thresh: float,
    transpose: bool,
    flip_x: bool,
    level: bool,
) -> tuple[PrecitecSurfaceAnalyzer, PrecitecSurfaceAnalyzer]:
    """Build both analyzers once per (files_hash, thresh); this is where the costly to_surface()/level() work happens."""
    analyzer_alt = PrecitecSurfaceAnalyzer(_data, signal="altitude", level=level)
    analyzer_int = PrecitecSurfaceAnalyzer(_data, signal="intensity", level=False)
    return analyzer_alt, analyzer_int


@st.cache_resource(show_spinner="Building 3D surface …")
def get_plot_3d(
    _analyzer: PrecitecSurfaceAnalyzer,
    files_hash: str,
    thresh: float,
    transpose: bool,
    flip_x: bool,
    level: bool,
):
    return _analyzer.plot_3d()


def _sync_widget_value(source_key: str, target_key: str) -> None:
    st.session_state[target_key] = st.session_state[source_key]


def main():
    st.set_page_config(page_title="Precitec Measurement Viewer", layout="wide")
    st.title("Precitec Measurement Viewer")

    with st.sidebar:
        st.header("Files")
        altitude_upload = st.file_uploader("Altitude file (.csv/.bcrf)", type=["csv", "bcrf"])
        intensity_upload = st.file_uploader("Intensity file (.csv/.bcrf)", type=["csv", "bcrf"])

    if altitude_upload is None or intensity_upload is None:
        st.info("Upload both an altitude and an intensity file to begin.")
        st.stop()

    altitude_bytes = altitude_upload.getvalue()
    intensity_bytes = intensity_upload.getvalue()
    files_hash = hashlib.sha256(altitude_bytes + intensity_bytes).hexdigest()

    original_data = load_data(altitude_bytes, altitude_upload.name, intensity_bytes, intensity_upload.name)

    with st.sidebar:
        st.header("Data preparation")
        transpose = st.checkbox("Transpose axes", key="transpose_axes")
        flip_x = st.checkbox("Flip X", key="flip_x")
        level = st.checkbox("Level altitude surface", value=True, key="level_altitude")

    measured_intensity = original_data.intensity[np.isfinite(original_data.intensity)]
    if measured_intensity.size == 0:
        st.warning("No measured intensity values are available for thresholding.")
        st.stop()

    with st.sidebar:
        st.header("Threshold")
        threshold_min = float(measured_intensity.min())
        threshold_max = float(measured_intensity.max())
        threshold_key = f"threshold_{files_hash[:12]}"
        threshold_input_key = f"{threshold_key}_exact"
        st.session_state.setdefault(threshold_key, threshold_min)
        st.session_state.setdefault(threshold_input_key, st.session_state[threshold_key])
        threshold_slider_col, threshold_input_col = st.columns([3, 1])
        with threshold_slider_col:
            st.slider(
                "Intensity threshold (mask below)",
                min_value=threshold_min,
                max_value=threshold_max,
                key=threshold_key,
                on_change=_sync_widget_value,
                args=(threshold_key, threshold_input_key),
            )
        with threshold_input_col:
            st.number_input(
                "Exact threshold",
                min_value=threshold_min,
                max_value=threshold_max,
                step=0.01,
                format="%.6f",
                key=threshold_input_key,
                on_change=_sync_widget_value,
                args=(threshold_input_key, threshold_key),
            )
        thresh = float(st.session_state[threshold_key])

    data = threshold_copy(original_data, files_hash, thresh, transpose, flip_x)
    analyzer_alt, analyzer_int = get_analyzers(
        data, files_hash, thresh, transpose, flip_x, level
    )
    plot_revision = (
        f"{files_hash[:12]}_{int(transpose)}_{int(flip_x)}_{int(level)}"
    )

    @st.fragment
    def profile_and_plots():
        with st.sidebar:
            st.header("Profile")
            orientation = st.radio("Orientation", ["Horizontal", "Vertical"], horizontal=True)
            if orientation == "Horizontal":
                axis_values, label, state_key = data.y, "Y position (µm)", "position_h"
            else:
                axis_values, label, state_key = data.x, "X position (µm)", "position_v"

            min_val, max_val = float(axis_values.min()), float(axis_values.max())
            step = float(axis_values[1] - axis_values[0]) if len(axis_values) > 1 else 1.0
            state_key = (
                f"{state_key}_{files_hash[:12]}_{int(transpose)}_"
                f"{int(flip_x)}"
            )
            input_key = f"{state_key}_exact"
            st.session_state.setdefault(
                state_key, float(axis_values[len(axis_values) // 2])
            )
            st.session_state.setdefault(input_key, st.session_state[state_key])

            col_minus, col_slider, col_plus = st.columns([1, 8, 1])
            with col_minus:
                if st.button("➖", key=f"{state_key}_minus"):
                    position = max(min_val, st.session_state[state_key] - step)
                    st.session_state[state_key] = position
                    st.session_state[input_key] = position
            with col_plus:
                if st.button("➕", key=f"{state_key}_plus"):
                    position = min(max_val, st.session_state[state_key] + step)
                    st.session_state[state_key] = position
                    st.session_state[input_key] = position
            with col_slider:
                st.slider(
                    label,
                    min_val,
                    max_val,
                    key=state_key,
                    on_change=_sync_widget_value,
                    args=(state_key, input_key),
                )
            position_input_col, _ = st.columns([1, 2])
            with position_input_col:
                st.number_input(
                    f"Exact {label.lower()}",
                    min_value=min_val,
                    max_value=max_val,
                    step=step,
                    format="%.6f",
                    key=input_key,
                    on_change=_sync_widget_value,
                    args=(input_key, state_key),
                )
            position = float(st.session_state[state_key])

            if orientation == "Horizontal":
                profile = analyzer_alt.horizontal_profile(y=position)
            else:
                profile = analyzer_alt.vertical_profile(x=position)

        with st.sidebar:
            st.header("Display")
            intensity_grayscale = st.checkbox("Display intensity in grayscale")

        with st.sidebar:
            st.header("Filter")
            method = st.selectbox("Method", ["None", "gaussian", "hampel"])
            filter_args: dict = {}
            if method == "gaussian":
                filter_args["cutoff"] = st.number_input("Cutoff (µm)", min_value=0.1, value=10.0)
                filter_args["filter_type"] = st.selectbox("Filter type", ["lowpass", "highpass", "bandpass"])
                if filter_args["filter_type"] == "bandpass":
                    filter_args["cutoff2"] = st.number_input("Cutoff 2 (µm, > cutoff)", min_value=filter_args["cutoff"] + 0.1, value=filter_args["cutoff"] + 10.0)
            elif method == "hampel":
                filter_args["window_size"] = st.slider("Window size", 1, 20, 5)
                filter_args["n_sigmas"] = st.slider("N sigmas", 0.5, 10.0, 3.0)

        filtered = analyzer_alt.filter_profile(profile, method=method, filter_args=filter_args) if method != "None" else None

        col_alt, col_int = st.columns(2)
        for col, analyzer, title in ((col_alt, analyzer_alt, "Altitude"), (col_int, analyzer_int, "Intensity")):
            plot_kwargs = {"colorscale": "Greys_r"} if title == "Intensity" and intensity_grayscale else {}
            fig = analyzer.plot_2d(**plot_kwargs)
            fig.add_scatter(
                x=[profile.location.x0, profile.location.x1],
                y=[profile.location.y0, profile.location.y1],
                mode="lines", line=dict(color="red", width=2), showlegend=False,
            )
            fig.update_xaxes(
                range=[float(data.x[0]), float(data.x[-1])],
                constrain="domain",
            )
            fig.update_layout(title=title, uirevision=plot_revision)
            col.plotly_chart(
                fig,
                width="stretch",
                key=f"surface_{title.lower()}_{plot_revision}",
            )

        st.subheader("Extracted profile")
        fig = analyzer_alt.plot_profile(profile, filtered=filtered, show_2d=False)
        fig.update_layout(uirevision=plot_revision)
        st.plotly_chart(
            fig, width="stretch", key=f"profile_{plot_revision}"
        )

    profile_and_plots()

    with st.sidebar:
        st.header("3D view")
        show_3d = st.checkbox("Show 3D surface (altitude)", value=False)

    if show_3d:
        st.subheader("3D surface (altitude)")
        figure_3d = get_plot_3d(
            analyzer_alt, files_hash, thresh, transpose, flip_x, level
        )
        figure_3d.update_layout(uirevision=plot_revision)
        st.plotly_chart(
            figure_3d,
            width="stretch",
            key=f"surface_3d_{plot_revision}",
        )


if __name__ == "__main__":
    main()

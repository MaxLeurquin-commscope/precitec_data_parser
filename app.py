"""
Streamlit viewer for Precitec CLS2 measurements (altitude + intensity).

Usage:
    streamlit run app.py
"""
import copy
import hashlib
import tempfile
from pathlib import Path

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
def threshold_copy(_original: PrecitecData, files_hash: str, thresh: float) -> PrecitecData:
    """Thresholded deep copy, built once per (files_hash, thresh) and reused (never mutated) across reruns."""
    data = copy.deepcopy(_original)
    if thresh > float(data.intensity.min()):
        data.threshold_data(thresh)
    return data


@st.cache_resource(show_spinner="Building surface analyzers …")
def get_analyzers(_data: PrecitecData, files_hash: str, thresh: float) -> tuple[PrecitecSurfaceAnalyzer, PrecitecSurfaceAnalyzer]:
    """Build both analyzers once per (files_hash, thresh); this is where the costly to_surface()/level() work happens."""
    analyzer_alt = PrecitecSurfaceAnalyzer(_data, signal="altitude", level=True)
    analyzer_int = PrecitecSurfaceAnalyzer(_data, signal="intensity", level=False)
    return analyzer_alt, analyzer_int


@st.cache_resource(show_spinner="Building 3D surface …")
def get_plot_3d(_analyzer: PrecitecSurfaceAnalyzer, files_hash: str, thresh: float):
    return _analyzer.plot_3d()


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
        st.header("Threshold")
        thresh = st.slider(
            "Intensity threshold (mask below)",
            min_value=float(original_data.intensity.min()),
            max_value=float(original_data.intensity.max()),
            value=float(original_data.intensity.min()),
        )

    data = threshold_copy(original_data, files_hash, thresh)
    analyzer_alt, analyzer_int = get_analyzers(data, files_hash, thresh)

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
            if state_key not in st.session_state:
                st.session_state[state_key] = float(axis_values[len(axis_values) // 2])

            col_minus, col_slider, col_plus = st.columns([1, 8, 1])
            with col_minus:
                if st.button("➖", key=f"{state_key}_minus"):
                    st.session_state[state_key] = max(min_val, st.session_state[state_key] - step)
            with col_plus:
                if st.button("➕", key=f"{state_key}_plus"):
                    st.session_state[state_key] = min(max_val, st.session_state[state_key] + step)
            with col_slider:
                position = st.slider(label, min_val, max_val, key=state_key)

            if orientation == "Horizontal":
                profile = analyzer_alt.horizontal_profile(y=position)
            else:
                profile = analyzer_alt.vertical_profile(x=position)

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
            fig = analyzer.plot_2d()
            fig.add_scatter(
                x=[profile.location.x0, profile.location.x1],
                y=[profile.location.y0, profile.location.y1],
                mode="lines", line=dict(color="red", width=2), showlegend=False,
            )
            fig.update_layout(title=title)
            col.plotly_chart(fig, width="stretch")

        st.subheader("Extracted profile")
        fig = analyzer_alt.plot_profile(profile, filtered=filtered, show_2d=False)
        st.plotly_chart(fig, width="stretch")

    profile_and_plots()

    with st.sidebar:
        st.header("3D view")
        show_3d = st.checkbox("Show 3D surface (altitude)", value=False)

    if show_3d:
        st.subheader("3D surface (altitude)")
        st.plotly_chart(get_plot_3d(analyzer_alt, files_hash, thresh), width="stretch")


if __name__ == "__main__":
    main()

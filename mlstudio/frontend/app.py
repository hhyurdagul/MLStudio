import streamlit as st

from .pages import (
    render_test_page,
    render_training_page,
    render_validation_page,
)


_WORKSPACE_KEY = "_navigation_workspace"
_SUPERVISED_MODE_KEY = "_navigation_supervised_mode"
_TIMESERIES_MODE_KEY = "_navigation_timeseries_mode"


def _clear_page_state(*preserved_keys: str) -> None:
    preserved_state = {
        key: st.session_state[key]
        for key in preserved_keys
        if key in st.session_state
    }
    st.session_state.clear()
    st.session_state.update(preserved_state)
    st.cache_data.clear()
    st.cache_resource.clear()


def _clear_workspace_state() -> None:
    _clear_page_state(_WORKSPACE_KEY)


def _clear_supervised_mode_state() -> None:
    _clear_page_state(_WORKSPACE_KEY, _SUPERVISED_MODE_KEY)


def _clear_timeseries_mode_state() -> None:
    _clear_page_state(_WORKSPACE_KEY, _TIMESERIES_MODE_KEY)


def run_app(
    attach_feature_selection: bool = False, attach_lookback: bool = False
) -> None:
    st.set_page_config(page_title="MLStudio", layout="wide")
    st.title("MLStudio")
    st.caption("Train, validate, and test regression and time-series models.")

    workspace = st.sidebar.radio(
        "Workspace",
        ["Time Series", "Supervised"],
        key=_WORKSPACE_KEY,
        on_change=_clear_workspace_state,
    )
    if workspace == "Time Series":
        from .pages.timeseries import render_timeseries_page

        render_timeseries_page(
            on_mode_change=_clear_timeseries_mode_state,
        )
        return

    st.header("Supervised")
    st.caption(
        "Train machine learning models on supervised data."
    )
    mode = st.radio(
        "Mode",
        ["Model Training", "Model Testing", "Train-Test On Training Data"],
        horizontal=True,
        key=_SUPERVISED_MODE_KEY,
        on_change=_clear_supervised_mode_state,
    )
    if mode == "Model Training":
        render_training_page(
            attach_feature_selection=attach_feature_selection,
            attach_lookback=attach_lookback,
        )
    elif mode == "Model Testing":
        render_test_page()
    else:
        render_validation_page(
            attach_feature_selection=attach_feature_selection,
            attach_lookback=attach_lookback,
        )

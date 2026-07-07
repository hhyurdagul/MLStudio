import streamlit as st

from .pages import (
    render_test_page,
    render_training_page,
    render_validation_page,
)


def run_app(
    attach_feature_selection: bool = False, attach_lookback: bool = False
) -> None:
    st.set_page_config(page_title="MLStudio", layout="wide")
    st.title("MLStudio")
    st.caption("Train, validate, and test regression and time-series models.")

    workspace = st.sidebar.radio(
        "Workspace",
        ["Time Series", "Supervised"],
    )
    if workspace == "Time Series":
        from .pages.timeseries import render_timeseries_page

        render_timeseries_page()
        return

    st.header("Supervised")
    st.caption(
        "Train machine learning models on supervised data."
    )
    mode = st.radio(
        "Mode",
        ["Model Training", "Model Testing", "Train-Test On Training Data"],
        horizontal=True,
    )
    if mode == "Training":
        render_training_page(
            attach_feature_selection=attach_feature_selection,
            attach_lookback=attach_lookback,
        )
    elif mode == "Test":
        render_test_page()
    else:
        render_validation_page(
            attach_feature_selection=attach_feature_selection,
            attach_lookback=attach_lookback,
        )

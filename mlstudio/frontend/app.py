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
        ["Supervised", "Time Series"],
    )
    if workspace == "Time Series":
        from .pages.deep_learning import render_deep_learning_page

        render_deep_learning_page()
        return

    mode = st.radio(
        "Mode",
        ["Training", "Validation", "Test"],
        horizontal=True,
    )
    if mode == "Training":
        render_training_page(
            attach_feature_selection=attach_feature_selection,
            attach_lookback=attach_lookback,
        )
    elif mode == "Validation":
        render_validation_page(
            attach_feature_selection=attach_feature_selection,
            attach_lookback=attach_lookback,
        )
    elif mode == "Test":
        render_test_page()

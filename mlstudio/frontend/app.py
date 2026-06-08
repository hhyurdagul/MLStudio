import streamlit as st

from .pages import render_test_page, render_training_page, render_validation_page


def run_app() -> None:
    st.set_page_config(page_title="MLStudio", layout="wide")
    st.title("MLStudio")
    st.caption("Train, validate, and test regression models.")

    mode = st.radio(
        "Mode",
        ["Training", "Validation", "Test"],
        horizontal=True
    )
    if mode == "Training":
        render_training_page()
    elif mode == "Validation":
        render_validation_page()
    else:
        render_test_page()

from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Callable
import streamlit as st
import polars as pl


def render_input_training_data_component(
    read_data: Callable[[UploadedFile | None], pl.DataFrame | None],
) -> tuple[pl.DataFrame | None, list[str], str | None]:
    col1, col2, col3 = st.columns([6, 7, 6])
    file_selector = col1.file_uploader(
        "Upload Train File", type=[".csv", ".xlsx"], accept_multiple_files=False
    )

    df = read_data(file_selector)
    columns = [] if df is None else df.columns

    target_selector = col3.selectbox("Select target", columns)
    feature_selector = col2.multiselect(
        "Select features", [i for i in columns if i != target_selector]
    )

    return df, feature_selector, target_selector


def render_data_component(df: pl.DataFrame, features: list[str], target: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.write("Loaded Data")
        st.dataframe(df)
    with col2:
        st.write("Selected Training Data")
        st.dataframe(
            df.select(pl.col(features + [target]))
            .to_pandas()
            .style.map(lambda _: "color: darkorange;", subset=[target]),
            hide_index=True,
        )

from collections.abc import Callable

import polars as pl
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def render_dataset_selector(
    label: str,
    read_data: Callable[[UploadedFile | None], pl.DataFrame | None],
    *,
    key: str,
) -> pl.DataFrame | None:
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key=key,
    )
    if uploaded_file is None:
        return None
    try:
        return read_data(uploaded_file)
    except Exception as error:
        st.error(f"Could not read {label.lower()}: {error}")
        return None


def render_feature_target_selector(
    df: pl.DataFrame,
    *,
    key_prefix: str,
) -> tuple[list[str], str]:
    feature_column, target_column = st.columns(2)
    target = target_column.selectbox(
        "Target",
        df.columns,
        key=f"{key_prefix}_target",
    )
    available_features = [column for column in df.columns if column != target]
    features = feature_column.multiselect(
        "Features",
        available_features,
        default=[],
        key=f"{key_prefix}_features",
    )
    return features, target


def render_data_preview(
    df: pl.DataFrame,
    features: list[str],
    target: str | None = None,
) -> None:
    selected_columns = list(features)
    loaded, selected, selected_2 = st.columns([3, 3, 1])

    loaded.write("Loaded data")
    loaded.dataframe(df, hide_index=True)

    selected.write("Selected features")
    selected.dataframe(df.select(selected_columns), hide_index=True)

    if target is not None and target not in selected_columns:
        selected_2.write("Selected target")
        selected_2.dataframe(df.select(target), hide_index=True)

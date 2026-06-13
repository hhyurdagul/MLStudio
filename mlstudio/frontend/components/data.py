import polars as pl
import streamlit as st

from mlstudio.backend import read_tabular_data


def render_dataset_selector(
    label: str,
) -> pl.DataFrame | None:
    uploaded_file = st.file_uploader(
        label,
        type=["csv", "xlsx"],
        accept_multiple_files=False,
    )
    if uploaded_file is None:
        return None
    try:
        return read_tabular_data(uploaded_file, uploaded_file.name)
    except Exception as error:
        st.error(f"Could not read {label.lower()}: {error}")
        return None


def render_feature_target_selector(
    data: pl.DataFrame,
) -> tuple[list[str], str]:
    feature_column, target_column = st.columns(2)
    target = target_column.selectbox(
        "Target",
        data.columns,
    )
    features = feature_column.multiselect(
        "Features",
        data.select(pl.exclude(target)).columns,
        default=[],
    )
    return features, target


def render_data_preview(
    data: pl.DataFrame,
    features: list[str],
    target: str,
) -> None:
    loaded, selected_features, selected_target = st.columns([3, 3, 1])
    loaded.write("Loaded data")
    loaded.dataframe(data, hide_index=True)
    selected_features.write("Selected features")
    selected_features.dataframe(data.select(features), hide_index=True)
    selected_target.write("Selected target")
    selected_target.dataframe(data.select(target), hide_index=True)

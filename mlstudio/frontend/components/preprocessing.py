import polars as pl
import streamlit as st


def render_preprocessing_config(
    preprocessing: pl.DataFrame,
    *,
    key_prefix: str,
) -> pl.DataFrame:
    customize = st.checkbox(
        "Customize preprocessing",
        key=f"{key_prefix}_customize_preprocessing",
    )
    if not customize:
        st.dataframe(preprocessing, hide_index=True)
        return preprocessing

    string_steps = st.data_editor(
        preprocessing.filter(pl.col("Type") == "String"),
        disabled=["Variable", "Type", "Unique Count"],
        column_order=["Variable", "Preprocessing"],
        column_config={
            "Preprocessing": st.column_config.SelectboxColumn(
                "Preprocessing",
                options=["OrdinalEncoder", "OneHotEncoder"],
                required=True,
            )
        },
        key=f"{key_prefix}_string_preprocessing",
        hide_index=True,
    )
    numeric_steps = st.data_editor(
        preprocessing.filter(pl.col("Type").is_in(["Numeric", "Boolean"])),
        disabled=["Variable", "Type", "Unique Count"],
        column_order=["Variable", "Preprocessing"],
        column_config={
            "Preprocessing": st.column_config.SelectboxColumn(
                "Preprocessing",
                options=["StandardScaler", "MinMaxScaler", "None"],
                required=True,
            )
        },
        key=f"{key_prefix}_numeric_preprocessing",
        hide_index=True,
    )
    return pl.DataFrame(string_steps).vstack(pl.DataFrame(numeric_steps))

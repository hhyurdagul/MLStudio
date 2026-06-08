import polars as pl
import streamlit as st


def render_preprocessing_config(
    preprocessing: pl.DataFrame,
) -> pl.DataFrame:
    customize = st.checkbox(
        "Customize preprocessing",
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
        hide_index=True,
    )
    return pl.DataFrame(string_steps).vstack(pl.DataFrame(numeric_steps))

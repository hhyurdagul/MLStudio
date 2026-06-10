from typing import cast

import polars as pl
import streamlit as st

from mlstudio.backend import TargetProcessing


def render_preprocessing_config(
    preprocessing: pl.DataFrame,
) -> pl.DataFrame:
    customize = st.checkbox(
        "Customize preprocessing",
    )
    if not customize:
        st.dataframe(preprocessing, hide_index=True)
        return preprocessing

    configured_steps: list[pl.DataFrame] = []
    string_steps = preprocessing.filter(pl.col("Type") == "String")
    if not string_steps.is_empty():
        configured_steps.append(
            pl.DataFrame(
                st.data_editor(
                    string_steps,
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
            )
        )
    numeric_steps = preprocessing.filter(
        pl.col("Type").is_in(["Numeric", "Boolean"])
    )
    if not numeric_steps.is_empty():
        configured_steps.append(
            pl.DataFrame(
                st.data_editor(
                    numeric_steps,
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
            )
        )
    return pl.concat(configured_steps)


def render_target_processing() -> TargetProcessing:
    return cast(
        TargetProcessing,
        st.selectbox(
            "Target processing",
            ["None", "StandardScaler", "MinMaxScaler"],
        ),
    )

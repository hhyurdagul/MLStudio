from typing import cast

import polars as pl
import streamlit as st

from mlstudio.backend import (
    SCORE_FUNCTIONS,
    FeatureSelectionConfig,
    TargetProcessing,
)


def render_preprocessing_config(
    preprocessing: pl.DataFrame,
) -> pl.DataFrame:
    st.text("Preprocessing Table")
    configured_steps: list[pl.DataFrame] = []
    string_steps = preprocessing.filter(pl.col("Type") == "String")
    if not string_steps.is_empty():
        configured_steps.append(
            pl.DataFrame(
                st.data_editor(
                    string_steps,
                    disabled=["Variable", "Type", "Unique Count"],
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
    numeric_steps = preprocessing.filter(pl.col("Type").is_in(["Numeric", "Boolean"]))
    if not numeric_steps.is_empty():
        selected = st.selectbox(
            "Numeric Scaling", ["None", "StandardScaler", "MinMaxScaler"]
        )
        configured_steps.append(
            numeric_steps.with_columns(pl.lit(selected).alias("Preprocessing"))
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


def render_feature_selection(
    transformed_feature_count: int,
) -> FeatureSelectionConfig | None:
    if not st.checkbox("Use feature selection"):
        return None

    st.caption(f"Preprocessing will produce {transformed_feature_count} features.")
    method_column, count_column = st.columns(2)
    method = method_column.selectbox(
        "Feature selection method",
        tuple(SCORE_FUNCTIONS),
    )
    k = int(
        count_column.number_input(
            "Number of selected features",
            min_value=1,
            max_value=transformed_feature_count,
            value=min(10, transformed_feature_count),
            step=1,
        )
    )
    return FeatureSelectionConfig(method=method, count=k)


def render_lookback() -> int | None:
    if not st.checkbox("Use target lookback"):
        return None
    lookback = int(
        st.number_input(
            "Target lookback",
            min_value=1,
            value=3,
            step=1,
            help="Adds target lags 1 through this value to the model.",
        )
    )
    return lookback

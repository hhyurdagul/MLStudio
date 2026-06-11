from typing import cast

import polars as pl
import streamlit as st

from mlstudio.backend import (
    SCORE_FUNCTIONS,
    EstimatorWrapper,
    PipelineStep,
    TargetProcessing,
    create_feature_selection_step,
    create_lookback_wrapper,
)


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


def render_feature_selection(
    feature_count: int,
) -> PipelineStep | None:
    if not st.checkbox("Use feature selection"):
        return None

    method_column, count_column = st.columns(2)
    method = method_column.selectbox(
        "Feature selection method",
        tuple(SCORE_FUNCTIONS),
    )
    k = int(
        count_column.number_input(
            "Number of selected features",
            min_value=1,
            max_value=feature_count,
            value=min(10, feature_count),
            step=1,
        )
    )
    return create_feature_selection_step(method, k)


def render_lookback() -> EstimatorWrapper | None:
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
    return create_lookback_wrapper(lookback)

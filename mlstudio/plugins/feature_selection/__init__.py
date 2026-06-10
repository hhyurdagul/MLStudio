from collections.abc import Callable
from typing import Any

import streamlit as st
from sklearn.feature_selection import SelectKBest, f_regression

from mlstudio.backend import PipelineStep

from .score_functions import mrmr_regression, relief_regression

ScoreFunction = Callable[[Any, Any], Any]

SCORE_FUNCTIONS: dict[str, ScoreFunction] = {
    "F Regression": f_regression,
    "MRMR Regression": mrmr_regression,
    "Relief Regression": relief_regression,
}


def render_feature_selection(
    feature_count: int,
) -> PipelineStep | None:
    enabled = st.checkbox(
        "Use feature selection",
    )
    if not enabled:
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
            value=min(10, feature_count),
            step=1,
        )
    )
    return create_feature_selection_step(method, k)


def create_feature_selection_step(method: str, k: int) -> PipelineStep:
    if method not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown feature selection method: {method}")
    if k < 1:
        raise ValueError("The number of selected features must be at least one.")
    return PipelineStep(
        name="feature_selection",
        transformer=SelectKBest(score_func=SCORE_FUNCTIONS[method], k=k),
    )

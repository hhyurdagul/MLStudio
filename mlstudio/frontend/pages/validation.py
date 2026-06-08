from typing import cast

import streamlit as st

from mlstudio.backend import (
    ValidationStrategy,
    get_preprocessing_data,
    validate,
)
from mlstudio.frontend.components import (
    render_dataset_selector,
    render_feature_target_selector,
    render_grid_search,
    render_metrics,
    render_model_config,
    render_processed_data,
    render_predictions,
    render_preprocessing_config,
)
from mlstudio.plugins.feature_selection import render_pipeline_step


def render_validation_page() -> None:
    st.header("Validation")
    data = render_dataset_selector("Upload training data")
    if data is None:
        st.info("Upload training data to validate a model.")
        return

    features, target = render_feature_target_selector(
        data,
    )
    if not features:
        st.info("Select at least one feature to configure validation.")
        return

    strategy_column, value_column = st.columns(2)
    strategy = strategy_column.selectbox(
        "Validation strategy",
        ["Random split", "Last split", "Cross-validation"],
    )
    if strategy == "Cross-validation":
        folds = value_column.slider(
            "Validation folds",
            2,
            10,
            5,
        )
        validation_percent = 20
    else:
        validation_percent = value_column.slider(
            "Validation percent",
            1,
            50,
            20,
        )
        folds = 5

    try:
        preprocessing = render_preprocessing_config(
            get_preprocessing_data(data.select(features)),
        )
    except ValueError as error:
        st.error(str(error))
        return

    feature_selection = render_pipeline_step(
        feature_count=len(features)
    )
    model, model_is_valid = render_model_config()
    if not st.button(
        f"Validate {model.definition.label}",
        type="primary",
        disabled=not model_is_valid,
    ):
        return

    try:
        with st.spinner("Validating model..."):
            result = validate(
                data=data,
                features=features,
                target=target,
                preprocessing=preprocessing,
                model=model,
                strategy=cast(ValidationStrategy, strategy),
                validation_percent=validation_percent,
                folds=folds,
                pipeline_steps=(
                    (feature_selection,) if feature_selection is not None else ()
                ),
            )
        render_grid_search(result.grid_search)
        if result.prediction is not None:
            render_processed_data(result.prediction.processed)
            render_metrics(result.metrics)
            render_predictions(
                result.prediction.data,
            )
    except Exception as error:
        st.error(f"Validation failed: {error}")

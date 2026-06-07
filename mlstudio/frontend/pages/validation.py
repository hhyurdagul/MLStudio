from typing import cast

import streamlit as st

from mlstudio.backend import (
    CrossValidationMetrics,
    ValidationStrategy,
    get_preprocessing_data,
    validate,
)
from mlstudio.frontend.components import (
    render_cross_validation_metrics,
    render_dataset_selector,
    render_feature_target_selector,
    render_grid_search,
    render_metrics,
    render_model_config,
    render_predictions,
    render_preprocessing_config,
)


def render_validation_page() -> None:
    st.header("Validation")
    data = render_dataset_selector("Upload training data", key="validation_data")
    if data is None:
        st.info("Upload training data to validate a model.")
        return

    features, target = render_feature_target_selector(
        data,
        key_prefix="validation",
    )
    if not features:
        st.info("Select at least one feature to configure validation.")
        return

    strategy_column, value_column = st.columns(2)
    strategy = strategy_column.selectbox(
        "Validation strategy",
        ["Random split", "Last split", "Cross-validation"],
        key="validation_strategy",
    )
    if strategy == "Cross-validation":
        folds = value_column.slider(
            "Validation folds",
            2,
            10,
            5,
            key="validation_folds",
        )
        validation_percent = 20
    else:
        validation_percent = value_column.slider(
            "Validation percent",
            1,
            50,
            20,
            key="validation_percent",
        )
        folds = 5

    try:
        preprocessing = render_preprocessing_config(
            get_preprocessing_data(data.select(features)),
            key_prefix="validation",
        )
    except ValueError as error:
        st.error(str(error))
        return

    model, model_is_valid = render_model_config(key_prefix="validation")
    if not st.button(
        f"Validate {model.definition.label}",
        type="primary",
        disabled=not model_is_valid,
        key="validate_model",
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
            )
        if isinstance(result.metrics, CrossValidationMetrics):
            render_cross_validation_metrics(result.metrics)
        else:
            render_metrics(result.metrics)
        render_grid_search(result.grid_search)
        if result.prediction is not None:
            render_predictions(
                result.prediction.data,
                key="download_validation_predictions",
            )
    except Exception as error:
        st.error(f"Validation failed: {error}")

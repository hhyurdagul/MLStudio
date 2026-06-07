from typing import cast

import streamlit as st

from mlstudio.backend import (
    RowSelection,
    get_preprocessing_data,
    serialize_artifact,
    train,
)
from mlstudio.frontend.components import (
    render_data_preview,
    render_dataset_selector,
    render_feature_target_selector,
    render_grid_search,
    render_metrics,
    render_model_config,
    render_predictions,
    render_preprocessing_config,
)


def render_training_page() -> None:
    st.header("Training")
    training_column, test_column = st.columns(2)
    with training_column:
        training_data = render_dataset_selector(
            "Upload training data",
            key="training_data",
        )
    with test_column:
        test_data = render_dataset_selector(
            "Upload test data",
            key="training_test_data",
        )
    if training_data is None:
        st.info("Upload training data to configure a model.")
        return

    features, target = render_feature_target_selector(
        training_data,
        key_prefix="training",
    )
    with st.expander("Data preview"):
        render_data_preview(training_data, features, target)
    if not features:
        st.info("Select at least one feature to configure preprocessing and modeling.")
        return

    selection_column, percent_column = st.columns(2)
    row_selection = selection_column.selectbox(
        "Training rows",
        ["Random percent", "Last percent"],
        key="training_row_method",
    )
    training_percent = percent_column.slider(
        "Percent used for training",
        1,
        100,
        100,
        key="training_percent",
    )

    try:
        preprocessing = render_preprocessing_config(
            get_preprocessing_data(training_data.select(features)),
            key_prefix="training",
        )
    except ValueError as error:
        st.error(str(error))
        return

    model, model_is_valid = render_model_config(key_prefix="training")
    if not st.button(
        f"Train {model.definition.label}",
        type="primary",
        disabled=not model_is_valid,
        key="train_model",
    ):
        return

    try:
        with st.spinner("Training model..."):
            result = train(
                training_data=training_data,
                test_data=test_data,
                features=features,
                target=target,
                preprocessing=preprocessing,
                model=model,
                row_selection=cast(RowSelection, row_selection),
                training_percent=training_percent,
            )
        st.success(f"Trained on {result.trained_rows} rows.")
        render_grid_search(result.grid_search)
        st.download_button(
            "Download model bundle",
            data=serialize_artifact(result.artifact),
            file_name="mlstudio-model.joblib",
            mime="application/octet-stream",
            key="download_model",
            on_click="ignore",
        )
        if result.prediction is not None:
            if result.prediction.metrics is not None:
                render_metrics(result.prediction.metrics)
            render_predictions(
                result.prediction.data,
                key="download_training_predictions",
            )
    except Exception as error:
        st.error(f"Training failed: {error}")

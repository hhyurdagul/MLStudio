from typing import cast

import streamlit as st

from mlstudio.backend import (
    PipelineConfig,
    RowSelection,
    TrainingConfig,
    get_preprocessing_data,
    get_transformed_feature_count,
    predict,
    serialize_artifact,
    train,
)
from mlstudio.frontend.components import (
    render_data_preview,
    render_dataset_selector,
    render_feature_selection,
    render_feature_target_selector,
    render_grid_search,
    render_lookback,
    render_metrics,
    render_model_config,
    render_predictions,
    render_preprocessing_config,
    render_processed_data,
    render_target_processing,
)


def render_training_page(
    attach_feature_selection: bool = False, attach_lookback: bool = False
) -> None:
    st.header("Training")
    training_column, test_column = st.columns(2)
    with training_column:
        training_data = render_dataset_selector(
            "Upload training data",
        )
    with test_column:
        test_data = render_dataset_selector(
            "Upload test data",
        )
    if training_data is None:
        st.info("Upload training data to configure a model.")
        return

    features, target = render_feature_target_selector(
        training_data,
    )
    with st.expander("Data preview"):
        render_data_preview(training_data, features, target)
    if not features:
        st.info("Select at least one feature to configure preprocessing and modeling.")
        return

    with st.expander("Preprocessing"):
        try:
            preprocessing = render_preprocessing_config(
                get_preprocessing_data(training_data.select(features)),
            )
        except ValueError as error:
            st.error(str(error))
            return

        target_processing = render_target_processing()
        feature_selection = (
            render_feature_selection(
                transformed_feature_count=get_transformed_feature_count(
                    preprocessing
                ),
            )
            if attach_feature_selection
            else None
        )
        lookback = render_lookback() if attach_lookback else None

    selection_column, percent_column = st.columns(2)
    row_selection = selection_column.selectbox(
        "Training rows",
        ["Last percent"]
        if lookback is not None
        else ["Random percent", "Last percent"],
    )
    training_percent = percent_column.slider(
        "Percent used for training",
        1,
        100,
        100,
    )


    model, model_is_valid = render_model_config()
    train_requested = st.button(
        f"Train {model.definition.label}",
        type="primary",
        disabled=not model_is_valid,
    )

    training_result = None
    if train_requested:
        try:
            with st.spinner("Training model..."):
                training_result = train(
                    training_data,
                    TrainingConfig(
                        pipeline=PipelineConfig(
                            features=tuple(features),
                            target=target,
                            preprocessing=preprocessing,
                            model=model,
                            target_processing=target_processing,
                            feature_selection=feature_selection,
                            lookback=lookback,
                        ),
                        row_selection=cast(RowSelection, row_selection),
                        percent=training_percent,
                    ),
                    None,
                )
            st.session_state["supervised_training_artifact"] = (
                training_result.artifact
            )
        except Exception as error:
            st.error(f"Training failed: {error}")

    if training_result is not None:
        st.success(f"Trained on {training_result.trained_rows} rows.")
        render_grid_search(training_result.grid_search)
        render_processed_data(training_result.processed)

    artifact = st.session_state.get("supervised_training_artifact")
    if artifact is None:
        return

    st.download_button(
        "Download model bundle",
        data=serialize_artifact(artifact),
        file_name="mlstudio-model.joblib",
        mime="application/octet-stream",
        on_click="ignore",
    )
    if test_data is None:
        return
    if test_data.is_empty():
        st.error("The test dataset is empty.")
        return

    st.subheader("Predict test data")
    prediction_count = int(
        st.number_input(
            "Prediction count",
            min_value=1,
            max_value=test_data.height,
            value=test_data.height,
            step=1,
            help=(
                "Choose how many rows to predict from the start of the test "
                "dataset."
            ),
        )
    )
    if not st.button("Run predictions", type="primary"):
        return

    try:
        with st.spinner("Running predictions..."):
            prediction_result = predict(
                artifact,
                test_data.head(prediction_count),
            )
        render_processed_data(prediction_result.processed)
        if prediction_result.prediction.metrics is not None:
            render_metrics(prediction_result.prediction.metrics)
        render_predictions(prediction_result.prediction.data)
    except Exception as error:
        st.error(f"Prediction failed: {error}")

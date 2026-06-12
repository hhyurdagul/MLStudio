from typing import cast

import streamlit as st

from mlstudio.backend import (
    PipelineConfig,
    RowSelection,
    TrainingConfig,
    get_preprocessing_data,
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
                feature_count=len(features),
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
    if not st.button(
        f"Train {model.definition.label}",
        type="primary",
        disabled=not model_is_valid,
    ):
        return

    try:
        with st.spinner("Training model..."):
            result = train(
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
                test_data,
            )
        st.success(f"Trained on {result.trained_rows} rows.")
        render_grid_search(result.grid_search)
        st.download_button(
            "Download model bundle",
            data=serialize_artifact(result.artifact),
            file_name="mlstudio-model.joblib",
            mime="application/octet-stream",
            on_click="ignore",
        )

        render_processed_data(result.processed)
        if result.prediction is not None:
            if result.prediction.metrics is not None:
                render_metrics(result.prediction.metrics)
            render_predictions(
                result.prediction.data,
            )
    except Exception as error:
        st.error(f"Training failed: {error}")

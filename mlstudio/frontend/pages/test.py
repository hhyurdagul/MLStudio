import streamlit as st

from mlstudio.backend import deserialize_artifact, predict
from mlstudio.frontend.components import (
    render_dataset_selector,
    render_metrics,
    render_processed_data,
    render_predictions,
)


def render_test_page() -> None:
    st.header("Test")
    st.warning("Only upload model bundles that you created or trust.")
    model_column, data_column = st.columns(2)
    with model_column:
        model_file = st.file_uploader(
            "Upload model bundle",
            type=["joblib"],
        )
    with data_column:
        test_data = render_dataset_selector("Upload test data")
    if model_file is None or test_data is None:
        st.info("Upload a model bundle and test data to run predictions.")
        return

    try:
        artifact = deserialize_artifact(model_file.getvalue())
        st.caption(
            f"Model: {artifact.model_label} | Target: {artifact.target} | "
            f"Features: {', '.join(artifact.features)}"
        )
    except Exception as error:
        st.error(f"Could not load model bundle: {error}")
        return

    if not st.button("Run test predictions", type="primary"):
        return

    try:
        with st.spinner("Running predictions..."):
            result = predict(artifact, test_data)
        render_processed_data(result.processed)
        if result.prediction.metrics is not None:
            render_metrics(result.prediction.metrics)
        render_predictions(result.prediction.data)
    except Exception as error:
        st.error(f"Prediction failed: {error}")

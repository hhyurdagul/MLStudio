from typing import cast

import polars as pl
import streamlit as st

from mlstudio.backend import TargetProcessing
from mlstudio.backend.deep_learning import (
    DEEP_MODEL_NAMES,
    HIDDEN_ACTIVATIONS,
    OUTPUT_ACTIVATIONS,
    DeepLearningConfig,
    deserialize_deep_artifact,
    forecast_deep_model,
    serialize_deep_artifact,
    train_deep_model,
)
from mlstudio.frontend.components import (
    render_dataset_selector,
    render_metrics,
    render_predictions,
)


def render_deep_learning_page() -> None:
    st.header("Deep Learning Time Series")
    st.caption(
        "Train univariate PyTorch models on ordered target history and forecast "
        "future values recursively."
    )
    mode = st.radio(
        "Mode",
        ["Training", "Test"],
        horizontal=True,
    )
    if mode == "Training":
        _render_training()
    else:
        _render_test()


def _render_training() -> None:
    training_column, backtest_column = st.columns(2)
    with training_column:
        training_data = render_dataset_selector(
            "Upload ordered training data",
            key="deep_training_data",
        )
    with backtest_column:
        backtest_data = render_dataset_selector(
            "Upload optional backtest data",
            key="deep_backtest_data",
        )
    if training_data is None:
        st.info("Upload training data to configure a deep-learning model.")
        return

    numeric_targets = training_data.select(pl.selectors.numeric()).columns
    if not numeric_targets:
        st.error("The training dataset needs at least one numeric target column.")
        return
    target = st.selectbox("Target", numeric_targets)
    with st.expander("Target preview"):
        st.dataframe(training_data.select(target), hide_index=True)

    training_percent = st.slider(
        "Last percent used for training",
        min_value=1,
        max_value=100,
        value=100,
        help="Uses only the final contiguous percentage of the ordered series.",
    )
    config = _render_config()
    if config.output_activation != "Linear":
        st.warning(
            "A non-linear output activation constrains forecasts in scaled target "
            "space and can limit the values the model can predict."
        )
    if not st.button(f"Train {config.model_name}", type="primary"):
        return

    try:
        with st.spinner("Training deep-learning model..."):
            result = train_deep_model(
                training_data,
                target,
                config,
                backtest_data,
                training_percent,
            )
        st.success(f"Trained on {result.trained_windows} sliding windows.")
        _render_loss_history(result.losses)
        st.download_button(
            "Download deep-learning bundle",
            data=serialize_deep_artifact(result.artifact),
            file_name="mlstudio-deep-model.pt",
            mime="application/octet-stream",
            on_click="ignore",
        )
        if result.prediction is not None:
            st.subheader("Backtest")
            if result.prediction.metrics is not None:
                render_metrics(result.prediction.metrics)
            render_predictions(result.prediction.data)
    except Exception as error:
        st.error(f"Deep-learning training failed: {error}")


def _render_test() -> None:
    st.warning("Only upload deep-learning bundles that you created or trust.")
    bundle = st.file_uploader(
        "Upload deep-learning bundle",
        type=["pt"],
        key="deep_model_bundle",
    )
    if bundle is None:
        st.info("Upload a deep-learning bundle to forecast future values.")
        return

    try:
        artifact = deserialize_deep_artifact(bundle.getvalue())
    except Exception as error:
        st.error(f"Could not load deep-learning bundle: {error}")
        return

    st.caption(
        f"Model: {artifact.config.model_name} | Target: {artifact.target} | "
        f"Lookback: {artifact.config.lookback}"
    )
    horizon_column, actual_column = st.columns(2)
    with horizon_column:
        horizon = int(
            st.number_input(
                "Forecast horizon",
                min_value=1,
                value=10,
                step=1,
            )
        )
    with actual_column:
        actual_data = render_dataset_selector(
            "Upload optional actual target data",
            key="deep_actual_data",
        )

    if not st.button("Forecast", type="primary"):
        return
    try:
        with st.spinner("Generating recursive forecast..."):
            result = forecast_deep_model(artifact, horizon, actual_data)
        if result.prediction.metrics is not None:
            render_metrics(result.prediction.metrics)
        render_predictions(result.prediction.data)
    except Exception as error:
        st.error(f"Forecasting failed: {error}")


def _render_config() -> DeepLearningConfig:
    model_column, lookback_column, neurons_column = st.columns(3)
    model_name = model_column.selectbox("Model", DEEP_MODEL_NAMES)
    lookback = int(
        lookback_column.number_input(
            "Lookback",
            min_value=1,
            value=10,
            step=1,
        )
    )
    neurons = int(
        neurons_column.number_input(
            "Neurons",
            min_value=1,
            value=64,
            step=1,
        )
    )

    layers_column, rate_column, epochs_column = st.columns(3)
    layers = int(
        layers_column.number_input(
            "Layers",
            min_value=1,
            value=2,
            step=1,
        )
    )
    learning_rate = float(
        rate_column.number_input(
            "Learning rate",
            min_value=0.000001,
            value=0.001,
            step=0.0001,
            format="%.6f",
        )
    )
    epochs = int(
        epochs_column.number_input(
            "Epochs",
            min_value=1,
            value=100,
            step=1,
        )
    )

    batch_column, hidden_column, output_column = st.columns(3)
    batch_size = int(
        batch_column.number_input(
            "Batch size",
            min_value=1,
            value=32,
            step=1,
        )
    )
    hidden_activation = hidden_column.selectbox(
        "Hidden activation",
        HIDDEN_ACTIVATIONS,
    )
    output_activation = output_column.selectbox(
        "Output activation",
        OUTPUT_ACTIVATIONS,
    )
    target_processing = cast(
        TargetProcessing,
        st.selectbox(
            "Target processing",
            ["None", "StandardScaler", "MinMaxScaler"],
            index=1,
        ),
    )
    return DeepLearningConfig(
        model_name=model_name,
        lookback=lookback,
        neurons=neurons,
        layers=layers,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        target_processing=target_processing,
    )


def _render_loss_history(losses: tuple[float, ...]) -> None:
    st.subheader("Training loss")
    loss_data = pl.DataFrame(
        {
            "Epoch": range(1, len(losses) + 1),
            "MSE loss": losses,
        }
    )
    st.line_chart(loss_data, x="Epoch", y="MSE loss")

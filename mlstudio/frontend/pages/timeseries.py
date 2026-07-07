from math import ceil
from typing import cast

import polars as pl
import streamlit as st

from mlstudio.backend import TargetProcessing
from mlstudio.backend.autoregression import acf_graph, select_lags
from mlstudio.backend.timeseries import (
    HIDDEN_ACTIVATIONS,
    OUTPUT_ACTIVATIONS,
    TIMESERIES_MODEL_NAMES,
    TimeSeriesConfig,
    deserialize_timeseries_artifact,
    forecast_timeseries_model,
    serialize_timeseries_artifact,
    train_timeseries_model,
)
from mlstudio.frontend.components import (
    render_dataset_selector,
    render_metrics,
    render_predictions,
)


def render_timeseries_page() -> None:
    st.header("Time Series")
    st.caption(
        "Train univariate PyTorch models on ordered target history and forecast "
        "future values recursively."
    )
    mode = st.radio(
        "Mode",
        ["Model Training", "Model Testing"],
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
        )
    with backtest_column:
        backtest_data = render_dataset_selector(
            "Upload optional backtest data",
        )
    if training_data is None:
        st.info("Upload training data to configure a time-series model.")
        return

    numeric_targets = training_data.select(pl.selectors.numeric()).columns
    if not numeric_targets:
        st.error("The training dataset needs at least one numeric target column.")
        return
    target_column, training_percent_column = st.columns(2)
    with target_column:
        target = st.selectbox("Target", numeric_targets)
    with training_percent_column:
        training_percent = st.slider(
            "Last percent used for training",
            min_value=1,
            max_value=100,
            value=100,
            help="Uses only the final contiguous percentage of the ordered series.",
        )
    with st.expander("Target preview"):
        st.dataframe(training_data.select(target), hide_index=True)

    max_time_lag, lag_indices, lags_are_valid = _render_lag_preview(
        training_data,
        target,
        training_percent,
    )
    config = _render_config(max_time_lag, lag_indices)
    if config.output_activation != "Linear":
        st.warning(
            "A non-linear output activation constrains forecasts in scaled target "
            "space and can limit the values the model can predict."
        )
    train_requested = st.button(
        f"Train {config.model_name}",
        type="primary",
        disabled=not lags_are_valid,
    )

    training_result = None
    if train_requested:
        try:
            with st.spinner("Training time-series model..."):
                training_result = train_timeseries_model(
                    training_data,
                    target,
                    config,
                    backtest_data,
                    training_percent,
                )
            st.session_state["timeseries_training_artifact"] = (
                training_result.artifact
            )
        except Exception as error:
            st.error(f"Time-series training failed: {error}")

    if training_result is not None:
        st.success(f"Trained on {training_result.trained_windows} sliding windows.")
        _render_loss_history(training_result.losses)
        if training_result.prediction is not None:
            st.subheader("Backtest")
            if training_result.prediction.metrics is not None:
                render_metrics(training_result.prediction.metrics)
            render_predictions(
                training_result.prediction.data,
            )

    artifact = st.session_state.get("timeseries_training_artifact")
    if artifact is None:
        return

    st.download_button(
        "Download time-series bundle",
        data=serialize_timeseries_artifact(artifact),
        file_name="mlstudio-timeseries-model.pt",
        mime="application/octet-stream",
        on_click="ignore",
    )
    if backtest_data is not None:
        return

    st.subheader("Forecast trained model")
    forecast_count = int(
        st.number_input(
            "Forecast count",
            min_value=1,
            value=10,
            step=1,
        )
    )
    if st.button(
        "Generate forecast",
        type="primary",
    ):
        try:
            with st.spinner("Generating recursive forecast..."):
                forecast = forecast_timeseries_model(
                    artifact,
                    forecast_count,
                )
            render_predictions(
                forecast.prediction.data,
            )
        except Exception as error:
            st.error(f"Forecasting failed: {error}")


def _render_test() -> None:
    bundle = st.file_uploader(
        "Upload time-series bundle",
        type=["pt"],
    )
    if bundle is None:
        st.info("Upload a time-series bundle to forecast future values.")
        return

    try:
        artifact = deserialize_timeseries_artifact(bundle.getvalue())
    except Exception as error:
        st.error(f"Could not load time-series bundle: {error}")
        return

    st.caption(
        f"Model: {artifact.config.model_name} | Target: {artifact.target} | "
        f"Max Time Lag: {artifact.config.max_time_lag} | "
        f"Lag Indices: {', '.join(map(str, artifact.config.lag_indices))} | "
        f"Layer Sizes: {', '.join(map(str, artifact.config.layer_sizes))}"
    )
    actual_data = render_dataset_selector(
        "Upload optional actual target data",
    )
    if actual_data is not None and actual_data.is_empty():
        st.error("The actual target dataset is empty.")
        return

    maximum_horizon = actual_data.height if actual_data is not None else None
    horizon = int(
        st.number_input(
            "Forecast horizon",
            min_value=1,
            max_value=maximum_horizon,
            value=min(10, maximum_horizon) if maximum_horizon is not None else 10,
            step=1,
            help=(
                "Limited to the number of rows in the actual target dataset."
                if maximum_horizon is not None
                else None
            ),
        )
    )

    if not st.button("Forecast", type="primary"):
        return
    try:
        with st.spinner("Generating recursive forecast..."):
            result = forecast_timeseries_model(
                artifact,
                horizon,
                actual_data.head(horizon) if actual_data is not None else None,
            )
        if result.prediction.metrics is not None:
            render_metrics(result.prediction.metrics)
        render_predictions(result.prediction.data)
    except Exception as error:
        st.error(f"Forecasting failed: {error}")


def _render_lag_preview(
    training_data: pl.DataFrame,
    target: str,
    training_percent: int,
) -> tuple[int, tuple[int, ...], bool]:
    selected_rows = max(1, ceil(training_data.height * training_percent / 100))
    values = training_data[target].tail(selected_rows).to_numpy()
    if len(values) < 2:
        st.error("Lag analysis requires at least two selected training rows.")
        return 1, (), False

    max_time_lag = int(
        st.number_input(
            "Max Time Lag",
            min_value=1,
            max_value=len(values) - 1,
            value=min(10, len(values) - 1),
            step=1,
            help="Controls ACF diagnostics and the available lag indices.",
        )
    )

    mode_labels = {
        "All": "Select all",
        "Manual": "Select indices",
        "Best N ACF": "Best N by absolute ACF",
        "ACF threshold": "Absolute ACF threshold",
    }
    mode = st.selectbox(
        "Lag selection",
        ["All", "Manual", "Best N ACF", "ACF threshold"],
        format_func=lambda value: mode_labels[value],
    )
    indices: tuple[int, ...] = ()
    count: int | None = None
    threshold: float | None = None
    if mode == "Manual":
        indices = tuple(
            st.multiselect(
                "Lag indices",
                options=range(1, max_time_lag + 1),
                default=range(1, max_time_lag + 1),
            )
        )
    elif mode == "Best N ACF":
        count = int(
            st.number_input(
                "Number of lags",
                min_value=1,
                max_value=max_time_lag,
                value=min(5, max_time_lag),
                step=1,
            )
        )
    elif mode == "ACF threshold":
        threshold = float(
            st.number_input(
                "Minimum absolute ACF",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
            )
        )

    try:
        lag_indices = select_lags(
            values,
            max_time_lag,
            mode,
            indices=indices,
            count=count,
            threshold=threshold,
        )
        st.caption(
            "Selected lag indices: " + ", ".join(str(lag) for lag in lag_indices)
        )

        with st.expander("Lag Preview"):
            try:
                st.altair_chart(
                    acf_graph(values, max_time_lag),
                    width="stretch",
                )
            except (TypeError, ValueError) as error:
                st.error(f"Could not create ACF diagnostics: {error}")

        return max_time_lag, lag_indices, True
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return max_time_lag, (), False


def _render_config(
    max_time_lag: int,
    lag_indices: tuple[int, ...],
) -> TimeSeriesConfig:
    model_column, layers_column = st.columns(2)
    model_name = model_column.selectbox("Model", TIMESERIES_MODEL_NAMES)
    layer_count = int(
        layers_column.number_input(
            "Layers",
            min_value=1,
            value=1,
            step=1,
        )
    )

    st.caption("Neurons per layer")
    neuron_columns = st.columns(min(4, layer_count))
    layer_sizes = tuple(
        int(
            neuron_columns[index % len(neuron_columns)].number_input(
                f"Layer {index + 1}",
                min_value=1,
                value=64,
                step=1,
            )
        )
        for index in range(layer_count)
    )

    rate_column, epochs_column = st.columns(2)
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
            index=0,
        ),
    )
    return TimeSeriesConfig(
        model_name=model_name,
        max_time_lag=max_time_lag,
        lag_indices=lag_indices,
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        target_processing=target_processing,
    )


def _render_loss_history(losses: tuple[float, ...]) -> None:
    with st.expander("Training loss"):
        loss_data = pl.DataFrame(
            {
                "Epoch": range(1, len(losses) + 1),
                "MSE loss": losses,
            }
        )
        st.line_chart(loss_data, x="Epoch", y="MSE loss")

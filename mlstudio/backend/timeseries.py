from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from io import BytesIO
from math import ceil
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluation import calculate_metrics
from .types import PredictionResult, RegressionMetrics, TargetProcessing

TimeSeriesModelName = Literal[
    "MLP",
    "CNN",
    "RNN",
    "GRU",
    "LSTM",
    "Bi-LSTM",
    "ConvLSTM",
]
HiddenActivation = Literal["ReLU", "Tanh", "Sigmoid", "GELU"]
OutputActivation = Literal["Linear", "ReLU", "Sigmoid", "Tanh"]

TIMESERIES_MODEL_NAMES: tuple[TimeSeriesModelName, ...] = (
    "MLP",
    "CNN",
    "RNN",
    "GRU",
    "LSTM",
    "Bi-LSTM",
    "ConvLSTM",
)
HIDDEN_ACTIVATIONS: tuple[HiddenActivation, ...] = (
    "ReLU",
    "Tanh",
    "Sigmoid",
    "GELU",
)
OUTPUT_ACTIVATIONS: tuple[OutputActivation, ...] = (
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
)
TIMESERIES_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class TimeSeriesConfig:
    model_name: TimeSeriesModelName
    max_time_lag: int = 10
    lag_indices: tuple[int, ...] = tuple(range(1, 11))
    layer_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    hidden_activation: HiddenActivation = "ReLU"
    output_activation: OutputActivation = "Linear"
    target_processing: TargetProcessing = "StandardScaler"


@dataclass(frozen=True)
class TargetScaler:
    processing: TargetProcessing
    offset: float
    scale: float


@dataclass(frozen=True)
class TimeSeriesArtifact:
    version: int
    target: str
    config: TimeSeriesConfig
    scaler: TargetScaler
    target_history: tuple[float, ...]
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class TimeSeriesTrainingResult:
    artifact: TimeSeriesArtifact
    trained_windows: int
    losses: tuple[float, ...]
    prediction: PredictionResult | None


@dataclass(frozen=True)
class TimeSeriesForecastResult:
    prediction: PredictionResult


def train_timeseries_model(
    training_data: pl.DataFrame,
    target: str,
    config: TimeSeriesConfig,
    backtest_data: pl.DataFrame | None = None,
    training_percent: int = 100,
) -> TimeSeriesTrainingResult:
    _validate_config(config)
    if not 1 <= training_percent <= 100:
        raise ValueError("Training percent must be between 1 and 100.")
    target_values = _target_values(training_data, target)
    selected_rows = max(1, ceil(len(target_values) * training_percent / 100))
    target_values = target_values[-selected_rows:]
    if len(target_values) <= config.max_time_lag:
        raise ValueError(
            "Selected training data must contain more rows than the max time lag."
        )

    _set_seed()
    scaler = _fit_scaler(target_values, config.target_processing)
    scaled_target = _transform(target_values, scaler)
    windows, labels = _build_windows(scaled_target, config.lag_indices)
    model = create_timeseries_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    dataset = TensorDataset(
        torch.tensor(windows, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        total_loss = 0.0
        for batch_windows, batch_labels in loader:
            optimizer.zero_grad()
            predictions = model(batch_windows)
            loss = loss_function(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_windows)
        losses.append(total_loss / len(dataset))

    artifact = TimeSeriesArtifact(
        version=TIMESERIES_ARTIFACT_VERSION,
        target=target,
        config=config,
        scaler=scaler,
        target_history=tuple(
            float(value) for value in target_values[-config.max_time_lag :]
        ),
        state_dict={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    )
    prediction = None
    if backtest_data is not None:
        actual = _target_values(backtest_data, target)
        prediction = forecast_timeseries_model(
            artifact,
            len(actual),
            actual,
        ).prediction
    return TimeSeriesTrainingResult(
        artifact=artifact,
        trained_windows=len(dataset),
        losses=tuple(losses),
        prediction=prediction,
    )


def forecast_timeseries_model(
    artifact: TimeSeriesArtifact,
    horizon: int,
    actual: pl.DataFrame | pl.Series | np.ndarray | None = None,
) -> TimeSeriesForecastResult:
    if artifact.version != TIMESERIES_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported time-series artifact version {artifact.version}; "
            f"expected {TIMESERIES_ARTIFACT_VERSION}."
        )
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    actual_values = _optional_actual_values(actual, artifact.target)
    if actual_values is not None and len(actual_values) != horizon:
        raise ValueError("Actual target rows must match the forecast horizon.")

    model = create_timeseries_model(artifact.config)
    model.load_state_dict(artifact.state_dict)
    model.eval()
    history = _transform(np.asarray(artifact.target_history), artifact.scaler).tolist()
    scaled_predictions: list[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            selected_lags = [
                history[-lag] for lag in reversed(artifact.config.lag_indices)
            ]
            window = torch.tensor(
                selected_lags,
                dtype=torch.float32,
            ).reshape(1, len(artifact.config.lag_indices), 1)
            prediction = float(model(window).item())
            scaled_predictions.append(prediction)
            history.append(prediction)

    predictions = _inverse_transform(
        np.asarray(scaled_predictions),
        artifact.scaler,
    )
    metrics: RegressionMetrics | None = None
    if actual_values is None:
        prediction_data = pl.DataFrame({"Prediction": predictions})
    else:
        metrics = calculate_metrics(actual_values, predictions)
        prediction_data = pl.DataFrame(
            {"Real": actual_values, "Prediction": predictions}
        )
    return TimeSeriesForecastResult(
        prediction=PredictionResult(data=prediction_data, metrics=metrics)
    )


def serialize_timeseries_artifact(artifact: TimeSeriesArtifact) -> bytes:
    payload = {
        "version": artifact.version,
        "target": artifact.target,
        "config": asdict(artifact.config),
        "scaler": asdict(artifact.scaler),
        "target_history": list(artifact.target_history),
        "state_dict": artifact.state_dict,
    }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def deserialize_timeseries_artifact(data: bytes) -> TimeSeriesArtifact:
    try:
        payload = torch.load(
            BytesIO(data),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("Artifact payload must be a mapping.")
        version = int(payload["version"])
        if version != TIMESERIES_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported time-series artifact version {version}; "
                f"expected {TIMESERIES_ARTIFACT_VERSION}."
            )
        artifact = TimeSeriesArtifact(
            version=version,
            target=str(payload["target"]),
            config=TimeSeriesConfig(**payload["config"]),
            scaler=TargetScaler(**payload["scaler"]),
            target_history=tuple(float(value) for value in payload["target_history"]),
            state_dict=payload["state_dict"],
        )
        _validate_config(artifact.config)
        if len(artifact.target_history) != artifact.config.max_time_lag:
            raise ValueError("Artifact target history does not match its max time lag.")
        model = create_timeseries_model(artifact.config)
        model.load_state_dict(artifact.state_dict)
        return artifact
    except (
        EOFError,
        KeyError,
        TypeError,
        RuntimeError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        if isinstance(error, ValueError) and str(error).startswith("Unsupported"):
            raise
        raise ValueError(
            "The uploaded file is not a valid MLStudio time-series artifact."
        ) from error


def create_timeseries_model(config: TimeSeriesConfig) -> nn.Module:
    _validate_config(config)
    if config.model_name == "MLP":
        return _MLP(config)
    if config.model_name == "CNN":
        return _CNN(config)
    if config.model_name == "ConvLSTM":
        return _ConvLSTM(config)
    return _Recurrent(config)


def _build_windows(
    values: np.ndarray,
    lag_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    max_lag = max(lag_indices)
    windows = np.asarray(
        [
            [values[index - lag] for lag in reversed(lag_indices)]
            for index in range(max_lag, len(values))
        ],
        dtype=np.float32,
    ).reshape(-1, len(lag_indices), 1)
    labels = np.asarray(values[max_lag:], dtype=np.float32).reshape(-1, 1)
    return windows, labels


class _MLP(nn.Module):
    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_size = len(config.lag_indices)
        for layer_size in config.layer_sizes:
            blocks.extend(
                [
                    nn.Linear(input_size, layer_size),
                    _activation(config.hidden_activation),
                ]
            )
            input_size = layer_size
        blocks.extend(
            [
                nn.Linear(input_size, 1),
                _activation(config.output_activation),
            ]
        )
        self.network = nn.Sequential(*blocks)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.flatten(start_dim=1))


class _CNN(nn.Module):
    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = 1
        for layer_size in config.layer_sizes:
            blocks.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        layer_size,
                        kernel_size=3,
                        padding=1,
                    ),
                    _activation(config.hidden_activation),
                ]
            )
            input_channels = layer_size
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Linear(config.layer_sizes[-1], 1),
            _activation(config.output_activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features(values.transpose(1, 2))
        return self.output(self.pool(features).squeeze(-1))


class _Recurrent(nn.Module):
    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        recurrent_types: dict[str, type[nn.RNN] | type[nn.GRU] | type[nn.LSTM]] = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "Bi-LSTM": nn.LSTM,
        }
        bidirectional = config.model_name == "Bi-LSTM"
        recurrent_type = recurrent_types[config.model_name]
        self.recurrent_layers = nn.ModuleList()
        input_size = 1
        for layer_size in config.layer_sizes:
            recurrent_arguments: dict[str, object] = {
                "input_size": input_size,
                "hidden_size": layer_size,
                "num_layers": 1,
                "batch_first": True,
                "bidirectional": bidirectional,
            }
            if config.model_name == "RNN":
                recurrent_arguments["nonlinearity"] = (
                    "relu" if config.hidden_activation == "ReLU" else "tanh"
                )
            self.recurrent_layers.append(recurrent_type(**recurrent_arguments))
            input_size = layer_size * (2 if bidirectional else 1)
        self.output = nn.Sequential(
            _activation(config.hidden_activation),
            nn.Linear(input_size, 1),
            _activation(config.output_activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = values
        for recurrent_layer in self.recurrent_layers:
            sequence, _ = recurrent_layer(sequence)
        return self.output(sequence[:, -1, :])


class _ConvLSTM(nn.Module):
    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        convolution_size = config.layer_sizes[0]
        self.convolution = nn.Sequential(
            nn.Conv1d(1, convolution_size, kernel_size=3, padding=1),
            _activation(config.hidden_activation),
        )
        self.recurrent_layers = nn.ModuleList()
        input_size = convolution_size
        for layer_size in config.layer_sizes:
            self.recurrent_layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=layer_size,
                    num_layers=1,
                    batch_first=True,
                )
            )
            input_size = layer_size
        self.output = nn.Sequential(
            _activation(config.hidden_activation),
            nn.Linear(config.layer_sizes[-1], 1),
            _activation(config.output_activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        for recurrent_layer in self.recurrent_layers:
            sequence, _ = recurrent_layer(sequence)
        return self.output(sequence[:, -1, :])


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "Linear": nn.Identity,
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "GELU": nn.GELU,
    }
    return activations[name]()


def _validate_config(config: TimeSeriesConfig) -> None:
    if config.model_name not in TIMESERIES_MODEL_NAMES:
        raise ValueError(f"Unsupported time-series model: {config.model_name}.")
    if config.hidden_activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(f"Unsupported hidden activation: {config.hidden_activation}.")
    if config.output_activation not in OUTPUT_ACTIVATIONS:
        raise ValueError(f"Unsupported output activation: {config.output_activation}.")
    if config.max_time_lag < 1:
        raise ValueError("Max time lag must be at least 1.")
    if not config.lag_indices:
        raise ValueError("Select at least one lag index.")
    if tuple(sorted(set(config.lag_indices))) != config.lag_indices:
        raise ValueError("Lag indices must be unique and sorted.")
    if config.lag_indices[0] < 1:
        raise ValueError("Lag indices must be positive.")
    if config.lag_indices[-1] > config.max_time_lag:
        raise ValueError("Lag indices cannot exceed the max time lag.")
    if not config.layer_sizes:
        raise ValueError("At least one model layer is required.")
    if any(size < 1 for size in config.layer_sizes):
        raise ValueError("Every layer must contain at least one neuron.")
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be greater than zero.")
    if config.epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("Batch size must be at least 1.")


def _target_values(data: pl.DataFrame, target: str) -> np.ndarray:
    if target not in data.columns:
        raise ValueError(f"Missing target column: {target}.")
    series = data[target]
    if not series.dtype.is_numeric():
        raise ValueError("The target column must be numeric.")
    if series.null_count() > 0:
        raise ValueError("The target column cannot contain missing values.")
    values = series.to_numpy().astype(float)
    if len(values) == 0:
        raise ValueError("The target dataset is empty.")
    if not np.isfinite(values).all():
        raise ValueError("The target column must contain only finite values.")
    return values


def _optional_actual_values(
    actual: pl.DataFrame | pl.Series | np.ndarray | None,
    target: str,
) -> np.ndarray | None:
    if actual is None:
        return None
    if isinstance(actual, pl.DataFrame):
        return _target_values(actual, target)
    if isinstance(actual, pl.Series):
        return _target_values(pl.DataFrame({target: actual}), target)
    values = np.asarray(actual, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(
            "Actual target values must be a finite one-dimensional series."
        )
    return values


def _fit_scaler(
    values: np.ndarray,
    processing: TargetProcessing,
) -> TargetScaler:
    if processing == "None":
        return TargetScaler(processing, 0.0, 1.0)
    if processing == "StandardScaler":
        scale = float(values.std())
        return TargetScaler(processing, float(values.mean()), scale or 1.0)
    span = float(values.max() - values.min())
    return TargetScaler(processing, float(values.min()), span or 1.0)


def _transform(values: np.ndarray, scaler: TargetScaler) -> np.ndarray:
    return (np.asarray(values, dtype=float) - scaler.offset) / scaler.scale


def _inverse_transform(values: np.ndarray, scaler: TargetScaler) -> np.ndarray:
    return np.asarray(values, dtype=float) * scaler.scale + scaler.offset


def _set_seed() -> None:
    np.random.seed(42)
    torch.set_num_threads(1)
    torch.manual_seed(42)

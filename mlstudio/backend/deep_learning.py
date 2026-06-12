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

DeepModelName = Literal[
    "MLP",
    "1D-CNN",
    "RNN",
    "GRU",
    "LSTM",
    "Bi-LSTM",
    "ConvLSTM",
]
HiddenActivation = Literal["ReLU", "Tanh", "Sigmoid", "GELU"]
OutputActivation = Literal["Linear", "ReLU", "Sigmoid", "Tanh"]

DEEP_MODEL_NAMES: tuple[DeepModelName, ...] = (
    "MLP",
    "1D-CNN",
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
DEEP_ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class DeepLearningConfig:
    model_name: DeepModelName
    lookback: int = 10
    neurons: int = 64
    layers: int = 2
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
class DeepLearningArtifact:
    version: int
    target: str
    config: DeepLearningConfig
    scaler: TargetScaler
    target_history: tuple[float, ...]
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class DeepTrainingResult:
    artifact: DeepLearningArtifact
    trained_windows: int
    losses: tuple[float, ...]
    prediction: PredictionResult | None


@dataclass(frozen=True)
class DeepForecastResult:
    prediction: PredictionResult


def train_deep_model(
    training_data: pl.DataFrame,
    target: str,
    config: DeepLearningConfig,
    backtest_data: pl.DataFrame | None = None,
    training_percent: int = 100,
) -> DeepTrainingResult:
    _validate_config(config)
    if not 1 <= training_percent <= 100:
        raise ValueError("Training percent must be between 1 and 100.")
    target_values = _target_values(training_data, target)
    selected_rows = max(1, ceil(len(target_values) * training_percent / 100))
    target_values = target_values[-selected_rows:]
    if len(target_values) <= config.lookback:
        raise ValueError(
            "Selected training data must contain more rows than the lookback."
        )

    _set_seed()
    scaler = _fit_scaler(target_values, config.target_processing)
    scaled_target = _transform(target_values, scaler)
    windows, labels = _build_windows(scaled_target, config.lookback)
    model = create_deep_model(config)
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

    artifact = DeepLearningArtifact(
        version=DEEP_ARTIFACT_VERSION,
        target=target,
        config=config,
        scaler=scaler,
        target_history=tuple(float(value) for value in target_values[-config.lookback :]),
        state_dict={
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    )
    prediction = None
    if backtest_data is not None:
        actual = _target_values(backtest_data, target)
        prediction = forecast_deep_model(
            artifact,
            len(actual),
            actual,
        ).prediction
    return DeepTrainingResult(
        artifact=artifact,
        trained_windows=len(dataset),
        losses=tuple(losses),
        prediction=prediction,
    )


def forecast_deep_model(
    artifact: DeepLearningArtifact,
    horizon: int,
    actual: pl.DataFrame | pl.Series | np.ndarray | None = None,
) -> DeepForecastResult:
    if artifact.version != DEEP_ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported deep-learning artifact version {artifact.version}; "
            f"expected {DEEP_ARTIFACT_VERSION}."
        )
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    actual_values = _optional_actual_values(actual, artifact.target)
    if actual_values is not None and len(actual_values) != horizon:
        raise ValueError("Actual target rows must match the forecast horizon.")

    model = create_deep_model(artifact.config)
    model.load_state_dict(artifact.state_dict)
    model.eval()
    history = _transform(np.asarray(artifact.target_history), artifact.scaler).tolist()
    scaled_predictions: list[float] = []
    with torch.no_grad():
        for _ in range(horizon):
            window = torch.tensor(
                history[-artifact.config.lookback :],
                dtype=torch.float32,
            ).reshape(1, artifact.config.lookback, 1)
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
    return DeepForecastResult(
        prediction=PredictionResult(data=prediction_data, metrics=metrics)
    )


def serialize_deep_artifact(artifact: DeepLearningArtifact) -> bytes:
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


def deserialize_deep_artifact(data: bytes) -> DeepLearningArtifact:
    try:
        payload = torch.load(
            BytesIO(data),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(payload, dict):
            raise ValueError("Artifact payload must be a mapping.")
        artifact = DeepLearningArtifact(
            version=int(payload["version"]),
            target=str(payload["target"]),
            config=DeepLearningConfig(**payload["config"]),
            scaler=TargetScaler(**payload["scaler"]),
            target_history=tuple(float(value) for value in payload["target_history"]),
            state_dict=payload["state_dict"],
        )
        _validate_config(artifact.config)
        if artifact.version != DEEP_ARTIFACT_VERSION:
            raise ValueError(
                f"Unsupported deep-learning artifact version {artifact.version}; "
                f"expected {DEEP_ARTIFACT_VERSION}."
            )
        if len(artifact.target_history) != artifact.config.lookback:
            raise ValueError("Artifact target history does not match its lookback.")
        model = create_deep_model(artifact.config)
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
            "The uploaded file is not a valid MLStudio deep-learning artifact."
        ) from error


def create_deep_model(config: DeepLearningConfig) -> nn.Module:
    _validate_config(config)
    if config.model_name == "MLP":
        return _MLP(config)
    if config.model_name == "1D-CNN":
        return _CNN(config)
    if config.model_name == "ConvLSTM":
        return _ConvLSTM(config)
    return _Recurrent(config)


def _build_windows(
    values: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(
        [values[index - lookback : index] for index in range(lookback, len(values))],
        dtype=np.float32,
    ).reshape(-1, lookback, 1)
    labels = np.asarray(values[lookback:], dtype=np.float32).reshape(-1, 1)
    return windows, labels


class _MLP(nn.Module):
    def __init__(self, config: DeepLearningConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_size = config.lookback
        for _ in range(config.layers):
            blocks.extend(
                [
                    nn.Linear(input_size, config.neurons),
                    _activation(config.hidden_activation),
                ]
            )
            input_size = config.neurons
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
    def __init__(self, config: DeepLearningConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = 1
        for _ in range(config.layers):
            blocks.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        config.neurons,
                        kernel_size=3,
                        padding=1,
                    ),
                    _activation(config.hidden_activation),
                ]
            )
            input_channels = config.neurons
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Linear(config.neurons, 1),
            _activation(config.output_activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.features(values.transpose(1, 2))
        return self.output(self.pool(features).squeeze(-1))


class _Recurrent(nn.Module):
    def __init__(self, config: DeepLearningConfig) -> None:
        super().__init__()
        recurrent_types: dict[str, type[nn.RNN] | type[nn.GRU] | type[nn.LSTM]] = {
            "RNN": nn.RNN,
            "GRU": nn.GRU,
            "LSTM": nn.LSTM,
            "Bi-LSTM": nn.LSTM,
        }
        bidirectional = config.model_name == "Bi-LSTM"
        recurrent_type = recurrent_types[config.model_name]
        recurrent_arguments: dict[str, object] = {
            "input_size": 1,
            "hidden_size": config.neurons,
            "num_layers": config.layers,
            "batch_first": True,
            "bidirectional": bidirectional,
        }
        if config.model_name == "RNN":
            recurrent_arguments["nonlinearity"] = (
                "relu" if config.hidden_activation == "ReLU" else "tanh"
            )
        self.recurrent = recurrent_type(**recurrent_arguments)
        output_size = config.neurons * (2 if bidirectional else 1)
        self.output = nn.Sequential(
            _activation(config.hidden_activation),
            nn.Linear(output_size, 1),
            _activation(config.output_activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(values)
        return self.output(sequence[:, -1, :])


class _ConvLSTM(nn.Module):
    def __init__(self, config: DeepLearningConfig) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(1, config.neurons, kernel_size=3, padding=1),
            _activation(config.hidden_activation),
        )
        self.recurrent = nn.LSTM(
            input_size=config.neurons,
            hidden_size=config.neurons,
            num_layers=config.layers,
            batch_first=True,
        )
        self.output = nn.Sequential(
            _activation(config.hidden_activation),
            nn.Linear(config.neurons, 1),
            _activation(config.output_activation),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.convolution(values.transpose(1, 2)).transpose(1, 2)
        sequence, _ = self.recurrent(features)
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


def _validate_config(config: DeepLearningConfig) -> None:
    if config.model_name not in DEEP_MODEL_NAMES:
        raise ValueError(f"Unsupported deep-learning model: {config.model_name}.")
    if config.hidden_activation not in HIDDEN_ACTIVATIONS:
        raise ValueError(f"Unsupported hidden activation: {config.hidden_activation}.")
    if config.output_activation not in OUTPUT_ACTIVATIONS:
        raise ValueError(f"Unsupported output activation: {config.output_activation}.")
    if config.lookback < 1:
        raise ValueError("Lookback must be at least 1.")
    if config.neurons < 1:
        raise ValueError("Neurons must be at least 1.")
    if config.layers < 1:
        raise ValueError("Layers must be at least 1.")
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
        raise ValueError("Actual target values must be a finite one-dimensional series.")
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

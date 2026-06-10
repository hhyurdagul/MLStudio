from dataclasses import dataclass
from collections.abc import Callable
from typing import Literal

import polars as pl
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .models import ModelDefinition, ParameterValue

RowSelection = Literal["Random percent", "Last percent"]
ValidationStrategy = Literal["Random split", "Last split", "Cross-validation"]
TargetProcessing = Literal["None", "StandardScaler", "MinMaxScaler"]
FittedEstimator = Pipeline | GridSearchCV


@dataclass(frozen=True)
class PipelineStep:
    name: str
    transformer: BaseEstimator | Literal["passthrough"]


@dataclass(frozen=True)
class EstimatorWrapper:
    name: str
    wrap: Callable[[BaseEstimator], BaseEstimator]
    requires_ordered_data: bool = False
    supports_grid_search: bool = True
    grid_parameter_prefix: str | None = None
    use_estimator_score: bool = False


@dataclass(frozen=True)
class ModelConfig:
    definition: ModelDefinition
    parameters: dict[str, ParameterValue]
    use_grid_search: bool
    param_grid: dict[str, list[ParameterValue]]
    cv: int


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    mae: float
    rmse: float
    mape: float | None


@dataclass(frozen=True)
class GridSearchSummary:
    best_parameters: dict[str, object]
    best_score: float


@dataclass(frozen=True)
class ModelArtifact:
    version: int
    pipeline: FittedEstimator
    features: tuple[str, ...]
    target: str
    feature_dtypes: dict[str, str]
    model_label: str


@dataclass(frozen=True)
class ProcessedData:
    preprocessed: pl.DataFrame
    model_input: pl.DataFrame
    selected_features: tuple[str, ...]


@dataclass(frozen=True)
class PredictionResult:
    data: pl.DataFrame
    metrics: RegressionMetrics | None
    processed: ProcessedData


@dataclass(frozen=True)
class TrainingResult:
    artifact: ModelArtifact
    trained_rows: int
    prediction: PredictionResult | None
    grid_search: GridSearchSummary | None


@dataclass(frozen=True)
class ValidationResult:
    metrics: RegressionMetrics
    prediction: PredictionResult | None
    grid_search: GridSearchSummary | None

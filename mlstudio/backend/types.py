from dataclasses import dataclass
from typing import Literal

import polars as pl
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .models import ModelDefinition, ParameterValue

RowSelection = Literal["Random percent", "Last percent"]
ValidationStrategy = Literal["Random split", "Last split", "Cross-validation"]
FittedEstimator = Pipeline | GridSearchCV


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
class CrossValidationMetrics:
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    mape_mean: float | None
    mape_std: float | None


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
class PredictionResult:
    data: pl.DataFrame
    metrics: RegressionMetrics | None


@dataclass(frozen=True)
class TrainingResult:
    artifact: ModelArtifact
    trained_rows: int
    prediction: PredictionResult | None
    grid_search: GridSearchSummary | None


@dataclass(frozen=True)
class ValidationResult:
    metrics: RegressionMetrics | CrossValidationMetrics
    prediction: PredictionResult | None
    grid_search: GridSearchSummary | None

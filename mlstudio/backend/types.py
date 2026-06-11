from dataclasses import dataclass
from typing import Literal

import polars as pl
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .models import ModelDefinition, ParameterValue

RowSelection = Literal["Random percent", "Last percent"]
ValidationStrategy = Literal["Random split", "Last split", "Cross-validation"]
TargetProcessing = Literal["None", "StandardScaler", "MinMaxScaler"]
FittedEstimator = Pipeline | GridSearchCV


@dataclass(frozen=True)
class ModelConfig:
    definition: ModelDefinition
    parameters: dict[str, ParameterValue]
    use_grid_search: bool
    param_grid: dict[str, list[ParameterValue]]
    cv: int


@dataclass(frozen=True)
class FeatureSelectionConfig:
    method: str
    count: int


@dataclass(frozen=True)
class PipelineConfig:
    features: tuple[str, ...]
    target: str
    preprocessing: pl.DataFrame
    model: ModelConfig
    target_processing: TargetProcessing = "None"
    feature_selection: FeatureSelectionConfig | None = None
    lookback: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    pipeline: PipelineConfig
    row_selection: RowSelection = "Random percent"
    percent: int = 100


@dataclass(frozen=True)
class ValidationConfig:
    pipeline: PipelineConfig
    strategy: ValidationStrategy = "Random split"
    percent: int = 20
    folds: int = 5


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


@dataclass(frozen=True)
class TestResult:
    processed: ProcessedData
    prediction: PredictionResult


@dataclass(frozen=True)
class TrainingResult:
    artifact: ModelArtifact
    trained_rows: int
    processed: ProcessedData
    prediction: PredictionResult | None
    grid_search: GridSearchSummary | None


@dataclass(frozen=True)
class ValidationResult:
    metrics: RegressionMetrics
    processed: ProcessedData
    prediction: PredictionResult
    grid_search: GridSearchSummary | None

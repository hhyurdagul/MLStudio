from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import ceil
from typing import Literal

import joblib
import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from mlstudio.ml.models import ModelDefinition, ParameterValue

RowSelection = Literal["Random percent", "Last percent"]
ValidationStrategy = Literal["Random split", "Last split", "Cross-validation"]


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
class ModelArtifact:
    version: int
    pipeline: Pipeline | GridSearchCV
    features: tuple[str, ...]
    target: str
    feature_dtypes: dict[str, str]
    model_label: str


ARTIFACT_VERSION = 1


def create_estimator(
    transformer: ColumnTransformer,
    config: ModelConfig,
) -> Pipeline | GridSearchCV:
    pipeline = Pipeline(
        [
            ("preprocessing", transformer),
            ("model", config.definition.create_estimator(config.parameters)),
        ]
    )
    if not config.use_grid_search:
        return pipeline

    return GridSearchCV(
        pipeline,
        config.param_grid,
        cv=config.cv,
        n_jobs=-1,
        scoring="r2",
    )


def select_training_rows(
    df: pl.DataFrame,
    method: RowSelection,
    percent: int,
    *,
    seed: int = 42,
) -> pl.DataFrame:
    if df.is_empty():
        raise ValueError("The training dataset is empty.")
    if not 1 <= percent <= 100:
        raise ValueError("Training percent must be between 1 and 100.")

    row_count = max(1, ceil(df.height * percent / 100))
    if method == "Random percent":
        return df.sample(n=row_count, shuffle=True, seed=seed)
    if method == "Last percent":
        return df.tail(row_count)
    raise ValueError(f"Unknown row selection method: {method}")


def split_validation_data(
    df: pl.DataFrame,
    strategy: ValidationStrategy,
    validation_percent: int,
    *,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if df.height < 2:
        raise ValueError("Validation requires at least two rows.")
    if not 1 <= validation_percent <= 99:
        raise ValueError("Validation percent must be between 1 and 99.")

    if strategy == "Random split":
        train, validation = train_test_split(
            df,
            test_size=validation_percent / 100,
            random_state=seed,
            shuffle=True,
        )
        return pl.DataFrame(train), pl.DataFrame(validation)

    if strategy == "Last split":
        validation_rows = max(1, ceil(df.height * validation_percent / 100))
        if validation_rows >= df.height:
            raise ValueError("The validation split leaves no training rows.")
        return df.head(df.height - validation_rows), df.tail(validation_rows)

    raise ValueError(f"{strategy} does not create a single validation split.")


def calculate_metrics(
    actual: pl.Series | np.ndarray,
    predicted: np.ndarray,
) -> RegressionMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    nonzero = actual_values != 0
    mape = (
        float(np.mean(np.abs((actual_values[nonzero] - predicted_values[nonzero]) / actual_values[nonzero])) * 100)
        if nonzero.any()
        else None
    )
    return RegressionMetrics(
        r2=float(r2_score(actual_values, predicted_values)),
        mae=float(mean_absolute_error(actual_values, predicted_values)),
        rmse=float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        mape=mape,
    )


def _mape_without_zero_targets(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    nonzero = actual_values != 0
    if not nonzero.any():
        return float("nan")
    return float(
        np.mean(
            np.abs(
                (actual_values[nonzero] - predicted_values[nonzero])
                / actual_values[nonzero]
            )
        )
    )


def cross_validate_estimator(
    estimator: BaseEstimator,
    x: pl.DataFrame,
    y: pl.Series,
    folds: int,
) -> CrossValidationMetrics:
    if folds > x.height:
        raise ValueError("Cross-validation folds cannot exceed the number of rows.")
    if x.height < folds * 2:
        raise ValueError(
            "R² cross-validation requires at least two validation rows per fold."
        )

    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(
        estimator,
        x,
        y,
        cv=splitter,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "mape": make_scorer(
                _mape_without_zero_targets,
                greater_is_better=False,
            ),
        },
        n_jobs=-1,
    )
    mape_values = -scores["test_mape"] * 100
    finite_mape = mape_values[np.isfinite(mape_values)]
    return CrossValidationMetrics(
        r2_mean=float(np.mean(scores["test_r2"])),
        r2_std=float(np.std(scores["test_r2"])),
        mae_mean=float(np.mean(-scores["test_mae"])),
        mae_std=float(np.std(-scores["test_mae"])),
        rmse_mean=float(np.mean(-scores["test_rmse"])),
        rmse_std=float(np.std(-scores["test_rmse"])),
        mape_mean=float(np.mean(finite_mape)) if finite_mape.size else None,
        mape_std=float(np.std(finite_mape)) if finite_mape.size else None,
    )


def create_artifact(
    pipeline: Pipeline | GridSearchCV,
    df: pl.DataFrame,
    features: list[str],
    target: str,
    model_label: str,
) -> ModelArtifact:
    return ModelArtifact(
        version=ARTIFACT_VERSION,
        pipeline=pipeline,
        features=tuple(features),
        target=target,
        feature_dtypes={feature: str(df.schema[feature]) for feature in features},
        model_label=model_label,
    )


def serialize_artifact(artifact: ModelArtifact) -> bytes:
    buffer = BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def deserialize_artifact(data: bytes) -> ModelArtifact:
    artifact = joblib.load(BytesIO(data))
    if not isinstance(artifact, ModelArtifact):
        raise ValueError("The uploaded file is not an MLStudio model artifact.")
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {artifact.version}; expected {ARTIFACT_VERSION}."
        )
    return artifact


def validate_feature_columns(df: pl.DataFrame, features: list[str] | tuple[str, ...]) -> None:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))


def validate_feature_schema(
    df: pl.DataFrame,
    expected_dtypes: dict[str, str],
) -> None:
    def family(dtype: pl.DataType | type[pl.DataType]) -> str:
        if dtype.is_numeric():
            return "numeric"
        if dtype == pl.Boolean:
            return "boolean"
        if dtype in (pl.String, pl.Categorical, pl.Enum):
            return "string"
        return str(dtype)

    incompatible = [
        feature
        for feature, expected in expected_dtypes.items()
        if family(df.schema[feature])
        != (
            "numeric"
            if expected.startswith(("Int", "UInt", "Float", "Decimal"))
            else "boolean"
            if expected == "Boolean"
            else "string"
            if expected.startswith(("String", "Categorical", "Enum"))
            else expected
        )
    ]
    if incompatible:
        raise ValueError(
            "Incompatible feature types for: " + ", ".join(incompatible)
        )


def prediction_frame(
    df: pl.DataFrame,
    predictions: np.ndarray,
    target: str | None = None,
) -> pl.DataFrame:
    result = pl.DataFrame({"Prediction": predictions})
    if target is None or target not in df.columns:
        return result
    return pl.DataFrame(
        {
            "Real": df[target],
            "Prediction": predictions,
        }
    )

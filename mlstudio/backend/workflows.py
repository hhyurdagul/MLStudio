from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .artifacts import ARTIFACT_VERSION
from .evaluation import (
    calculate_metrics,
    cross_validate_estimator,
    select_training_rows,
    split_validation_data,
)
from .preprocessing import create_preprocessing_transformer
from .types import (
    GridSearchSummary,
    ModelArtifact,
    ModelConfig,
    PipelineStep,
    PredictionResult,
    ProcessedData,
    RowSelection,
    TrainingResult,
    ValidationResult,
    ValidationStrategy,
)


def train(
    training_data: pl.DataFrame,
    test_data: pl.DataFrame | None,
    features: list[str],
    target: str,
    preprocessing: pl.DataFrame,
    model: ModelConfig,
    row_selection: RowSelection,
    training_percent: int,
    pipeline_steps: tuple[PipelineStep, ...] = (),
) -> TrainingResult:
    _validate_labeled_data(training_data, features, target)
    selected_data = select_training_rows(
        training_data,
        row_selection,
        training_percent,
    )
    _validate_grid_folds(model, selected_data.height)

    estimator = _create_estimator(preprocessing, model, pipeline_steps)
    estimator.fit(selected_data.select(features), selected_data[target])
    artifact = ModelArtifact(
        version=ARTIFACT_VERSION,
        pipeline=estimator,
        features=tuple(features),
        target=target,
        feature_dtypes={
            feature: str(training_data.schema[feature]) for feature in features
        },
        model_label=model.definition.label,
    )
    return TrainingResult(
        artifact=artifact,
        trained_rows=selected_data.height,
        prediction=(
            _predict_with_artifact(artifact, test_data)
            if test_data is not None
            else None
        ),
        grid_search=_grid_search_summary(estimator),
    )


def validate(
    data: pl.DataFrame,
    features: list[str],
    target: str,
    preprocessing: pl.DataFrame,
    model: ModelConfig,
    strategy: ValidationStrategy,
    validation_percent: int,
    folds: int,
    pipeline_steps: tuple[PipelineStep, ...] = (),
) -> ValidationResult:
    _validate_labeled_data(data, features, target)
    estimator = _create_estimator(preprocessing, model, pipeline_steps)

    if strategy == "Cross-validation":
        metrics = cross_validate_estimator(
            estimator,
            data.select(features),
            data[target],
            folds,
        )
        return ValidationResult(metrics, None, None)

    training_data, validation_data = split_validation_data(
        data,
        strategy,
        validation_percent,
    )
    _validate_grid_folds(model, training_data.height)
    estimator.fit(training_data.select(features), training_data[target])
    predictions = estimator.predict(validation_data.select(features))
    metrics = calculate_metrics(validation_data[target], predictions)
    prediction = PredictionResult(
        data=_prediction_frame(validation_data, predictions, target),
        metrics=metrics,
        processed=_processed_data(estimator, validation_data.select(features)),
    )
    return ValidationResult(
        metrics=metrics,
        prediction=prediction,
        grid_search=_grid_search_summary(estimator),
    )


def predict(artifact: ModelArtifact, data: pl.DataFrame) -> PredictionResult:
    return _predict_with_artifact(artifact, data)


def _create_estimator(
    preprocessing: pl.DataFrame,
    config: ModelConfig,
    pipeline_steps: tuple[PipelineStep, ...],
) -> Pipeline | GridSearchCV:
    pipeline = Pipeline(
        [
            (
                "preprocessing",
                create_preprocessing_transformer(preprocessing),
            ),
            *((step.name, step.transformer) for step in pipeline_steps),
            ("model", config.definition.create_estimator(config.parameters)),
        ]
    )
    if not config.use_grid_search:
        return pipeline
    if not config.param_grid or any(not values for values in config.param_grid.values()):
        raise ValueError("Every grid-search parameter needs at least one value.")
    return GridSearchCV(
        pipeline,
        config.param_grid,
        cv=config.cv,
        n_jobs=-1,
        scoring="r2",
    )


def _predict_with_artifact(
    artifact: ModelArtifact,
    data: pl.DataFrame,
) -> PredictionResult:
    _validate_feature_data(data, artifact.features, artifact.feature_dtypes)
    predictions = artifact.pipeline.predict(data.select(artifact.features))
    metrics = None
    if artifact.target in data.columns:
        _validate_target(data, artifact.target)
        metrics = calculate_metrics(data[artifact.target], predictions)
    return PredictionResult(
        data=_prediction_frame(data, predictions, artifact.target),
        metrics=metrics,
        processed=_processed_data(
            artifact.pipeline,
            data.select(artifact.features),
        ),
    )


def _prediction_frame(
    data: pl.DataFrame,
    predictions: np.ndarray,
    target: str,
) -> pl.DataFrame:
    if target not in data.columns:
        return pl.DataFrame({"Prediction": predictions})
    return pl.DataFrame(
        {
            "Real": data[target],
            "Prediction": predictions,
        }
    )


def _processed_data(
    estimator: Pipeline | GridSearchCV,
    features: pl.DataFrame,
) -> ProcessedData:
    pipeline = (
        estimator.best_estimator_
        if isinstance(estimator, GridSearchCV)
        else estimator
    )
    preprocessing = pipeline.named_steps["preprocessing"]
    transformed = preprocessing.transform(features)
    feature_names = list(preprocessing.get_feature_names_out())
    preprocessed = _to_frame(transformed, feature_names)

    model_input = transformed
    selected_names = feature_names
    for name, transformer in pipeline.steps[1:-1]:
        if transformer == "passthrough":
            continue
        model_input = transformer.transform(model_input)
        selected_names = _transformed_feature_names(
            transformer,
            selected_names,
            step_name=name,
        )

    return ProcessedData(
        preprocessed=preprocessed,
        model_input=_to_frame(model_input, selected_names),
        selected_features=tuple(selected_names),
    )


def _transformed_feature_names(
    transformer: object,
    input_features: list[str],
    *,
    step_name: str,
) -> list[str]:
    get_support = getattr(transformer, "get_support", None)
    if callable(get_support):
        support = get_support()
        return [
            feature
            for feature, selected in zip(input_features, support, strict=True)
            if selected
        ]

    get_feature_names_out = getattr(transformer, "get_feature_names_out", None)
    if callable(get_feature_names_out):
        return list(get_feature_names_out(input_features))

    output_count = getattr(transformer, "n_features_out_", len(input_features))
    if output_count == len(input_features):
        return input_features
    return [f"{step_name}_{index}" for index in range(output_count)]


def _to_frame(values: Any, columns: list[str]) -> pl.DataFrame:
    toarray = getattr(values, "toarray", None)
    if callable(toarray):
        values = toarray()
    return pl.DataFrame(values, schema=columns, orient="row")


def _validate_labeled_data(
    data: pl.DataFrame,
    features: list[str],
    target: str,
) -> None:
    if not features:
        raise ValueError("Select at least one feature.")
    missing = [column for column in [*features, target] if column not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    _validate_target(data, target)
    null_columns = [
        column for column in [*features, target] if data[column].null_count() > 0
    ]
    if null_columns:
        raise ValueError(
            "Missing values are not supported yet. Found nulls in: "
            + ", ".join(null_columns)
        )


def _validate_target(data: pl.DataFrame, target: str) -> None:
    if not data.schema[target].is_numeric():
        raise ValueError("Regression requires a numeric target.")
    if data[target].null_count() > 0:
        raise ValueError("The target contains missing values.")


def _validate_feature_data(
    data: pl.DataFrame,
    features: tuple[str, ...],
    expected_dtypes: dict[str, str],
) -> None:
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    incompatible = [
        feature
        for feature, expected in expected_dtypes.items()
        if _dtype_family(data.schema[feature]) != _dtype_name_family(expected)
    ]
    if incompatible:
        raise ValueError(
            "Incompatible feature types for: " + ", ".join(incompatible)
        )
    null_features = [feature for feature in features if data[feature].null_count() > 0]
    if null_features:
        raise ValueError(
            "Missing values are not supported yet. Found nulls in: "
            + ", ".join(null_features)
        )


def _dtype_family(dtype: pl.DataType | type[pl.DataType]) -> str:
    if dtype.is_numeric():
        return "numeric"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype in (pl.String, pl.Categorical, pl.Enum):
        return "string"
    return str(dtype)


def _dtype_name_family(dtype: str) -> str:
    if dtype.startswith(("Int", "UInt", "Float", "Decimal")):
        return "numeric"
    if dtype == "Boolean":
        return "boolean"
    if dtype.startswith(("String", "Categorical", "Enum")):
        return "string"
    return dtype


def _validate_grid_folds(config: ModelConfig, rows: int) -> None:
    if config.use_grid_search and config.cv > rows:
        raise ValueError("Grid-search folds cannot exceed the training rows.")


def _grid_search_summary(
    estimator: Pipeline | GridSearchCV,
) -> GridSearchSummary | None:
    if not isinstance(estimator, GridSearchCV):
        return None
    return GridSearchSummary(
        best_parameters=estimator.best_params_,
        best_score=float(estimator.best_score_),
    )

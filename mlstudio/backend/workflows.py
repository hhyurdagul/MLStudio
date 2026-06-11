from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .artifacts import ARTIFACT_VERSION
from .evaluation import (
    calculate_metrics,
    cross_validation_predictions,
    select_training_rows,
    split_validation_data,
    time_series_validation_predictions,
)
from .feature_selection import create_feature_selector
from .lookback import AutoregressiveRegressor
from .preprocessing import (
    create_preprocessing_transformer,
    wrap_target_processing,
)
from .types import (
    GridSearchSummary,
    ModelArtifact,
    ModelConfig,
    PipelineConfig,
    PredictionResult,
    ProcessedData,
    RowSelection,
    TargetProcessing,
    TestResult,
    TrainingConfig,
    TrainingResult,
    ValidationConfig,
    ValidationResult,
    ValidationStrategy,
)


def train(
    training_data: pl.DataFrame,
    config: TrainingConfig,
    test_data: pl.DataFrame | None = None,
) -> TrainingResult:
    pipeline = config.pipeline
    _validate_labeled_data(training_data, pipeline.features, pipeline.target)
    selected_data = select_training_rows(
        training_data,
        config.row_selection,
        config.percent,
    )
    _validate_ordered_training(pipeline.lookback, config.row_selection)
    _validate_grid_folds(pipeline.model, selected_data.height)

    estimator = _create_estimator(pipeline)
    estimator.fit(
        selected_data.select(pipeline.features),
        selected_data[pipeline.target],
    )
    artifact = ModelArtifact(
        version=ARTIFACT_VERSION,
        pipeline=estimator,
        features=pipeline.features,
        target=pipeline.target,
        feature_dtypes={
            feature: str(training_data.schema[feature]) for feature in pipeline.features
        },
        model_label=pipeline.model.definition.label,
    )
    return TrainingResult(
        artifact=artifact,
        trained_rows=selected_data.height,
        processed=_processed_data(
            estimator,
            selected_data.select(pipeline.features),
        ),
        prediction=(
            _predict_with_artifact(artifact, test_data)
            if test_data is not None
            else None
        ),
        grid_search=_grid_search_summary(estimator),
    )


def validate(
    data: pl.DataFrame,
    config: ValidationConfig,
) -> ValidationResult:
    pipeline = config.pipeline
    _validate_labeled_data(data, pipeline.features, pipeline.target)
    _validate_ordered_validation(pipeline.lookback, config.strategy)
    estimator = _create_estimator(pipeline)

    if config.strategy == "Cross-validation":
        if pipeline.lookback is not None:
            prediction_indices, predictions = time_series_validation_predictions(
                estimator,
                data.select(pipeline.features),
                data[pipeline.target],
                config.folds,
            )
            prediction_data = data[prediction_indices]
        else:
            predictions = cross_validation_predictions(
                estimator,
                data.select(pipeline.features),
                data[pipeline.target],
                config.folds,
            )
            prediction_data = data
        metrics = calculate_metrics(prediction_data[pipeline.target], predictions)
        estimator.fit(data.select(pipeline.features), data[pipeline.target])
        prediction = PredictionResult(
            data=_prediction_frame(prediction_data, predictions, pipeline.target),
            metrics=metrics,
        )
        return ValidationResult(
            metrics=metrics,
            processed=_processed_data(
                estimator,
                prediction_data.select(pipeline.features),
            ),
            prediction=prediction,
            grid_search=_grid_search_summary(estimator),
        )

    training_data, validation_data = split_validation_data(
        data,
        config.strategy,
        config.percent,
    )
    _validate_grid_folds(pipeline.model, training_data.height)
    estimator.fit(
        training_data.select(pipeline.features),
        training_data[pipeline.target],
    )
    predictions = estimator.predict(validation_data.select(pipeline.features))
    metrics = calculate_metrics(validation_data[pipeline.target], predictions)
    prediction = PredictionResult(
        data=_prediction_frame(validation_data, predictions, pipeline.target),
        metrics=metrics,
    )
    return ValidationResult(
        metrics=metrics,
        processed=_processed_data(
            estimator,
            validation_data.select(pipeline.features),
        ),
        prediction=prediction,
        grid_search=_grid_search_summary(estimator),
    )


def predict(artifact: ModelArtifact, data: pl.DataFrame) -> TestResult:
    prediction = _predict_with_artifact(artifact, data)
    return TestResult(
        processed=_processed_data(
            artifact.pipeline,
            data.select(artifact.features),
        ),
        prediction=prediction,
    )


def _create_estimator(config: PipelineConfig) -> Pipeline | GridSearchCV:
    model_config = config.model
    model = model_config.definition.create_estimator(model_config.parameters)
    if config.lookback is not None:
        model = AutoregressiveRegressor(model, lookback=config.lookback)
    model = wrap_target_processing(model, config.target_processing)

    steps: list[tuple[str, Any]] = [
        ("preprocessing", create_preprocessing_transformer(config.preprocessing))
    ]
    if config.feature_selection is not None:
        steps.append(
            ("feature_selection", create_feature_selector(config.feature_selection))
        )
    steps.append(("model", model))
    pipeline = Pipeline(steps)

    if not model_config.use_grid_search:
        return pipeline
    if not model_config.param_grid or any(
        not values for values in model_config.param_grid.values()
    ):
        raise ValueError("Every grid-search parameter needs at least one value.")
    param_grid = _wrapped_param_grid(
        model_config.param_grid,
        config.lookback,
        config.target_processing,
    )
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=(
            TimeSeriesSplit(n_splits=model_config.cv)
            if config.lookback is not None
            else model_config.cv
        ),
        n_jobs=-1,
        scoring=None if config.lookback is not None else "r2",
    )


def _wrapped_param_grid(
    param_grid: dict[str, list[Any]],
    lookback: int | None,
    target_processing: TargetProcessing,
) -> dict[str, list[Any]]:
    prefixes = []
    if target_processing != "None":
        prefixes.append("regressor")
    if lookback is not None:
        prefixes.append("estimator")
    if not prefixes:
        return param_grid
    prefix = "__".join(prefixes)
    return {
        (
            f"model__{prefix}__{name.removeprefix('model__')}"
            if name.startswith("model__")
            else name
        ): values
        for name, values in param_grid.items()
    }


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
        estimator.best_estimator_ if isinstance(estimator, GridSearchCV) else estimator
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

    model = pipeline.named_steps["model"]
    processed_model = (
        model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    )
    prediction_features = getattr(processed_model, "prediction_features", None)
    if callable(prediction_features):
        model_input = prediction_features(model_input)
        selected_names = _transformed_feature_names(
            processed_model,
            selected_names,
            step_name="model",
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
    features: tuple[str, ...],
    target: str,
) -> None:
    if not features:
        raise ValueError("Select at least one feature.")
    missing = set([*features, target]) - set(data.columns)
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
        raise ValueError("Incompatible feature types for: " + ", ".join(incompatible))
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
    if config.use_grid_search and config.cv < 2:
        raise ValueError("Grid-search folds cannot be lower than 2.")
    if config.use_grid_search and config.cv > rows:
        raise ValueError("Grid-search folds cannot exceed the training rows.")


def _validate_ordered_training(
    lookback: int | None,
    row_selection: RowSelection,
) -> None:
    _validate_lookback(lookback)
    if lookback is not None and row_selection != "Last percent":
        raise ValueError(
            "Target lookback requires training with the last contiguous rows."
        )


def _validate_ordered_validation(
    lookback: int | None,
    strategy: ValidationStrategy,
) -> None:
    _validate_lookback(lookback)
    if lookback is not None and strategy not in ("Last split", "Cross-validation"):
        raise ValueError(
            "Target lookback requires validation with a chronological last split."
        )


def _validate_lookback(lookback: int | None) -> None:
    if lookback is not None and lookback < 1:
        raise ValueError("Lookback must be at least one.")


def _grid_search_summary(
    estimator: Pipeline | GridSearchCV,
) -> GridSearchSummary | None:
    if not isinstance(estimator, GridSearchCV):
        return None
    return GridSearchSummary(
        best_parameters=estimator.best_params_,
        best_score=float(estimator.best_score_),
    )

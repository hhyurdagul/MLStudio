from math import ceil

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split

from .types import (
    CrossValidationMetrics,
    RegressionMetrics,
    RowSelection,
    ValidationStrategy,
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
    return df.tail(row_count)


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
        train_data, validation_data = train_test_split(
            df,
            test_size=validation_percent / 100,
            random_state=seed,
            shuffle=True,
        )
        return pl.DataFrame(train_data), pl.DataFrame(validation_data)
    if strategy == "Last split":
        validation_rows = max(1, ceil(df.height * validation_percent / 100))
        if validation_rows >= df.height:
            raise ValueError("The validation split leaves no training rows.")
        return df.head(df.height - validation_rows), df.tail(validation_rows)
    raise ValueError("Cross-validation does not create a single validation split.")


def calculate_metrics(
    actual: pl.Series | np.ndarray,
    predicted: np.ndarray,
) -> RegressionMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    nonzero = actual_values != 0
    mape = (
        float(
            np.mean(
                np.abs(
                    (actual_values[nonzero] - predicted_values[nonzero])
                    / actual_values[nonzero]
                )
            )
            * 100
        )
        if nonzero.any()
        else None
    )
    return RegressionMetrics(
        r2=float(r2_score(actual_values, predicted_values)),
        mae=float(mean_absolute_error(actual_values, predicted_values)),
        rmse=float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        mape=mape,
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

    scores = cross_validate(
        estimator,
        x,
        y,
        cv=KFold(n_splits=folds, shuffle=True, random_state=42),
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "mape": make_scorer(_mape_without_zero_targets, greater_is_better=False),
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

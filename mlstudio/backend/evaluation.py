from math import ceil

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from .types import (
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


def cross_validation_predictions(
    estimator: BaseEstimator,
    x: pl.DataFrame,
    y: pl.Series,
    folds: int,
) -> np.ndarray:
    if folds > x.height:
        raise ValueError("Cross-validation folds cannot exceed the number of rows.")
    return np.asarray(
        cross_val_predict(
            estimator,
            x,
            y,
            cv=KFold(n_splits=folds, shuffle=True, random_state=42),
            n_jobs=-1,
        )
    )

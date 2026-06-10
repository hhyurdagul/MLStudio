import polars as pl
import numpy as np
from typing import Any
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from .types import TargetProcessing


def get_preprocessing_data(df: pl.DataFrame) -> pl.DataFrame:
    supported_columns = set(
        df.select(
            pl.selectors.string(include_categorical=True)
            | pl.selectors.numeric()
            | pl.selectors.boolean()
        ).columns
    )
    unsupported = [column for column in df.columns if column not in supported_columns]
    if unsupported:
        raise ValueError("Unsupported feature types for: " + ", ".join(unsupported))

    return (
        pl.DataFrame(
            schema={
                "Variable": str,
                "Unique Count": pl.UInt32,
                "Type": str,
                "Preprocessing": str,
            }
        )
        .vstack(
            df.select(pl.selectors.string(include_categorical=True).n_unique())
            .unpivot(variable_name="Variable", value_name="Unique Count")
            .with_columns(
                pl.lit("String").alias("Type"),
                pl.when(pl.col("Unique Count") > 10)
                .then(pl.lit("OrdinalEncoder"))
                .otherwise(pl.lit("OneHotEncoder"))
                .alias("Preprocessing"),
            )
        )
        .vstack(
            df.select(pl.selectors.numeric().n_unique())
            .unpivot(variable_name="Variable", value_name="Unique Count")
            .with_columns(
                pl.lit("Numeric").alias("Type"),
                pl.lit("StandardScaler").alias("Preprocessing"),
            )
        )
        .vstack(
            df.select(pl.selectors.boolean().n_unique())
            .unpivot(variable_name="Variable", value_name="Unique Count")
            .with_columns(
                pl.lit("Boolean").alias("Type"),
                pl.lit("None").alias("Preprocessing"),
            )
        )
    ).select("Variable", "Type", "Unique Count", "Preprocessing")


def create_preprocessing_transformer(
    preprocessing: pl.DataFrame,
) -> ColumnTransformer:
    def variables(step: str) -> list[str]:
        return preprocessing.filter(pl.col("Preprocessing") == step)[
            "Variable"
        ].to_list()

    return ColumnTransformer(
        [
            (
                "OneHotEncoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                variables("OneHotEncoder"),
            ),
            (
                "OrdinalEncoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                variables("OrdinalEncoder"),
            ),
            ("StandardScaler", StandardScaler(), variables("StandardScaler")),
            ("MinMaxScaler", MinMaxScaler(), variables("MinMaxScaler")),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


class TargetProcessedRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        regressor: BaseEstimator,
        transformer: TransformerMixin,
    ) -> None:
        self.regressor = regressor
        self.transformer = transformer

    def fit(self, X: Any, y: Any) -> "TargetProcessedRegressor":
        target = np.asarray(y, dtype=float).reshape(-1, 1)
        self.transformer_ = clone(self.transformer)
        transformed_target = self.transformer_.fit_transform(target).reshape(-1)
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, transformed_target)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["regressor_", "transformer_"])
        transformed = np.asarray(self.regressor_.predict(X)).reshape(-1, 1)
        return self.transformer_.inverse_transform(transformed).reshape(-1)

    def score(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> float:
        check_is_fitted(self, ["regressor_", "transformer_"])
        transformed_target = self.transformer_.transform(
            np.asarray(y, dtype=float).reshape(-1, 1)
        ).reshape(-1)
        return float(
            self.regressor_.score(
                X,
                transformed_target,
                sample_weight=sample_weight,
            )
        )

    def prediction_features(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        prediction_features = getattr(self.regressor_, "prediction_features", None)
        return np.asarray(
            prediction_features(X)
            if callable(prediction_features)
            else _dense_array(X)
        )

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        get_feature_names_out = getattr(
            self.regressor_,
            "get_feature_names_out",
            None,
        )
        if callable(get_feature_names_out):
            return np.asarray(get_feature_names_out(input_features))
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


def wrap_target_processing(
    regressor: BaseEstimator,
    processing: TargetProcessing,
) -> BaseEstimator:
    if processing == "None":
        return regressor
    transformer = (
        StandardScaler()
        if processing == "StandardScaler"
        else MinMaxScaler()
    )
    return TargetProcessedRegressor(regressor, transformer)


def _dense_array(values: Any) -> np.ndarray:
    toarray = getattr(values, "toarray", None)
    if callable(toarray):
        values = toarray()
    return np.asarray(values)

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_array, check_is_fitted


class AutoregressiveRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator: BaseEstimator, lookback: int = 1) -> None:
        self.estimator = estimator
        self.lookback = lookback

    def fit(self, x: Any, y: Any) -> "AutoregressiveRegressor":
        features = _dense_array(x)
        target = np.asarray(y, dtype=float).reshape(-1)
        if self.lookback < 1:
            raise ValueError("Lookback must be at least one.")
        if len(target) <= self.lookback:
            raise ValueError("Training data must contain more rows than the lookback.")
        if features.shape[0] != target.shape[0]:
            raise ValueError("Features and target must contain the same rows.")

        lagged = np.column_stack(
            [target[self.lookback - lag : -lag] for lag in range(1, self.lookback + 1)]
        )
        model_features = np.column_stack((features[self.lookback :], lagged))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(model_features, target[self.lookback :])
        self.target_history_ = target[-self.lookback :].copy()
        self.n_features_in_ = features.shape[1]
        self.n_features_out_ = self.n_features_in_ + self.lookback
        return self

    def predict(self, x: Any) -> np.ndarray:
        predictions, _ = self._recursive_prediction(x)
        return predictions

    def score(
        self,
        X: Any,
        y: Any,
        sample_weight: Any = None,
    ) -> float:
        target = np.asarray(y, dtype=float).reshape(-1)
        predictions = self.predict_with_actual_history(X, target)
        return float(
            r2_score(
                target,
                predictions,
                sample_weight=sample_weight,
            )
        )

    def predict_with_actual_history(self, x: Any, y: Any) -> np.ndarray:
        check_is_fitted(self, ["estimator_", "target_history_"])
        features = _dense_array(x)
        target = np.asarray(y, dtype=float).reshape(-1)
        if features.shape[0] != target.shape[0]:
            raise ValueError("Features and target must contain the same rows.")
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )

        history = [*self.target_history_, *target]
        model_rows = [
            np.concatenate(
                (
                    row,
                    np.asarray(
                        [
                            history[self.lookback + index - lag]
                            for lag in range(1, self.lookback + 1)
                        ]
                    ),
                )
            )
            for index, row in enumerate(features)
        ]
        return np.asarray(self.estimator_.predict(np.asarray(model_rows)))

    def prediction_features(self, x: Any) -> np.ndarray:
        _, features = self._recursive_prediction(x)
        return features

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        if input_features is None:
            names = [f"x{index}" for index in range(self.n_features_in_)]
        else:
            names = [str(name) for name in input_features]
        return np.asarray(
            [
                *names,
                *(f"target_lag_{lag}" for lag in range(1, self.lookback + 1)),
            ],
            dtype=object,
        )

    def _recursive_prediction(self, x: Any) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, ["estimator_", "target_history_"])
        features = _dense_array(x)
        if features.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {features.shape[1]}."
            )

        history = list(self.target_history_)
        predictions: list[float] = []
        model_rows: list[np.ndarray] = []
        for row in features:
            lags = np.asarray([history[-lag] for lag in range(1, self.lookback + 1)])
            model_row = np.concatenate((row, lags))
            prediction = float(self.estimator_.predict(model_row.reshape(1, -1))[0])
            model_rows.append(model_row)
            predictions.append(prediction)
            history.append(prediction)
        augmented = (
            np.asarray(model_rows)
            if model_rows
            else np.empty((0, self.n_features_out_))
        )
        return np.asarray(predictions), augmented


def _dense_array(values: Any) -> np.ndarray:
    checked = check_array(values, accept_sparse=True)
    toarray = getattr(checked, "toarray", None)
    if callable(toarray):
        checked = toarray()
    return np.asarray(checked)

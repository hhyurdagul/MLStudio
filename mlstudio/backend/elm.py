from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data

ELMActivation = Literal["relu", "tanh", "sigmoid", "sine"]


class _ELMBase(BaseEstimator):
    def __init__(
        self,
        n_hidden: int = 100,
        activation: ELMActivation = "tanh",
        alpha: float = 1e-3,
        random_state: int | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.random_state = random_state

    def _fit_hidden_layer(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.n_hidden < 1:
            raise ValueError("n_hidden must be at least 1.")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if self.activation not in {"relu", "tanh", "sigmoid", "sine"}:
            raise ValueError("activation must be one of: relu, tanh, sigmoid, sine.")

        random = check_random_state(self.random_state)
        self.hidden_weights_ = random.normal(size=(X.shape[1], self.n_hidden))
        self.hidden_biases_ = random.normal(size=self.n_hidden)
        return self._hidden_output(X)

    def _hidden_output(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        hidden_input = X @ self.hidden_weights_ + self.hidden_biases_
        if self.activation == "relu":
            return np.maximum(hidden_input, 0.0)
        if self.activation == "tanh":
            return np.tanh(hidden_input)
        if self.activation == "sigmoid":
            clipped = np.clip(hidden_input, -709.0, 709.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        return np.sin(hidden_input)

    def _solve_output_weights(
        self,
        hidden_output: NDArray[np.float64],
        targets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if self.alpha == 0:
            return np.asarray(
                np.linalg.lstsq(hidden_output, targets, rcond=None)[0],
                dtype=np.float64,
            )
        system = hidden_output.T @ hidden_output
        system.flat[:: system.shape[0] + 1] += self.alpha
        return np.asarray(
            np.linalg.solve(system, hidden_output.T @ targets),
            dtype=np.float64,
        )


class ELMRegressor(RegressorMixin, _ELMBase):
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> "ELMRegressor":
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
            y_numeric=True,
            multi_output=True,
        )
        hidden_output = self._fit_hidden_layer(X)
        self.output_weights_ = self._solve_output_weights(hidden_output, y)
        return self

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        check_is_fitted(
            self,
            ("hidden_weights_", "hidden_biases_", "output_weights_"),
        )
        X = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
        )
        return self._hidden_output(X) @ self.output_weights_


class ELMClassifier(ClassifierMixin, _ELMBase):
    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.object_],
    ) -> "ELMClassifier":
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
        )
        check_classification_targets(y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        targets = np.eye(len(self.classes_), dtype=np.float64)[encoded]
        hidden_output = self._fit_hidden_layer(X)
        self.output_weights_ = self._solve_output_weights(
            hidden_output,
            targets,
        )
        return self

    def decision_function(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        check_is_fitted(
            self,
            (
                "classes_",
                "hidden_weights_",
                "hidden_biases_",
                "output_weights_",
            ),
        )
        X = validate_data(
            self,
            X,
            reset=False,
            accept_sparse=False,
            ensure_2d=True,
            dtype=np.float64,
        )
        return self._hidden_output(X) @ self.output_weights_

    def predict_proba(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        scores = self.decision_function(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.object_]:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

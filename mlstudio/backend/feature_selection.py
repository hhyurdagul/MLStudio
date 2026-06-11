"""Feature selection steps and scoring functions."""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y

from .types import PipelineStep

ScoreFunction = Callable[[Any, Any], Any]


def _validate_positive_integer(value: int, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _mrmr(
    X: object,
    y: object,
    *,
    classification: bool,
    method: str,
    n_neighbors: int,
    discrete_features: str | bool | np.ndarray,
    random_state: int | np.random.RandomState | None,
) -> np.ndarray:
    if method not in {"difference", "quotient"}:
        raise ValueError("method must be either 'difference' or 'quotient'.")
    n_neighbors = _validate_positive_integer(n_neighbors, "n_neighbors")
    X, y = check_X_y(X, y, dtype="numeric")

    mutual_info = mutual_info_classif if classification else mutual_info_regression
    relevance = mutual_info(
        X,
        y,
        discrete_features=discrete_features,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    correlations = np.corrcoef(X, rowvar=False)
    if X.shape[1] == 1:
        correlations = np.array([[1.0]])
    correlations = np.nan_to_num(np.abs(correlations), nan=0.0)
    np.fill_diagonal(correlations, 0.0)

    selected: list[int] = []
    remaining = np.ones(X.shape[1], dtype=bool)
    ranking = np.empty(X.shape[1], dtype=int)
    eps = np.finfo(float).eps

    for rank in range(X.shape[1]):
        candidates = np.flatnonzero(remaining)
        if not selected:
            candidate_scores = relevance[candidates]
        else:
            redundancy = correlations[np.ix_(candidates, selected)].mean(axis=1)  # type: ignore
            if method == "difference":
                candidate_scores = relevance[candidates] - redundancy
            else:
                candidate_scores = relevance[candidates] / np.maximum(redundancy, eps)

        best_feature = int(candidates[np.argmax(candidate_scores)])
        selected.append(best_feature)
        remaining[best_feature] = False
        ranking[best_feature] = X.shape[1] - rank

    return ranking.astype(float)


def mrmr_classif(
    X: object,
    y: object,
    *,
    method: str = "difference",
    n_neighbors: int = 3,
    discrete_features: str | bool | np.ndarray = "auto",
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Return an mRMR ranking for classification.

    The scores encode the greedy selection order, making the function suitable
    for ``SelectKBest(score_func=mrmr_classif)``.
    """
    return _mrmr(
        X,
        y,
        classification=True,
        method=method,
        n_neighbors=n_neighbors,
        discrete_features=discrete_features,
        random_state=random_state,
    )


def mrmr_regression(
    X: object,
    y: object,
    *,
    method: str = "difference",
    n_neighbors: int = 3,
    discrete_features: str | bool | np.ndarray = "auto",
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Return an mRMR ranking for regression."""
    return _mrmr(
        X,
        y,
        classification=False,
        method=method,
        n_neighbors=n_neighbors,
        discrete_features=discrete_features,
        random_state=random_state,
    )


def _relief(
    X: object,
    y: object,
    *,
    classification: bool,
    n_neighbors: int,
    n_samples: int | None,
    random_state: int | np.random.RandomState | None,
) -> np.ndarray:
    n_neighbors = _validate_positive_integer(n_neighbors, "n_neighbors")
    if n_samples is not None:
        n_samples = _validate_positive_integer(n_samples, "n_samples")

    X, y = check_X_y(X, y, dtype="numeric")
    if X.shape[0] < 2:
        raise ValueError("Relief requires at least two samples.")
    n_neighbors = min(n_neighbors, X.shape[0] - 1)

    feature_range = np.ptp(X, axis=0)
    X_scaled = (X - X.min(axis=0)) / np.where(feature_range == 0, 1, feature_range)
    sample_count = X.shape[0] if n_samples is None else min(n_samples, X.shape[0])
    rng = check_random_state(random_state)
    sample_indices = rng.choice(X.shape[0], size=sample_count, replace=False)
    distances = pairwise_distances(X_scaled[sample_indices], X_scaled)

    if classification:
        return _relief_classification(
            X_scaled, y, sample_indices, distances, n_neighbors
        )
    return _relief_regression(X_scaled, y, sample_indices, distances, n_neighbors)


def _relief_classification(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    distances: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("Classification Relief requires at least two classes.")

    priors = dict(zip(classes, counts / counts.sum()))
    scores = np.zeros(X.shape[1], dtype=float)

    for row_position, sample_index in enumerate(sample_indices):
        label = y[sample_index]
        differences = np.abs(X - X[sample_index])
        row_distances = distances[row_position]

        hit_candidates = np.flatnonzero(y == label)
        hit_candidates = hit_candidates[hit_candidates != sample_index]
        if hit_candidates.size:
            hits = hit_candidates[
                np.argsort(row_distances[hit_candidates])[:n_neighbors]
            ]
            scores -= differences[hits].mean(axis=0) / sample_indices.size

        other_class_probability = 1.0 - priors[label]
        for other_class in classes[classes != label]:
            miss_candidates = np.flatnonzero(y == other_class)
            misses = miss_candidates[
                np.argsort(row_distances[miss_candidates])[:n_neighbors]
            ]
            scores += (
                priors[other_class]
                / other_class_probability
                * differences[misses].mean(axis=0)
                / sample_indices.size
            )

    return scores


def _relief_regression(
    X: np.ndarray,
    y: np.ndarray,
    sample_indices: np.ndarray,
    distances: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    target_range = np.ptp(y)
    if target_range == 0:
        raise ValueError("Regression Relief requires a non-constant target.")

    target_difference_sum = 0.0
    feature_difference_sum = np.zeros(X.shape[1], dtype=float)
    joint_difference_sum = np.zeros(X.shape[1], dtype=float)
    total_weight = 0.0

    for row_position, sample_index in enumerate(sample_indices):
        row_distances = distances[row_position].copy()
        row_distances[sample_index] = np.inf
        neighbors = np.argsort(row_distances)[:n_neighbors]
        neighbor_distances = row_distances[neighbors]

        scale = max(float(np.mean(neighbor_distances)), np.finfo(float).eps)
        weights = np.exp(-neighbor_distances / scale)
        feature_differences = np.abs(X[neighbors] - X[sample_index])
        target_differences = np.abs(y[neighbors] - y[sample_index]) / target_range

        total_weight += weights.sum()
        target_difference_sum += np.dot(weights, target_differences)
        feature_difference_sum += (weights[:, None] * feature_differences).sum(axis=0)
        joint_difference_sum += (
            weights[:, None] * target_differences[:, None] * feature_differences
        ).sum(axis=0)

    target_probability = target_difference_sum / total_weight
    feature_probability = feature_difference_sum / total_weight
    joint_probability = joint_difference_sum / total_weight
    eps = np.finfo(float).eps

    return joint_probability / max(target_probability, eps) - (
        feature_probability - joint_probability
    ) / max(1.0 - target_probability, eps)


def relief_classif(
    X: object,
    y: object,
    *,
    n_neighbors: int = 10,
    n_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Return ReliefF feature scores for classification."""
    return _relief(
        X,
        y,
        classification=True,
        n_neighbors=n_neighbors,
        n_samples=n_samples,
        random_state=random_state,
    )


def relief_regression(
    X: object,
    y: object,
    *,
    n_neighbors: int = 10,
    n_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """Return RReliefF feature scores for regression."""
    return _relief(
        X,
        y,
        classification=False,
        n_neighbors=n_neighbors,
        n_samples=n_samples,
        random_state=random_state,
    )


SCORE_FUNCTIONS: dict[str, ScoreFunction] = {
    "F Regression": f_regression,
    "MRMR Regression": mrmr_regression,
    "Relief Regression": relief_regression,
}


def create_feature_selection_step(method: str, k: int) -> PipelineStep:
    if method not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown feature selection method: {method}")
    if k < 1:
        raise ValueError("The number of selected features must be at least one.")
    return PipelineStep(
        name="feature_selection",
        transformer=SelectKBest(score_func=SCORE_FUNCTIONS[method], k=k),
    )


__all__ = [
    "SCORE_FUNCTIONS",
    "create_feature_selection_step",
    "mrmr_classif",
    "mrmr_regression",
    "relief_classif",
    "relief_regression",
]

"""Autocorrelation and partial autocorrelation utilities."""

from __future__ import annotations

import altair as alt
import numpy as np
import polars as pl


def _prepare_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size < 2:
        raise ValueError("values must contain at least two observations")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite numbers")
    if np.all(array == array[0]):
        raise ValueError("autocorrelation is undefined for a constant series")
    return array


def _prepare_nlags(nlags: int | None, size: int) -> int:
    if nlags is None:
        return min(40, size - 1)
    if isinstance(nlags, bool) or not isinstance(nlags, (int, np.integer)):
        raise TypeError("nlags must be an integer or None")
    if nlags < 0:
        raise ValueError("nlags must be non-negative")
    if nlags >= size:
        raise ValueError("nlags must be smaller than the number of observations")
    return int(nlags)


def acf(values, nlags: int | None = None) -> np.ndarray:
    """Return biased sample autocorrelations from lag zero through ``nlags``."""
    array = _prepare_values(values)
    nlags = _prepare_nlags(nlags, array.size)
    centered = array - array.mean()
    denominator = np.dot(centered, centered)

    correlations = np.empty(nlags + 1, dtype=float)
    correlations[0] = 1.0
    for lag in range(1, nlags + 1):
        correlations[lag] = np.dot(centered[:-lag], centered[lag:]) / denominator
    return correlations


def pacf(values, nlags: int | None = None) -> np.ndarray:
    """Return partial autocorrelations using the Durbin-Levinson recursion."""
    array = _prepare_values(values)
    nlags = _prepare_nlags(nlags, array.size)
    correlations = acf(array, nlags)

    partial = np.empty(nlags + 1, dtype=float)
    partial[0] = 1.0
    if nlags == 0:
        return partial

    coefficients = np.zeros((nlags + 1, nlags + 1), dtype=float)
    coefficients[1, 1] = correlations[1]
    partial[1] = correlations[1]

    for order in range(2, nlags + 1):
        previous = coefficients[order - 1, 1:order]
        denominator = 1.0 - np.dot(previous, correlations[1:order][::-1])
        if np.isclose(denominator, 0.0):
            raise ValueError(f"PACF recursion became singular at lag {order}")

        reflection = (
            correlations[order] - np.dot(previous, correlations[1:order][::-1])
        ) / denominator
        coefficients[order, order] = reflection
        coefficients[order, 1:order] = previous - reflection * previous[::-1]
        partial[order] = reflection

    return partial


def select_lags(
    values,
    max_lag: int,
    mode: str,
    *,
    indices: tuple[int, ...] = (),
    count: int | None = None,
    threshold: float | None = None,
) -> tuple[int, ...]:
    """Select positive lag indices using explicit or ACF-based criteria."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if array.size < 2:
        raise ValueError("values must contain at least two observations")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must contain only finite numbers")
    max_lag = _prepare_nlags(max_lag, array.size)
    available = tuple(range(1, max_lag + 1))

    if mode == "All":
        return available
    if mode == "Manual":
        selected = tuple(sorted(set(indices)))
        if not selected:
            raise ValueError("Select at least one lag index.")
        if selected[0] < 1 or selected[-1] > max_lag:
            raise ValueError("Lag indices must be between 1 and max lag.")
        return selected

    magnitudes = np.abs(acf(array, max_lag)[1:])
    if mode == "Best N ACF":
        if count is None or not 1 <= count <= max_lag:
            raise ValueError("Best-lag count must be between 1 and max lag.")
        strongest = np.argsort(-magnitudes, kind="stable")[:count] + 1
        return tuple(sorted(int(lag) for lag in strongest))
    if mode == "ACF threshold":
        if threshold is None or not 0 <= threshold <= 1:
            raise ValueError("ACF threshold must be between 0 and 1.")
        selected = tuple(
            lag
            for lag, magnitude in zip(available, magnitudes, strict=True)
            if magnitude >= threshold
        )
        if not selected:
            raise ValueError("No lags meet the ACF threshold.")
        return selected
    raise ValueError(f"Unsupported lag-selection mode: {mode}.")


def _correlation_graph(
    correlations: np.ndarray,
    sample_size: int,
    title: str,
    width: int,
    height: int,
) -> alt.Chart:
    confidence = 1.96 / np.sqrt(sample_size)
    frame = pl.DataFrame(
        {
            "lag": np.arange(correlations.size, dtype=int),
            "correlation": correlations,
            "zero": np.zeros(correlations.size, dtype=float),
        }
    )
    bounds = pl.DataFrame(
        {
            "bound": [confidence, -confidence],
            "kind": ["95% confidence", "95% confidence"],
        }
    )

    stems = (
        alt.Chart(frame)
        .mark_rule(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X("lag:Q", title="Lag", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("correlation:Q", title="Correlation"),
            y2=alt.Y2("zero:Q"),
        )
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=50, color="#1f77b4")
        .encode(
            x="lag:Q",
            y="correlation:Q",
            tooltip=[
                alt.Tooltip("lag:Q", format=".0f"),
                alt.Tooltip("correlation:Q", format=".4f"),
            ],
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]})).mark_rule(color="#666666").encode(y="y:Q")
    )
    confidence_lines = (
        alt.Chart(bounds)
        .mark_rule(color="#d62728", strokeDash=[5, 5])
        .encode(y="bound:Q")
    )

    return (
        (confidence_lines + zero + stems + points)
        .properties(title=title, width=width, height=height)
        .configure_view(stroke=None)
    )


def acf_graph(
    values,
    nlags: int | None = None,
    *,
    title: str = "Autocorrelation Function",
    width: int = 600,
    height: int = 300,
) -> alt.Chart:
    """Return an Altair autocorrelation chart with approximate 95% bounds."""
    array = _prepare_values(values)
    return _correlation_graph(acf(array, nlags), array.size, title, width, height)


def pacf_graph(
    values,
    nlags: int | None = None,
    *,
    title: str = "Partial Autocorrelation Function",
    width: int = 600,
    height: int = 300,
) -> alt.Chart:
    """Return an Altair partial-autocorrelation chart with 95% bounds."""
    array = _prepare_values(values)
    return _correlation_graph(pacf(array, nlags), array.size, title, width, height)

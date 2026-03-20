from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd


def population_stability_index(
    baseline: Sequence[float] | np.ndarray | pd.Series,
    incoming: Sequence[float] | np.ndarray | pd.Series,
    bins: int | Sequence[float] = 10,
    strategy: Literal["quantile", "uniform"] = "quantile",
    epsilon: float = 1e-6,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two numerical distributions.

    PSI measures how much an incoming distribution has shifted relative to a baseline
    distribution. Lower values indicate more similar distributions, while larger values
    indicate stronger drift.

    The function is designed to be reusable in drift pipelines and handles several common
    edge cases:
    - null and non-finite values are removed before calculation
    - zero-probability bins are smoothed with `epsilon` to avoid divide-by-zero
    - quantile-based bins gracefully fall back to uniform bins when the baseline has too
      few unique values

    Parameters
    ----------
    baseline:
        Baseline numerical samples.
    incoming:
        Incoming numerical samples to compare with the baseline.
    bins:
        Either the number of bins to create, or an explicit sequence of bin edges.
        When an integer is provided, the edges are derived from `strategy`.
    strategy:
        Bin generation strategy used when `bins` is an integer.
        Supported values are `"quantile"` and `"uniform"`.
    epsilon:
        Small positive constant used to stabilize zero-probability bins.

    Returns
    -------
    float
        The computed PSI value.

    Raises
    ------
    ValueError
        If either dataset has no valid numerical values, if `bins` is invalid,
        or if `epsilon` is not strictly positive.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    baseline_values = _prepare_numerical_array(baseline, dataset_name="baseline")
    incoming_values = _prepare_numerical_array(incoming, dataset_name="incoming")
    bin_edges = _resolve_bin_edges(baseline_values, incoming_values, bins=bins, strategy=strategy)

    baseline_counts, _ = np.histogram(baseline_values, bins=bin_edges)
    incoming_counts, _ = np.histogram(incoming_values, bins=bin_edges)

    baseline_distribution = _to_stable_distribution(baseline_counts, epsilon=epsilon)
    incoming_distribution = _to_stable_distribution(incoming_counts, epsilon=epsilon)

    psi_values = (incoming_distribution - baseline_distribution) * np.log(
        incoming_distribution / baseline_distribution,
    )
    return float(np.sum(psi_values))


def _prepare_numerical_array(
    values: Sequence[float] | np.ndarray | pd.Series,
    dataset_name: str,
) -> np.ndarray:
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]

    if finite_values.size == 0:
        raise ValueError(f"{dataset_name} must contain at least one finite numerical value.")

    return finite_values


def _resolve_bin_edges(
    baseline: np.ndarray,
    incoming: np.ndarray,
    bins: int | Sequence[float],
    strategy: Literal["quantile", "uniform"],
) -> np.ndarray:
    if isinstance(bins, int):
        if bins < 2:
            raise ValueError("bins must be at least 2 when provided as an integer.")
        if strategy not in {"quantile", "uniform"}:
            raise ValueError("strategy must be either 'quantile' or 'uniform'.")

        if strategy == "quantile":
            edges = _quantile_bin_edges(baseline, bins)
        else:
            edges = _uniform_bin_edges(baseline, incoming, bins)
    else:
        edges = np.asarray(list(bins), dtype=float)
        if edges.ndim != 1 or edges.size < 3:
            raise ValueError("Explicit bin edges must contain at least three values.")
        if not np.all(np.isfinite(edges)):
            raise ValueError("Explicit bin edges must be finite numeric values.")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("Explicit bin edges must be strictly increasing.")

    bounded_edges = edges.copy()
    bounded_edges[0] = -np.inf
    bounded_edges[-1] = np.inf
    return bounded_edges


def _quantile_bin_edges(baseline: np.ndarray, bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(baseline, quantiles)
    unique_edges = np.unique(edges)

    if unique_edges.size < 3:
        return _uniform_bin_edges(baseline, baseline, bins)

    if unique_edges.size != edges.size:
        return _uniform_bin_edges(baseline, baseline, bins)

    return edges.astype(float)


def _uniform_bin_edges(
    baseline: np.ndarray,
    incoming: np.ndarray,
    bins: int,
) -> np.ndarray:
    combined = np.concatenate((baseline, incoming))
    minimum = float(np.min(combined))
    maximum = float(np.max(combined))

    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    return np.linspace(minimum, maximum, bins + 1, dtype=float)


def _to_stable_distribution(counts: np.ndarray, epsilon: float) -> np.ndarray:
    total = counts.sum()
    if total <= 0:
        raise ValueError("Histogram counts must sum to a positive value.")

    distribution = counts.astype(float) / float(total)
    stabilized = np.maximum(distribution, epsilon)
    return stabilized / stabilized.sum()

"""Prediction quality metrics for spatio-temporal data."""
from __future__ import annotations

import numpy as np


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis: int | None = None,
) -> np.ndarray:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : array of matching shape
    axis : int or None
        Axis along which the mean is taken.  None → scalar over all elements.

    Returns
    -------
    rmse : float or array
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))


def compute_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Per-snapshot relative L2 error  ||ê_t||₂ / ||u_t||₂.

    Parameters
    ----------
    y_true, y_pred : array (n_time, n_features)

    Returns
    -------
    rel_err : array (n_time,)
    """
    err = np.linalg.norm(y_true - y_pred, axis=1)
    ref = np.linalg.norm(y_true, axis=1)
    ref = np.where(ref < 1e-12, 1.0, ref)
    return err / ref


def compute_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Global Pearson correlation (all elements flattened)."""
    r = np.corrcoef(y_true.ravel(), y_pred.ravel())
    return float(r[0, 1])


def compute_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Global R2 score (all elements flattened)."""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < 1e-20:
        return 0.0
    return 1.0 - ss_res / ss_tot


def prediction_horizon(
    rel_err: np.ndarray,
    threshold: float = 0.10,
    dt: float = 0.2,
) -> float:
    """Time at which the relative error first exceeds *threshold*.

    Returns the total test duration (in time units) if the threshold is
    never exceeded.
    """
    idx = np.argmax(rel_err > threshold)
    if rel_err[idx] <= threshold:
        return len(rel_err) * dt
    return (idx + 1) * dt

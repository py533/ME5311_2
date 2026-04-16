"""Proper Orthogonal Decomposition (POD) via randomised truncated SVD."""
from __future__ import annotations

import numpy as np
from sklearn.utils.extmath import randomized_svd


class POD:
    """Truncated POD via randomised SVD.

    Parameters
    ----------
    n_modes : int
        Number of POD modes to retain.
    random_state : int
        Seed for the randomised SVD algorithm.
    """

    def __init__(self, n_modes: int = 50, random_state: int = 42) -> None:
        self.n_modes = n_modes
        self.random_state = random_state
        self.modes_: np.ndarray | None = None
        self.singular_values_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "POD":
        """Fit POD on snapshot matrix X of shape (n_time, n_spatial).

        Uses randomised SVD so that only `n_modes` components are computed,
        which is much cheaper than the full SVD when n_modes << min(n_t, n_sp).
        """
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        _, s, Vt = randomized_svd(
            Xc, n_components=self.n_modes, random_state=self.random_state
        )
        self.singular_values_ = s
        self.modes_ = Vt.T
        self._total_variance = float(np.linalg.norm(Xc, "fro") ** 2)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project snapshots onto the POD basis.

        Parameters
        ----------
        X : array (n_time, n_spatial)

        Returns
        -------
        Z : array (n_time, r)
        """
        return (X - self.mean_) @ self.modes_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct the spatial field from POD coefficients.

        Parameters
        ----------
        Z : array (n_time, r)

        Returns
        -------
        X_rec : array (n_time, n_spatial)
        """
        return Z @ self.modes_.T + self.mean_

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fractional energy explained by each mode (relative to total variance)."""
        s2 = self.singular_values_ ** 2
        return s2 / self._total_variance

    @property
    def cumulative_energy_(self) -> np.ndarray:
        """Cumulative energy fraction (useful for choosing truncation rank)."""
        return np.cumsum(self.explained_variance_ratio_)

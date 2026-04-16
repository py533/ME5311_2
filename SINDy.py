"""Sparse Identification of Nonlinear Dynamics (SINDy)."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


class SINDy:
    """SINDy predictor in a truncated POD-coefficient space.

    Parameters
    ----------
    dt : float
        Temporal spacing of training snapshots.
    poly_degree : int
        Polynomial library degree (1 or 2).
    threshold : float
        STLSQ sparsity threshold λ.
    ridge_alpha : float
        Ridge regularisation in each STLSQ step.
    n_iter : int
        Number of STLSQ iterations.
    n_sub : int
        RK4 sub-steps per dt interval (continuous mode only).
    discrete : bool
        If True, use discrete-time formulation (no derivatives, no RK4).
    """

    def __init__(
        self,
        dt: float = 0.2,
        poly_degree: int = 2,
        threshold: float = 0.005,
        ridge_alpha: float = 1e-4,
        n_iter: int = 20,
        n_sub: int = 10,
        discrete: bool = False,
        clip_sigma: float = 2.0,
    ) -> None:
        self.dt          = dt
        self.poly_degree = poly_degree
        self.threshold   = threshold
        self.ridge_alpha = ridge_alpha
        self.n_iter      = n_iter
        self.n_sub       = n_sub
        self.discrete    = discrete
        self.clip_sigma  = clip_sigma
        self.Xi_: np.ndarray | None = None
        self._triu: tuple | None    = None
        self._z_lo: np.ndarray | None = None
        self._z_hi: np.ndarray | None = None
        self._r: int = 0

    def _build_library(self, Z: np.ndarray) -> np.ndarray:
        """Polynomial library for Z (n_samples, r) → (n_samples, n_feat)."""
        n = Z.shape[0]
        bias = np.ones((n, 1))
        if self.poly_degree == 1:
            return np.concatenate([bias, Z], axis=1)
        i, j = self._triu
        ZZ   = Z[:, i] * Z[:, j]
        return np.concatenate([bias, Z, ZZ], axis=1)

    def _eval_library_1d(self, z: np.ndarray) -> np.ndarray:
        """Library for a single state vector z (r,)."""
        if self.poly_degree == 1:
            return np.concatenate([[1.0], z])
        i, j = self._triu
        return np.concatenate([[1.0], z, z[i] * z[j]])

    @staticmethod
    def _derivatives(Z: np.ndarray, dt: float) -> np.ndarray:
        """Smoothed time derivatives via Savitzky-Golay filter."""
        n = Z.shape[0]
        wl = min(11, max(5, n // 1000))
        if wl % 2 == 0:
            wl += 1
        wl = max(wl, 5)
        return savgol_filter(Z, window_length=wl, polyorder=3,
                             deriv=1, delta=dt, axis=0)

    def _stlsq(self, Theta: np.ndarray, dZ: np.ndarray) -> np.ndarray:
        """Sequential Thresholded Least Squares."""
        r      = dZ.shape[1]
        n_feat = Theta.shape[1]
        ThTh = Theta.T @ Theta + self.ridge_alpha * np.eye(n_feat)
        ThDZ = Theta.T @ dZ
        Xi   = np.linalg.solve(ThTh, ThDZ)

        for _ in range(self.n_iter):
            active = np.abs(Xi) >= self.threshold
            for j in range(r):
                idx = np.where(active[:, j])[0]
                if len(idx) == 0:
                    Xi[:, j] = 0.0
                    continue
                Th_j    = Theta[:, idx]
                A       = Th_j.T @ Th_j + self.ridge_alpha * np.eye(len(idx))
                b       = Th_j.T @ dZ[:, j]
                xi_j    = np.linalg.solve(A, b)
                Xi[:, j]    = 0.0
                Xi[idx, j]  = xi_j

        return Xi

    def fit(self, Z: np.ndarray) -> "SINDy":
        """Fit SINDy on POD-coefficient time series Z (n_time, r)."""
        self._r    = Z.shape[1]
        self._triu = np.triu_indices(self._r)
        self._z_lo = Z.mean(0) - self.clip_sigma * Z.std(0)
        self._z_hi = Z.mean(0) + self.clip_sigma * Z.std(0)

        if self.discrete:
            targets  = Z[1:]
            features = self._build_library(Z[:-1])
        else:
            targets  = self._derivatives(Z, self.dt)
            features = self._build_library(Z)

        n_feat = features.shape[1]
        ThTh = features.T @ features + self.ridge_alpha * np.eye(n_feat)
        Xi_full = np.linalg.solve(ThTh, features.T @ targets)
        res_full = np.linalg.norm(features @ Xi_full - targets) / np.linalg.norm(targets)
        mode_tag = "discrete" if self.discrete else "continuous"
        print(f"  SINDy ({mode_tag}): full fit residual = {res_full:.4f}")

        self.Xi_ = self._stlsq(features, targets)

        nnz = int(np.count_nonzero(self.Xi_))
        residual = np.linalg.norm(features @ self.Xi_ - targets) / np.linalg.norm(targets)
        print(f"  SINDy: {nnz}/{self.Xi_.size} non-zero  "
              f"(library {n_feat} features × {self._r} modes, "
              f"sparsity {100*(1-nnz/self.Xi_.size):.1f}%, "
              f"sparse fit residual {residual:.4f})")
        return self

    def _rhs(self, z: np.ndarray) -> np.ndarray:
        """RHS evaluation:  Θ(z) · Ξ  (works for both modes)."""
        z_c = np.clip(z, self._z_lo, self._z_hi)
        return self._eval_library_1d(z_c) @ self.Xi_

    def predict(self, n_steps: int, z0: np.ndarray) -> np.ndarray:
        """Predict n_steps ahead."""
        z     = z0.copy()
        preds = np.empty((n_steps, self._r))

        if self.discrete:
            for i in range(n_steps):
                z = self._rhs(z)
                z = np.clip(z, self._z_lo, self._z_hi)
                preds[i] = z
        else:
            h_sub = self.dt / self.n_sub
            h6    = h_sub / 6.0
            for i in range(n_steps):
                for _ in range(self.n_sub):
                    k1 = self._rhs(z)
                    k2 = self._rhs(z + 0.5 * h_sub * k1)
                    k3 = self._rhs(z + 0.5 * h_sub * k2)
                    k4 = self._rhs(z + h_sub * k3)
                    z  = z + h6 * (k1 + 2.0*k2 + 2.0*k3 + k4)
                z = np.clip(z, self._z_lo, self._z_hi)
                preds[i] = z

        return preds

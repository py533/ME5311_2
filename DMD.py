"""Dynamic Mode Decomposition (DMD) with optional time-delay embedding."""
from __future__ import annotations

import numpy as np


class DMD:
    """Exact DMD with optional time-delay (Hankel) embedding.

    Parameters
    ----------
    n_modes : int or None
        SVD truncation rank for the (augmented) snapshot matrix.
        None = full rank (no truncation).
    stabilise : bool
        If True, clamp *unstable* eigenvalues (|λ| > 1) to the unit
        circle.  Stable eigenvalues are left untouched so the prediction
        decays naturally to the mean — which dominates long-horizon
        correlation.
    delay : int
        Number of time-delay embeddings (0 = standard DMD).
        With delay d, the augmented state is [z_t; z_{t-1}; ...; z_{t-d}],
        giving dimension (d+1)*r.
    """

    def __init__(self, n_modes: int | None = None, stabilise: bool = True,
                 delay: int = 0) -> None:
        self.n_modes = n_modes
        self.stabilise = stabilise
        self.delay = delay
        self.eigenvalues_: np.ndarray | None = None
        self._U: np.ndarray | None = None       # SVD basis for projection
        self._A_tilde: np.ndarray | None = None  # reduced operator
        self._r_state: int = 0                   # original state dimension

    def fit(self, Z: np.ndarray) -> "DMD":
        """Fit DMD on time series Z of shape (n_time, n_features)."""
        self._r_state = Z.shape[1]
        n, r = Z.shape
        d = self.delay

        if d > 0:
            cols = n - d
            H = np.empty(((d + 1) * r, cols))
            for k in range(d + 1):
                start = d - k
                H[k * r:(k + 1) * r, :] = Z[start:start + cols].T
            X1 = H[:, :-1]
            X2 = H[:, 1:]
        else:
            X1 = Z[:-1].T
            X2 = Z[1:].T

        U, s, Vt = np.linalg.svd(X1, full_matrices=False)
        rank = self.n_modes if self.n_modes is not None else len(s)
        rank = min(rank, len(s))
        U  = U[:, :rank]
        s  = s[:rank]
        Vt = Vt[:rank, :]

        S_inv = np.diag(1.0 / s)

        A_tilde = U.T @ X2 @ Vt.T @ S_inv

        evals, W = np.linalg.eig(A_tilde)

        mag = np.abs(evals)
        print(f"  DMD eigenvalues: "
              f"|λ| min={mag.min():.4f}  max={mag.max():.4f}  mean={mag.mean():.4f}")
        if self.stabilise:
            unstable = mag > 1.0
            if unstable.any():
                evals[unstable] = evals[unstable] / mag[unstable]
                print(f"  DMD: clamped {unstable.sum()} unstable eigenvalues to unit circle")

        self.eigenvalues_ = evals

        A_tilde_s = W @ np.diag(evals) @ np.linalg.inv(W)
        self._A_tilde = np.real(A_tilde_s)
        self._U = U
        return self

    def predict(self, n_steps: int, z0: np.ndarray) -> np.ndarray:
        """Predict *n_steps* future states.

        Parameters
        ----------
        z0 : array
            * delay == 0 → shape ``(n_features,)`` — single initial state.
            * delay >  0 → shape ``(delay+1, n_features)`` — window of
              consecutive states in **chronological** order
              ``[z_{t-d}, z_{t-d+1}, …, z_t]``.

        Returns
        -------
        preds : array (n_steps, n_features)
        """
        r = self._r_state
        d = self.delay

        if d > 0:
            h = z0[::-1].ravel()
        else:
            h = z0.copy()

        h_t = self._U.T @ h

        preds = np.empty((n_steps, r))
        for i in range(n_steps):
            h_t = self._A_tilde @ h_t
            h_full = self._U @ h_t
            preds[i] = h_full[:r]

        return preds

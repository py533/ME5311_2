"""LSTM predictor for POD coefficient dynamics using PyTorch.

This implementation is designed to run natively on Windows and can use
NVIDIA GPU acceleration when a CUDA-enabled PyTorch build is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover - import availability is runtime-dependent.
    _HAS_TORCH = False


class _LSTMRegressor(nn.Module):
    """Compact stacked LSTM regressor: sequence -> next state."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


@dataclass
class _EarlyStopper:
    """Simple early stopping monitor for validation loss."""

    patience: int = 5
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class LSTMPredictor:
    """Sequence-to-one LSTM for one-step POD-coefficient prediction.

    Parameters
    ----------
    seq_len : int
        Number of past time steps used as input context.
    n_features : int
        POD coefficients to model.
    hidden_size : int
        LSTM hidden width.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between recurrent blocks.
    learning_rate : float
        Adam learning rate.
    epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early stopping patience.
    verbose : int
        Training verbosity. 0 = silent, 1 = print epoch summary.
    random_state : int
        Random seed for reproducibility.
    require_cuda : bool
        If True, raise an error when CUDA is unavailable.
    weight_decay : float
        L2 regularization for Adam.
    grad_clip : float
        Max gradient norm for clipping. Non-positive value disables clipping.
    """

    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 128,
        patience: int = 5,
        verbose: int = 0,
        random_state: int = 42,
        require_cuda: bool = True,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for LSTM model. "
                "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu128"
            )

        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.verbose = int(verbose)
        self.random_state = int(random_state)
        self.require_cuda = bool(require_cuda)
        self.weight_decay = float(weight_decay)
        self.grad_clip = float(grad_clip)

        self.model_: _LSTMRegressor | None = None
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.device_: torch.device | None = None

    # ------------------------------------------------------------------
    @staticmethod
    def _build_sequences(Z: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build supervised windows: X[t-seq_len:t] -> y[t]."""
        if Z.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={Z.shape}")
        if Z.shape[0] <= seq_len:
            raise ValueError(
                f"Need more than seq_len snapshots, got n={Z.shape[0]}, seq_len={seq_len}"
            )

        xs = np.empty((Z.shape[0] - seq_len, seq_len, Z.shape[1]), dtype=np.float32)
        ys = np.empty((Z.shape[0] - seq_len, Z.shape[1]), dtype=np.float32)
        for i, t in enumerate(range(seq_len, Z.shape[0])):
            xs[i] = Z[t - seq_len:t]
            ys[i] = Z[t]
        return xs, ys

    # ------------------------------------------------------------------
    def _set_seed(self) -> None:
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    # ------------------------------------------------------------------
    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    # ------------------------------------------------------------------
    def fit(self, Z_train: np.ndarray, Z_val: np.ndarray | None = None) -> "LSTMPredictor":
        """Fit LSTM on POD coefficient trajectories."""
        self._set_seed()
        if self.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for LSTM but no CUDA device is available.")
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        Z_train = np.asarray(Z_train, dtype=np.float32)
        self.mu_ = Z_train.mean(axis=0)
        self.sigma_ = Z_train.std(axis=0)
        self.sigma_ = np.where(self.sigma_ < 1e-8, 1.0, self.sigma_)

        Z_train_n = (Z_train - self.mu_) / self.sigma_
        X_train, y_train = self._build_sequences(Z_train_n, self.seq_len)

        tr_ds = TensorDataset(self._to_tensor(X_train), self._to_tensor(y_train))
        tr_loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if Z_val is not None and len(Z_val) > self.seq_len:
            Z_val = np.asarray(Z_val, dtype=np.float32)
            Z_val_n = (Z_val - self.mu_) / self.sigma_
            X_val, y_val = self._build_sequences(Z_val_n, self.seq_len)
            va_ds = TensorDataset(self._to_tensor(X_val), self._to_tensor(y_val))
            val_loader = DataLoader(va_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model_ = _LSTMRegressor(
            n_features=self.n_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device_)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=max(2, self.patience // 2),
            min_lr=1e-5,
        )
        stopper = _EarlyStopper(patience=self.patience)

        best_state = None
        best_metric = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            train_loss = 0.0
            n_train = 0
            for xb, yb in tr_loader:
                xb = xb.to(self.device_, non_blocking=True)
                yb = yb.to(self.device_, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = criterion(pred, yb)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=self.grad_clip)
                optimizer.step()

                train_loss += float(loss.item()) * xb.shape[0]
                n_train += xb.shape[0]

            train_loss /= max(n_train, 1)

            if val_loader is None:
                monitor = train_loss
            else:
                self.model_.eval()
                val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device_, non_blocking=True)
                        yb = yb.to(self.device_, non_blocking=True)
                        pred = self.model_(xb)
                        loss = criterion(pred, yb)
                        val_loss += float(loss.item()) * xb.shape[0]
                        n_val += xb.shape[0]
                monitor = val_loss / max(n_val, 1)

            scheduler.step(monitor)

            if monitor < best_metric:
                best_metric = monitor
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}

            if self.verbose:
                if val_loader is None:
                    print(f"  Epoch {epoch:>3}: train_loss={train_loss:.6f}")
                else:
                    print(f"  Epoch {epoch:>3}: train_loss={train_loss:.6f}, val_loss={monitor:.6f}")

            if stopper.step(monitor):
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    # ------------------------------------------------------------------
    def predict_one_step(self, history_window: np.ndarray) -> np.ndarray:
        """Predict one step ahead from a window of shape (seq_len, n_features)."""
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model is not fitted.")

        hist = np.asarray(history_window, dtype=np.float32)
        if hist.shape != (self.seq_len, self.n_features):
            raise ValueError(
                f"Expected history shape {(self.seq_len, self.n_features)}, got {hist.shape}"
            )

        hist_n = (hist - self.mu_) / self.sigma_
        x = self._to_tensor(hist_n[None, :, :]).to(self.device_)

        with torch.no_grad():
            y_n = self.model_(x).detach().cpu().numpy()[0]

        return y_n * self.sigma_ + self.mu_

    # ------------------------------------------------------------------
    def predict_windows(self, windows: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Predict one-step outputs for many windows of shape (n_windows, seq_len, n_features)."""
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model is not fitted.")

        ws = np.asarray(windows, dtype=np.float32)
        if ws.ndim != 3 or ws.shape[1:] != (self.seq_len, self.n_features):
            raise ValueError(
                "Expected windows shape "
                f"(n_windows, {self.seq_len}, {self.n_features}), got {ws.shape}"
            )

        ws_n = (ws - self.mu_) / self.sigma_
        ds = TensorDataset(self._to_tensor(ws_n))
        loader = DataLoader(ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

        preds_n = []
        self.model_.eval()
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device_, non_blocking=True)
                yb = self.model_(xb).detach().cpu().numpy()
                preds_n.append(yb)

        y_n = np.vstack(preds_n)
        return y_n * self.sigma_ + self.mu_

    # ------------------------------------------------------------------
    def predict(self, n_steps: int, z0: np.ndarray) -> np.ndarray:
        """Autoregressively predict n_steps from initial window z0."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted.")

        window = np.asarray(z0, dtype=np.float32).copy()
        if window.shape != (self.seq_len, self.n_features):
            raise ValueError(
                f"Expected z0 shape {(self.seq_len, self.n_features)}, got {window.shape}"
            )

        preds = np.empty((int(n_steps), self.n_features), dtype=np.float32)
        for i in range(int(n_steps)):
            nxt = self.predict_one_step(window)
            preds[i] = nxt
            window = np.vstack([window[1:], nxt])
        return preds

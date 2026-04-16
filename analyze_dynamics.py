"""Analyze the nature of the dynamics to understand prediction limits."""
import numpy as np
from pathlib import Path
from load_data import load_vector_field
from POD import POD
from metrics import compute_relative_error, compute_correlation

DATA_PATH = Path("data/vector_64.npy")
NT_TRAIN = 10_500; NT_VAL = 12_000; NT_TOTAL = 15_000; DT = 0.2

raw = load_vector_field(DATA_PATH)
nt, ny, nx, nc = raw.shape
n_sp = ny * nx * nc
X_flat = raw.reshape(nt, n_sp)
X_train = np.array(X_flat[:NT_TRAIN], dtype=np.float64)
X_val = np.array(X_flat[NT_TRAIN:NT_VAL], dtype=np.float64)
X_test = np.array(X_flat[NT_VAL:], dtype=np.float64)

pod = POD(n_modes=50)
pod.fit(X_train)
Z_train = pod.transform(X_train)
Z_test = pod.transform(X_test)

print("=== POD coefficient autocorrelation (mode 0, 1, 2) ===")
for mode in [0, 1, 2]:
    z = Z_train[:, mode]
    z_centered = z - z.mean()
    var = np.var(z_centered)
    for lag in [1, 10, 50, 100, 500, 1000]:
        if lag < len(z):
            acf = np.mean(z_centered[:-lag] * z_centered[lag:]) / var
            print(f"  mode {mode}, lag {lag:>5} ({lag*DT:>6.1f} t.u.): autocorr = {acf:.4f}")
    print()

print("=== One-step-ahead vs multi-step prediction correlation ===")
from DMD import DMD
from SINDy import SINDy

dmd = DMD(n_modes=None, stabilise=True, delay=0)
dmd.fit(Z_train[:, :28])

Z_test_28 = Z_test[:, :28]
preds_1step = np.empty_like(Z_test_28[1:])
for i in range(len(Z_test_28) - 1):
    h = dmd._U.T @ Z_test_28[i]
    h_next = dmd._A_tilde @ h
    preds_1step[i] = (dmd._U @ h_next)[:28]

def pad(Z, r):
    out = np.zeros((Z.shape[0], 50))
    out[:, :r] = Z
    return out

X_1step = pod.inverse_transform(pad(preds_1step, 28))
corr_1step = compute_correlation(X_test[1:], X_1step)
re_1step = compute_relative_error(X_test[1:], X_1step).mean()
print(f"  DMD  one-step-ahead: corr = {corr_1step:.4f}, rel.err = {re_1step:.4f}")

print("\n=== Correlation vs prediction horizon (DMD, delay=4) ===")
dmd4 = DMD(n_modes=None, stabilise=True, delay=4)
dmd4.fit(Z_train[:, :28])

Z_val_28 = np.array(X_flat[NT_TRAIN:NT_VAL], dtype=np.float64)
Z_val_28 = pod.transform(Z_val_28)[:, :28]

for n_steps in [10, 50, 100, 200, 500, 1000, 3000]:
    z0 = Z_val_28[-(4+1):]
    Z_pred = dmd4.predict(n_steps, z0=z0)
    X_pred = pod.inverse_transform(pad(Z_pred, 28))
    corr = compute_correlation(X_test[:n_steps], X_pred)
    re = compute_relative_error(X_test[:n_steps], X_pred).mean()
    print(f"  {n_steps:>5} steps ({n_steps*DT:>6.1f} t.u.): corr = {corr:.4f}, rel.err = {re:.4f}")

print("\n=== Correlation vs prediction horizon (SINDy, discrete) ===")
sindy = SINDy(dt=DT, poly_degree=2, threshold=0.005, ridge_alpha=1e-4,
              n_iter=20, n_sub=10, discrete=True, clip_sigma=1.8)
sindy.fit(Z_train[:, :30])

Z_val_30 = pod.transform(np.array(X_flat[NT_TRAIN:NT_VAL], dtype=np.float64))[:, :30]

for n_steps in [10, 50, 100, 200, 500, 1000, 3000]:
    Z_pred = sindy.predict(n_steps, z0=Z_val_30[-1])
    X_pred = pod.inverse_transform(pad(Z_pred, 30))
    corr = compute_correlation(X_test[:n_steps], X_pred)
    re = compute_relative_error(X_test[:n_steps], X_pred).mean()
    print(f"  {n_steps:>5} steps ({n_steps*DT:>6.1f} t.u.): corr = {corr:.4f}, rel.err = {re:.4f}")

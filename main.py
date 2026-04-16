"""ME5311 Project 2 pipeline."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from load_data import load_vector_field
from POD import POD
from DMD import DMD
from SINDy import SINDy
from LSTM import LSTMPredictor
from metrics import (
    compute_relative_error,
    compute_rmse,
    compute_correlation,
    compute_r2,
    prediction_horizon,
)
import plot

DATA_PATH   = Path("data/vector_64.npy")
OUTPUT_DIR  = Path("output")
NT_TRAIN    = 10_500
NT_VAL      = 12_000
NT_TOTAL    = 15_000
N_POD       = 50
R_DMD       = 20
DMD_DELAY   = 4
R_SINDY     = 30
R_LSTM      = 10
LSTM_SEQ    = 16
LSTM_HIDDEN = 96
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.10
LSTM_LR     = 1e-3
LSTM_BATCH  = 256
LSTM_PATIENCE = 6
LSTM_EPOCHS = 40
DT          = 0.2
SNAP_IDX    = 3
SAVE_FIG    = str(OUTPUT_DIR / "report_figure.png")
SAVE_TXT    = str(OUTPUT_DIR / "results_summary.txt")
RECEDING_ONE_STEP = False


def _write_results_txt(path: str, records: list[dict], pod_energy: float,
                       elapsed: float, n_sp: int) -> None:
    """Write a human-readable results summary to *path*."""
    sep  = "=" * 100
    sep2 = "-" * 100
    lines = [
        sep,
        "  ME5311 Project 2 — Results Summary",
        sep,
        "",
        "Dataset",
        sep2,
        f"  Spatial DOF          : {n_sp:,}  (64 × 64 × 2 components)",
        f"  Total snapshots      : {NT_TOTAL:,}  (Δt = {DT}, span = {NT_TOTAL*DT:.0f} t.u.)",
        f"  Train / Val / Test   : {NT_TRAIN:,} / {NT_VAL-NT_TRAIN:,} / {NT_TOTAL-NT_VAL:,}"
        f"  ({100*NT_TRAIN//NT_TOTAL}% / "
        f"{100*(NT_VAL-NT_TRAIN)//NT_TOTAL}% / "
        f"{100*(NT_TOTAL-NT_VAL)//NT_TOTAL}%)",
        "",
        "Dimensionality Reduction (POD)",
        sep2,
        f"  Modes retained       : {N_POD}",
        f"  Cumulative energy    : {pod_energy:.4f} %",
        "",
        "Prediction Results",
        sep2,
        f"  {'Method':<8}  {'Val RelErr':>10}  {'Test RelErr':>11}  "
        f"{'Horizon(s)':>11}  {'RMSE':>8}  {'Corr':>7}  {'R2':>7}  "
        f"{'Train(s)':>9}  {'Infer(ms/step)':>14}  {'Complexity (train/infer)'}",
        f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*7}  {'-'*7}  "
        f"{'-'*9}  {'-'*14}  {'-'*24}",
    ]
    for r in records:
        lines.append(
            f"  {r['name']:<8}  {r['val_re']:>10.4f}  {r['test_re']:>11.4f}  "
            f"{r['horizon']:>11.2f}  {r['rmse']:>8.4f}  {r['corr']:>7.4f}  {r['r2']:>7.4f}  "
            f"{r['train_time']:>9.2f}  {r['infer_time']*1e3:>14.4f}  {r['complexity']}"
        )
    lines += [
        "",
        "Notes",
        sep2,
        "  Horizon    : time (s) at which relative error first exceeds 50 %.",
        "  RMSE / Corr: computed over the full test set (all snapshots).",
        "  Train(s)   : wall-clock seconds for model fitting.",
        "  Infer(ms)  : wall-clock milliseconds per predicted step (test set / nt_test).",
        "  Complexity : theoretical big-O for training / per-step inference.",
        "               r = POD modes, n = training snapshots.",
        "",
        f"  Total wall-clock time : {elapsed/60:.1f} min",
        sep,
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results summary saved →  {path}")


def main() -> None:
    t_wall = time.perf_counter()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 64)
    print("Loading dataset …")
    raw = load_vector_field(DATA_PATH)
    nt, ny, nx, nc = raw.shape
    n_sp = ny * nx * nc

    X_flat  = raw.reshape(nt, n_sp)
    X_train = np.array(X_flat[:NT_TRAIN],        dtype=np.float64)
    X_val   = np.array(X_flat[NT_TRAIN:NT_VAL],  dtype=np.float64)
    X_test  = np.array(X_flat[NT_VAL:],          dtype=np.float64)
    nt_val  = len(X_val)
    nt_test = len(X_test)
    print(f"  train = {NT_TRAIN:,}  |  val = {nt_val:,}  |  test = {nt_test:,}  |  DOF = {n_sp:,}")

    print("\n[1/4]  Fitting POD …")
    pod = POD(n_modes=N_POD)
    pod.fit(X_train)
    ce = pod.cumulative_energy_
    print(f"       {N_POD} modes  →  {ce[-1]*100:.2f} % of total energy")
    print(f"       DMD uses first {R_DMD} modes  |  SINDy uses first {R_SINDY} modes"
          f"  |  LSTM uses first {R_LSTM} modes")

    Z_train = pod.transform(X_train)
    Z_val   = pod.transform(X_val)
    Z_test  = pod.transform(X_test)

    re_rec_val  = compute_relative_error(X_val,  pod.inverse_transform(Z_val))
    re_rec_test = compute_relative_error(X_test, pod.inverse_transform(Z_test))
    print(f"       POD-{N_POD} reconstruction error — val: {re_rec_val.mean():.4f}"
          f"  |  test: {re_rec_test.mean():.4f}")

    print(f"\n[2/4]  Fitting Hankel-DMD ({R_DMD} of {N_POD} POD modes, delay={DMD_DELAY}) …")
    Z_train_dmd = Z_train[:, :R_DMD]
    Z_val_dmd   = Z_val[:, :R_DMD]
    dmd = DMD(n_modes=None, stabilise=True, delay=DMD_DELAY)
    _t0 = time.perf_counter(); dmd.fit(Z_train_dmd); dmd_train_time = time.perf_counter() - _t0

    def _pad_dmd(Z_d):
        Z_full = np.zeros((Z_d.shape[0], N_POD))
        Z_full[:, :R_DMD] = Z_d
        return Z_full

    if RECEDING_ONE_STEP:
        Z_dmd_val = np.empty((nt_val, R_DMD))
        ref_val = np.vstack([Z_train_dmd[-(DMD_DELAY + 1):], Z_val_dmd])
        for t in range(nt_val):
            Z_dmd_val[t] = dmd.predict(1, z0=ref_val[t:t + DMD_DELAY + 1])[0]
    else:
        Z_dmd_val = dmd.predict(nt_val, z0=Z_train_dmd[-(DMD_DELAY + 1):])
    re_dmd_val = compute_relative_error(X_val, pod.inverse_transform(_pad_dmd(Z_dmd_val)))
    print(f"       val  mean relative error = {re_dmd_val.mean():.4f}")

    _t0 = time.perf_counter()
    if RECEDING_ONE_STEP:
        Z_dmd = np.empty((nt_test, R_DMD))
        ref_test = np.vstack([Z_val_dmd[-(DMD_DELAY + 1):], Z_test[:, :R_DMD]])
        for t in range(nt_test):
            Z_dmd[t] = dmd.predict(1, z0=ref_test[t:t + DMD_DELAY + 1])[0]
    else:
        Z_dmd = dmd.predict(nt_test, z0=Z_val_dmd[-(DMD_DELAY + 1):])
    dmd_infer_time = (time.perf_counter() - _t0) / nt_test
    X_dmd  = pod.inverse_transform(_pad_dmd(Z_dmd))
    re_dmd = compute_relative_error(X_test, X_dmd)
    print(f"       test mean relative error = {re_dmd.mean():.4f}")
    print(f"       train {dmd_train_time:.2f}s  |  infer {dmd_infer_time*1e3:.3f} ms/step")

    print(f"\n[3/4]  Fitting SINDy ({R_SINDY} POD modes, degree-2 quadratic library) …")
    sindy = SINDy(dt=DT, poly_degree=2, threshold=0.005, ridge_alpha=1e-4,
                  n_iter=20, n_sub=10, discrete=True, clip_sigma=1.8)
    _t0 = time.perf_counter()
    sindy.fit(Z_train[:, :R_SINDY])
    sindy_train_time = time.perf_counter() - _t0

    def _pad_sindy(Z_s):
        if Z_s.shape[1] == N_POD:
            return Z_s
        Z_full = np.zeros((Z_s.shape[0], N_POD))
        Z_full[:, :R_SINDY] = Z_s
        return Z_full

    if RECEDING_ONE_STEP:
        Z_sindy_val = np.empty((nt_val, R_SINDY))
        ref_val_s = np.vstack([Z_train[-1:, :R_SINDY], Z_val[:, :R_SINDY]])
        for t in range(nt_val):
            Z_sindy_val[t] = sindy.predict(1, z0=ref_val_s[t])[0]
    else:
        Z_sindy_val = sindy.predict(nt_val, z0=Z_train[-1, :R_SINDY])
    re_sindy_val = compute_relative_error(X_val, pod.inverse_transform(_pad_sindy(Z_sindy_val)))
    print(f"       val  mean relative error = {re_sindy_val.mean():.4f}")

    _t0 = time.perf_counter()
    if RECEDING_ONE_STEP:
        Z_sindy = np.empty((nt_test, R_SINDY))
        ref_test_s = np.vstack([Z_val[-1:, :R_SINDY], Z_test[:, :R_SINDY]])
        for t in range(nt_test):
            Z_sindy[t] = sindy.predict(1, z0=ref_test_s[t])[0]
    else:
        Z_sindy = sindy.predict(nt_test, z0=Z_val[-1, :R_SINDY])
    sindy_infer_time = (time.perf_counter() - _t0) / nt_test
    X_sindy  = pod.inverse_transform(_pad_sindy(Z_sindy))
    re_sindy = compute_relative_error(X_test, X_sindy)
    print(f"       test mean relative error = {re_sindy.mean():.4f}")
    print(f"       train {sindy_train_time:.2f}s  |  infer {sindy_infer_time*1e3:.3f} ms/step")

    re_lstm_val = None
    re_lstm = None
    X_lstm = None
    lstm_train_time = 0.0
    lstm_infer_time = 0.0

    print(f"\n[4/4]  Fitting LSTM ({R_LSTM} POD modes, seq_len={LSTM_SEQ}) …")
    try:
        lstm = LSTMPredictor(
            seq_len=LSTM_SEQ,
            n_features=R_LSTM,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            dropout=LSTM_DROPOUT,
            learning_rate=LSTM_LR,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH,
            patience=LSTM_PATIENCE,
            verbose=0,
        )

        _t0 = time.perf_counter()
        lstm.fit(Z_train[:, :R_LSTM], Z_val[:, :R_LSTM])
        lstm_train_time = time.perf_counter() - _t0
        print(f"       device = {lstm.device_}")

        def _pad_lstm(Z_l):
            if Z_l.shape[1] == N_POD:
                return Z_l
            Z_full = np.zeros((Z_l.shape[0], N_POD))
            Z_full[:, :R_LSTM] = Z_l
            return Z_full

        if RECEDING_ONE_STEP:
            ref_val_l = np.vstack([Z_train[-LSTM_SEQ:, :R_LSTM], Z_val[:, :R_LSTM]])
            win_val = np.stack([ref_val_l[t:t + LSTM_SEQ] for t in range(nt_val)], axis=0)
            Z_lstm_val = lstm.predict_windows(win_val)
        else:
            Z_lstm_val = lstm.predict(nt_val, z0=Z_train[-LSTM_SEQ:, :R_LSTM])
        re_lstm_val = compute_relative_error(X_val, pod.inverse_transform(_pad_lstm(Z_lstm_val)))
        print(f"       val  mean relative error = {re_lstm_val.mean():.4f}")

        _t0 = time.perf_counter()
        if RECEDING_ONE_STEP:
            ref_test_l = np.vstack([Z_val[-LSTM_SEQ:, :R_LSTM], Z_test[:, :R_LSTM]])
            win_test = np.stack([ref_test_l[t:t + LSTM_SEQ] for t in range(nt_test)], axis=0)
            Z_lstm = lstm.predict_windows(win_test)
        else:
            Z_lstm = lstm.predict(nt_test, z0=Z_val[-LSTM_SEQ:, :R_LSTM])
        lstm_infer_time = (time.perf_counter() - _t0) / nt_test

        X_lstm = pod.inverse_transform(_pad_lstm(Z_lstm))
        re_lstm = compute_relative_error(X_test, X_lstm)
        print(f"       test mean relative error = {re_lstm.mean():.4f}")
        print(f"       train {lstm_train_time:.2f}s  |  infer {lstm_infer_time*1e3:.3f} ms/step")
    except ImportError as exc:
        print(f"       LSTM skipped: {exc}")

    print("\n  Per-step relative error (test):")
    if re_lstm is None:
        print(f"  {'step':>6}  {'DMD':>8}  {'SINDy':>8}  {'POD-rec':>8}")
    else:
        print(f"  {'step':>6}  {'DMD':>8}  {'SINDy':>8}  {'LSTM':>8}  {'POD-rec':>8}")
    re_rec_per = compute_relative_error(X_test, pod.inverse_transform(Z_test))
    for t in [0, 1, 5, 10, 50, 100, 500, 1000]:
        if t < nt_test:
            if re_lstm is None:
                print(f"  {t:>6}  {re_dmd[t]:>8.4f}  {re_sindy[t]:>8.4f}  {re_rec_per[t]:>8.4f}")
            else:
                print(f"  {t:>6}  {re_dmd[t]:>8.4f}  {re_sindy[t]:>8.4f}  {re_lstm[t]:>8.4f}"
                      f"  {re_rec_per[t]:>8.4f}")

    X_mean_pred = np.tile(pod.mean_, (nt_test, 1))
    corr_mean = compute_correlation(X_test, X_mean_pred)
    re_mean = compute_relative_error(X_test, X_mean_pred).mean()
    print(f"\n  Baseline — mean-field prediction: "
          f"corr = {corr_mean:.4f},  rel.err = {re_mean:.4f}")

    print("\n" + "=" * 64)
    thresh = 0.50
    print(f"{'Method':<10}  {'Val rel.err':>12}  {'Test rel.err':>13}  "
            f"{'Horizon(s)':>10}  {'RMSE':>8}  {'Corr':>6}  {'R2':>7}")
    print("-" * 66)
    rmse_list = []
    corr_list = []
    r2_list = []
    records   = []
    _complexity = {
        "DMD":   "O(r²n + r³) / O(r²)",
        "SINDy": "O(r²n) / O(r²)",
        "LSTM":  "O(E·n·s·h²) / O(s·h²)",
    }
    _train_times  = {"DMD": dmd_train_time, "SINDy": sindy_train_time, "LSTM": lstm_train_time}
    _infer_times  = {"DMD": dmd_infer_time, "SINDy": sindy_infer_time, "LSTM": lstm_infer_time}

    method_results = [
        ("DMD",   re_dmd_val,   re_dmd,   X_dmd),
        ("SINDy", re_sindy_val, re_sindy, X_sindy),
    ]
    if re_lstm is not None and re_lstm_val is not None and X_lstm is not None:
        method_results.append(("LSTM", re_lstm_val, re_lstm, X_lstm))

    for name, re_val, re, Xp in method_results:
        ph   = prediction_horizon(re, threshold=thresh, dt=DT)
        rmse = compute_rmse(X_test, Xp)
        corr = compute_correlation(X_test, Xp)
        r2   = compute_r2(X_test, Xp)
        rmse_list.append(float(rmse))
        corr_list.append(corr)
        r2_list.append(r2)
        records.append({"name": name, "val_re": re_val.mean(),
                        "test_re": re.mean(), "horizon": ph,
                "rmse": float(rmse), "corr": corr, "r2": r2,
                        "train_time": _train_times[name],
                        "infer_time": _infer_times[name],
                        "complexity": _complexity[name]})
        print(f"{name:<10}  {re_val.mean():>12.4f}  {re.mean():>13.4f}  "
              f"{ph:>8.1f} s   {rmse:>8.4f}  {corr:>6.4f}  {r2:>7.4f}")

    horizon_list = [r["horizon"] for r in records]
    metrics_data = {"methods": [r["name"] for r in records], "rmse": rmse_list,
                    "corr": corr_list, "horizon": horizon_list}

    print("\nSweeping correlation vs POD modes …")
    r_values = [10, 20, 30, 40, 50]
    corr_vs_r = {"r": r_values, "DMD": [], "SINDy": [], "LSTM": []}
    for r_sweep in r_values:
        r_d = min(r_sweep, R_DMD) if r_sweep < R_DMD else R_DMD
        r_d = r_sweep
        d_delay = DMD_DELAY
        Ztr_d = Z_train[:, :r_d]
        Zval_d = Z_val[:, :r_d]
        Ztest_d = Z_test[:, :r_d]
        dmd_s = DMD(n_modes=None, stabilise=True, delay=d_delay)
        dmd_s.fit(Ztr_d)
        if RECEDING_ONE_STEP:
            ref = np.vstack([Zval_d[-(d_delay + 1):], Ztest_d])
            Zpd = np.empty((nt_test, r_d))
            for t in range(nt_test):
                Zpd[t] = dmd_s.predict(1, z0=ref[t:t + d_delay + 1])[0]
        else:
            Zpd = dmd_s.predict(nt_test, z0=Zval_d[-(d_delay + 1):])
        Zpad = np.zeros((nt_test, N_POD)); Zpad[:, :r_d] = Zpd
        corr_vs_r["DMD"].append(compute_correlation(X_test, pod.inverse_transform(Zpad)))

        sindy_s = SINDy(dt=DT, poly_degree=2, threshold=0.005, ridge_alpha=1e-4,
                        n_iter=20, n_sub=10, discrete=True, clip_sigma=1.8)
        sindy_s.fit(Z_train[:, :r_sweep])
        if RECEDING_ONE_STEP:
            ref_s = np.vstack([Z_val[-1:, :r_sweep], Z_test[:, :r_sweep]])
            Zps = np.empty((nt_test, r_sweep))
            for t in range(nt_test):
                Zps[t] = sindy_s.predict(1, z0=ref_s[t])[0]
        else:
            Zps = sindy_s.predict(nt_test, z0=Z_val[-1, :r_sweep])
        Zpad_s = np.zeros((nt_test, N_POD)); Zpad_s[:, :r_sweep] = Zps
        corr_vs_r["SINDy"].append(compute_correlation(X_test, pod.inverse_transform(Zpad_s)))

        try:
            lstm_s = LSTMPredictor(
                seq_len=LSTM_SEQ, n_features=r_sweep,
                hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
                dropout=LSTM_DROPOUT, learning_rate=LSTM_LR,
                epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
                patience=LSTM_PATIENCE, verbose=0,
            )
            lstm_s.fit(Z_train[:, :r_sweep], Z_val[:, :r_sweep])
            if RECEDING_ONE_STEP:
                ref_l = np.vstack([Z_val[-LSTM_SEQ:, :r_sweep], Z_test[:, :r_sweep]])
                win_l = np.stack([ref_l[t:t + LSTM_SEQ] for t in range(nt_test)], axis=0)
                Zpl = lstm_s.predict_windows(win_l)
            else:
                Zpl = lstm_s.predict(nt_test, z0=Z_val[-LSTM_SEQ:, :r_sweep])
            Zpad_l = np.zeros((nt_test, N_POD)); Zpad_l[:, :r_sweep] = Zpl
            corr_vs_r["LSTM"].append(compute_correlation(X_test, pod.inverse_transform(Zpad_l)))
        except Exception:
            corr_vs_r["LSTM"].append(float("nan"))

        print(f"  r={r_sweep:>2}:  DMD={corr_vs_r['DMD'][-1]:.4f}"
              f"  SINDy={corr_vs_r['SINDy'][-1]:.4f}"
              f"  LSTM={corr_vs_r['LSTM'][-1]:.4f}")

    print("\nGenerating report figure …")
    plot.plot_report_figure(
        pod=pod,
        rel_err_dmd=re_dmd,
        rel_err_sindy=re_sindy,
        rel_err_lstm=re_lstm,
        rel_err_pod=re_rec_per,
        corr_vs_r=corr_vs_r,
        X_test=X_test,
        X_dmd_pred=X_dmd,
        X_sindy_pred=X_sindy,
        X_lstm_pred=X_lstm,
        metrics_data=metrics_data,
        nt_test=nt_test,
        dt=DT,
        snapshot_idx=SNAP_IDX,
        nx=nx,
        ny=ny,
        nc=nc,
        save_path=SAVE_FIG,
    )

    elapsed = time.perf_counter() - t_wall
    print(f"\nTotal wall-clock time: {elapsed / 60:.1f} min")

    _write_results_txt(
        path=SAVE_TXT,
        records=records,
        pod_energy=ce[-1] * 100,
        elapsed=elapsed,
        n_sp=n_sp,
    )
    print("=" * 64)


if __name__ == "__main__":
    main()

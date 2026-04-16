"""Plotting utilities for report figures."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "legend.framealpha": 0.85,
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi":        150,
})

METHOD_COLORS = {
    "DMD":   "#2166ac",   # blue
    "SINDy": "#1a9850",   # green
    "LSTM":  "#b2182b",   # red
}


def plot_report_figure(
    pod,
    rel_err_dmd: np.ndarray,
    rel_err_sindy: np.ndarray,
    rel_err_lstm: np.ndarray | None,
    rel_err_pod: np.ndarray | None,
    corr_vs_r: dict | None,
    X_test: np.ndarray,
    X_dmd_pred: np.ndarray,
    X_sindy_pred: np.ndarray,
    X_lstm_pred: np.ndarray | None,
    metrics_data: dict,
    nt_test: int,
    dt: float = 0.2,
    snapshot_idx: int = 100,
    nx: int = 64,
    ny: int = 64,
    nc: int = 2,
    save_path: str = "report_figure.pdf",
    **kwargs,
) -> None:
    """Create and save the SCI-quality 5-panel report figure."""
    t_pred = np.arange(1, nt_test + 1) * dt

    fig = plt.figure(figsize=(13, 14))
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.07, right=0.96,
        bottom=0.04, top=0.96,
        hspace=0.28, wspace=0.35,
        height_ratios=[1, 1, 0.8],
    )

    ax_pod = fig.add_subplot(gs[0, 0])
    r  = pod.n_modes
    ev = pod.explained_variance_ratio_
    ce = pod.cumulative_energy_

    ax_pod.bar(np.arange(1, r + 1), ev * 100,
               color="#4393c3", alpha=0.85, linewidth=0)
    ax_pod.set_xlabel("POD mode index")
    ax_pod.set_ylabel("Individual energy (%)", color="#4393c3")
    ax_pod.tick_params(axis="y", labelcolor="#4393c3")
    ax_pod.set_xlim(0.5, r + 0.5)

    ax_p2 = ax_pod.twinx()
    ax_p2.plot(np.arange(1, r + 1), ce * 100,
               color="#d6604d", lw=1.4, marker="o", markersize=2.0, zorder=3)
    ax_p2.set_ylabel("Cumulative energy (%)", color="#d6604d")
    ax_p2.tick_params(axis="y", labelcolor="#d6604d")
    ax_p2.set_ylim(0, 108)
    ax_p2.axhline(99, color="#d6604d", lw=0.7, ls="--", alpha=0.6)

    ax_pod.set_title("(a) POD energy spectrum")

    ax_cvr = fig.add_subplot(gs[0, 1])

    if corr_vs_r is not None:
        r_vals = corr_vs_r["r"]
        for name in ["DMD", "SINDy", "LSTM"]:
            if name in corr_vs_r and len(corr_vs_r[name]) == len(r_vals):
                vals = corr_vs_r[name]
                ax_cvr.plot(r_vals, vals, marker="o", markersize=5, lw=1.6,
                            color=METHOD_COLORS[name], label=name)
                best_idx = int(np.nanargmax(vals))
                ax_cvr.annotate(f"{vals[best_idx]:.3f}",
                                xy=(r_vals[best_idx], vals[best_idx]),
                                xytext=(0, 6), textcoords="offset points",
                                ha="center", fontsize=7, color=METHOD_COLORS[name])

    ax_cvr.set_xlabel("Number of POD modes (r)")
    ax_cvr.set_ylabel("Pearson correlation")
    ax_cvr.set_title("(b) Correlation vs POD truncation rank")
    ax_cvr.legend(loc="lower right", frameon=True)
    ax_cvr.grid(True, which="major", ls="--", lw=0.5, alpha=0.4)
    if corr_vs_r is not None:
        all_vals = [v for name in ["DMD", "SINDy", "LSTM"]
                    if name in corr_vs_r for v in corr_vs_r[name]
                    if not np.isnan(v)]
        if all_vals:
            lo = min(all_vals) - 0.05
            hi = max(all_vals) + 0.05
            ax_cvr.set_ylim(lo, hi)

    ax_bar = fig.add_subplot(gs[1, 0])

    methods_b    = metrics_data.get("methods", ["DMD", "SINDy"])
    horizon_vals = metrics_data.get("horizon", [0.0] * len(methods_b))
    rmse_vals    = metrics_data["rmse"]
    corr_vals    = metrics_data["corr"]
    cols         = [METHOD_COLORS[m] for m in methods_b]
    n_methods    = len(methods_b)
    x            = np.arange(n_methods)
    w            = 0.25

    bars_h = ax_bar.bar(x - w, horizon_vals, w, color=cols, alpha=0.88,
                        linewidth=0.5, edgecolor="white")
    ax_bar.set_ylabel("Prediction horizon (s)", labelpad=4)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(methods_b)
    ax_bar.set_ylim(bottom=0)
    ax_bar.tick_params(axis="y", which="minor", left=False)

    ax_b2 = ax_bar.twinx()
    bars_r = ax_b2.bar(x, rmse_vals, w, color=cols, alpha=0.55,
                       hatch="...", linewidth=0.5, edgecolor="white")
    bars_c = ax_b2.bar(x + w, corr_vals, w, color=cols, alpha=0.40,
                       hatch="///", linewidth=0.5, edgecolor="white")
    ax_b2.set_ylabel("RMSE / Correlation", labelpad=4)
    ax_b2.set_ylim(0, max(max(rmse_vals), 1.0) * 1.35)
    ax_b2.tick_params(axis="y", which="minor", right=False)

    for i in range(n_methods):
        ax_bar.text(x[i] - w, horizon_vals[i] + 0.1,
                    f"{horizon_vals[i]:.1f}", ha="center", va="bottom", fontsize=6.5)
        ax_b2.text(x[i], rmse_vals[i] + 0.01,
                   f"{rmse_vals[i]:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax_b2.text(x[i] + w, corr_vals[i] + 0.01,
                   f"{corr_vals[i]:.3f}", ha="center", va="bottom", fontsize=6.5)

    legend_handles = [
        Patch(facecolor="gray", alpha=0.88, linewidth=0,
              label="Horizon (rel.err > 50%)"),
        Patch(facecolor="gray", alpha=0.55, hatch="...", linewidth=0,
              label="RMSE"),
        Patch(facecolor="gray", alpha=0.40, hatch="///", linewidth=0,
              label="Correlation"),
    ]
    ax_bar.legend(handles=legend_handles, loc="upper right",
                  handlelength=1.2, handleheight=0.9, fontsize=6.5)
    ax_bar.set_title("(c) Prediction metrics (test set)")

    idx    = snapshot_idx
    t_snap = t_pred[idx]

    true_snap  = X_test[idx].reshape(ny, nx, nc)[:, :, 0]
    dmd_snap   = X_dmd_pred[idx].reshape(ny, nx, nc)[:, :, 0]
    sindy_snap = X_sindy_pred[idx].reshape(ny, nx, nc)[:, :, 0]
    lstm_snap  = None if X_lstm_pred is None else X_lstm_pred[idx].reshape(ny, nx, nc)[:, :, 0]

    vmin = true_snap.min()
    vmax = true_snap.max()
    kw   = dict(origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="none")

    fig.text(0.73, 0.635, f"(d) Snapshot comparison (t = {t_snap:.1f} s)",
             ha="center", fontsize=10, fontweight="normal")

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs[1, 1],
        hspace=0.30,
        wspace=0.08,
        width_ratios=[1, 1, 0.06],
    )
    ax_e1 = fig.add_subplot(inner[0, 0])
    ax_e2 = fig.add_subplot(inner[0, 1])
    ax_e3 = fig.add_subplot(inner[1, 0])
    ax_e4 = fig.add_subplot(inner[1, 1])
    cax   = fig.add_subplot(inner[:, 2])

    sub_titles = [
        (ax_e1, true_snap,  "Ground truth"),
        (ax_e2, dmd_snap,   "DMD prediction"),
        (ax_e3, sindy_snap, "SINDy prediction"),
    ]
    if lstm_snap is not None:
        sub_titles.append((ax_e4, lstm_snap, "LSTM prediction"))
    else:
        ax_e4.axis("off")

    im = None
    for ax, data, subtitle in sub_titles:
        im = ax.imshow(data, **kw)
        ax.set_title(subtitle, fontsize=8, pad=2)
        ax.axis("off")

    fig.colorbar(im, cax=cax, label=r"$u_x$")
    cax.tick_params(labelsize=7)

    ax_err = fig.add_subplot(gs[2, :])

    win = max(1, nt_test // 60)

    def _rolling_mean(arr, w):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="same")

    curves = [("DMD", rel_err_dmd), ("SINDy", rel_err_sindy)]
    if rel_err_lstm is not None:
        curves.append(("LSTM", rel_err_lstm))

    for name, re in curves:
        c = METHOD_COLORS[name]
        ax_err.plot(t_pred, re, color=c, alpha=0.15, lw=0.4)
        ax_err.plot(t_pred, _rolling_mean(re, win), color=c, lw=1.6,
                    label=f"{name} (mean {re.mean():.3f})")

    ax_err.axhline(0.5, color="black", lw=0.9, ls="--", alpha=0.6,
                   label="Horizon threshold (50%)")

    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Relative error")
    ax_err.set_title("(e) Prediction error over test set")
    ax_err.set_xlim(t_pred[0], t_pred[-1])
    ax_err.set_ylim(bottom=0)
    ax_err.legend(loc="lower right", frameon=True, fontsize=7)
    ax_err.grid(True, which="major", ls="--", lw=0.5, alpha=0.4)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Report figure saved  →  {save_path}")
    plt.close(fig)

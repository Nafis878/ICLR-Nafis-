"""Figures (spec section 5/M8). Rendered to figures/ as PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("figures")


def _save(fig, name: str) -> str:
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")
    return str(p)


def regret_surface_by_family(tab: pd.DataFrame, tau: float) -> str:
    """Per-family TabPFN Bayes-regret distributions against the tau level set."""
    fams = list(tab.groupby("family").regret.median().sort_values().index)
    data = [tab.loc[tab.family == f, "regret"].dropna().to_numpy() for f in fams]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
    for i, arr in enumerate(data, start=1):
        ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(arr)), arr,
                   s=5, alpha=0.35, color="#08519c", zorder=3)
    ax.axhline(tau, color="crimson", ls="--", lw=1.4,
               label=f"tau = {tau:.3f} (75th pct on scm_prior_control)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=8)
    ax.set_ylabel("TabPFN Bayes regret (log-loss)")
    ax.set_title("M3 regret surface: TabPFN v2 Bayes regret by DGP family")
    ax.legend(fontsize=8)
    return _save(fig, "m3_regret_by_family.png")


def degradation_axes_plot(axes: pd.DataFrame, k: int = 10) -> str:
    a = axes.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(a) + 1.4))
    colors = ["#c6362f" if v > 0 else "#2f6fc6" for v in a.spearman]
    ax.barh(a.axis.str.replace("param_", "").str.replace("mod_", ""), a.spearman, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman rho with TabPFN Bayes regret")
    ax.set_title("DGP axes on which TabPFN degrades most sharply")
    return _save(fig, "m3_degradation_axes.png")


def oop_calibration_synthetic(fit: dict) -> str:
    y, pred = fit["y"], fit["oof_pred"]
    ok = np.isfinite(pred)
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(pred[ok], y[ok], s=8, alpha=0.4, color="#08519c")
    lim = [min(pred[ok].min(), y[ok].min()), max(pred[ok].max(), y[ok].max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("predicted regret (grouped-CV, out of fold)")
    ax.set_ylabel("actual Bayes regret")
    ax.set_title(f"M4: g on synthetic data\n{fit['model_name']}, "
                 f"Spearman={fit['cv'][fit['model_name']]['cv_spearman']:.3f}")
    return _save(fig, "m4_synthetic_fit.png")


def feature_importance_plot(imp: pd.DataFrame, k: int = 15) -> str:
    a = imp.head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(a) + 1.4))
    ax.barh(a.feature.str.replace("stat_", ""), a.importance, color="#4a8c4a")
    ax.set_xlabel("importance in g")
    ax.set_title("M4: which observable statistics drive the OOP score")
    return _save(fig, "m4_feature_importances.png")


def real_vs_synthetic_projection(syn: pd.DataFrame, real: pd.DataFrame, feats: list[str]) -> str:
    """PCA of the statistic space: do real datasets sit inside the probe space?"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    S = np.nan_to_num(syn.reindex(columns=feats).to_numpy(dtype=float))
    R = np.nan_to_num(real.reindex(columns=feats).to_numpy(dtype=float))
    sc = StandardScaler().fit(np.vstack([S, R]))
    p = PCA(n_components=2, random_state=0).fit(sc.transform(S))
    ps, pr = p.transform(sc.transform(S)), p.transform(sc.transform(R))

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.scatter(ps[:, 0], ps[:, 1], s=7, alpha=0.30, color="#7f9fc4", label="synthetic probes (M3)")
    inr = real["in_range"].to_numpy(dtype=bool) if "in_range" in real else np.ones(len(pr), bool)
    ax.scatter(pr[inr, 0], pr[inr, 1], s=44, marker="o", color="#c6362f",
               edgecolor="black", lw=0.5, label="real, in TabPFN range")
    ax.scatter(pr[~inr, 0], pr[~inr, 1], s=54, marker="^", color="#f0a202",
               edgecolor="black", lw=0.5, label="real, out of range")
    ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]*100:.0f}% var)")
    ax.set_title("M5: real datasets vs the synthetic probe space")
    ax.legend(fontsize=8)
    return _save(fig, "m5_real_vs_synthetic.png")


def m0_cost_plot(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for model, mark in (("tabpfn", "o"), ("lightgbm", "s")):
        s = df[(df.model == model) & (df.status == "ok")]
        if s.empty:
            continue
        g = s.groupby("n").t.median()
        ax.plot(g.index, g.values, marker=mark, label=model)
    cen = df[df.censored]
    if len(cen):
        ax.axhline(cen.wall.median(), color="crimson", ls=":", lw=1.2,
                   label="timeout cap (censored above)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("median fit+predict seconds (CPU)")
    ax.set_title("M0: CPU cost of TabPFN v2 vs tuned LightGBM")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    return _save(fig, "m0_cost.png")

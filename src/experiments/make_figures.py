"""Render all pre-freeze figures (M0, M3, M4, M5)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis import figures  # noqa: E402
from analysis.oop_score import feature_importances  # noqa: E402


def rotation_figure() -> None:
    df = pd.read_csv("results/m3b_rotation_controlled.csv")
    ok = df[df.status == "ok"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for model, colour in (("tabpfn", "#08519c"), ("strong_classical", "#c6362f")):
        s = ok[ok.model == model].groupby("rotation_strength").regret
        m, sd = s.mean(), s.std()
        ax.errorbar(m.index, m.values, yerr=sd.values, marker="o", capsize=3,
                    color=colour, label=model)
    ax.set_xlabel("rotation strength (0 = axis-aligned, 1 = fully rotated)")
    ax.set_ylabel("Bayes regret (log-loss)")
    ax.set_title("Controlled rotation probe: everything fixed but the angle\n"
                 "Bayes risk is invariant, so this is pure inductive bias", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    figures._save(fig, "m3b_rotation_controlled.png")


def main() -> None:
    m0 = pd.read_parquet("results/m0_costs.parquet")
    figures.m0_cost_plot(m0)

    df = pd.read_parquet("results/m3_sweep.parquet")
    tab = df[(df.status == "ok") & (df.model == "tabpfn") & df.regret.notna()]
    meta = json.load(open("results/m4_oop_score.json"))
    tau = meta["tau"]
    figures.regret_surface_by_family(tab, tau)

    axes = pd.read_csv("results/m3_degradation_axes.csv")
    figures.degradation_axes_plot(axes)

    fit = joblib.load("results/m4_oop_score.joblib")
    figures.oop_calibration_synthetic(fit)
    figures.feature_importance_plot(feature_importances(fit))

    real = pd.read_parquet("results/m5_real_statistics.parquet")
    figures.real_vs_synthetic_projection(tab, real, fit["features"])

    rotation_figure()
    print("\n[figures] all pre-freeze figures rendered to figures/")


if __name__ == "__main__":
    main()

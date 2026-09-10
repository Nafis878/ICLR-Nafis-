"""Controlled rotation probe -- a matched-pairs follow-up to M3.

WHY THIS EXISTS. The M3 sweep gives two readings that appear to conflict:
  * BETWEEN families, rotated_piecewise_constant has ~2.8x the median regret of
    piecewise_constant (0.180 vs 0.065);
  * WITHIN rotated_piecewise_constant, rotation_strength correlates NEGATIVELY
    with regret (rho = -0.35).
Both are confounded: the two families get independent Latin hypercube designs,
so they differ in depth, leaf purity, d and n as well as rotation, and 15 LHS
points per family cannot separate five parameters.

This holds EVERYTHING fixed except the rotation angle, on one tree structure, so
the rotation axis is measured rather than inferred. It adds no DGP family and no
statistic -- it re-uses RotatedPiecewiseConstant -- so it is analysis, not a
change to the frozen artefacts.

Bayes risk is invariant to rotation (an orthogonal map is information
preserving, asserted in tests/test_dgp.py), so any regret difference here is
PURE inductive bias.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from dgp.registry import draw  # noqa: E402
from models.wrappers import fit_model, strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402

ANGLES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_SEEDS = 5
N_TRAIN, N_TEST = 800, 1500
BASE = dict(d=12, n_classes=3, depth=4, leaf_purity=0.9, struct_seed=17)


def run_one(cfg: dict) -> dict:
    X, y, bayes, exact = draw(
        cfg["family"], cfg["params"], None, cfg["n_train"] + cfg["n_test"], cfg["seed"]
    )
    ntr = cfg["n_train"]
    K = int(cfg["params"]["n_classes"])
    if cfg["model"] == "strong_classical":
        proba, info = strong_classical(X[:ntr], y[:ntr], X[ntr:], K, cfg["model_cfgs"], cfg["seed"])
    else:
        proba, info = fit_model(cfg["model"], X[:ntr], y[:ntr], X[ntr:], K,
                                cfg["model_cfgs"][cfg["model"]], cfg["seed"])
    return {**evaluate(proba, y[ntr:], K, bayes[ntr:] if exact else None), "exact_bayes": exact}


def main() -> None:
    env.preflight()
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for n in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[n] = {**mc[n], "n_random_configs": 8, "cv_folds": 3}

    cells = []
    for s in ANGLES:
        for seed in range(N_SEEDS):
            for model in ("tabpfn", "strong_classical"):
                cells.append({
                    "milestone": "m3b",
                    "family": "rotated_piecewise_constant",
                    "params": {**BASE, "rotation_strength": float(s)},
                    "n_train": N_TRAIN, "n_test": N_TEST, "seed": seed,
                    "model": model, "model_cfgs": mc,
                })
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)

    recs = [{"rotation_strength": r["config"]["params"]["rotation_strength"],
             "model": r["config"]["model"], "seed": r["config"]["seed"],
             "status": r["status"], "regret": r.get("regret"),
             "bayes_logloss": r.get("bayes_logloss")} for r in rows]
    df = pd.DataFrame(recs)
    ok = df[df.status == "ok"]

    print("=" * 74)
    print("CONTROLLED ROTATION PROBE (everything fixed except the rotation angle)")
    print("=" * 74)
    piv = ok.pivot_table(index="rotation_strength", columns="model", values="regret",
                         aggfunc=["mean", "std"])
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

    b = ok.groupby("rotation_strength").bayes_logloss.mean()
    print(f"\nBayes log-loss across angles: {b.min():.4f} .. {b.max():.4f} "
          f"(spread {b.max()-b.min():.5f}; rotation is information-preserving, so this "
          f"must be ~constant)")

    tab = ok[ok.model == "tabpfn"]
    lo = tab[tab.rotation_strength == 0.0].regret
    hi = tab[tab.rotation_strength == 1.0].regret
    from scipy.stats import mannwhitneyu, spearmanr

    rho = spearmanr(tab.rotation_strength, tab.regret)
    u = mannwhitneyu(hi, lo, alternative="greater")
    delta = hi.mean() - lo.mean()
    pooled_sd = np.sqrt((lo.var(ddof=1) + hi.var(ddof=1)) / 2)
    print(f"\nTabPFN regret  s=0: {lo.mean():+.4f} (sd {lo.std(ddof=1):.4f})   "
          f"s=1: {hi.mean():+.4f} (sd {hi.std(ddof=1):.4f})")
    print(f"  difference {delta:+.4f}, pooled sd {pooled_sd:.4f} "
          f"-> {'LARGER' if abs(delta) > pooled_sd else 'SMALLER'} than seed-to-seed noise")
    print(f"  Spearman(rotation, regret) = {rho.statistic:+.3f} (p={rho.pvalue:.3g}); "
          f"Mann-Whitney one-sided p={u.pvalue:.3g}")
    if abs(delta) <= pooled_sd:
        print("  Effect is NOT larger than its own seed-to-seed spread; per spec section 6")
        print("  this must not be reported as a result.")

    df.to_csv("results/m3b_rotation_controlled.csv", index=False)
    json.dump({"angles": ANGLES, "n_seeds": N_SEEDS, "base_params": BASE,
               "tabpfn_regret_by_angle": tab.groupby("rotation_strength").regret.mean().to_dict(),
               "delta_s1_minus_s0": float(delta), "pooled_sd": float(pooled_sd),
               "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
               "bayes_logloss_spread": float(b.max() - b.min())},
              open("results/m3b_rotation_controlled.json", "w"), indent=2)
    print("\n[m3b] wrote results/m3b_rotation_controlled.{csv,json}")


if __name__ == "__main__":
    main()

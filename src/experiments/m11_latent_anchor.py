"""W5 -- does the in-prior anchor survive LATENT CONFOUNDING? (prereg 18552a4)

Matched comparison of TabPFN Bayes regret on the restricted anchor
(scm_prior_control, label reads observed nodes only) against the confounded one
(scm_latent_confounded, a discrete latent drives both X and the decision rule).
Same d, n_classes, n_train and seeds on both sides, so only the confounding differs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from dgp.registry import draw  # noqa: E402
from models.wrappers import fit_model, strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402

N_TRAIN, N_TEST, N_SEEDS = 800, 2000, 5
D, K = 8, 3


def run_one(cfg: dict) -> dict:
    X, y, bayes, exact = draw(cfg["family"], cfg["params"], None,
                              cfg["n_train"] + cfg["n_test"], cfg["seed"])
    ntr = cfg["n_train"]
    kk = int(cfg["params"]["n_classes"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg["model"] == "strong_classical":
            proba, _ = strong_classical(X[:ntr], y[:ntr], X[ntr:], kk,
                                        cfg["model_cfgs"], cfg["seed"])
        else:
            proba, _ = fit_model(cfg["model"], X[:ntr], y[:ntr], X[ntr:], kk,
                                 cfg["model_cfgs"][cfg["model"]], cfg["seed"])
    return {**evaluate(proba, y[ntr:], kk, bayes[ntr:] if exact else None),
            "exact_bayes": exact}


def main() -> None:
    env.preflight()
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for n in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[n] = {**mc[n], "n_random_configs": 8, "cv_folds": 3}

    fams = {
        "scm_prior_control": dict(d=D, n_classes=K, max_parents=3,
                                  activation="tanh", noise_scale=0.3, label_noise=0.05),
        "scm_latent_confounded": dict(d=D, n_classes=K, n_latent_states=4,
                                      confounding=1.0, separation=1.5,
                                      signal_scale=1.5, label_noise=0.05),
    }
    cells = [{"milestone": "m11anchor", "family": f, "params": p, "n_train": N_TRAIN,
              "n_test": N_TEST, "seed": s, "model": m, "model_cfgs": mc}
             for f, p in fams.items() for s in range(N_SEEDS)
             for m in ("tabpfn", "strong_classical")]
    rows = runner.run_many(cells, run_one, n_jobs=4, verbose=5)

    recs = [{"family": r["config"]["family"], "model": r["config"]["model"],
             "seed": r["config"]["seed"], "status": r["status"],
             "regret": r.get("regret"), "bayes_logloss": r.get("bayes_logloss")}
            for r in rows]
    df = pd.DataFrame(recs)
    ok = df[df.status == "ok"]
    print("=" * 76)
    print("W5 -- IN-PRIOR ANCHOR UNDER LATENT CONFOUNDING  (prereg 18552a4)")
    print("=" * 76)
    piv = ok.pivot_table(index="family", columns="model", values="regret",
                         aggfunc=["mean", "std"])
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nBayes log-loss (oracle) by family:")
    print(ok.groupby("family").bayes_logloss.mean().to_string(float_format=lambda v: f"{v:.4f}"))

    tab = ok[ok.model == "tabpfn"]
    a = tab[tab.family == "scm_prior_control"].regret
    b = tab[tab.family == "scm_latent_confounded"].regret
    from scipy.stats import mannwhitneyu

    u = mannwhitneyu(b, a, alternative="greater")
    print(f"\nTabPFN regret  restricted {a.mean():+.4f} (sd {a.std(ddof=1):.4f})"
          f"   confounded {b.mean():+.4f} (sd {b.std(ddof=1):.4f})")
    print(f"  difference {b.mean()-a.mean():+.4f}   Mann-Whitney one-sided p={u.pvalue:.4g}")
    print(f"  H_W5 (anchor SURVIVES confounding): "
          f"{'SUPPORTED -- regret stays low' if b.mean() < 0.10 else 'NOT SUPPORTED -- regret rises'}")
    print("  reference: out-of-prior families reach ~0.29 (high_frequency)")
    df.to_csv("results/m11_latent_anchor.csv", index=False)
    json.dump({"restricted_mean": float(a.mean()), "confounded_mean": float(b.mean()),
               "p_one_sided": float(u.pvalue)},
              open("results/m11_latent_anchor.json", "w"), indent=2)
    print("\n[w5] wrote results/m11_latent_anchor.{csv,json}")


if __name__ == "__main__":
    main()

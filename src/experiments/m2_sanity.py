"""M2 -- metric definitions plus the in-prior sanity run (spec section 5/M2)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import numpy as np  # noqa: E402

import runner  # noqa: E402
from dgp.registry import draw  # noqa: E402
from models.wrappers import fit_model, strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402


def run_one(cfg: dict) -> dict:
    X, y, bayes, exact = draw(
        cfg["family"], cfg["params"], cfg.get("modifiers"),
        cfg["n_train"] + cfg["n_test"], cfg["seed"],
    )
    ntr = cfg["n_train"]
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    bayes_te = bayes[ntr:]
    K = cfg["params"]["n_classes"]

    if cfg["model"] == "strong_classical":
        proba, info = strong_classical(Xtr, ytr, Xte, K, cfg["model_cfgs"], cfg["seed"])
    else:
        proba, info = fit_model(
            cfg["model"], Xtr, ytr, Xte, K, cfg["model_cfgs"][cfg["model"]], cfg["seed"]
        )
    metrics = evaluate(proba, yte, K, bayes_te if exact else None)
    return {**metrics, "exact_bayes": exact, "info": info}


def main() -> None:
    env.preflight()
    cfg = runner.load_config("configs/m2_baselines.yaml")
    s = cfg["sanity"]
    cells = []
    for model in ["tabpfn", "strong_classical", "lightgbm", "knn"]:
        for seed in range(cfg["n_seeds"]):
            cells.append({
                "milestone": "m2", "model": model, "family": s["family"],
                "params": s["params"], "modifiers": None,
                "n_train": cfg["n_train"], "n_test": cfg["n_test"],
                "seed": seed, "model_cfgs": cfg["models"],
            })
    rows = runner.run_many(cells, run_one, n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 3)

    print("\n=== M2 in-prior sanity: scm_prior_control ===")
    by = {}
    for r in rows:
        if r["status"] != "ok":
            print("FAILED CELL:", r.get("error", "")[:200])
            continue
        by.setdefault(r["config"]["model"], []).append(r)
    bayes_ll = np.mean([r["bayes_logloss"] for r in by.get("tabpfn", []) if "bayes_logloss" in r])
    print(f"Bayes log-loss (oracle): {bayes_ll:.4f}")
    for m, rs in by.items():
        reg = np.array([r["regret"] for r in rs])
        ll = np.array([r["logloss"] for r in rs])
        sig = sum(r["regret_significant"] for r in rs)
        print(f"  {m:17s} logloss={ll.mean():.4f}  regret={reg.mean():+.4f} "
              f"(sd {reg.std(ddof=1):.4f}, {sig}/{len(rs)} seeds significant)")


if __name__ == "__main__":
    main()

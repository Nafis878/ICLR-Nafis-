"""W2 -- re-run the classical reference with a MUCH stronger tuning budget.

The spec named under-tuned baselines the single most common reviewer objection, and
M7 used best-of-4 with 8 random configs (32 total). This re-runs strong_classical as
best-of-6 -- adding XGBoost and CatBoost, both listed as core deps in the spec and
previously unused -- with 165 configs per dataset.

Budgets are cost-aware rather than uniform: measured on CPU, one CatBoost config costs
~40x one LightGBM config, so a flat budget would spend nearly all compute on a single
family instead of broadening the ensemble.

Only strong_classical is re-run; TabPFN's M7 results are cached and untouched, so the
comparison changes on exactly one side and every frozen prediction set can be re-scored
against the stronger bar.

Datasets run in a SEEDED SHUFFLED order so that a partial run is a balanced sample
rather than "all the small ones"; coverage and any size skew are reported explicitly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from models.wrappers import strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    Xtr, ytr, Xte, yte, _ = prepare_split(X, y, cfg["m5"]["eval_policy"], cfg["seed"])
    kk = int(len(np.unique(y)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba, info = strong_classical(Xtr, ytr, Xte, kk, cfg["model_cfgs"], cfg["seed"])
    return {**evaluate(proba, yte, kk, None), "dataset_id": cfg["dataset_id"],
            "name": name, "winner": info.get("winner"),
            "val_scores": {k: v for k, v in info.get("val_scores", {}).items()
                           if not str(k).endswith("_error")},
            "n_train": len(Xtr), "n_test": len(Xte)}


def main() -> None:
    env.preflight()
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    mc = runner.load_config("configs/m2_baselines.yaml")["models"]
    total = sum(v["n_random_configs"] for k, v in mc.items() if k != "tabpfn")
    print(f"[w2] best-of-{len([k for k in mc if k!='tabpfn'])}, {total} configs/dataset "
          f"(M7 used best-of-4 with 32)")

    outcomes = pd.read_csv("results/m7_outcomes.csv")
    dsets = sorted(outcomes.dataset_id.unique().tolist())
    rng = np.random.default_rng(20260912)
    order = {d: i for i, d in enumerate(rng.permutation(dsets).tolist())}

    cells = [{"milestone": "m12base", "dataset_id": int(d), "seed": 0,
              "m5": m5cfg, "model_cfgs": mc} for d in dsets]
    cells.sort(key=lambda c: order[c["dataset_id"]])
    print(f"[w2] {len(cells)} datasets, seeded shuffle so partial coverage stays balanced")
    runner.run_many(cells, run_one, n_jobs=4, verbose=5)
    print("[w2] done")


if __name__ == "__main__":
    main()

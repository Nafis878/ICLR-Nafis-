"""W1 -- scale ladder to TabPFN v2's stated 10k-row ceiling (prereg 18552a4).

The earlier evaluation capped n_train at 1500-4000, inviting "you evaluated outside
the model's operating range". TabPFN v2's stated range is <=10k rows, so this runs the
paired rotation probe AT that ceiling.

Feasible because cost is n^0.77 * d^0.76: choosing low-dimensional datasets makes
n=10000 cheap (~300 s/cell) where n=4000 on d=500 was ~30 min/cell.
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
from experiments.m9_rotation_transfer import TUNING, numeric_mask, rotate_block  # noqa: E402
from models.wrappers import fit_model  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402

LEVELS = [500, 2000, 10000]
MAX_CLASSES = 10          # TabPFN v2 hard limit; >10 errors out
MAX_D = 20                # keeps n=10000 affordable on CPU
N_DATASETS = 13


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    pol = dict(cfg["m5"]["eval_policy"])
    pol["max_train_rows"] = cfg["n_train_cap"]
    pol["max_test_rows"] = 2000
    Xtr, ytr, Xte, yte, _ = prepare_split(X, y, pol, cfg["split_seed"])
    kk = int(len(np.unique(y)))
    mask = numeric_mask(Xtr)
    if mask.sum() < 4:
        raise ValueError(f"only {int(mask.sum())} numeric columns; rotation undefined")
    if cfg["condition"] == "rotated":
        Xtr, Xte = rotate_block(Xtr, Xte, mask, cfg["rotation_seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba, _ = fit_model("tabpfn", Xtr, ytr, Xte, kk,
                             cfg["model_cfgs"]["tabpfn"], cfg["split_seed"])
    return {**evaluate(proba, yte, kk, None), "dataset_id": cfg["dataset_id"],
            "name": name, "n_train": len(Xtr), "n_numeric": int(mask.sum())}


def main() -> None:
    env.preflight()
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for n in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[n] = {**mc[n], **TUNING}

    a = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
    b = pd.read_parquet("results/m5rep_real_statistics.parquet")
    d = pd.concat([a, b], ignore_index=True).drop_duplicates("dataset_id")
    d["numeric"] = (d.d_eval * (1 - d.stat_categorical_fraction)).round()
    # need ~14.3k rows for a 10k train split at test_size=0.3
    cand = d[(d.n_full >= 14300) & (d.numeric >= 4)
             & (d.n_classes <= MAX_CLASSES) & (d.d_eval <= MAX_D)]
    cand = cand.nsmallest(N_DATASETS, "d_eval")
    print(f"[w1] {len(cand)} datasets reach n_train=10000 within TabPFN's class/feature limits")
    print(cand[["name", "n_full", "d_eval", "n_classes"]].to_string(index=False))

    cells = [{"milestone": "m11scale", "dataset_id": int(r.dataset_id),
              "n_train_cap": n, "condition": c, "split_seed": 0,
              "rotation_seed": 10_000 + int(r.dataset_id), "m5": m5cfg, "model_cfgs": mc}
             for r in cand.itertuples() for n in LEVELS for c in ("original", "rotated")]
    cells.sort(key=lambda c: (c["n_train_cap"], c["dataset_id"], c["condition"]))
    print(f"[w1] {len(cells)} cells (cheap levels first)")
    runner.run_many(cells, run_one, n_jobs=4, verbose=5)
    print("[w1] done")


if __name__ == "__main__":
    main()

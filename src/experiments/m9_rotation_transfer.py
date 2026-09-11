"""M9 -- does the recovered prior transfer? A PAIRED rotation probe on real data.

WHY THIS EXPERIMENT EXISTS. M7 asked "which real datasets does TabPFN lose on?"
and could not answer it: TabPFN lost on only 9/53 datasets, and the bootstrap
95% CI on AUROC was [0.133, 0.579] -- it contains chance. Detecting a useful
AUROC of 0.70 at that 17% base rate would need ~381 datasets. The task was
underpowered by roughly 7x, so the M7 "negative result" is really "no evidence
either way". Adding datasets cannot fix that; the question has to be better posed.

This experiment poses it better, and it is the question the prior recovery
actually licenses. M3's controlled probe showed TabPFN's Bayes regret rises
0.158 -> 0.360 with rotation angle while Bayes risk is constant to five decimals
-- a large effect with a non-circular oracle. Here we apply the same perturbation
to REAL datasets and ask whether the recovered prior predicts the damage.

Why this is far better powered than M7:
  * PAIRED within dataset -- each dataset is its own control, so between-dataset
    variance (which swamped M7) cancels;
  * the effect is LARGE by construction rather than rare -- we are measuring a
    degradation we induce, not waiting for a spontaneous 17% failure;
  * the target is CONTINUOUS (degradation magnitude), not a 9-positive binary.

PRE-REGISTERED HYPOTHESES, fixed before the first cell was run:
  H1  TabPFN's test log-loss INCREASES under rotation of the numeric block
      (paired one-sided Wilcoxon, alpha = 0.05).
  H2  strong_classical degrades MORE than TabPFN, because its tree members are
      axis-aligned while TabPFN is not (paired, one-sided).
  H3  The rotation_alignment statistic -- computed on the UNROTATED training
      split, frozen at M6 -- positively predicts the magnitude of TabPFN's
      degradation (Spearman > 0, one-sided).
H3 is the out-of-prior claim restated on a target that has variance to predict.

Rotation is applied to the standardized NUMERIC block only. That is not a
convenience: an orthogonal map mixes columns, so it is undefined on categorical
codes, and the rotation_alignment statistic is defined the same way.
Standardization uses TRAINING moments only, and the same rotation matrix is
applied to train and test, so no information leaks across the split.
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
from dgp.base import random_rotation  # noqa: E402
from models.wrappers import fit_model, strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402
from stats.features import CATEGORICAL_MAX_UNIQUE  # noqa: E402

MAX_TRAIN = 1500
MAX_TEST = 1500
MIN_NUMERIC_COLS = 4
TUNING = {"n_random_configs": 16, "cv_folds": 3}  # 2x the M7 budget


def numeric_mask(X: np.ndarray) -> np.ndarray:
    """Columns with many distinct values are treated as numeric (same rule as
    the frozen statistic vector)."""
    out = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j][np.isfinite(X[:, j])]
        out[j] = col.size > 0 and len(np.unique(col)) > CATEGORICAL_MAX_UNIQUE
    return out


def rotate_block(Xtr, Xte, mask, seed):
    """Standardize the numeric block on TRAIN moments, rotate, write back."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    A, B = Xtr[:, mask], Xte[:, mask]
    mu = np.nanmean(A, axis=0)
    sd = np.nanstd(A, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    A = np.nan_to_num((A - mu) / sd)
    B = np.nan_to_num((B - mu) / sd)
    R = random_rotation(int(mask.sum()), np.random.default_rng(seed))
    Xtr[:, mask] = A @ R
    Xte[:, mask] = B @ R
    return Xtr, Xte


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    pol = dict(cfg["m5"]["eval_policy"])
    pol["max_train_rows"], pol["max_test_rows"] = MAX_TRAIN, MAX_TEST
    Xtr, ytr, Xte, yte, _ = prepare_split(X, y, pol, cfg["split_seed"])
    K = int(len(np.unique(y)))

    mask = numeric_mask(Xtr)
    if mask.sum() < MIN_NUMERIC_COLS:
        raise ValueError(f"only {int(mask.sum())} numeric columns; rotation undefined")

    if cfg["condition"] == "rotated":
        Xtr, Xte = rotate_block(Xtr, Xte, mask, cfg["rotation_seed"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg["model"] == "strong_classical":
            proba, info = strong_classical(Xtr, ytr, Xte, K, cfg["model_cfgs"], cfg["split_seed"])
        else:
            proba, info = fit_model(cfg["model"], Xtr, ytr, Xte, K,
                                    cfg["model_cfgs"][cfg["model"]], cfg["split_seed"])
    m = evaluate(proba, yte, K, None)
    return {**m, "dataset_id": cfg["dataset_id"], "name": name,
            "n_numeric": int(mask.sum()), "d_total": int(Xtr.shape[1]),
            "n_train": len(Xtr), "n_test": len(Xte)}


def build_cells(m5cfg, mc, dsets):
    return [
        {"milestone": "m9", "dataset_id": int(d), "condition": cond, "model": model,
         "split_seed": 0, "rotation_seed": 10_000 + int(d),
         "m5": m5cfg, "model_cfgs": mc}
        for d in dsets
        for cond in ("original", "rotated")
        for model in ("tabpfn", "strong_classical")
    ]


def main() -> None:
    env.preflight()
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for n in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[n] = {**mc[n], **TUNING}

    real = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
    dsets = sorted(real.dataset_id.tolist())
    cells = build_cells(m5cfg, mc, dsets)
    # Cheap (small-n) cells first so an early stop still covers many datasets.
    cells.sort(key=lambda c: (real.set_index("dataset_id").loc[c["dataset_id"], "n_eval_train"],
                              c["dataset_id"], c["condition"], c["model"]))
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"[m9] {len(cells)} cells over {len(dsets)} datasets "
          f"(2 conditions x 2 models), n_jobs={n_jobs}")
    runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)
    print("[m9] done")


if __name__ == "__main__":
    main()

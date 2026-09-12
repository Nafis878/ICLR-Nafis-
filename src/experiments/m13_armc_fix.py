"""Arm-C confound fix: is the code-permutation effect a MODEL property or a
PREPROCESSING artefact? (pre-registered 18552a4)

M10 arm C permuted categorical codes and found TabPFN degraded (17/22, p=0.0138).
But the wrapper never passed `categorical_features_indices`, so TabPFN saw ordinal
codes as ordered numbers. That makes the original result a statement about a common
preprocessing DEFAULT, not about TabPFN's prior.

This runs the missing half of a 2x2:

                        codes original      codes permuted
    undeclared          cached (m10C)       cached (m10C)
    declared            HERE                HERE

If the effect persists when categoricals are declared, it is a property of the model.
If it vanishes, the original arm C measured only the cost of not declaring them -- a
sharp preprocessing warning, and the earlier claim gets restated. Neither outcome was
predicted in advance; both are reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import warnings  # noqa: E402

import numpy as np  # noqa: E402

import runner  # noqa: E402
from experiments.m10_robustness import excluded_ids, permute_codes  # noqa: E402
from experiments.m9_rotation_transfer import numeric_mask  # noqa: E402
from models.wrappers import fit_model  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402
from stats.features import CATEGORICAL_MAX_UNIQUE  # noqa: E402


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    pol = dict(cfg["m5"]["eval_policy"])
    pol["max_train_rows"], pol["max_test_rows"] = 1500, 1500
    Xtr, ytr, Xte, yte, _ = prepare_split(X, y, pol, cfg["split_seed"])
    kk = int(len(np.unique(y)))

    num = numeric_mask(Xtr)
    cat = ~num
    keep = np.zeros_like(cat)
    for j in np.where(cat)[0]:
        u = np.unique(Xtr[:, j][np.isfinite(Xtr[:, j])])
        keep[j] = 2 <= len(u) <= CATEGORICAL_MAX_UNIQUE
    if keep.sum() < 2:
        raise ValueError(f"only {int(keep.sum())} categorical columns; permutation undefined")

    if cfg["condition"] == "perturbed":
        Xtr, Xte = permute_codes(Xtr, Xte, keep, cfg["perturb_seed"])

    mcfg = dict(cfg["model_cfgs"]["tabpfn"])
    if cfg["declared"]:
        mcfg["categorical_features_indices"] = np.where(keep)[0].tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba, info = fit_model("tabpfn", Xtr, ytr, Xte, kk, mcfg, cfg["split_seed"])
    return {**evaluate(proba, yte, kk, None), "dataset_id": cfg["dataset_id"],
            "name": name, "declared": cfg["declared"], "n_categorical": int(keep.sum()),
            "declared_ok": bool(info.get("categoricals_declared"))}


def main() -> None:
    env.preflight()
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    mc = runner.load_config("configs/m2_baselines.yaml")["models"]
    ids = sorted(d for d, k in excluded_ids().items() if k < 2)
    cells = [{"milestone": "m13armc", "dataset_id": int(d), "condition": c,
              "declared": True, "split_seed": 0, "perturb_seed": 20_000 + int(d),
              "m5": m5cfg, "model_cfgs": mc}
             for d in ids for c in ("original", "perturbed")]
    print(f"[armc] {len(cells)} cells over {len(ids)} datasets (declared half of the 2x2)")
    runner.run_many(cells, run_one, n_jobs=4, verbose=5)
    print("[armc] done")


if __name__ == "__main__":
    main()

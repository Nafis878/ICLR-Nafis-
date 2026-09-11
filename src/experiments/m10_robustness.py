"""M10 -- three robustness arms (pre-registered at efcbabc).

A  scale        rotation at n_train in {500, 1500, 4000}: does the cap matter?
B  low-dim      the 11 datasets excluded only for having 2-3 numeric columns
C  categorical  the 23 datasets with 0-1 numeric columns, perturbed by PERMUTING
                category codes -- a bijection, so information and Bayes risk are
                preserved exactly, which makes it the categorical analogue of an
                orthogonal rotation. It removes only the ordinality that an
                ordinal encoding imposes by accident.
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
from experiments.m9_rotation_transfer import TUNING, numeric_mask, rotate_block  # noqa: E402
from models.wrappers import fit_model, strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402
from stats.features import CATEGORICAL_MAX_UNIQUE  # noqa: E402

SCALE_LEVELS = [500, 1500, 4000]


def permute_codes(Xtr, Xte, mask, seed):
    """Relabel each categorical column by a random bijection on its codes.

    Information preserving: a bijection on labels cannot change p(y | x), so the
    Bayes risk is identical. Only the arbitrary ORDER of the integer codes moves.
    The mapping is learned on the union of train/test values so that no category
    seen at test time is left unmapped, and the SAME mapping is applied to both.
    """
    Xtr, Xte = Xtr.copy(), Xte.copy()
    rng = np.random.default_rng(seed)
    for j in np.where(mask)[0]:
        vals = np.unique(np.concatenate([Xtr[:, j], Xte[:, j]]))
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        perm = rng.permutation(vals)
        lut = {float(v): float(p) for v, p in zip(vals, perm)}
        for X in (Xtr, Xte):
            col = X[:, j]
            out = np.array([lut.get(float(v), v) if np.isfinite(v) else v for v in col])
            X[:, j] = out
    return Xtr, Xte


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    pol = dict(cfg["m5"]["eval_policy"])
    pol["max_train_rows"] = cfg.get("n_train_cap", 1500)
    pol["max_test_rows"] = 1500
    Xtr, ytr, Xte, yte, _ = prepare_split(X, y, pol, cfg["split_seed"])
    K = int(len(np.unique(y)))

    num = numeric_mask(Xtr)
    arm = cfg["arm"]
    if arm in ("A", "B"):
        if num.sum() < cfg["min_numeric"]:
            raise ValueError(f"only {int(num.sum())} numeric columns; rotation undefined")
        if cfg["condition"] == "perturbed":
            Xtr, Xte = rotate_block(Xtr, Xte, num, cfg["perturb_seed"])
    else:  # arm C -- categorical code permutation
        cat = ~num
        # a column is only meaningfully categorical if it has >=2 and few levels
        keep = np.zeros_like(cat)
        for j in np.where(cat)[0]:
            u = np.unique(Xtr[:, j][np.isfinite(Xtr[:, j])])
            keep[j] = 2 <= len(u) <= CATEGORICAL_MAX_UNIQUE
        if keep.sum() < 2:
            raise ValueError(f"only {int(keep.sum())} categorical columns; permutation undefined")
        if cfg["condition"] == "perturbed":
            Xtr, Xte = permute_codes(Xtr, Xte, keep, cfg["perturb_seed"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg["model"] == "strong_classical":
            proba, info = strong_classical(Xtr, ytr, Xte, K, cfg["model_cfgs"], cfg["split_seed"])
        else:
            proba, info = fit_model(cfg["model"], Xtr, ytr, Xte, K,
                                    cfg["model_cfgs"][cfg["model"]], cfg["split_seed"])
    m = evaluate(proba, yte, K, None)
    return {**m, "dataset_id": cfg["dataset_id"], "name": name, "arm": arm,
            "n_numeric": int(num.sum()), "n_train": len(Xtr), "n_test": len(Xte)}


def all_stats() -> pd.DataFrame:
    a = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
    b = pd.read_parquet("results/m5rep_real_statistics.parquet")
    return pd.concat([a, b], ignore_index=True).drop_duplicates("dataset_id")


def excluded_ids() -> dict[int, int]:
    """dataset_id -> numeric-column count, for datasets M9/M9REP could not use."""
    import glob
    import json

    out = {}
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") in ("m9", "m9rep") and r.get("status") != "ok":
            e = str(r.get("error"))
            if "numeric columns" in e:
                out[c["dataset_id"]] = int(e.split("only ")[1].split(" numeric")[0])
    return out


def main() -> None:
    env.preflight()
    arm = sys.argv[1]
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for n in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[n] = {**mc[n], **TUNING}
    stats = all_stats()
    exc = excluded_ids()

    if arm == "A":
        done = pd.read_csv("results/m9_pooled.csv").dataset_id.tolist()
        elig = stats[stats.dataset_id.isin(done)].nlargest(20, "n_eval_train")
        cells = [{"milestone": "m10A", "arm": "A", "dataset_id": int(d), "n_train_cap": n,
                  "condition": c, "model": "tabpfn", "split_seed": 0,
                  "perturb_seed": 10_000 + int(d), "min_numeric": 4,
                  "m5": m5cfg, "model_cfgs": mc}
                 for d in elig.dataset_id for n in SCALE_LEVELS
                 for c in ("original", "perturbed")]
    elif arm == "B":
        ids = [d for d, k in exc.items() if k >= 2]
        cells = [{"milestone": "m10B", "arm": "B", "dataset_id": int(d), "n_train_cap": 1500,
                  "condition": c, "model": m, "split_seed": 0,
                  "perturb_seed": 10_000 + int(d), "min_numeric": 2,
                  "m5": m5cfg, "model_cfgs": mc}
                 for d in ids for c in ("original", "perturbed")
                 for m in ("tabpfn", "strong_classical")]
    else:
        ids = [d for d, k in exc.items() if k < 2]
        cells = [{"milestone": "m10C", "arm": "C", "dataset_id": int(d), "n_train_cap": 1500,
                  "condition": c, "model": m, "split_seed": 0,
                  "perturb_seed": 20_000 + int(d), "min_numeric": 0,
                  "m5": m5cfg, "model_cfgs": mc}
                 for d in ids for c in ("original", "perturbed")
                 for m in ("tabpfn", "strong_classical")]

    print(f"[m10{arm}] {len(cells)} cells, n_jobs={n_jobs}")
    runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)
    print(f"[m10{arm}] done")


if __name__ == "__main__":
    main()

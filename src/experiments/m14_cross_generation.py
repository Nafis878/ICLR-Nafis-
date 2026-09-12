"""M14 -- cross-generation rotation probe: TabPFN v1 (ICLR 2023) vs v2 (Nature 2025).

Pre-registered at d2ef887. Identical paired design to M9; only the model changes.

v1's stated limits are far tighter than v2's (<=1000 train rows, <=100 features,
<=10 classes), so the sample is restricted to datasets inside THOSE limits and
n_train is capped at 1000 for both arms. The comparison that matters is the paired
original-vs-rotated delta WITHIN each generation, never a raw log-loss comparison
across models.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402
import subprocess  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from experiments.m9_rotation_transfer import numeric_mask, rotate_block  # noqa: E402

V1_PY = str(Path(".venv-tabpfn1/Scripts/python.exe").resolve())
WORKER = str(Path(__file__).resolve().parent / "m14_v1_worker.py")
MAX_TRAIN, MAX_TEST = 1000, 1000     # v1's stated ceiling
MAX_D, MAX_CLASSES = 100, 10         # v1's stated limits


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    pol = dict(cfg["m5"]["eval_policy"])
    pol["max_train_rows"], pol["max_test_rows"] = MAX_TRAIN, MAX_TEST
    pol["max_features"] = MAX_D
    Xtr, ytr, Xte, yte, _ = prepare_split(X, y, pol, cfg["split_seed"])
    k = int(len(np.unique(y)))
    if k > MAX_CLASSES:
        raise ValueError(f"{k} classes exceeds TabPFN v1's limit of {MAX_CLASSES}")

    mask = numeric_mask(Xtr)
    if mask.sum() < 4:
        raise ValueError(f"only {int(mask.sum())} numeric columns; rotation undefined")
    if cfg["condition"] == "rotated":
        Xtr, Xte = rotate_block(Xtr, Xte, mask, cfg["rotation_seed"])

    payload = json.dumps({"Xtr": Xtr.tolist(), "ytr": ytr.tolist(),
                          "Xte": Xte.tolist(), "yte": yte.tolist(), "n_classes": k})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = subprocess.run([V1_PY, "-u", WORKER], input=payload,
                              capture_output=True, text=True, timeout=cfg["timeout_s"])
    if proc.returncode != 0:
        raise RuntimeError(f"v1 worker failed: {(proc.stderr or '')[-400:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return {**out, "dataset_id": cfg["dataset_id"], "name": name,
            "n_numeric": int(mask.sum()), "generation": "v1"}


def main() -> None:
    env.preflight()
    assert Path(V1_PY).exists(), "TabPFN v1 venv missing"
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")

    a = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
    b = pd.read_parquet("results/m5rep_real_statistics.parquet")
    d = pd.concat([a, b], ignore_index=True).drop_duplicates("dataset_id")
    d["numeric"] = (d.d_eval * (1 - d.stat_categorical_fraction)).round()
    cand = d[(d.numeric >= 4) & (d.n_classes <= MAX_CLASSES) & (d.d_eval <= MAX_D)]
    cand = cand.nsmallest(30, "d_eval")
    print(f"[m14] {len(cand)} datasets inside TabPFN v1's stated limits "
          f"(<={MAX_TRAIN} rows, <={MAX_D} features, <={MAX_CLASSES} classes)")

    cells = [{"milestone": "m14v1", "dataset_id": int(r.dataset_id), "condition": c,
              "split_seed": 0, "rotation_seed": 10_000 + int(r.dataset_id),
              "timeout_s": 1800, "m5": m5cfg}
             for r in cand.itertuples() for c in ("original", "rotated")]
    cells.sort(key=lambda c: (c["dataset_id"], c["condition"]))
    print(f"[m14] {len(cells)} cells")
    runner.run_many(cells, run_one, n_jobs=3, verbose=5)
    print("[m14] done")


if __name__ == "__main__":
    main()

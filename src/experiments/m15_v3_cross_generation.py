"""M15 -- cross-generation rotation probe: TabPFN-3 (2026) vs v2 (Nature 2025).

Identical paired design to M9 and M14; only the model changes.

RUN THIS ONLY AFTER accepting the TABPFN-3 Non-Commercial License yourself.
See scripts/TABPFN3_HANDOFF.md.

Limits are set to v2's so the v2 <-> v3 comparison is on matched data. The
comparison that matters is the paired original-vs-rotated delta WITHIN each
generation, never a raw log-loss comparison across models.
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

V3_PY = str(Path(".venv-tabpfn3/Scripts/python.exe").resolve())
WORKER = str(Path(__file__).resolve().parent / "m15_v3_worker.py")
MAX_TRAIN, MAX_TEST = 1500, 1500     # matches the M9 v2 probe
MAX_D, MAX_CLASSES = 500, 10         # matches the M9 v2 probe


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
        raise ValueError(f"{k} classes exceeds TabPFN's limit of {MAX_CLASSES}")

    mask = numeric_mask(Xtr)
    if mask.sum() < 4:
        raise ValueError(f"only {int(mask.sum())} numeric columns; rotation undefined")
    if cfg["condition"] == "rotated":
        Xtr, Xte = rotate_block(Xtr, Xte, mask, cfg["rotation_seed"])

    payload = json.dumps({"Xtr": Xtr.tolist(), "ytr": ytr.tolist(),
                          "Xte": Xte.tolist(), "yte": yte.tolist(), "n_classes": k})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = subprocess.run([V3_PY, "-u", WORKER], input=payload,
                              capture_output=True, text=True, timeout=cfg["timeout_s"])
    if proc.returncode != 0:
        raise RuntimeError(f"v3 worker failed: {(proc.stderr or '')[-400:]}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return {**out, "dataset_id": cfg["dataset_id"], "name": name,
            "n_numeric": int(mask.sum()), "generation": "v3"}


def main() -> None:
    env.preflight()
    assert Path(V3_PY).exists(), (
        "TabPFN-3 venv missing -- see scripts/TABPFN3_HANDOFF.md")
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")

    a = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
    b = pd.read_parquet("results/m5rep_real_statistics.parquet")
    d = pd.concat([a, b], ignore_index=True).drop_duplicates("dataset_id")
    d["numeric"] = (d.d_eval * (1 - d.stat_categorical_fraction)).round()
    cand = d[(d.numeric >= 4) & (d.n_classes <= MAX_CLASSES) & (d.d_eval <= MAX_D)]
    cand = cand.nsmallest(30, "d_eval")
    print(f"[m15] {len(cand)} datasets inside the M9 v2 probe's limits "
          f"(<={MAX_TRAIN} rows, <={MAX_D} features, <={MAX_CLASSES} classes)")

    cells = [{"milestone": "m15v3", "dataset_id": int(r.dataset_id), "condition": c,
              "split_seed": 0, "rotation_seed": 10_000 + int(r.dataset_id),
              "timeout_s": 1800, "m5": m5cfg}
             for r in cand.itertuples() for c in ("original", "rotated")]
    cells.sort(key=lambda c: (c["dataset_id"], c["condition"]))
    print(f"[m15] {len(cells)} cells")
    runner.run_many(cells, run_one, n_jobs=3, verbose=5)
    print("[m15] done")


if __name__ == "__main__":
    main()

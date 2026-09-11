"""M9-REP -- independent replication on the 91 datasets no model has ever seen.

Pre-registered at commit 4b14eb6. Two stages, both reusing the primary code paths
unchanged so the replication is procedurally identical:
  stage "stats"  -- eval-policy statistics (gives rotation_alignment for H3-REP)
  stage "run"    -- the paired original/rotated cells

The 91 datasets were dropped at M5 by a seeded budget sample taken before any
outcome existed, so they are a clean independent sample rather than an extension
of the primary test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from experiments.m6b_amend import run_one as stats_run_one  # noqa: E402
from experiments.m9_rotation_transfer import TUNING, run_one as m9_run_one  # noqa: E402


def replication_ids(cfg) -> list[int]:
    from experiments.m5_real_datasets import resolve_suites

    suites, _ = resolve_suites(cfg)
    allids = sorted({int(d) for v in suites.values() for d in v})
    used = set(pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
               .dataset_id.tolist())
    return [d for d in allids if d not in used]


def main() -> None:
    env.preflight()
    stage = sys.argv[1] if len(sys.argv) > 1 else "stats"
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    ids = replication_ids(m5cfg)
    print(f"[m9rep] replication pool: {len(ids)} datasets never modelled")

    if stage == "stats":
        cells = [{"milestone": "m5rep", "dataset_id": int(d), "m5": m5cfg} for d in ids]
        rows = runner.run_many(cells, stats_run_one, n_jobs=n_jobs, verbose=5)
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"[m9rep] statistics: {len(ok)} ok / {len(rows)} attempted")
        recs = []
        for r in ok:
            rec = {k: r[k] for k in ("dataset_id", "name", "n_full", "d_full",
                                     "n_classes", "n_eval_train", "d_eval", "in_range")}
            rec.update({f"stat_{k}": v for k, v in r["stats"].items()})
            recs.append(rec)
        df = pd.DataFrame(recs).sort_values("dataset_id")
        df.to_parquet("results/m5rep_real_statistics.parquet", index=False)
        print(f"[m9rep] wrote results/m5rep_real_statistics.parquet ({len(df)} datasets)")
        return

    stats = pd.read_parquet("results/m5rep_real_statistics.parquet")
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for n in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[n] = {**mc[n], **TUNING}

    dsets = sorted(stats.dataset_id.tolist())
    cells = [
        {"milestone": "m9rep", "dataset_id": int(d), "condition": cond, "model": model,
         "split_seed": 0, "rotation_seed": 10_000 + int(d), "m5": m5cfg, "model_cfgs": mc}
        for d in dsets for cond in ("original", "rotated")
        for model in ("tabpfn", "strong_classical")
    ]
    size = stats.set_index("dataset_id").n_eval_train.to_dict()
    cells.sort(key=lambda c: (size.get(c["dataset_id"], 0), c["dataset_id"],
                              c["condition"], c["model"]))
    print(f"[m9rep] {len(cells)} cells over {len(dsets)} datasets, n_jobs={n_jobs}")
    runner.run_many(cells, m9_run_one, n_jobs=n_jobs, verbose=5)
    print("[m9rep] done")


if __name__ == "__main__":
    main()

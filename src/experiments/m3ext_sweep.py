"""M3-EXT -- extend the prior recovery sweep to n_train in [1000, 4000].

Reuses the M3 machinery unchanged so the two sweeps pool cleanly; only the
config differs. See configs/m3ext_prior_sweep.yaml for why this exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import numpy as np  # noqa: E402

import runner  # noqa: E402
from experiments.m3_sweep import build_cells, run_one  # noqa: E402


def main() -> None:
    env.preflight()
    cfg = runner.load_config("configs/m3ext_prior_sweep.yaml")
    base = runner.load_config("configs/m2_baselines.yaml")
    models_cfg = base["models"]
    for name in ("lightgbm", "random_forest", "knn", "mlp"):
        models_cfg[name] = {
            **models_cfg[name],
            "n_random_configs": cfg["tuning"]["n_random_configs"],
            "cv_folds": cfg["tuning"]["cv_folds"],
        }
    cfg["models_cfg"] = models_cfg

    cells = build_cells(cfg)
    for c in cells:
        c["milestone"] = "m3ext"          # distinct cache namespace
    np.random.default_rng(cfg["seed"] + 1).shuffle(cells)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"[m3ext] {len(cells)} cells, n_train in {cfg['n_train_range']}, n_jobs={n_jobs}")
    runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)
    print("[m3ext] done")


if __name__ == "__main__":
    main()

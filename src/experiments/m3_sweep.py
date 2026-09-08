"""M3 -- prior recovery sweep (spec section 5/M3).

Latin hypercube over the DGP parameter space, >=5 seeds per cell, one tidy
parquet row per (family, params, seed, model, metric). The per-seed distribution
is kept, not just the mean, so error bars are available downstream.
"""
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
from stats.features import compute_statistics  # noqa: E402

INT_PARAMS = {"d", "n_classes", "depth", "max_parents", "n_directions", "n_nodes"}


def latin_hypercube(n_points: int, n_dims: int, rng) -> np.ndarray:
    """Centred LHS in the unit cube: one sample per stratum on every axis."""
    cut = np.linspace(0, 1, n_points + 1)
    u = rng.random((n_points, n_dims))
    pts = cut[:n_points, None] + u * (1.0 / n_points)
    for j in range(n_dims):
        pts[:, j] = rng.permutation(pts[:, j])
    return pts


def _scale(u: float, lo, hi, name: str):
    if name in INT_PARAMS:
        return int(round(lo + u * (hi - lo)))
    return float(lo + u * (hi - lo))


def build_cells(cfg: dict) -> list[dict]:
    """One LHS design per family, over that family's params + the modifier axes."""
    rng = np.random.default_rng(cfg["seed"])
    fams = cfg["families"]
    mod_ranges = cfg["modifier_ranges"]
    per_family = max(1, cfg["n_cells"] // len(fams))
    cells = []

    for fam, ranges in fams.items():
        pnames = list(ranges.keys())
        mnames = [(m, p) for m, ps in mod_ranges.items() for p in ps]
        dims = len(pnames) + len(mnames) + 1  # +1 for n_train
        design = latin_hypercube(per_family, dims, rng)

        for row in design:
            params = {
                name: _scale(row[i], ranges[name][0], ranges[name][1], name)
                for i, name in enumerate(pnames)
            }
            params["struct_seed"] = int(rng.integers(0, 10_000))
            mods: dict[str, dict] = {}
            for j, (mname, pname) in enumerate(mnames):
                lo, hi = mod_ranges[mname][pname]
                mods.setdefault(mname, {})[pname] = float(
                    _scale(row[len(pnames) + j], lo, hi, pname)
                )
            lo_n, hi_n = cfg["n_train_range"]
            n_train = int(round(lo_n + row[-1] * (hi_n - lo_n)))

            for seed in range(cfg["n_seeds"]):
                for model in cfg["models"]:
                    cells.append(
                        {
                            "milestone": "m3",
                            "family": fam,
                            "params": params,
                            "modifiers": mods,
                            "n_train": n_train,
                            "n_test": cfg["n_test"],
                            "seed": seed,
                            "model": model,
                            "model_cfgs": cfg["models_cfg"],
                        }
                    )
    return cells


def run_one(cfg: dict) -> dict:
    X, y, bayes, exact = draw(
        cfg["family"], cfg["params"], cfg.get("modifiers"),
        cfg["n_train"] + cfg["n_test"], cfg["seed"],
    )
    ntr = cfg["n_train"]
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    K = int(cfg["params"]["n_classes"])
    if len(np.unique(ytr)) < 2:
        raise ValueError("degenerate draw: training split has a single class")

    if cfg["model"] == "strong_classical":
        proba, info = strong_classical(Xtr, ytr, Xte, K, cfg["model_cfgs"], cfg["seed"])
    else:
        proba, info = fit_model(
            cfg["model"], Xtr, ytr, Xte, K, cfg["model_cfgs"][cfg["model"]], cfg["seed"]
        )
    metrics = evaluate(proba, yte, K, bayes[ntr:] if exact else None)

    # Statistics are computed on the TRAINING split only: on a real dataset that
    # is all the score would be allowed to see before deciding to run TabPFN.
    stats = compute_statistics(Xtr, ytr)
    return {**metrics, "exact_bayes": exact, "info": info, "stats": stats}


def main() -> None:
    env.preflight()
    cfg = runner.load_config("configs/m3_prior_sweep.yaml")
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
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"[m3] {len(cells)} cells "
          f"({len(cells) // (cfg['n_seeds'] * len(cfg['models']))} LHS points "
          f"x {cfg['n_seeds']} seeds x {len(cfg['models'])} models), n_jobs={n_jobs}")
    runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)
    print("[m3] done")


if __name__ == "__main__":
    main()

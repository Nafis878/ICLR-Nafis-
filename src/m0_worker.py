"""One M0 timing cell, run in its own process so the parent can impose a hard
wall-clock timeout (Windows has no signal.alarm). Reads a JSON config on stdin,
writes a JSON result to stdout.

Only fit+predict is timed: interpreter and checkpoint load are excluded so the
cost model describes the science, not the harness.
"""
from __future__ import annotations

import env  # noqa: F401  -- first import: CPU + single-threaded BLAS + no telemetry

import json
import sys
import time


def make_data(n_train: int, n_test: int, d: int, n_classes: int, seed: int):
    from sklearn.datasets import make_classification

    # make_classification requires n_classes * n_clusters_per_class <= 2**n_informative.
    import math

    n_informative = max(2, math.ceil(math.log2(n_classes)), min(d, d // 2))
    n_informative = min(n_informative, d)
    X, y = make_classification(
        n_samples=n_train + n_test,
        n_features=d,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def time_tabpfn(cfg, Xtr, ytr, Xte):
    from tabpfn import TabPFNClassifier

    mp = cfg["model_params"]
    # Construct + load checkpoint OUTSIDE the timed region.
    clf = TabPFNClassifier(
        device=mp.get("device", "cpu"),
        n_estimators=mp.get("n_estimators", 4),
        random_state=cfg["seed"],
    )
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    return time.perf_counter() - t0, proba


def time_lightgbm(cfg, Xtr, ytr, Xte):
    """Tuned LightGBM: random search over a fixed, recorded budget."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_score

    mp = cfg["model_params"]
    space = mp["search_space"]
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.perf_counter()
    best, best_score = None, -np.inf
    n_folds = min(mp["cv_folds"], int(np.bincount(ytr).min()))
    for _ in range(mp["n_random_configs"]):
        params = {k: rng.choice(v).item() for k, v in space.items()}
        clf = LGBMClassifier(**params, verbose=-1, n_jobs=1, random_state=cfg["seed"])
        if n_folds >= 2:
            s = cross_val_score(clf, Xtr, ytr, cv=n_folds, scoring="neg_log_loss", n_jobs=1).mean()
        else:
            s = 0.0
        if s > best_score:
            best, best_score = params, s
    clf = LGBMClassifier(**best, verbose=-1, n_jobs=1, random_state=cfg["seed"])
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    return time.perf_counter() - t0, proba


def main() -> None:
    cfg = json.loads(sys.stdin.read())
    import psutil

    proc = psutil.Process()
    rss0 = proc.memory_info().rss

    Xtr, ytr, Xte, _ = make_data(
        cfg["n_train"], cfg["n_test"], cfg["d"], cfg["n_classes"], cfg["seed"]
    )
    fn = {"tabpfn": time_tabpfn, "lightgbm": time_lightgbm}[cfg["model"]]
    elapsed, proba = fn(cfg, Xtr, ytr, Xte)

    json.dump(
        {
            "status": "ok",
            "fit_predict_s": elapsed,
            "peak_rss_mb": (proc.memory_info().rss - rss0) / 1e6,
            "proba_shape": list(proba.shape),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()

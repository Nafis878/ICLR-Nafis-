"""One TabPFN-3 cell. Runs inside .venv-tabpfn3, reads JSON on stdin.

Kept out of the main venv because tabpfn 8.x and 2.2.1 cannot coexist in one
process. Identical contract to m14_v1_worker.py, so the analysis is shared.
"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np  # noqa: E402


def main():
    cfg = json.loads(sys.stdin.read())
    Xtr = np.array(cfg["Xtr"], dtype=float)
    ytr = np.array(cfg["ytr"], dtype=int)
    Xte = np.array(cfg["Xte"], dtype=float)
    yte = np.array(cfg["yte"], dtype=int)

    import torch
    from tabpfn import TabPFNClassifier

    assert not torch.cuda.is_available(), "CUDA visible; this project is CPU-only"
    t0 = time.perf_counter()
    clf = TabPFNClassifier(device="cpu", random_state=cfg.get("seed", 0))
    clf.fit(Xtr, ytr)
    proba = np.asarray(clf.predict_proba(Xte), dtype=float)
    elapsed = time.perf_counter() - t0

    k = int(cfg["n_classes"])
    if proba.shape[1] < k:
        proba = np.hstack([proba, np.full((len(proba), k - proba.shape[1]), 1e-9)])
    proba = np.clip(proba, 1e-12, None)
    proba = proba / proba.sum(1, keepdims=True)
    ll = float(-np.log(proba[np.arange(len(yte)), yte]).mean())
    acc = float((proba.argmax(1) == yte).mean())
    json.dump({"status": "ok", "logloss": ll, "accuracy": acc,
               "fit_predict_s": elapsed, "n_train": int(len(Xtr)),
               "tabpfn_version": getattr(__import__("tabpfn"), "__version__", "?")},
              sys.stdout)


if __name__ == "__main__":
    main()

"""One TabPFN v1 cell. Runs inside .venv-tabpfn1, reads JSON on stdin.

Kept separate from the v2 code path because the two installs pin different
tabpfn versions and must never share a process.
"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    # v1 cannot ingest NaN; median-impute exactly as the v2 wrapper does.
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)

    t0 = time.perf_counter()
    clf = TabPFNClassifier(device="cpu", N_ensemble_configurations=4)
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
               "fit_predict_s": elapsed, "n_train": int(len(Xtr))}, sys.stdout)


if __name__ == "__main__":
    main()

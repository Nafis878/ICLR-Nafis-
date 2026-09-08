"""Regret metric (spec section 5/M2).

PRIMARY: Bayes regret in log-loss, regret = logloss(model) - logloss(bayes).
Log-loss because TabPFN is a Bayesian predictor and calibration should be the
first thing to degrade when a dataset leaves the prior. Accuracy and AUC are
recorded as secondary only.

An effect smaller than its own error bar is FLAGGED, never quietly reported
(spec section 3 and section 6).
"""
from __future__ import annotations

import numpy as np


def _safe(proba: np.ndarray, n_classes: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    if proba.shape[1] < n_classes:  # a model never saw some class in train
        pad = np.full((len(proba), n_classes - proba.shape[1]), 1e-9)
        proba = np.hstack([proba, pad])
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(1, keepdims=True)


def log_loss_per_row(proba: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    p = _safe(proba, n_classes)
    return -np.log(p[np.arange(len(y)), y])


def evaluate(proba, y, n_classes: int, bayes_proba=None) -> dict:
    """All metrics for one (model, dataset) pair, with regret error bars."""
    from sklearn.metrics import accuracy_score, roc_auc_score

    ll_rows = log_loss_per_row(proba, y, n_classes)
    out = {
        "logloss": float(ll_rows.mean()),
        "logloss_se": float(ll_rows.std(ddof=1) / np.sqrt(len(ll_rows))),
        "accuracy": float(accuracy_score(y, _safe(proba, n_classes).argmax(1))),
        "n_eval": int(len(y)),
    }
    try:
        p = _safe(proba, n_classes)
        out["auc"] = float(
            roc_auc_score(y, p[:, 1]) if n_classes == 2
            else roc_auc_score(y, p, multi_class="ovr", average="macro")
        )
    except Exception:
        out["auc"] = None

    if bayes_proba is not None:
        b_rows = log_loss_per_row(bayes_proba, y, n_classes)
        diff = ll_rows - b_rows
        regret = float(diff.mean())
        # Paired SE: the same rows are used for both, so pair before differencing.
        regret_se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        out.update(
            {
                "bayes_logloss": float(b_rows.mean()),
                "regret": regret,
                "regret_se": regret_se,
                # A regret indistinguishable from zero at 2 SE is not a finding.
                "regret_significant": bool(abs(regret) > 2 * regret_se),
            }
        )
    return out

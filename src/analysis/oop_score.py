"""M4 -- the out-of-prior (OOP) score (spec section 5/M4).

g maps the OBSERVABLE statistic vector to predicted TabPFN Bayes regret. It is
fit on synthetic M3 data only; real datasets are never used to fit it.

Grouped cross-validation is essential here: the 5 seeds of one LHS point share
DGP parameters, so a random split would leak near-duplicate rows between train
and test and inflate CV performance. Folds are therefore split by LHS point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAT_PREFIX = "stat_"


def statistic_columns(df: pd.DataFrame, drop: list[str] | None = None) -> list[str]:
    drop = set(drop or [])
    return sorted(
        c for c in df.columns
        if c.startswith(STAT_PREFIX) and c not in drop and df[c].notna().any()
    )


def _models(seed: int):
    """Simple and interpretable, per spec: regularized linear, or a shallow GBDT."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))
        ),
        "gbdt_depth4": LGBMRegressor(
            max_depth=4, num_leaves=15, n_estimators=300, learning_rate=0.05,
            min_child_samples=20, verbose=-1, n_jobs=1, random_state=seed,
        ),
    }


def fit_oop_score(
    df: pd.DataFrame,
    target: str = "regret",
    group_col: str = "cell_id",
    seed: int = 0,
    drop_stats: list[str] | None = None,
) -> dict:
    """Fit g and report grouped-CV performance ON SYNTHETIC DATA ONLY.

    This is not yet evidence about real data -- that claim is only made at M7,
    against predictions frozen at M6.
    """
    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupKFold

    feats = statistic_columns(df, drop_stats)
    work = df.dropna(subset=[target]).copy()
    X = work[feats].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = work[target].to_numpy(dtype=float)
    groups = work[group_col].to_numpy()

    n_splits = int(min(5, len(np.unique(groups))))
    cv = GroupKFold(n_splits=n_splits)
    results = {}
    for name, mk in _models(seed).items():
        preds = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, groups):
            from sklearn.base import clone

            m = clone(mk).fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
        ok = np.isfinite(preds)
        ss_res = float(((y[ok] - preds[ok]) ** 2).sum())
        ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
        results[name] = {
            "cv_r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "cv_spearman": float(spearmanr(y[ok], preds[ok]).statistic),
            "cv_mae": float(np.abs(y[ok] - preds[ok]).mean()),
            "oof_pred": preds,
        }

    best = max(results, key=lambda k: results[k]["cv_spearman"])
    from sklearn.base import clone

    final = clone(_models(seed)[best]).fit(X, y)
    return {
        "features": feats,
        "model_name": best,
        "model": final,
        "cv": {k: {kk: vv for kk, vv in v.items() if kk != "oof_pred"} for k, v in results.items()},
        "oof_pred": results[best]["oof_pred"],
        "y": y,
        "groups": groups,
        "n_rows": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
    }


def feature_importances(fit: dict) -> pd.DataFrame:
    m, feats = fit["model"], fit["features"]
    if hasattr(m, "feature_importances_"):
        imp = np.asarray(m.feature_importances_, dtype=float)
    else:
        coef = m[-1].coef_ if hasattr(m, "__getitem__") else m.coef_
        imp = np.abs(np.ravel(coef))
    return (
        pd.DataFrame({"feature": feats, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def predict(fit: dict, stats_df: pd.DataFrame) -> np.ndarray:
    """Apply g to any frame carrying the same statistic columns."""
    X = stats_df.reindex(columns=fit["features"]).to_numpy(dtype=float)
    return fit["model"].predict(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))


def serialize(fit: dict) -> dict:
    """Exact, human-readable parameters of g for the pre-registration record."""
    m = fit["model"]
    out = {"model_name": fit["model_name"], "features": fit["features"]}
    if fit["model_name"] == "ridge":
        ridge = m[-1]
        scaler = m[0]
        out["coefficients"] = dict(zip(fit["features"], map(float, np.ravel(ridge.coef_))))
        out["intercept"] = float(ridge.intercept_)
        out["alpha"] = float(ridge.alpha_)
        out["scaler_mean"] = dict(zip(fit["features"], map(float, scaler.mean_)))
        out["scaler_scale"] = dict(zip(fit["features"], map(float, scaler.scale_)))
    else:
        out["importances"] = dict(
            zip(fit["features"], map(float, np.asarray(m.feature_importances_)))
        )
        out["booster_txt_sha256"] = _sha(m.booster_.model_to_string())
    return out


def _sha(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()

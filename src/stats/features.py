"""The fixed observable-statistic vector (spec section 5/M4).

THE TRAP THIS AVOIDS: the M3 regret surface is a function of latent DGP
parameters, which cannot be observed on a real dataset. Every statistic here is
computable from (X, y) alone, and the SAME function is applied to synthetic and
real data. Anything not computable on real data is dropped at M5 before freezing.

All randomness is seeded from a module constant so the vector is deterministic.
Row subsampling caps the O(n^2) neighbour statistics; the cap is part of the
definition, applied identically everywhere.
"""
from __future__ import annotations

import warnings

import numpy as np

STAT_SEED = 20260908
KNN_SUBSAMPLE = 2000
ROTATION_REPEATS = 5
LOW_MI_THRESHOLD = 0.01
SKEW_THRESHOLD = 1.0
KURT_THRESHOLD = 3.0
ROTATION_TREE_DEPTH = 4
CATEGORICAL_MAX_UNIQUE = 20


def _subsample(X, y, cap, seed):
    if len(X) <= cap:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=cap, replace=False)
    return X[idx], y[idx]


def _impute_standardize(X):
    """Median-impute then standardize. Constant columns get scale 1."""
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), med, X)
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def _knn_label_agreement(Xs, y, k, seed):
    """Fraction of the k nearest neighbours (excluding self) sharing the label."""
    from sklearn.neighbors import NearestNeighbors

    n = len(Xs)
    k_eff = int(min(k, n - 1))
    if k_eff < 1:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k_eff + 1, n_jobs=1).fit(Xs)
    _, idx = nn.kneighbors(Xs)
    neigh = y[idx[:, 1:]]  # drop self
    return float((neigh == y[:, None]).mean())


def _categorical_mask(X):
    """A column is treated as categorical if it has few distinct finite values."""
    out = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            out[j] = len(np.unique(col)) <= CATEGORICAL_MAX_UNIQUE
    return out


def _axis_aligned_score(Z, y, seed) -> float:
    """Cross-validated accuracy of a depth-limited AXIS-ALIGNED decision tree."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    counts = np.bincount(y)
    n_folds = int(min(3, counts[counts > 0].min()))
    if n_folds < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return float(
        cross_val_score(
            DecisionTreeClassifier(max_depth=ROTATION_TREE_DEPTH, random_state=seed),
            Z, y, cv=cv, n_jobs=1,
        ).mean()
    )


def _rotation_alignment(Xs, y, seed) -> dict:
    """Axis-aligned-learner accuracy in the given basis / mean under rotations.

    ORIGINAL statistic, operationalizing the rotation-invariance axis of
    Grinsztajn et al. Values above 1 mean the labels are unusually well aligned
    with the coordinate axes -- exactly the regime where axis-aligned splits win
    and a rotation-equivariant learner does not.

    WHY NOT kNN: the spec proposes kNN label agreement in the original basis
    divided by agreement under rotation. Euclidean kNN is EXACTLY rotation
    invariant -- an orthogonal map is an isometry, so it preserves every pairwise
    distance and therefore every neighbour set. Measured here, rotating X changed
    pairwise distances by 2.7e-15 and left the agreement bit-identical, making
    that ratio identically 1.0 and carrying no signal. L1 kNN is only technically
    non-invariant and moved the statistic by 1e-4. A depth-limited decision tree
    is genuinely axis-sensitive (0.869 -> 0.788 on the same data), so the probe is
    rebased on it. This is a correction to the spec, made before the M6 freeze.

    Defined on the standardized numeric block: rotation mixes columns, so the
    statistic is only meaningful for numeric features, and that restriction is
    part of the definition rather than an implementation detail.
    """
    from dgp.base import random_rotation

    d = Xs.shape[1]
    base = _axis_aligned_score(Xs, y, seed)
    if d < 2 or not np.isfinite(base):
        return {
            "rotation_alignment": 1.0,
            "rotation_alignment_sd": 0.0,
            "rotation_base_score": float(base) if np.isfinite(base) else np.nan,
        }
    rng = np.random.default_rng(seed + 31)
    vals = np.array(
        [_axis_aligned_score(Xs @ random_rotation(d, rng), y, seed) for _ in range(ROTATION_REPEATS)],
        dtype=float,
    )
    mean_rot = float(np.nanmean(vals))
    return {
        "rotation_alignment": float(base / max(mean_rot, 1e-9)),
        # Stability of the statistic across rotation draws, reported as required.
        "rotation_alignment_sd": float(np.nanstd(vals, ddof=1) / max(mean_rot, 1e-9)),
        "rotation_base_score": float(base),
    }


def compute_statistics(X, y, *, seed: int = STAT_SEED) -> dict:
    """The fixed statistic vector. Identical code path for synthetic and real."""
    from scipy.stats import kurtosis, skew
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    n, d = X.shape
    classes, counts = np.unique(y, return_counts=True)
    K = len(classes)
    p = counts / counts.sum()

    out: dict[str, float] = {
        "n": float(n),
        "d": float(d),
        "n_over_d": float(n / max(d, 1)),
        "log_n": float(np.log10(max(n, 1))),
        "log_d": float(np.log10(max(d, 1))),
        "n_classes": float(K),
        "class_entropy": float(-(p * np.log(np.clip(p, 1e-12, None))).sum()),
        "class_imbalance_ratio": float(p.max() / max(p.min(), 1e-12)),
        "missing_rate": float(np.isnan(X).mean()),
    }

    cat = _categorical_mask(X)
    out["categorical_fraction"] = float(cat.mean())
    max_card = 0
    for j in np.where(cat)[0]:
        col = X[:, j][np.isfinite(X[:, j])]
        max_card = max(max_card, len(np.unique(col)))
    out["max_categorical_cardinality"] = float(max_card)

    # ---- marginal shape ----------------------------------------------------
    med = np.nanmedian(X, axis=0, keepdims=True)
    Xf = np.where(np.isnan(X), med, X)
    Xf = np.where(np.isnan(Xf), 0.0, Xf)
    sk = np.nan_to_num(np.abs(skew(Xf, axis=0, bias=False)))
    ku = np.nan_to_num(kurtosis(Xf, axis=0, bias=False, fisher=True))
    out.update(
        {
            "skew_mean": float(sk.mean()),
            "skew_max": float(sk.max()),
            "skew_frac_high": float((sk > SKEW_THRESHOLD).mean()),
            "kurtosis_mean": float(ku.mean()),
            "kurtosis_max": float(ku.max()),
            "kurtosis_frac_high": float((ku > KURT_THRESHOLD).mean()),
        }
    )

    # ---- covariance geometry ----------------------------------------------
    Xs = _impute_standardize(X)
    C = np.atleast_2d(np.cov(Xs, rowvar=False))
    ev = np.clip(np.linalg.eigvalsh((C + C.T) / 2), 0, None)
    tot = ev.sum()
    if tot > 0:
        q = ev / tot
        eff_rank = float(np.exp(-(q * np.log(np.clip(q, 1e-12, None))).sum()))
    else:
        eff_rank = 1.0
    pos = ev[ev > 1e-10]
    out["cov_cond_number"] = float(
        np.log10(max(ev.max(), 1e-12) / (pos.min() if pos.size else 1e-12))
    )
    out["cov_effective_rank"] = eff_rank
    out["cov_effective_rank_ratio"] = float(eff_rank / max(d, 1))

    # ---- target dependence -------------------------------------------------
    Xsub, ysub = _subsample(Xs, y, KNN_SUBSAMPLE, seed)
    try:
        mi = mutual_info_classif(Xsub, ysub, random_state=seed, n_neighbors=3)
    except Exception:
        mi = np.zeros(d)
    out.update(
        {
            "mi_mean": float(np.mean(mi)),
            "mi_max": float(np.max(mi)) if d else 0.0,
            "mi_frac_low": float((mi < LOW_MI_THRESHOLD).mean()),
            "mi_concentration": float(np.max(mi) / max(np.mean(mi), 1e-9)) if d else 0.0,
        }
    )

    # ---- target irregularity ----------------------------------------------
    for k in (1, 5, 15):
        out[f"knn_agree_k{k}"] = _knn_label_agreement(Xsub, ysub, k, seed)
    # 1-NN disagreement is dominated by label noise rather than curvature.
    out["est_label_noise"] = float(1.0 - out["knn_agree_k1"])

    # ---- rotation alignment (original statistic) ---------------------------
    out.update(_rotation_alignment(Xsub, ysub, seed))

    # ---- structural probes: stump vs linear --------------------------------
    try:
        acc_stump = float(
            DecisionTreeClassifier(max_depth=1, random_state=seed).fit(Xsub, ysub).score(Xsub, ysub)
        )
    except Exception:
        acc_stump = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc_lin = float(
                LogisticRegression(max_iter=200).fit(Xsub, ysub).score(Xsub, ysub)
            )
    except Exception:
        acc_lin = float("nan")
    out["stump_acc"] = acc_stump
    out["linear_acc"] = acc_lin
    out["stump_minus_linear"] = float(acc_stump - acc_lin)
    out["majority_acc"] = float(p.max())
    out["linear_lift"] = float(acc_lin - p.max())

    return out


_STAT_NAMES: list[str] | None = None


def statistic_names() -> list[str]:
    global _STAT_NAMES
    if _STAT_NAMES is None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 4))
        y = (X[:, 0] > 0).astype(int)
        _STAT_NAMES = sorted(compute_statistics(X, y).keys())
    return _STAT_NAMES

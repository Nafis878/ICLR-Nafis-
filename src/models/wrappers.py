"""Unified model interface (spec section 5/M2).

Every model exposes fit_predict_proba(Xtr, ytr, Xte, n_classes, cfg, seed).

Baselines get a FIXED, RECORDED random-search tuning budget. Under-tuned GBDT
baselines are the single most common reviewer objection to tabular papers
(spec section 6), so the protocol is config-driven and reported, never implicit.
"""
from __future__ import annotations

import numpy as np


def _impute(Xtr, Xte):
    """Median imputation for models that cannot take NaN. TabPFN handles NaN natively."""
    if not (np.isnan(Xtr).any() or np.isnan(Xte).any()):
        return Xtr, Xte
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    return Xtr, Xte


def _random_search(estimator_fn, space, Xtr, ytr, n_configs, cv_folds, seed):
    """Random search on a stratified CV split, scored by neg log-loss."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr)
    n_folds = int(min(cv_folds, counts[counts > 0].min()))
    best, best_score = None, -np.inf
    for _ in range(n_configs):
        params = {k: rng.choice(v).item() for k, v in space.items()}
        try:
            if n_folds >= 2:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                s = cross_val_score(
                    estimator_fn(params, seed), Xtr, ytr, cv=cv,
                    scoring="neg_log_loss", n_jobs=1,
                ).mean()
            else:
                s = 0.0
        except Exception:
            continue
        if s > best_score:
            best, best_score = params, s
    if best is None:
        best = {k: v[0] for k, v in space.items()}
    return best, best_score


# --------------------------------------------------------------------------- #

def fit_tabpfn(Xtr, ytr, Xte, n_classes, cfg, seed):
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(
        device=cfg.get("device", "cpu"),
        n_estimators=cfg.get("n_estimators", 4),
        random_state=seed,
        ignore_pretraining_limits=cfg.get("ignore_pretraining_limits", True),
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "tabpfn"}


def fit_lightgbm(Xtr, ytr, Xte, n_classes, cfg, seed):
    from lightgbm import LGBMClassifier

    Xtr, Xte = _impute(Xtr, Xte)
    fn = lambda p, s: LGBMClassifier(**p, verbose=-1, n_jobs=1, random_state=s)  # noqa: E731
    best, score = _random_search(
        fn, cfg["search_space"], Xtr, ytr, cfg["n_random_configs"], cfg["cv_folds"], seed
    )
    clf = fn(best, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "lightgbm", "best_params": best, "cv_score": score}


def fit_random_forest(Xtr, ytr, Xte, n_classes, cfg, seed):
    from sklearn.ensemble import RandomForestClassifier

    Xtr, Xte = _impute(Xtr, Xte)
    fn = lambda p, s: RandomForestClassifier(**p, n_jobs=1, random_state=s)  # noqa: E731
    best, score = _random_search(
        fn, cfg["search_space"], Xtr, ytr, cfg["n_random_configs"], cfg["cv_folds"], seed
    )
    clf = fn(best, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "random_forest", "best_params": best, "cv_score": score}


def fit_knn(Xtr, ytr, Xte, n_classes, cfg, seed):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = _impute(Xtr, Xte)
    space = dict(cfg["search_space"])
    space["n_neighbors"] = [k for k in space["n_neighbors"] if k < len(Xtr)] or [1]
    fn = lambda p, s: make_pipeline(  # noqa: E731
        StandardScaler(), KNeighborsClassifier(**p, n_jobs=1)
    )
    best, score = _random_search(
        fn, space, Xtr, ytr, cfg["n_random_configs"], cfg["cv_folds"], seed
    )
    clf = fn(best, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "knn", "best_params": best, "cv_score": score}


def fit_mlp(Xtr, ytr, Xte, n_classes, cfg, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = _impute(Xtr, Xte)

    def fn(p, s):
        p = dict(p)
        p["hidden_layer_sizes"] = (int(p.pop("width")),) * int(p.pop("depth"))
        return make_pipeline(
            StandardScaler(), MLPClassifier(**p, random_state=s, max_iter=400)
        )

    best, score = _random_search(
        fn, cfg["search_space"], Xtr, ytr, cfg["n_random_configs"], cfg["cv_folds"], seed
    )
    clf = fn(best, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "mlp", "best_params": best, "cv_score": score}


def fit_xgboost(Xtr, ytr, Xte, n_classes, cfg, seed):
    """Gradient boosting, second family. Listed as a core dep in the spec and
    previously unused -- adding it widens the best-of ensemble rather than merely
    deepening one search."""
    from xgboost import XGBClassifier

    Xtr, Xte = _impute(Xtr, Xte)
    fn = lambda p, s: XGBClassifier(  # noqa: E731
        **p, n_jobs=1, random_state=s, tree_method="hist",
        verbosity=0, eval_metric="mlogloss",
    )
    best, score = _random_search(
        fn, cfg["search_space"], Xtr, ytr, cfg["n_random_configs"], cfg["cv_folds"], seed
    )
    clf = fn(best, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "xgboost", "best_params": best, "cv_score": score}


def fit_catboost(Xtr, ytr, Xte, n_classes, cfg, seed):
    """Ordered-boosting family; often the strongest single tabular learner."""
    from catboost import CatBoostClassifier

    Xtr, Xte = _impute(Xtr, Xte)
    fn = lambda p, s: CatBoostClassifier(  # noqa: E731
        **p, random_seed=s, verbose=0, allow_writing_files=False, thread_count=1,
    )
    best, score = _random_search(
        fn, cfg["search_space"], Xtr, ytr, cfg["n_random_configs"], cfg["cv_folds"], seed
    )
    clf = fn(best, seed).fit(Xtr, ytr)
    return clf.predict_proba(Xte), {"model": "catboost", "best_params": best, "cv_score": score}


MODELS = {
    "tabpfn": fit_tabpfn,
    "lightgbm": fit_lightgbm,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "random_forest": fit_random_forest,
    "knn": fit_knn,
    "mlp": fit_mlp,
}

CLASSICAL = ["lightgbm", "xgboost", "catboost", "random_forest", "knn", "mlp"]


def fit_model(name, Xtr, ytr, Xte, n_classes, cfg, seed):
    return MODELS[name](Xtr, ytr, Xte, n_classes, cfg, seed)


def strong_classical(Xtr, ytr, Xte, n_classes, cfgs, seed, val_frac=0.25):
    """best-of{tuned LightGBM, RF, kNN, MLP} selected on a held-out validation split.

    Selection uses validation log-loss only; the winner is then refit on the full
    training set. This is the reference on real data, where no Bayes oracle exists.
    """
    from sklearn.model_selection import train_test_split

    from oracles.metrics import log_loss_per_row

    counts = np.bincount(ytr, minlength=n_classes)
    stratify = ytr if counts[counts > 0].min() >= 2 else None
    Xa, Xv, ya, yv = train_test_split(
        Xtr, ytr, test_size=val_frac, random_state=seed, stratify=stratify
    )
    scores = {}
    for name in CLASSICAL:
        try:
            proba, _ = fit_model(name, Xa, ya, Xv, n_classes, cfgs[name], seed)
            scores[name] = float(log_loss_per_row(proba, yv, n_classes).mean())
        except Exception as exc:  # noqa: BLE001 -- recorded, not dropped
            scores[name] = float("inf")
            scores[f"{name}_error"] = str(exc)[:200]
    valid = {k: v for k, v in scores.items() if not k.endswith("_error")}
    winner = min(valid, key=valid.get)
    proba, info = fit_model(winner, Xtr, ytr, Xte, n_classes, cfgs[winner], seed)
    return proba, {"model": "strong_classical", "winner": winner, "val_scores": scores, **info}

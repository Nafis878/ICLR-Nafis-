"""Single place that turns a config dict into a (X, y, bayes_proba) draw."""
from __future__ import annotations

import numpy as np

from dgp.closed_form import GaussianMixture, LogisticLinear
from dgp.latent import GPSmooth, HighFrequency, PiecewiseConstant, RotatedPiecewiseConstant
from dgp.modifiers import REGISTRY as MOD_REGISTRY, apply_chain
from dgp.scm import SCMLatentConfounded, SCMPriorControl

FAMILIES = {
    c.name: c
    for c in [
        GaussianMixture, LogisticLinear, PiecewiseConstant,
        RotatedPiecewiseConstant, GPSmooth, HighFrequency, SCMPriorControl,
        SCMLatentConfounded,
    ]
}


def build_dgp(family: str, params: dict):
    return FAMILIES[family](**params)


def build_modifiers(spec: dict | None) -> list:
    if not spec:
        return []
    return [MOD_REGISTRY[name](**(kw or {})) for name, kw in spec.items()]


def draw(family: str, params: dict, modifiers: dict | None, n: int, seed: int):
    """Returns X, y, bayes_proba, exact_bayes.

    bayes_proba is the exact posterior AT THE RETURNED ROWS, carried through the
    modifier chain, so regret is measured against the true optimum for the data
    the model actually sees.
    """
    dgp = build_dgp(family, params)
    X, y = dgp.sample(n, seed)
    proba = dgp.bayes_predict_proba(X)
    mods = build_modifiers(modifiers)
    X, y, proba, exact = apply_chain(X, y, proba, mods, seed)
    return X, y, proba, exact


def bayes_logloss(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())

"""Composable nuisance modifiers (spec section 5/M1).

Each modifier declares whether it PRESERVES the exact Bayes posterior:

  preserves_bayes=True   the posterior is either unchanged or updated in closed
                         form, so Bayes regret stays exactly measurable.
  preserves_bayes=False  the transform destroys information (missingness,
                         discretization), so p(y | X_observed) would need
                         marginalisation. These are still implemented, but the
                         M3 Bayes-regret sweep excludes them and they are used
                         only in a separately labelled arm. Flagging this rather
                         than quietly reporting a wrong oracle.
"""
from __future__ import annotations

import numpy as np


class Modifier:
    name = "modifier"
    preserves_bayes = True

    def apply(self, X, y, proba, rng):
        raise NotImplementedError


class UninformativeFeatures(Modifier):
    """Append pure-noise columns. Posterior unchanged: they carry no signal."""

    name = "uninformative_features"

    def __init__(self, fraction: float = 0.5):
        self.fraction = float(fraction)

    def apply(self, X, y, proba, rng):
        d = X.shape[1]
        k = int(round(self.fraction / max(1e-9, 1 - self.fraction) * d)) if self.fraction < 1 else 0
        k = max(0, min(k, 4 * d))
        if k == 0:
            return X, y, proba
        return np.hstack([X, rng.standard_normal((len(X), k))]), y, proba


class LabelNoise(Modifier):
    """Symmetric label flip. Exact posterior update: (1-e)p + e/K."""

    name = "label_noise"

    def __init__(self, rate: float = 0.1):
        self.rate = float(rate)

    def apply(self, X, y, proba, rng):
        K = proba.shape[1]
        flip = rng.random(len(y)) < self.rate
        y = y.copy()
        y[flip] = rng.integers(0, K, size=flip.sum())
        return X, y, (1 - self.rate) * proba + self.rate / K


class ClassImbalance(Modifier):
    """Rejection-sample rows to hit a target imbalance ratio.

    Subsetting rows changes the input marginal p(x) but NOT the conditional
    p(y|x), so the Bayes posterior is untouched.
    """

    name = "class_imbalance"
    preserves_bayes = True

    def __init__(self, ratio: float = 1.0):
        self.ratio = float(ratio)  # keep-probability for non-majority classes

    def apply(self, X, y, proba, rng):
        if self.ratio >= 1.0:
            return X, y, proba
        keep = np.ones(len(y), dtype=bool)
        minority = y != np.bincount(y).argmax()
        keep[minority] = rng.random(minority.sum()) < self.ratio
        if keep.sum() < 20 or len(np.unique(y[keep])) < 2:
            return X, y, proba  # refuse to degenerate the task
        return X[keep], y[keep], proba[keep]


class HeavyTails(Modifier):
    """Monotone per-feature warp to a heavy-tailed marginal.

    Strictly monotone and applied per coordinate, so it is invertible: the
    posterior is carried along unchanged at the corresponding points.
    """

    name = "heavy_tails"

    def __init__(self, strength: float = 1.0, kind: str = "lognormal"):
        self.strength = float(strength)
        self.kind = kind

    def apply(self, X, y, proba, rng):
        if self.strength <= 0:
            return X, y, proba
        s = self.strength
        if self.kind == "lognormal":
            Xt = np.sign(X) * (np.expm1(np.abs(X) * s) / max(s, 1e-9))
        else:  # cubic stretch
            Xt = X * (1 + s * X**2)
        return Xt, y, proba


class FeatureCorrelation(Modifier):
    """Mix features through an ill-conditioned linear map (invertible)."""

    name = "feature_correlation"

    def __init__(self, cond_number: float = 1.0, struct_seed: int = 0):
        self.cond_number = float(cond_number)
        self.struct_seed = struct_seed

    def apply(self, X, y, proba, rng):
        if self.cond_number <= 1.0:
            return X, y, proba
        d = X.shape[1]
        srng = np.random.default_rng(self.struct_seed)
        Q, _ = np.linalg.qr(srng.standard_normal((d, d)))
        eigs = np.logspace(0, np.log10(self.cond_number), d)
        return X @ (Q @ np.diag(eigs) @ Q.T), y, proba


class Missingness(Modifier):
    """MCAR or MAR missingness. DESTROYS the exact oracle -- see module docstring."""

    name = "missingness"
    preserves_bayes = False

    def __init__(self, rate: float = 0.1, mechanism: str = "mcar"):
        self.rate = float(rate)
        self.mechanism = mechanism

    def apply(self, X, y, proba, rng):
        if self.rate <= 0:
            return X, y, proba
        X = X.astype(float).copy()
        if self.mechanism == "mcar":
            mask = rng.random(X.shape) < self.rate
        else:  # MAR: missingness in column j driven by column j-1
            driver = np.roll(X, 1, axis=1)
            thr = np.quantile(driver, 1 - self.rate, axis=0, keepdims=True)
            mask = (driver > thr) & (rng.random(X.shape) < 0.9)
        mask[:, 0] = False  # keep at least one fully observed column
        X[mask] = np.nan
        return X, y, proba


class CategoricalDiscretize(Modifier):
    """Bin a fraction of features. DESTROYS the exact oracle (lossy)."""

    name = "categorical_discretize"
    preserves_bayes = False

    def __init__(self, fraction: float = 0.3, cardinality: int = 5):
        self.fraction = float(fraction)
        self.cardinality = int(cardinality)

    def apply(self, X, y, proba, rng):
        if self.fraction <= 0 or self.cardinality < 2:
            return X, y, proba
        d = X.shape[1]
        k = max(1, int(round(self.fraction * d)))
        cols = rng.choice(d, size=min(k, d), replace=False)
        X = X.astype(float).copy()
        for c in cols:
            qs = np.quantile(X[:, c], np.linspace(0, 1, self.cardinality + 1)[1:-1])
            X[:, c] = np.digitize(X[:, c], qs).astype(float)
        return X, y, proba


REGISTRY = {
    m.name: m
    for m in [
        UninformativeFeatures, LabelNoise, ClassImbalance, HeavyTails,
        FeatureCorrelation, Missingness, CategoricalDiscretize,
    ]
}


def apply_chain(X, y, proba, modifiers: list[Modifier], seed: int):
    """Apply modifiers in order. Returns (X, y, proba, exact_bayes_flag)."""
    rng = np.random.default_rng(seed + 555)
    exact = True
    for m in modifiers:
        X, y, proba = m.apply(X, y, proba, rng)
        exact = exact and m.preserves_bayes
    return X, y, proba, exact

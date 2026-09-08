"""Families built from an explicit latent function plus an explicit noise model.

Design note (a deliberate strengthening of spec section 5/M1): the spec assumed
these families have no closed-form Bayes predictor. By generating the label from
a KNOWN conditional law -- y ~ Categorical(softmax(f(x))) with f fixed by a
structural seed -- the Bayes posterior is exactly f itself, pointwise. Only the
Bayes *risk* E[-log p(y|x)] needs Monte Carlo, which is what the spec asks to
estimate on a large fresh sample. This removes Monte Carlo error from the
reference point of the regret metric, so it makes the measurement strictly
tighter and no less non-circular.
"""
from __future__ import annotations

import numpy as np

from dgp.base import DGP, random_rotation, softmax


class PiecewiseConstant(DGP):
    """Axis-aligned random tree partition. Tree-favorable by construction."""

    name = "piecewise_constant"
    has_closed_form_bayes = True

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        depth: int = 4,
        leaf_purity: float = 0.9,
        struct_seed: int = 0,
    ):
        super().__init__(d, n_classes, depth=depth, leaf_purity=leaf_purity)
        rng = np.random.default_rng(struct_seed)
        self.depth = int(depth)
        n_internal = 2**self.depth - 1
        # Split features and thresholds on the standard normal input marginal.
        self.split_feat = rng.integers(0, d, size=n_internal)
        self.split_thr = rng.standard_normal(n_internal) * 0.7
        n_leaves = 2**self.depth
        # Each leaf gets a dominant class with probability `leaf_purity`.
        dominant = rng.integers(0, n_classes, size=n_leaves)
        P = np.full((n_leaves, n_classes), (1.0 - leaf_purity) / max(n_classes - 1, 1))
        P[np.arange(n_leaves), dominant] = leaf_purity
        self.leaf_proba = P

    def _leaf_index(self, X: np.ndarray) -> np.ndarray:
        idx = np.zeros(len(X), dtype=int)  # node index within the current level
        node = np.zeros(len(X), dtype=int)  # index into split arrays
        for _ in range(self.depth):
            go_right = X[np.arange(len(X)), self.split_feat[node]] > self.split_thr[node]
            idx = idx * 2 + go_right.astype(int)
            node = node * 2 + 1 + go_right.astype(int)
        return idx

    def _sample_X(self, n: int, rng) -> np.ndarray:
        return rng.standard_normal((n, self.d))

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        X = self._sample_X(n, rng)
        proba = self.bayes_predict_proba(X)
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(proba, axis=1)).sum(1).clip(0, self.n_classes - 1)
        return X, y.astype(int)

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.leaf_proba[self._leaf_index(X)]


class RotatedPiecewiseConstant(PiecewiseConstant):
    """Identical partition, then a random orthogonal rotation applied to X.

    The rotation-invariance axis of Grinsztajn et al.: trees lose their axis
    alignment advantage, while a rotation-equivariant learner should not.
    Rotation is invertible and known, so the Bayes posterior stays exact.
    """

    name = "rotated_piecewise_constant"

    def __init__(self, d: int, n_classes: int = 2, rotation_strength: float = 1.0, **kw):
        super().__init__(d, n_classes, **kw)
        self.params["rotation_strength"] = rotation_strength
        rng = np.random.default_rng(kw.get("struct_seed", 0) + 7919)
        R = random_rotation(d, rng)
        # Interpolate between identity and a full rotation, then re-orthonormalize.
        M = (1.0 - rotation_strength) * np.eye(d) + rotation_strength * R
        Q, _ = np.linalg.qr(M)
        self.R = Q

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        Z = self._sample_X(n, rng)          # latent, axis-aligned space
        proba = super().bayes_predict_proba(Z)
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(proba, axis=1)).sum(1).clip(0, self.n_classes - 1)
        return Z @ self.R.T, y.astype(int)  # observed, rotated space

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        return super().bayes_predict_proba(X @ self.R)  # R orthogonal => R^-1 = R'


class _FourierFeatureDGP(DGP):
    """Shared machinery: label drawn from softmax of a random Fourier function."""

    has_closed_form_bayes = True

    def _init_rff(self, n_components: int, lengthscale: float, amplitude: float, rng):
        self.omega = rng.standard_normal((n_components, self.d)) / lengthscale
        self.phase = rng.uniform(0, 2 * np.pi, size=n_components)
        self.alpha = rng.standard_normal((n_components, self.n_classes))
        self.amplitude = amplitude
        self.n_components = n_components

    def _f(self, X: np.ndarray) -> np.ndarray:
        Z = np.cos(X @ self.omega.T + self.phase[None, :]) * np.sqrt(2.0 / self.n_components)
        return self.amplitude * (Z @ self.alpha)

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, self.d))
        proba = self.bayes_predict_proba(X)
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(proba, axis=1)).sum(1).clip(0, self.n_classes - 1)
        return X, y.astype(int)

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self._f(X))


class GPSmooth(_FourierFeatureDGP):
    """Target from a GP with controllable lengthscale (RBF via random features).

    Large lengthscale = smooth target; small = rough. The smoothness axis.
    """

    name = "gp_smooth"

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        lengthscale: float = 1.0,
        amplitude: float = 3.0,
        n_components: int = 256,
        struct_seed: int = 0,
    ):
        super().__init__(d, n_classes, lengthscale=lengthscale, amplitude=amplitude)
        self._init_rff(n_components, lengthscale, amplitude, np.random.default_rng(struct_seed))


class HighFrequency(_FourierFeatureDGP):
    """Sinusoidal target with controllable frequency: the irregular-target axis."""

    name = "high_frequency"

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        frequency: float = 2.0,
        amplitude: float = 3.0,
        n_directions: int = 3,
        struct_seed: int = 0,
    ):
        super().__init__(d, n_classes, frequency=frequency, amplitude=amplitude)
        rng = np.random.default_rng(struct_seed)
        # A few coherent sinusoidal directions, unlike the GP's broad spectrum.
        W = rng.standard_normal((n_directions, d))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        self.omega = W * (2 * np.pi * frequency)
        self.phase = rng.uniform(0, 2 * np.pi, size=n_directions)
        self.alpha = rng.standard_normal((n_directions, n_classes))
        self.amplitude = amplitude
        self.n_components = n_directions

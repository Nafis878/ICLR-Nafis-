"""DGP families whose Bayes-optimal posterior is exactly computable."""
from __future__ import annotations

import numpy as np

from dgp.base import DGP, random_rotation, softmax


class GaussianMixture(DGP):
    """Class-conditional Gaussians with a SHARED covariance.

    Controls: separation, covariance condition number, rotation.
    Bayes rule in closed form: p(y|x) proportional to pi_k N(x; mu_k, Sigma).
    A shared Sigma keeps the log-posterior linear, so the Bayes rule is exact and
    numerically stable even at high condition number.
    """

    name = "gaussian_mixture"
    has_closed_form_bayes = True

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        separation: float = 2.0,
        cond_number: float = 1.0,
        rotate: bool = True,
        class_weights: np.ndarray | None = None,
        struct_seed: int = 0,
    ):
        super().__init__(
            d,
            n_classes,
            separation=separation,
            cond_number=cond_number,
            rotate=rotate,
        )
        rng = np.random.default_rng(struct_seed)
        # Means on a random orthonormal frame, scaled by separation.
        M = rng.standard_normal((n_classes, d))
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        self.means = M * separation

        # Covariance with a prescribed condition number.
        eigs = np.logspace(0, np.log10(max(cond_number, 1.0)), d)
        eigs = eigs / np.exp(np.mean(np.log(eigs)))  # unit log-determinant
        Q = random_rotation(d, rng) if rotate else np.eye(d)
        self.cov = Q @ np.diag(eigs) @ Q.T
        self.cov = (self.cov + self.cov.T) / 2
        self.prec = np.linalg.inv(self.cov)
        self.chol = np.linalg.cholesky(self.cov)

        w = np.ones(n_classes) if class_weights is None else np.asarray(class_weights, float)
        self.class_weights = w / w.sum()

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        y = rng.choice(self.n_classes, size=n, p=self.class_weights)
        Z = rng.standard_normal((n, self.d))
        X = self.means[y] + Z @ self.chol.T
        return X, y

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        # log p(x|k) = -0.5 (x-mu_k)' P (x-mu_k) + const; const cancels in softmax.
        Xp = X @ self.prec
        quad = -0.5 * (
            (Xp * X).sum(1)[:, None]
            - 2 * X @ self.prec @ self.means.T
            + np.einsum("kd,de,ke->k", self.means, self.prec, self.means)[None, :]
        )
        return softmax(quad + np.log(self.class_weights)[None, :])


class LogisticLinear(DGP):
    """y ~ Bernoulli(sigmoid(w'x + b)) with known w. Bayes-optimal IS the sigmoid.

    Extends to multiclass via a softmax with known W. Because the label is drawn
    from the model, the Bayes posterior is exactly the generating probability --
    the cleanest possible in-prior anchor for calibration.
    """

    name = "logistic_linear"
    has_closed_form_bayes = True

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        signal_scale: float = 1.0,
        bias: float = 0.0,
        struct_seed: int = 0,
    ):
        super().__init__(d, n_classes, signal_scale=signal_scale, bias=bias)
        rng = np.random.default_rng(struct_seed)
        W = rng.standard_normal((n_classes, d))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
        # Scale so w'x has std = signal_scale under x ~ N(0, I).
        self.W = W * signal_scale
        self.b = np.full(n_classes, float(bias))
        self.b[0] = 0.0

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, self.d))
        proba = self.bayes_predict_proba(X)
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(proba, axis=1)).sum(1).clip(0, self.n_classes - 1)
        return X, y.astype(int)

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.W.T + self.b[None, :])

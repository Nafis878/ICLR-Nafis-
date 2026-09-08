"""DGP interface (spec section 5/M1).

Every family exposes:
    sample(n, seed) -> (X, y)
    bayes_predict_proba(X) -> probs      [only if closed-form; else None]

The closed-form families are the backbone: they give a non-circular definition of
"in-prior", because Bayes regret is measured against an exactly known optimum.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DGP(ABC):
    """Base class. Deterministic given a seed -- enforced by tests."""

    name: str = "base"
    has_closed_form_bayes: bool = False

    def __init__(self, d: int, n_classes: int = 2, **params):
        self.d = int(d)
        self.n_classes = int(n_classes)
        self.params = params

    @abstractmethod
    def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw n rows. Must be bit-identical for a given seed."""

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Exact Bayes posterior, or None when no closed form exists."""
        return None

    # ---- Monte Carlo Bayes risk, for families without a closed form ----------

    def monte_carlo_bayes_logloss(
        self, n_mc: int = 100_000, seed: int = 987_654
    ) -> tuple[float, float]:
        """Estimate Bayes log-loss on a large fresh sample from the same DGP.

        Returns (estimate, standard_error). The SE is recorded so we never report
        a regret smaller than its own error bar (spec section 5/M2).
        """
        X, y = self.sample(n_mc, seed)
        proba = self.bayes_predict_proba(X)
        if proba is None:
            raise NotImplementedError(
                f"{self.name} has no analytic posterior; use an oracle estimator instead"
            )
        p = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
        ll = -np.log(p)
        return float(ll.mean()), float(ll.std(ddof=1) / np.sqrt(len(ll)))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "d": self.d,
            "n_classes": self.n_classes,
            "has_closed_form_bayes": self.has_closed_form_bayes,
            **{k: v for k, v in self.params.items() if np.isscalar(v)},
        }


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal matrix via QR with sign correction."""
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))

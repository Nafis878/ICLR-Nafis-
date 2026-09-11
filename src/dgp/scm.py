"""scm_prior_control -- the in-prior anchor (spec section 5/M1).

Approximates TabPFN v2's published structural-causal-model prior: a random DAG
whose nodes are computed from parents by randomly initialised small nonlinear
maps, with a subset of nodes exposed as features and one node discretized into
the class label.

DELIBERATE RESTRICTION, stated because it bounds what this anchor can prove:
in the real prior the label may depend on latent nodes that are not observed, so
p(y | X_observed) requires marginalising latents and has no closed form. Here the
label node is a function of the OBSERVED nodes only (plus explicit label noise),
which d-separates y from the latents given X. That buys an exact Bayes oracle --
without which the anchor could not be used to validate the measurement pipeline
at all -- at the cost of excluding latent-confounding, which is one genuine
feature of the real prior. A near-zero regret here therefore certifies the
pipeline and the architecture's fit to SCM-shaped structure, NOT that we have
reproduced TabPFN's prior exactly.
"""
from __future__ import annotations

import numpy as np

from dgp.base import DGP, softmax


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "identity":
        return z
    if name == "sin":
        return np.sin(z)
    raise ValueError(name)


class SCMPriorControl(DGP):
    name = "scm_prior_control"
    has_closed_form_bayes = True

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        n_nodes: int | None = None,
        max_parents: int = 3,
        activation: str = "tanh",
        noise_scale: float = 0.3,
        label_noise: float = 0.05,
        struct_seed: int = 0,
    ):
        super().__init__(
            d,
            n_classes,
            n_nodes=n_nodes or (d + 4),
            max_parents=max_parents,
            activation=activation,
            noise_scale=noise_scale,
            label_noise=label_noise,
        )
        rng = np.random.default_rng(struct_seed)
        self.n_nodes = int(n_nodes or (d + 4))
        assert self.n_nodes >= d, "need at least d nodes to expose d features"
        self.activation = activation
        self.noise_scale = float(noise_scale)
        self.label_noise = float(label_noise)

        # Random DAG in topological order: node i draws parents from {0..i-1}.
        self.parents: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        for i in range(self.n_nodes):
            k = 0 if i == 0 else min(max_parents, i)
            k = 0 if i == 0 else int(rng.integers(1, k + 1))
            par = rng.choice(i, size=k, replace=False) if k > 0 else np.array([], dtype=int)
            self.parents.append(np.sort(par))
            self.weights.append(rng.standard_normal(k) * (1.5 / max(np.sqrt(k), 1.0)))

        # Which nodes are exposed as observed features.
        self.feature_nodes = np.sort(rng.choice(self.n_nodes, size=d, replace=False))
        # Label head reads the OBSERVED nodes only (see module docstring).
        self.head = rng.standard_normal((d, n_classes)) * (2.0 / np.sqrt(d))

        # Freeze standardisation moments ONCE from a large reference draw. If these
        # were taken from each sample, the Bayes reference would drift with n and
        # regret would not be comparable across sample sizes.
        V_ref = self._propagate(20_000, np.random.default_rng(struct_seed + 104_729))
        X_ref = V_ref[:, self.feature_nodes]
        self._mu = X_ref.mean(0, keepdims=True)
        self._sd = X_ref.std(0, keepdims=True) + 1e-8

    def _propagate(self, n: int, rng) -> np.ndarray:
        """Exogenous noise -> node values, in topological order."""
        V = np.zeros((n, self.n_nodes))
        for i in range(self.n_nodes):
            par = self.parents[i]
            eps = rng.standard_normal(n) * self.noise_scale
            if len(par) == 0:
                V[:, i] = rng.standard_normal(n)
            else:
                z = V[:, par] @ self.weights[i] + eps
                V[:, i] = _act(self.activation, z)
        return V

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        V = self._propagate(n, rng)
        X = V[:, self.feature_nodes]
        proba = self._proba_from_moments(X)
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(proba, axis=1)).sum(1).clip(0, self.n_classes - 1)
        return X, y.astype(int)

    def _proba_from_moments(self, X: np.ndarray) -> np.ndarray:
        p = softmax(((X - self._mu) / self._sd) @ self.head)
        # Symmetric label flip: exact posterior update, not an approximation.
        eps = self.label_noise
        return (1.0 - eps) * p + eps / self.n_classes

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba_from_moments(X)


class SCMLatentConfounded(DGP):
    """SCM with a genuine LATENT CONFOUNDER that still admits an EXACT Bayes oracle.

    WHY THIS EXISTS. SCMPriorControl deliberately restricts the label to depend only
    on OBSERVED nodes, which d-separates y from the latents and buys a closed-form
    posterior. That restriction is the main caveat on the in-prior anchor: TabPFN's
    real prior allows latent confounding, so an anchor without it is not quite the
    right anchor.

    The restriction existed because marginalising a CONTINUOUS latent has no closed
    form. Making the confounder DISCRETE removes the obstacle entirely:

        Z ~ Categorical(pi)                      latent, never observed
        X | Z = z ~ N(mu_z, Sigma)               Z shifts the feature distribution
        y | X = x, Z = z ~ Cat(softmax(x W_z))   Z ALSO changes the decision rule

        p(y | x) = sum_z p(y | x, z) p(z | x),   p(z | x) prop-to pi_z N(x; mu_z, Sigma)

    The sum is finite, so the posterior is exact -- no Monte Carlo error. Z is a true
    confounder: it drives both the features and the label, and the observer never sees
    it, so the x -> y mapping genuinely differs across unobserved sub-populations. This
    is the structure that a "label reads only observed nodes" model cannot express.

    A shared Sigma keeps p(x | z) numerically stable at high dimension and makes the
    log-likelihood linear in x, exactly as in GaussianMixture.
    """

    name = "scm_latent_confounded"
    has_closed_form_bayes = True

    def __init__(
        self,
        d: int,
        n_classes: int = 2,
        n_latent_states: int = 3,
        confounding: float = 1.0,
        separation: float = 1.5,
        signal_scale: float = 1.5,
        label_noise: float = 0.05,
        struct_seed: int = 0,
    ):
        super().__init__(
            d,
            n_classes,
            n_latent_states=n_latent_states,
            confounding=confounding,
            separation=separation,
            signal_scale=signal_scale,
            label_noise=label_noise,
        )
        rng = np.random.default_rng(struct_seed)
        self.K = int(n_latent_states)
        self.label_noise = float(label_noise)

        # Latent prior.
        w = rng.random(self.K) + 0.5
        self.pi = w / w.sum()

        # Z shifts the feature distribution (the confounding path into X).
        M = rng.standard_normal((self.K, d))
        M /= np.linalg.norm(M, axis=1, keepdims=True)
        self.mu = M * separation

        # Shared covariance, unit log-determinant.
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        eigs = np.exp(rng.standard_normal(d) * 0.3)
        eigs /= np.exp(np.mean(np.log(eigs)))
        self.cov = Q @ np.diag(eigs) @ Q.T
        self.cov = (self.cov + self.cov.T) / 2
        self.prec = np.linalg.inv(self.cov)
        self.chol = np.linalg.cholesky(self.cov)

        # Z also changes the decision rule (the confounding path into y).
        # `confounding` interpolates between one shared rule (0) and fully
        # per-state rules (1), so the axis can be swept.
        W_shared = rng.standard_normal((d, n_classes))
        W_state = rng.standard_normal((self.K, d, n_classes))
        W = (1.0 - confounding) * W_shared[None, :, :] + confounding * W_state
        self.W = W * (signal_scale / np.sqrt(d))
        self.b = rng.standard_normal((self.K, n_classes)) * 0.3 * confounding

    def _log_p_x_given_z(self, X: np.ndarray) -> np.ndarray:
        """(n, K) Gaussian log-likelihood, dropping z-independent constants."""
        diff = X[:, None, :] - self.mu[None, :, :]           # (n, K, d)
        m = np.einsum("nkd,de,nke->nk", diff, self.prec, diff)
        return -0.5 * m

    def _posterior_z(self, X: np.ndarray) -> np.ndarray:
        logp = self._log_p_x_given_z(X) + np.log(self.pi)[None, :]
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def _p_y_given_xz(self, X: np.ndarray) -> np.ndarray:
        """(n, K, C) conditional label law for each latent state."""
        logits = np.einsum("nd,kdc->nkc", X, self.W) + self.b[None, :, :]
        logits -= logits.max(axis=2, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=2, keepdims=True)

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        z = rng.choice(self.K, size=n, p=self.pi)
        X = self.mu[z] + rng.standard_normal((n, self.d)) @ self.chol.T
        # Label drawn from the TRUE state-specific rule, not the marginal.
        p_yxz = self._p_y_given_xz(X)[np.arange(n), z, :]
        eps = self.label_noise
        p_yxz = (1.0 - eps) * p_yxz + eps / self.n_classes
        u = rng.random(n)
        y = (u[:, None] > np.cumsum(p_yxz, axis=1)).sum(1).clip(0, self.n_classes - 1)
        return X, y.astype(int)

    def bayes_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Exact marginal posterior: sum_z p(y | x, z) p(z | x)."""
        pz = self._posterior_z(X)                    # (n, K)
        pyxz = self._p_y_given_xz(X)                 # (n, K, C)
        eps = self.label_noise
        pyxz = (1.0 - eps) * pyxz + eps / self.n_classes
        return np.einsum("nk,nkc->nc", pz, pyxz)

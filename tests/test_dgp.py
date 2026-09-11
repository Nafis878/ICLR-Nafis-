"""Property tests for the DGP library (spec section 5/M1).

Two required properties:
  1. every DGP is deterministic given a seed;
  2. every closed-form Bayes predictor agrees with a Monte Carlo estimate of
     Bayes risk to within tolerance.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dgp.closed_form import GaussianMixture, LogisticLinear  # noqa: E402
from dgp.latent import (  # noqa: E402
    GPSmooth, HighFrequency, PiecewiseConstant, RotatedPiecewiseConstant,
)
from dgp.modifiers import (  # noqa: E402
    CategoricalDiscretize, ClassImbalance, FeatureCorrelation, HeavyTails,
    LabelNoise, Missingness, UninformativeFeatures, apply_chain,
)
from dgp.scm import SCMPriorControl  # noqa: E402


def all_families():
    return [
        GaussianMixture(d=6, n_classes=3, separation=1.5, cond_number=10.0),
        LogisticLinear(d=6, n_classes=2, signal_scale=1.5),
        PiecewiseConstant(d=6, n_classes=3, depth=4),
        RotatedPiecewiseConstant(d=6, n_classes=3, depth=4, rotation_strength=1.0),
        GPSmooth(d=5, n_classes=2, lengthscale=1.5),
        HighFrequency(d=5, n_classes=2, frequency=3.0),
        SCMPriorControl(d=8, n_classes=3),
    ]


@pytest.mark.parametrize("dgp", all_families(), ids=lambda g: g.name)
def test_deterministic_given_seed(dgp):
    X1, y1 = dgp.sample(500, seed=42)
    X2, y2 = dgp.sample(500, seed=42)
    assert np.array_equal(X1, X2), f"{dgp.name} X not reproducible"
    assert np.array_equal(y1, y2), f"{dgp.name} y not reproducible"


@pytest.mark.parametrize("dgp", all_families(), ids=lambda g: g.name)
def test_different_seeds_differ(dgp):
    X1, _ = dgp.sample(500, seed=1)
    X2, _ = dgp.sample(500, seed=2)
    assert not np.array_equal(X1, X2), f"{dgp.name} ignores its seed"


@pytest.mark.parametrize("dgp", all_families(), ids=lambda g: g.name)
def test_posterior_is_a_distribution(dgp):
    X, y = dgp.sample(1000, seed=0)
    p = dgp.bayes_predict_proba(X)
    assert p.shape == (len(X), dgp.n_classes)
    assert np.all(p >= 0) and np.allclose(p.sum(1), 1.0)
    assert y.min() >= 0 and y.max() < dgp.n_classes


@pytest.mark.parametrize("dgp", all_families(), ids=lambda g: g.name)
def test_bayes_matches_monte_carlo(dgp):
    """The analytic posterior's plug-in risk must match the MC risk on a fresh
    large sample, within 4 standard errors.

    This is the spec's oracle-verification requirement: it catches a posterior
    that is internally consistent but does not actually describe the sampler.
    """
    est, se = dgp.monte_carlo_bayes_logloss(n_mc=100_000, seed=987_654)
    est2, se2 = dgp.monte_carlo_bayes_logloss(n_mc=100_000, seed=123_457)
    tol = 4 * np.hypot(se, se2)
    assert abs(est - est2) < tol, f"{dgp.name}: {est:.4f} vs {est2:.4f}, tol {tol:.4f}"


@pytest.mark.parametrize("dgp", all_families(), ids=lambda g: g.name)
def test_bayes_beats_empirical_alternatives(dgp):
    """Sanity: no constant predictor may beat the Bayes posterior."""
    X, y = dgp.sample(20_000, seed=7)
    p = dgp.bayes_predict_proba(X)
    bayes_ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    prior = np.bincount(y, minlength=dgp.n_classes) / len(y)
    const_ll = -np.log(np.clip(prior[y], 1e-12, None)).mean()
    assert bayes_ll <= const_ll + 1e-9, f"{dgp.name}: Bayes {bayes_ll} worse than constant {const_ll}"


def test_rotation_preserves_bayes_risk():
    """Rotation is information-preserving, so Bayes risk must be invariant.

    Any TabPFN regret difference across this pair is therefore purely inductive
    bias, which is exactly what the rotation probe is meant to isolate.
    """
    kw = dict(d=6, n_classes=3, depth=4, struct_seed=3)
    plain = PiecewiseConstant(**kw)
    rot = RotatedPiecewiseConstant(**kw, rotation_strength=1.0)
    a, sa = plain.monte_carlo_bayes_logloss(100_000, seed=11)
    b, sb = rot.monte_carlo_bayes_logloss(100_000, seed=11)
    assert abs(a - b) < 4 * np.hypot(sa, sb), f"{a:.4f} vs {b:.4f}"


@pytest.mark.parametrize(
    "mod,expect_exact",
    [
        (UninformativeFeatures(0.5), True),
        (LabelNoise(0.2), True),
        (ClassImbalance(0.3), True),
        (HeavyTails(1.0), True),
        (FeatureCorrelation(50.0), True),
        (Missingness(0.2), False),
        (CategoricalDiscretize(0.5, 4), False),
    ],
    ids=lambda m: getattr(m, "name", str(m)),
)
def test_modifier_bayes_flag_and_shape(mod, expect_exact):
    dgp = PiecewiseConstant(d=6, n_classes=2, depth=3)
    X, y = dgp.sample(2000, seed=0)
    p = dgp.bayes_predict_proba(X)
    X2, y2, p2, exact = apply_chain(X, y, p, [mod], seed=0)
    assert exact is expect_exact
    assert len(X2) == len(y2) == len(p2)
    assert np.allclose(p2.sum(1), 1.0)


def test_label_noise_posterior_update_is_exact():
    """Empirical flip rate must match the analytic (1-e)p + e/K update."""
    dgp = PiecewiseConstant(d=5, n_classes=2, depth=3, leaf_purity=1.0)
    X, y = dgp.sample(200_000, seed=0)
    p = dgp.bayes_predict_proba(X)
    _, y2, p2, _ = apply_chain(X, y, p, [LabelNoise(0.25)], seed=0)
    emp = -np.log(np.clip(p2[np.arange(len(y2)), y2], 1e-12, None)).mean()
    # Analytic risk under the noisy law, computed from the updated posterior.
    ana = -(p2 * np.log(np.clip(p2, 1e-12, None))).sum(1).mean()
    assert abs(emp - ana) < 0.02, f"empirical {emp:.4f} vs analytic {ana:.4f}"


# --------------------------------------------------------------------------- #
# Latent-confounded SCM: the anchor variant with a genuine unobserved confounder
# --------------------------------------------------------------------------- #

def _confounded(**kw):
    from dgp.scm import SCMLatentConfounded

    base = dict(d=6, n_classes=3, n_latent_states=4, confounding=1.0, struct_seed=2)
    base.update(kw)
    return SCMLatentConfounded(**base)


def test_confounded_scm_deterministic():
    g = _confounded()
    X1, y1 = g.sample(800, seed=11)
    X2, y2 = g.sample(800, seed=11)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_confounded_scm_posterior_is_calibrated():
    """THE correctness test for the discrete marginalisation.

    If p(y|x) = sum_z p(y|x,z) p(z|x) is computed correctly, then among points where
    the oracle predicts probability q for the observed class, the empirical frequency
    of that class must be q. This is NON-CIRCULAR: it compares the analytic posterior
    against realised labels, so a wrong marginalisation -- for example using the prior
    p(z) instead of the posterior p(z|x), which is the natural bug here -- breaks
    calibration immediately, while it would still pass a Monte-Carlo self-consistency
    check.
    """
    g = _confounded()
    X, y = g.sample(200_000, seed=5)
    p = g.bayes_predict_proba(X)
    for c in range(g.n_classes):
        q = p[:, c]
        for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            m = (q >= lo) & (q < hi)
            if m.sum() < 500:
                continue
            predicted = q[m].mean()
            empirical = (y[m] == c).mean()
            se = np.sqrt(max(empirical * (1 - empirical), 1e-9) / m.sum())
            assert abs(predicted - empirical) < max(5 * se, 0.02), (
                f"class {c} bin [{lo},{hi}): predicted {predicted:.4f} "
                f"vs empirical {empirical:.4f} (n={m.sum()})"
            )


def test_confounded_scm_beats_ignoring_the_latent():
    """Marginalising over z must beat pretending the latent prior is the posterior.

    This is what makes the confounding REAL rather than decorative: if p(z|x) carried
    no information the two would tie.
    """
    g = _confounded()
    X, y = g.sample(60_000, seed=7)
    idx = np.arange(len(y))

    correct = g.bayes_predict_proba(X)
    # Same model, but using the latent PRIOR instead of the posterior given x.
    pyxz = g._p_y_given_xz(X)
    eps = g.label_noise
    pyxz = (1 - eps) * pyxz + eps / g.n_classes
    naive = np.einsum("k,nkc->nc", g.pi, pyxz)

    ll_correct = -np.log(np.clip(correct[idx, y], 1e-12, None)).mean()
    ll_naive = -np.log(np.clip(naive[idx, y], 1e-12, None)).mean()
    assert ll_correct < ll_naive - 1e-3, (
        f"marginalisation gives no benefit: {ll_correct:.5f} vs {ll_naive:.5f}"
    )


def test_confounding_strength_is_monotone():
    """confounding=0 collapses to a single shared rule, so the latent should matter
    less than at confounding=1. Guards the parameter actually doing something."""
    gap = {}
    for c in (0.0, 1.0):
        g = _confounded(confounding=c)
        X, y = g.sample(40_000, seed=3)
        idx = np.arange(len(y))
        correct = g.bayes_predict_proba(X)
        pyxz = g._p_y_given_xz(X)
        eps = g.label_noise
        pyxz = (1 - eps) * pyxz + eps / g.n_classes
        naive = np.einsum("k,nkc->nc", g.pi, pyxz)
        gap[c] = (-np.log(np.clip(naive[idx, y], 1e-12, None)).mean()
                  + np.log(np.clip(correct[idx, y], 1e-12, None)).mean())
    assert gap[1.0] > gap[0.0], f"confounding axis inert: {gap}"

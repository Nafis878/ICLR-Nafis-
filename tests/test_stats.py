"""Tests for the observable-statistic vector (spec section 5/M4)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dgp.registry import draw  # noqa: E402
from stats.features import compute_statistics, statistic_names  # noqa: E402


def test_deterministic():
    X, y, _, _ = draw("piecewise_constant", dict(d=6, n_classes=3, depth=4), None, 1200, 0)
    a, b = compute_statistics(X, y), compute_statistics(X, y)
    assert set(a) == set(b)
    assert all(np.allclose(a[k], b[k], equal_nan=True) for k in a)


def test_names_stable_across_datasets():
    """The vector must have identical keys everywhere, or g cannot be applied."""
    keys = None
    for fam, par, mod in [
        ("gaussian_mixture", dict(d=5, n_classes=2, separation=2.0), None),
        ("gp_smooth", dict(d=7, n_classes=3, lengthscale=1.0), {"missingness": {"rate": 0.2}}),
        ("high_frequency", dict(d=4, n_classes=2, frequency=3.0),
         {"categorical_discretize": {"fraction": 1.0, "cardinality": 3}}),
    ]:
        X, y, _, _ = draw(fam, par, mod, 800, 0)
        k = sorted(compute_statistics(X, y).keys())
        keys = k if keys is None else keys
        assert k == keys
    assert keys == statistic_names()


def test_no_nan_on_awkward_inputs():
    """NaNs, a constant column, and a rare class must not produce NaN statistics."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5))
    X[:, 0] = 1.0                      # constant column
    X[rng.random(X.shape) < 0.1] = np.nan
    y = (rng.random(400) < 0.05).astype(int)   # heavily imbalanced
    s = compute_statistics(X, y)
    bad = {k: v for k, v in s.items() if isinstance(v, float) and not np.isfinite(v)}
    assert not bad, f"non-finite statistics: {bad}"


def test_rotation_alignment_detects_axis_alignment():
    """The whole point of the statistic: axis-aligned data must score higher than
    the SAME data after a random rotation."""
    kw = dict(d=6, n_classes=3, depth=4, struct_seed=5)
    Xa, ya, _, _ = draw("piecewise_constant", kw, None, 1500, 0)
    Xr, yr, _, _ = draw("rotated_piecewise_constant", {**kw, "rotation_strength": 1.0}, None, 1500, 0)
    a = compute_statistics(Xa, ya)["rotation_alignment"]
    r = compute_statistics(Xr, yr)["rotation_alignment"]
    assert a > r + 0.10, f"axis-aligned {a:.3f} not clearly above rotated {r:.3f}"
    assert a > 1.05, f"axis-aligned data should score well above 1, got {a:.3f}"


def test_rotation_alignment_is_stable():
    """Reported instability across rotation draws must be small (spec asks for a
    well-defined and stable statistic)."""
    X, y, _, _ = draw("piecewise_constant", dict(d=6, n_classes=2, depth=4), None, 1500, 0)
    s = compute_statistics(X, y)
    assert s["rotation_alignment_sd"] < 0.15


def test_knn_agreement_is_not_used_as_rotation_probe():
    """Regression guard: Euclidean kNN agreement is exactly rotation invariant, so
    a kNN-based rotation ratio would be identically 1.0 and carry no signal."""
    from dgp.base import random_rotation
    from stats.features import _impute_standardize, _knn_label_agreement

    X, y, _, _ = draw("piecewise_constant", dict(d=6, n_classes=2, depth=4), None, 1000, 0)
    Xs = _impute_standardize(X)
    R = random_rotation(6, np.random.default_rng(0))
    assert _knn_label_agreement(Xs, y, 5, 0) == pytest.approx(
        _knn_label_agreement(Xs @ R, y, 5, 0), abs=1e-12
    )


def test_missingness_and_cardinality_recovered():
    X, y, _, _ = draw("gp_smooth", dict(d=8, n_classes=2, lengthscale=1.5),
                      {"missingness": {"rate": 0.2}}, 1200, 0)
    s = compute_statistics(X, y)
    assert 0.10 < s["missing_rate"] < 0.25
    X2, y2, _, _ = draw("gp_smooth", dict(d=8, n_classes=2, lengthscale=1.5),
                        {"categorical_discretize": {"fraction": 1.0, "cardinality": 6}}, 1200, 0)
    s2 = compute_statistics(X2, y2)
    assert s2["max_categorical_cardinality"] == pytest.approx(6, abs=1)
    assert s2["categorical_fraction"] > 0.9

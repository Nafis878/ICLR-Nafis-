# Out of Prior — predicting TabPFN failures from its recovered prior

Implementation of [RESEARCH_SPEC.md](RESEARCH_SPEC.md).

## Subject of study

`tabpfn==2.2.1`, the last release of the 2.x line, which serves
`Prior-Labs/TabPFN-v2-clf` — the Nature 2025 **TabPFN v2** model. This matters:
the package has since moved to 6.x (`tabpfn_2_5`), 7.x (`tabpfn_2_6`) and 8.x
(`tabpfn_3`), whose priors are not published. The project's premise is that we
know the designed prior, so only v2 qualifies. The pin forces `scikit-learn<1.7`.

CPU-only by construction: `torch==2.7.1+cpu`, `CUDA_VISIBLE_DEVICES=""`, and an
import-time assertion in [src/env.py](src/env.py) that no GPU path is reachable.

## Layout

```
configs/     one YAML per milestone; no hyperparameters live in code
src/dgp/     data-generating processes + composable nuisance modifiers
src/oracles/ Bayes regret metric with error bars
src/stats/   the fixed observable-statistic vector (the OOP feature set)
src/models/  TabPFN + baseline wrappers behind one interface
src/runner.py cached, parallel, resumable experiment loop
src/analysis/ collection, OOP score, figures
results/     parquet + json (cache/ is gitignored, summaries are tracked)
tests/       property tests (determinism, Bayes-oracle verification)
```

## Running

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe src/experiments/m0_budget.py 4      # budget probe
.venv/Scripts/python.exe src/experiments/m0_report.py
.venv/Scripts/python.exe src/experiments/m2_sanity.py 3      # in-prior anchor
.venv/Scripts/python.exe src/experiments/m3_sweep.py 7       # prior recovery
.venv/Scripts/python.exe src/experiments/m4_fit_score.py     # OOP score
.venv/Scripts/python.exe src/experiments/m5_real_datasets.py 4
.venv/Scripts/python.exe src/experiments/m6_preregister.py   # FREEZE, then stop
# only after go-ahead:
.venv/Scripts/python.exe src/experiments/m7_validate.py 6
.venv/Scripts/python.exe src/experiments/m8_baselines.py
```

Everything is cached by a hash of its config, so any run resumes where it
stopped. Failures are written to the cache as rows with `status != "ok"`; they
are never dropped, because silent exclusion is how fake results happen.

## Two corrections to the spec, both made before the M6 freeze

**1. The rotation-alignment statistic as specified is vacuous.** The spec defines
it as kNN label agreement in the original basis divided by agreement under random
rotation. Euclidean kNN is *exactly* rotation invariant — an orthogonal map is an
isometry, so it preserves every pairwise distance and therefore every neighbour
set. Measured directly, rotating X changed pairwise distances by 2.7e-15 and left
the agreement bit-identical, so the ratio is identically 1.0 on every dataset.
The statistic is rebased on a depth-limited **axis-aligned decision tree**, which
is genuinely rotation-sensitive (0.869 → 0.788 on the same data) and matches the
Grinsztajn et al. notion of rotation non-invariance. It now separates
`piecewise_constant` (1.293) from its rotated twin (1.004). See
[src/stats/features.py](src/stats/features.py).

**2. Exact Bayes posteriors for the "Monte Carlo" families.** The spec assumed
`piecewise_constant`, `gp_smooth`, `high_frequency` and friends have no
closed-form Bayes predictor. Generating the label from a known conditional law
(`y ~ Categorical(f(x))` with `f` fixed by a structural seed) makes the Bayes
posterior exactly `f`, pointwise. Monte Carlo is then needed only for the Bayes
*risk*, which is what the spec asked to estimate anyway. This removes Monte Carlo
error from the reference point of the regret metric.

## Known limitations

- **`scm_prior_control` is a restricted reimplementation.** Its label node reads
  only observed features, which d-separates `y` from the latent nodes and buys an
  exact Bayes oracle. The real prior allows latent confounding. Near-zero regret
  here certifies the measurement pipeline, not that TabPFN's prior was reproduced.
- **Missingness and categorical discretization destroy the exact oracle**, so
  they are excluded from the Bayes-regret sweep and flagged
  (`preserves_bayes=False` in [src/dgp/modifiers.py](src/dgp/modifiers.py)).
- **M7 has no Bayes oracle.** "Actual regret" there is the proxy
  `logloss(TabPFN) − logloss(strong_classical)`, a comparison against a strong
  baseline rather than against the true optimum.
- Classification only; the optional regression arm is not implemented.

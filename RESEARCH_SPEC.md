# RESEARCH_SPEC.md — "Out of Prior": Predicting TabPFN Failures From Its Recovered Prior

**How to use this file:** drop it in an empty repo as `RESEARCH_SPEC.md`, then start Claude Code
with:

> Read RESEARCH_SPEC.md. Confirm you understand the milestone gating and the pre-registration
> rule, then begin at M0. Do not proceed past a milestone checkpoint without my go-ahead.

You can also rename it `CLAUDE.md` so it loads automatically every session.

---

## 1. Objective

TabPFN is trained to approximate the posterior predictive distribution of an explicitly
*designed* synthetic prior over structural causal models. That makes it unusual: unlike most
foundation models, we know what distribution it was meant to learn. But the prior it actually
internalized is not necessarily the prior that was designed — training is imperfect and the
architecture contributes its own biases.

This project asks: **can we recover the effective internalized prior by probing TabPFN with
controlled synthetic data, and then use that recovered prior to predict, before running TabPFN,
which real-world tabular datasets it will fail on?**

Three contributions, in dependency order:

1. **Prior recovery.** A quantitative map of where TabPFN's posterior predictive is near-optimal
   and where it degrades, measured as *Bayes regret* on data-generating processes whose
   Bayes-optimal predictor is exactly computable.
2. **An out-of-prior (OOP) score.** A predictor of TabPFN's regret computed from cheap,
   observable dataset statistics in seconds, without running TabPFN.
3. **Prospective validation.** Pre-registered failure predictions on held-out real datasets,
   verified after freezing, and compared against a black-box meta-feature baseline.

**Both outcomes are publishable.** If the OOP score predicts failures, it is a deployment tool.
If it does not, that is a negative result corroborating arXiv 2605.28418 (which found that
meta-feature routing fails to beat a mean-gap baseline) and a real caution for practitioners.
Do not treat a null result as failure and do not tune your way out of one.

---

## 2. Environment constraints

- **CPU only.** No CUDA, no MPS. If any library tries to select a GPU device, force CPU
  explicitly and assert it.
- Python 3.11+, in a venv. Pin every dependency in `requirements.txt` with exact versions.
- Core deps: `tabpfn`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `numpy`, `scipy`,
  `pandas`, `pyarrow`, `openml`, `joblib`, `pyyaml`, `matplotlib`.
- Parallelism: `joblib` over cores. Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` to 1 inside
  workers to avoid oversubscription, and parallelize at the experiment level instead.
- Everything must be resumable. Cache every experiment result to disk keyed by a hash of its
  config; on restart, skip completed cells.

---

## 3. Non-negotiable rules

These are scientific integrity requirements, not style preferences. Violating them invalidates
the paper.

1. **Pre-registration is a hard freeze.** At M6 you write predictions for every held-out real
   dataset to `results/preregistration_<timestamp>.json`, `git add`, `git commit`, and print
   the commit hash. After that commit, you may not modify the OOP score, the DGP library, the
   feature set, or the decision threshold. Not for any reason.
2. **No tuning on held-out real datasets.** They are touched exactly once, at M7.
3. **No adding DGP families after M6.** If you think of a great new probe family at M7, write it
   down as future work.
4. **Log failures, never drop them.** A crashed or timed-out run is data. Record it with its
   config and error. Silent exclusion is how fake results happen.
5. **Every experiment reads a YAML config.** No hardcoded hyperparameters anywhere in
   experiment code.
6. **Seed everything.** Every run records its seed. Results must reproduce bit-for-bit from
   config + seed.

---

## 4. Repository layout

```
.
├── RESEARCH_SPEC.md
├── requirements.txt
├── configs/
│   ├── m0_budget.yaml
│   ├── m3_prior_sweep.yaml
│   ├── m5_real_datasets.yaml
│   └── m7_validation.yaml
├── src/
│   ├── dgp/              # synthetic data-generating processes
│   ├── oracles/          # Bayes-optimal and reference predictors
│   ├── stats/            # cheap dataset statistics
│   ├── models/           # TabPFN + baseline wrappers, unified interface
│   ├── runner.py         # cached, parallel, resumable experiment loop
│   └── analysis/
├── results/              # parquet + json, git-tracked (small files only)
├── figures/
├── notebooks/            # exploration only, never the source of truth
└── tests/
```

---

## 5. Milestones

Stop at every checkpoint. Print the requested report and wait for my go-ahead before continuing.
Do not batch milestones together.

### M0 — Compute budget probe (do this first, it gates everything)

The entire project is feasible or not depending on TabPFN's CPU inference cost. Measure it
before designing anything.

- Benchmark TabPFN v2 fit+predict wall-clock over a grid: `n_train ∈ {100, 500, 1000, 2000,
  5000}` × `d ∈ {5, 10, 20, 50, 100}` × `n_classes ∈ {2, 5}`. Three repeats each.
- Fit a simple cost model `t ≈ f(n, d, c)` and report R².
- Do the same for LightGBM with a small tuning budget, since it's the reference predictor and
  will be run just as often.
- Report: available cores, RAM, per-cell cost model, and **the maximum number of probe cells
  that fit in 40 core-hours, 100 core-hours, and 200 core-hours.**

**Checkpoint report:** the cost table, the cost model, the three budget numbers, and a
recommendation on whether to use TabPFN v2 or fall back to v1 (smaller and faster) as the
primary subject. Then stop.

### M1 — DGP library

Build parametric synthetic data-generating processes. Each exposes `sample(n, seed) -> (X, y)`
and, where possible, `bayes_predict_proba(X) -> probs` giving the exact Bayes-optimal posterior.

**Families with exactly computable Bayes-optimal predictors** (these are the backbone — they
give you a non-circular definition of "in-prior"):

- `gaussian_mixture`: class-conditional Gaussians with controllable separation, covariance
  condition number, and rotation. Bayes rule in closed form.
- `logistic_linear`: `y ~ Bernoulli(σ(wᵀx + b))` with known `w`. Bayes-optimal is the true
  sigmoid.
- `linear_gaussian_regression`: for the regression arm if you include one.

**Families without closed-form Bayes-optimal, where you estimate Bayes risk by Monte Carlo on a
large fresh sample** (10⁵+ rows from the same DGP):

- `piecewise_constant`: axis-aligned random tree partition. Tree-favorable by construction.
- `rotated_piecewise_constant`: identical, then a random orthogonal rotation applied to X. This
  is the rotation-invariance axis from Grinsztajn et al. and is one of the most interesting
  probes here — trees lose their advantage under rotation, neural approaches shouldn't.
- `gp_smooth`: target sampled from a GP with controllable lengthscale. Smoothness axis.
- `high_frequency`: sinusoidal target with controllable frequency. The "irregular target
  function" axis.
- `scm_prior_control`: a reimplementation approximating TabPFN's own published SCM prior. **This
  is essential** — it is your in-prior anchor. If TabPFN doesn't show near-zero regret here,
  something is wrong with your measurement pipeline, not with TabPFN.

**Nuisance modifiers**, composable on top of any family:

- fraction of uninformative (pure-noise) features
- label noise rate (symmetric flip)
- class imbalance ratio
- heavy-tailed marginals (t-distributed or lognormal transforms)
- feature correlation / covariance condition number
- missingness rate (MCAR and MAR)
- categorical discretization with controllable cardinality

Write property tests: every DGP must be deterministic given a seed, and every closed-form Bayes
predictor must be verified against a Monte Carlo estimate of Bayes risk to within tolerance.

**Checkpoint report:** the family list with parameter ranges, and the Bayes-oracle verification
table. Then stop.

### M2 — Reference predictors and the regret metric

- **Primary metric: Bayes regret in log-loss.** `regret = logloss(model) − logloss(bayes)`.
  Log-loss because TabPFN is a Bayesian predictor and calibration is the thing that should
  degrade first when a dataset leaves the prior. Report accuracy and AUC as secondary only.
- Where Bayes is not closed-form, use the Monte Carlo estimate from M1 and record its standard
  error. Never report a regret smaller than its own error bar without flagging it.
- **Practical reference:** tuned LightGBM, RandomForest, kNN, and a small MLP. Define
  `strong_classical = best-of` on a validation split. This is what you compare TabPFN against
  on real data where no Bayes oracle exists.
- Fix the tuning budget for baselines and record it. Under-tuned baselines are the single most
  common reviewer objection to tabular papers.

**Checkpoint report:** metric definitions, baseline tuning protocol, and a sanity run showing
near-zero TabPFN regret on `scm_prior_control`. Then stop.

### M3 — Prior recovery sweep

- Latin hypercube (not full grid — it wastes budget) over the DGP parameter space, sized to the
  M0 budget.
- ≥5 seeds per cell. Record the full per-seed distribution, not just the mean.
- Output a tidy parquet: one row per (family, params, seed, model, metric).
- Produce the **regret surface**: TabPFN's Bayes regret as a function of DGP parameters. Fit a
  level set at a threshold τ (choose τ by a stated rule, e.g. the 75th percentile of regret on
  `scm_prior_control`, and justify it in writing).

**Checkpoint report:** regret surface plots per family, the fitted boundary, and the three DGP
axes on which TabPFN degrades most sharply. Then stop.

### M4 — The out-of-prior (OOP) score

The trap here: the regret surface is a function of *latent DGP parameters*, which you cannot
observe on a real dataset. The OOP score must be a function of *observable statistics only*.

Compute this fixed statistic vector on every synthetic dataset (and later, identically, on every
real one):

- `n`, `d`, `n/d`, `n_classes`, class-distribution entropy
- per-feature skewness and kurtosis (summarized: mean, max, fraction exceeding thresholds)
- covariance condition number and effective rank
- per-feature mutual information with the target (discretized); fraction of features below a
  low-MI threshold — the uninformative-feature proxy
- kNN label agreement at k ∈ {1, 5, 15} — the target-irregularity proxy
- **rotation-alignment score:** kNN label agreement in the original basis divided by mean
  agreement over several random orthogonal rotations. Directly operationalizes the
  axis-alignment axis. This is an original statistic; make sure it's well-defined and stable.
- depth-1 decision stump accuracy vs. a linear probe accuracy — a structural tree-vs-linear proxy
- estimated label noise via kNN disagreement rate
- missingness rate, max categorical cardinality

Fit `predicted_regret = g(statistics)` on the M3 synthetic data. Use a simple, interpretable
model (regularized linear, or a shallow GBDT with ≤4 depth). Report cross-validated performance
*on synthetic data* — this is not yet evidence about real data.

**Checkpoint report:** statistic definitions, `g`'s synthetic CV performance, and feature
importances. Then stop.

### M5 — Real dataset mapping

- Pull the TabArena suite plus the Grinsztajn et al. benchmark plus OpenML CC-18. Deduplicate.
- Filter to what fits TabPFN's stated operating range, but **also keep a deliberate
  out-of-range set** — those are the easy positives and you need them to check the score isn't
  trivially just "n is too big."
- Compute the identical statistic vector on every dataset. Any statistic that can't be computed
  on real data must be removed from `g` and `g` refit — do this now, before freezing.

**Checkpoint report:** dataset inventory, statistic distributions, and how real datasets sit
relative to the synthetic probe space (a projection plot). Then stop.

### M6 — PRE-REGISTRATION FREEZE

This is the milestone that makes the paper credible.

1. Run `g` on every held-out real dataset. For each, write predicted regret, predicted
   win/loss against `strong_classical`, and a confidence.
2. Write it all to `results/preregistration_<timestamp>.json`, including the decision threshold,
   the exact `g` coefficients, and a hash of the statistic-computation code.
3. `git commit`. Print the hash. Echo it into the JSON.
4. **Do not run TabPFN on the held-out real datasets before this commit exists.**

**Checkpoint report:** the commit hash and a summary of predictions. Then stop and confirm with
me before M7.

### M7 — Prospective validation

- Now run TabPFN and `strong_classical` on the held-out real datasets.
- Compute actual regret proxy (TabPFN log-loss minus `strong_classical` log-loss, since no
  Bayes oracle exists here — state this substitution explicitly, it's a real limitation).
- Score the frozen predictions: Spearman correlation predicted vs. actual, AUROC for the
  binary failure prediction, and a calibration plot.

### M8 — Baselines, ablations, figures

Three comparisons, all against the same frozen predictions:

- **(a) Mean-gap baseline** — predict the average gap for every dataset. This is what the
  meta-feature approach in arXiv 2605.28418 failed to beat. It is the bar.
- **(b) Black-box meta-feature regression** — same statistics, fit directly to real-data
  performance via cross-validation, with no synthetic prior grounding. This isolates whether the
  prior recovery contributed anything.
- **(c) Our prior-grounded OOP score.**

Ablations: drop the rotation-alignment statistic; drop the kNN-irregularity statistics; use only
`n`/`d`/`n_classes` (the trivial baseline).

Figures: regret surface, real-vs-synthetic projection, predicted-vs-actual scatter with the
pre-registration commit hash in the caption, calibration curve, ablation bar chart.

---

## 6. Things not to do

- Do not use any GPU code path, even if one is available.
- Do not re-open the pre-registration after seeing M7 results. If the prediction fails, the
  paper is a negative result — write that.
- Do not add DGP families, statistics, or datasets after M6.
- Do not report a result whose effect size is smaller than its seed-to-seed standard deviation.
- Do not under-tune the GBDT baselines. Reviewers will assume you did and you need the tuning
  protocol documented to rebut it.
- Do not exceed 25 core-hours on any single milestone without checking in first.
- Do not write the paper's claims into the code comments. Let the results decide.

---

## 7. Checkpoint report format

At every milestone, output exactly this:

```
MILESTONE: <id>
STATUS: complete | blocked
COMPUTE USED: <core-hours>, cumulative <core-hours>
KEY NUMBERS: <3–6 numbers that matter>
SURPRISES: <anything that contradicts the spec's assumptions>
NEXT: <what M<id+1> will do>
BLOCKERS / DECISIONS NEEDED: <or "none">
```

If any milestone's result contradicts a premise of this spec — for example if TabPFN shows large
regret on `scm_prior_control`, or if M0 shows the budget is 10× short — **stop and say so
plainly.** Do not work around a broken premise silently. A spec that turns out to be wrong is a
finding, not an obstacle.

---

## 8. Key related work to stay differentiated from

Read the abstracts before writing any framing text. The paper must state its delta from each of
these in the first two paragraphs.

- **McCarter, "What Exactly Has TabPFN Learned to Do?"** (arXiv 2502.08978) — characterizes
  learned biases qualitatively via black-box function approximation. **Our delta:** quantitative
  regret-based prior recovery, plus a predictive claim they do not make.
- **"Explaining Tabular Foundation Model Differences Through Meta-Features"** (arXiv 2605.28418)
  — predicts performance gaps from black-box meta-features; reports failure to beat a mean-gap
  baseline. **Our delta:** prior-grounded rather than black-box; this paper is our baseline (b),
  and its negative result is what motivates the approach.
- **Grinsztajn, Oyallon & Varoquaux** (NeurIPS 2022 D&B, arXiv 2207.08815) — the three
  inductive-bias challenges (irregular targets, uninformative features, rotation
  non-invariance). Our probe axes are deliberately built on theirs.
- **Nagler, "Statistical Foundations of Prior-Data Fitted Networks"** (ICML 2023) — theory that
  PFN consistency is not guaranteed under task shift. This is the theoretical justification for
  expecting out-of-prior failure at all; cite it as motivation.
- **Müller et al., PFNs** (ICLR 2022) and **Hollmann et al., TabPFN** (ICLR 2023) and **TabPFN
  v2** (Nature 2025) — the objects of study. Verify v2's prior details against the Nature paper
  directly, not secondary summaries.
- **Gupta, Sethi & Kumar, "TabPFN Through the Looking Glass"** (arXiv 2601.08181) — internal
  representation probing. Adjacent but mechanistic rather than predictive.

---

## 9. Timeline

Target AISTATS 2027 (Oct 8, 2026). Register an ICLR 2027 abstract by Sep 18 as a free hedge.

- Days 1–3: M0–M2
- Days 4–10: M3–M4
- Days 11–14: M5–M6 (freeze)
- Days 15–18: M7–M8
- Days 19–28: writing, with experiments frozen

If M0 shows the compute budget is short by more than 3×, cut the DGP parameter space rather than
the seed count. Seed count protects your error bars; probe density only affects resolution.

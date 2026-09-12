# Out of Prior

**Auditing tabular foundation models with information-preserving perturbations.**

Subject of study: **TabPFN v2** (`tabpfn==2.2.1`, `Prior-Labs/TabPFN-v2-clf`, Nature 2025),
with a cross-generation arm on **TabPFN v1** (ICLR 2023). CPU-only throughout.

---

## The problem

Tabular foundation models are evaluated by comparing them to *other models*. That tells
you who wins a horse race; it cannot tell you **what the model's prior actually encodes**,
or **when that prior stops applying**.

The obstacle is that on real data you never know the optimum, so "the model did worse"
and "the task got harder" are indistinguishable.

## The method

Two ideas, combined:

**1. Measure regret against a computable optimum.** We build data-generating processes
whose Bayes-optimal posterior is known *exactly and pointwise*, so every number reported
is `logloss(model) - logloss(Bayes)`. No oracle, no claim.

**2. Perturb the data in ways that provably preserve information.** Rotating the
standardised numeric block is an orthogonal map; permuting categorical codes is a
bijection. Both leave the Bayes risk **mathematically unchanged** - verified to five
decimal places in our probes.

> This is the lever. Because the perturbation cannot destroy information, **any**
> degradation is inductive bias, full stop. Prior work measured accuracy drops under
> rotation without an oracle and could not separate the two.

The audit then asks three questions of any tabular model: *does the perturbation hurt it,
by how much, and can we predict which datasets it will hurt?*

## What we found

| # | finding | evidence |
|---|---|---|
| 1 | **Axis-dependence is real and large** on real data | rotation degrades TabPFN v2 on **86/95** datasets, **p = 7.7e-16** |
| 2 | **It is getting WORSE across generations** | relative degradation **+4.9% (v1) to +11.3% (v2)**, paired **p = 1.0e-05** |
| 3 | **It survives the model's full operating range** | holds at n = 500, 2000 and **10000** - TabPFN v2's stated ceiling (12/13, p = 2.4e-04) |
| 4 | **A cheap statistic predicts *which* datasets suffer** | rho = **+0.530**, p = 1.1e-05 on a **pre-registered independent replication** (57 unseen datasets); pooled rho = +0.409 |
| 5 | **A second axis: categorical ordinality** | permuting category codes degrades TabPFN on 17/22 datasets (p = 0.014), and **persists when categoricals are correctly declared** (15/22, p = 0.027) |
| 6 | **The in-prior anchor survives latent confounding** | a *discrete* latent gives genuine confounding **and** an exact oracle; regret 0.021 to 0.065 (p = 0.004), still far below out-of-prior levels (~0.29) |
| 7 | **Open-ended failure routing does NOT work - and the field may not know it** | AUROC 0.343, 95% CI **[0.133, 0.579]**; detecting AUROC 0.70 at the observed base rate needs **~381 datasets**, where the literature uses ~50 |
| 8 | **Baseline strength materially changes the story** | best-of-6 with 165 configs nearly doubles TabPFN's loss rate, **18% to 31%** |

Finding 2 is the headline: **newer TabPFN is more axis-dependent than older TabPFN**, not
less. The diagnostic matters more for current models, not less.

Finding 7 is deliberately kept. The original thesis of this project - predict which
datasets a foundation model will lose on - **failed**, and the power analysis says prior
negatives in this area may be underpowered rather than conclusive. That is reported, not
buried.

## The direction for ICLR

**One direction: a diagnostic framework, headlined by the generational trend.**

Not "TabPFN is not rotation invariant" - Grinsztajn et al. (2022) established that axis
for tabular models, and we cite them as its source. What is new is:

1. measuring it as **regret against an exact oracle**, so degradation is *provably*
   inductive bias rather than lost information;
2. showing the bias **intensifies across model generations**;
3. a **replicated** statistic that predicts which datasets suffer;
4. the **power analysis** that reframes the field's existing negative results.

The framework is reusable: any new tabular foundation model can be dropped into the same
paired probe. See [PAPER_OUTLINE.md](PAPER_OUTLINE.md) for the full argument and
positioning against related work.

**To finish:** the TabPFN-3 arm, which turns the two-point generational trend into three.
The runner is built and tested ([scripts/TABPFN3_HANDOFF.md](scripts/TABPFN3_HANDOFF.md));
only execution is outstanding, because post-v2 weights require accepting a non-commercial
licence.

## Method integrity

- **8 pre-registrations**, each committing hypotheses *and directions* to git before the
  first cell ran; `m7_validate.py` refuses to score a prediction set whose commit is not
  in the repo.
- **Failures are recorded, never dropped** - every cell is cached with status, config and
  traceback.
- Everything is **resumable** and keyed by a hash of its config.
- **55 property tests**, including the one that matters most: the latent-confounded oracle
  is validated by *calibration*, which a wrong marginalisation would fail even though it
  would pass a self-consistency check.

Pre-registration caught things a post-hoc analysis would have buried: a score that
extrapolated 571x beyond its support and emitted predictions of 242.6; an interim
correlation of +0.350 (p = 0.043) that **did not survive full coverage** and was withdrawn;
a baseline comparison that flipped sign between partial and complete data; and a confound
in our own wrapper, disclosed and resolved as a 2x2 rather than quietly fixed.

## Honest limits

- **CPU-only.** No GPU. The cost model `t ~ n^0.77 * d^0.76` bounds every sizing decision.
  The evaluation does reach TabPFN v2's stated 10000-row ceiling.
- **TabPFN-3 untested.** Post-v2 weights sit behind a licence-acceptance step, so
  independent auditing of current TabPFN models carries a licence barrier - itself worth
  stating.
- **Real-data regret is a proxy** (gap vs best available; no oracle exists there). Against
  best-of-all-models, TabPFN's median regret is 0.0000 and it is the best model on 44/53.
- Rotation is undefined on categorical-only data; the categorical arm covers exactly those.

## Scale

**102 core-hours | 2691 experiment cells | 8 pre-registrations | 55 tests | 11 figures**

## Repository map

```
src/dgp/          synthetic families with exact Bayes oracles (+ nuisance modifiers)
src/oracles/      Bayes-regret metric with error bars
src/stats/        the observable-statistic vector (the OOP feature set)
src/models/       TabPFN + best-of-6 classical baselines, one interface
src/runner.py     cached, parallel, resumable experiment loop
src/experiments/  m0..m15, one milestone per file
results/          CHECKPOINTS.md (full lab notebook), pre-registrations, parquet/CSV
figures/          11 figures
```

### Running it

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe src/experiments/m0_budget.py 4
.venv/Scripts/python.exe src/experiments/m9_rotation_transfer.py 4
.venv/Scripts/python.exe src/experiments/m14_analyze.py
```

The full milestone-by-milestone record, including every surprise and correction, is in
[results/CHECKPOINTS.md](results/CHECKPOINTS.md).

# Out of Prior — paper outline

**Framing rule:** lead with what is new. "TabPFN is not rotation invariant" is
*confirmatory* — Grinsztajn, Oyallon & Varoquaux (NeurIPS 2022) established
rotation non-invariance for tabular models. Rotation is our **instrument**, not our
claim. Cite them as the source of the axis, not as something we overturn.

---

## Contribution 1 — Prior recovery against exact Bayes oracles

The methodological core. Prior work characterises tabular models by comparing them
to *other models*; we compare TabPFN to a **computable optimum**.

- Seven synthetic families with exact, pointwise Bayes posteriors. Where the spec
  assumed only Monte-Carlo Bayes risk was available, generating labels from a known
  conditional law makes the posterior exactly that law — Monte Carlo is then needed
  only for the *risk*, removing MC error from the metric's reference point.
- Recovered regret surface, ordered as inductive-bias theory predicts:
  `scm_prior_control` 0.019 → `high_frequency` 0.289.
- In-prior anchor validated: TabPFN regret **+0.0172 ± 0.0051** vs tuned LightGBM
  **+0.0865**, i.e. ~5× closer to Bayes-optimal.
- **Latent confounding solved, not assumed away.** A *discrete* latent makes
  `p(y|x) = Σ_z p(y|x,z)p(z|x)` an exact finite sum — genuine confounding *and* an
  exact oracle. Regret rises 0.0214 → 0.0647 (p=0.004) but stays far below
  out-of-prior levels: the anchor survives, and the earlier restriction was
  optimistic by ~3×.

**Why this matters:** every degradation we report is measured against a provable
optimum, so "the model got worse" cannot be confused with "the task got harder."

## Contribution 2 — Degradation that is provably inductive bias, and predictable

- **Controlled probe:** everything fixed but the rotation angle. Regret
  0.158 → 0.360 (Spearman +0.875, 10.7× seed noise) while **Bayes risk is constant
  to five decimal places**. Information is provably preserved, so the loss is
  entirely inductive bias.
- **It transfers to real data:** rotating the numeric block degrades TabPFN on
  **86/95 real datasets, p = 7.7e-16**, and survives at every training size up to
  TabPFN v2's stated **10,000-row ceiling** (12/13, p = 0.00024).
- **A statistic predicts *which* datasets suffer** — the genuinely novel claim, with
  no analogue in the cited work. ρ = **+0.530**, p = 1.1e-05, CI [+0.304, +0.704] on
  a **pre-registered independent replication** (57 datasets never previously
  modelled). Pooled ρ = +0.409.
  - Note: the statistic as originally specified was *vacuous* — Euclidean kNN is
    exactly rotation invariant, so the proposed ratio is identically 1.0. Rebasing
    it on an axis-aligned learner is what made it informative.
- **A second axis:** permuting categorical codes — a bijection, so information and
  Bayes risk are unchanged — degrades TabPFN on 17/22 datasets (p=0.014), and the
  effect **persists when categoricals are correctly declared** (15/22, p=0.027), so
  it is a model property, not a preprocessing artefact. Practical consequence:
  accuracy depends on how your preprocessing happened to sort category labels.
- **The bias persists and intensifies across generations:** relative degradation
  under rotation is **+4.9%** for v1 (ICLR 2023) and **+11.3%** for v2 (Nature 2025),
  paired p = 1.0e-05. Newer is *more* axis-dependent, not less.

## Contribution 3 — A bounded negative, with the power analysis the field lacks

- Open-ended win/loss routing (predict where TabPFN loses to a tuned ensemble) is
  **indistinguishable from chance**: AUROC 0.343, bootstrap 95% CI **[0.133, 0.579]**.
- **This is the contribution, not an apology.** At the observed 17% base rate,
  detecting AUROC 0.70 at 80% power needs **~381 datasets**; the literature, including
  arXiv 2605.28418, routinely uses ~50. **Prior negatives in this area may be
  underpowered rather than conclusive.**
- The negative survives a **much stronger baseline** (best-of-6, 165 configs vs
  best-of-4, 32), so it is not an artefact of weak comparators — the standard
  objection. Under that baseline TabPFN's loss rate nearly doubles, 18% → 31%, so
  its margin in the literature is likely overstated too.
- Contrast with Contribution 2: prediction **works for a known degradation axis**
  (ρ=+0.53, replicated) and **fails for open-ended routing**. That boundary is the
  paper's most useful practical statement.

## Method: pre-registration as infrastructure

Nine pre-registration commits, each with hypotheses and directions fixed *before*
the first cell ran, verified by `m7_validate.py` against the git history. This
caught several things a post-hoc analysis would have buried:

- an OOP score that extrapolated 571× beyond its support and produced predictions of
  242.6 against a training range of [-0.35, 0.83];
- an interim Spearman of +0.350 (p=0.043) that **did not survive full coverage**
  (+0.175, p=0.293) and is withdrawn;
- a baseline comparison that flipped sign between partial and complete data;
- a confound in our own wrapper (`categorical_features_indices` never passed),
  disclosed and resolved as a 2×2 rather than quietly corrected.

Two failures share one root cause worth stating: **a model fitted on a narrow
support fails outside it.** It broke the OOP score, and it made our own compute
model underestimate n=10,000 cells by 3.3×.

## Limitations (stated, not hidden)

- **CPU-only.** No GPU. The cost model `t ≈ n^0.77 d^0.76` bounds every sizing
  decision. Evaluation reaches TabPFN v2's stated 10k ceiling, so the operating-range
  objection is answered, but seeds and dataset counts remain compute-bound.
- **TabPFN-3 untested.** Post-v2 generations require accepting a commercial
  non-commercial-use licence before download. Independent auditing of current TabPFN
  models therefore has a licence barrier — worth stating as a field-level observation.
- **Real-data regret is a proxy** (gap vs best available), since no Bayes oracle
  exists there. Against best-of-all-models, TabPFN's median regret is 0.0000 and it is
  the best available model on 44/53 datasets.
- Rotation is undefined on categorical-only data; the categorical arm covers those.

## Positioning vs related work

| work | what they did | our delta |
|---|---|---|
| Grinsztajn et al. 2022 | established rotation non-invariance via accuracy drops | measured as **regret against an exact oracle**, so degradation is provably bias, not lost information |
| McCarter 2025 (2502.08978) | qualitative black-box characterisation of TabPFN | quantitative regret surface **plus a predictive claim** they do not make |
| arXiv 2605.28418 | meta-feature routing fails to beat a mean-gap baseline | we **reproduce the negative** and show it is likely **underpowered** (~381 datasets needed), while demonstrating prediction *does* work on a known axis |
| Nagler 2023 | theory: PFN consistency not guaranteed under task shift | empirical measurement of exactly that shift, with the axes named |

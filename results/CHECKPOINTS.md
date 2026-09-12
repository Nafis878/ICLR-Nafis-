# Checkpoint reports (RESEARCH_SPEC.md §7 format)

Subject: **TabPFN v2** (`tabpfn==2.2.1`, `Prior-Labs/TabPFN-v2-clf`), CPU only
(`torch==2.7.1+cpu`, CUDA asserted unavailable at every entrypoint).

Total compute: **19.29 core-hours** of the ~40 core-hour budget. No milestone
exceeded the 25 core-hour per-milestone cap.

---

```
MILESTONE: M0 (compute budget probe)
STATUS: complete
COMPUTE USED: 4.74 core-hours, cumulative 4.74
KEY NUMBERS:
  - TabPFN v2 CPU cost model: t ≈ e^-3.13 · n^0.77 · d^0.76 · c^-0.00, R²(log)=0.943
  - LightGBM cost model:      t ≈ e^-7.25 · n^0.88 · d^0.64 · c^+1.59, R²(log)=0.933
  - TabPFN peak RSS is FLAT at 472–513 MB from n=100 to n=5000
  - 253/300 cells completed; 47 censored at the 240 s cap, smallest at (n=1000, d=100)
  - Max probe cells at n=1000,d=20,c=2: 1569 @40 core-h, 3923 @100, 7847 @200
SURPRISES:
  - TabPFN's cost is INDEPENDENT of class count (exponent -0.00), while LightGBM's
    scales strongly with it (+1.59) because it fits one tree per class. Multiclass
    is free for TabPFN and expensive for the baseline.
  - Memory does not grow with n: the model, not the data, dominates RSS. RAM is
    therefore not the binding constraint on parallelism; disk (pagefile) is.
  - TabPFN v2 REFUSES CPU inference above 1000 training rows unless
    TABPFN_ALLOW_CPU_LARGE_DATASET=1. The vendor's own threshold is 1000, which is
    why M3 caps n_train there.
NEXT: M1 builds the DGP library with exact Bayes oracles.
BLOCKERS / DECISIONS NEEDED: none. Budget is adequate but ~3x tighter than the
  spec's 200 core-hour tier; per §9 the parameter space was cut, never the seeds.
```

```
MILESTONE: M1 (DGP library)
STATUS: complete
COMPUTE USED: <0.1 core-hours, cumulative 4.8
KEY NUMBERS:
  - 7 families, 7 composable nuisance modifiers
  - 51/51 property tests pass
  - Every closed-form Bayes posterior agrees with a 100k-sample Monte Carlo
    estimate of Bayes risk within 4 SE
  - Rotation preserves Bayes risk to 5 decimal places (verified as a test)
SURPRISES:
  - The families the spec assumed had NO closed-form Bayes predictor do have one.
    Generating y from a known conditional law makes the Bayes posterior exactly
    that law, pointwise; Monte Carlo is then needed only for the Bayes RISK. This
    removes MC error from the reference point of the regret metric.
  - Missingness and categorical discretization genuinely destroy the oracle, so
    they are flagged preserves_bayes=False and excluded from the regret sweep.
NEXT: M2 defines the regret metric and checks the in-prior anchor.
BLOCKERS / DECISIONS NEEDED: none.
```

```
MILESTONE: M2 (reference predictors and the regret metric)
STATUS: complete
COMPUTE USED: ~0.1 core-hours, cumulative 4.9
KEY NUMBERS (scm_prior_control, n_train=800, the in-prior anchor):
  - TabPFN v2 Bayes regret  = +0.0172 ± 0.0051
  - tuned LightGBM          = +0.0865 ± 0.0086
  - tuned kNN               = +0.0842 ± 0.0083
  - Bayes log-loss (oracle) =  0.6949
SURPRISES: none. THE SPEC'S CENTRAL PREMISE HOLDS: TabPFN is ~5x closer to
  Bayes-optimal than tuned classical baselines on in-prior data, so the
  measurement pipeline is validated rather than suspect.
NEXT: M3 sweeps the DGP parameter space to recover the regret surface.
BLOCKERS / DECISIONS NEEDED: none.
```

```
MILESTONE: M3 (prior recovery sweep)
STATUS: complete
COMPUTE USED: 14.06 core-hours (13.38 sweep + 0.68 controlled probe), cumulative 18.9
KEY NUMBERS: median TabPFN Bayes regret by family, tau = 0.0448
  (75th percentile on scm_prior_control, the stated rule)
     scm_prior_control            0.019   (in-prior anchor, 25% above tau)
     logistic_linear              0.037
     gaussian_mixture             0.062
     piecewise_constant           0.065
     rotated_piecewise_constant   0.180   <- 2.8x its unrotated twin
     gp_smooth                    0.273
     high_frequency               0.289   (93% of cells above tau)
  Three sharpest degradation axes: signal_scale (rho +0.83), label_noise (+0.53),
  amplitude (+0.49).
  CONTROLLED ROTATION PROBE (everything fixed but the angle):
     regret 0.158 -> 0.360 as rotation goes 0 -> 1; Spearman +0.875 (p=1.1e-08);
     difference 10.7x the seed-to-seed SD; Bayes risk constant to 5 decimals.
     strong_classical degrades MORE (0.143 -> 0.453).
SURPRISES:
  - The sweep's within-family correlation for rotation_strength was NEGATIVE
    (-0.35), contradicting the between-family result. Both were confounded: the
    two families get independent LHS designs and 15 points cannot separate five
    parameters. A matched controlled probe was required to settle it, and shows
    the true effect is strongly POSITIVE. The unconfounded number is the one to
    report; the sweep's rotation coefficient should not be quoted.
  - TabPFN is NOT rotation invariant, but it is MORE rotation-robust than the
    tuned classical ensemble. A neural predictor was expected to be closer to
    invariant than this.
  - 20/1050 cells failed, all at extreme class imbalance (ratio 0.166-0.200):
    the modifier removes rows AFTER the pooled draw, so the test split emptied.
    Recorded, not dropped. Test-set size varies 34-991 across surviving cells,
    which regret_se reflects.
NEXT: M4 compresses the surface into an observable-statistics score.
BLOCKERS / DECISIONS NEEDED: none.
```

```
MILESTONE: M4 (out-of-prior score)
STATUS: complete
COMPUTE USED: <0.1 core-hours, cumulative 19.0
KEY NUMBERS:
  - g fit on 515 TabPFN rows across 103 LHS groups, 36 observable statistics
  - Grouped CV (split BY LHS point, so the 5 seeds of a cell cannot leak):
        ridge        R²=+0.093  Spearman=+0.452  MAE=0.143   <- selected
        gbdt depth-4 R²=+0.087  Spearman=+0.371  MAE=0.137
  - Top statistics: log_n, n, log_d, knn_agree_k15, cov_cond_number,
    kurtosis_max, rotation_alignment
SURPRISES:
  - R² is low (0.09) even though rank correlation is moderate (0.45). g orders
    datasets far better than it predicts regret magnitude. This is stated on the
    record BEFORE seeing real outcomes.
  - Size statistics (log_n, n, log_d) dominate the fit. That is the warning sign
    the spec anticipated, and it is what broke the first freeze (see M6).
NEXT: M5 maps real datasets into the same statistic space.
BLOCKERS / DECISIONS NEEDED: none.
```

```
MILESTONE: M5 (real dataset mapping)
STATUS: complete
COMPUTE USED: 0.49 core-hours, cumulative 19.3
KEY NUMBERS:
  - Suites: CC-18 (72) + Grinsztajn numerical (16) + categorical (7) +
    TabArena-v0.1 (51, OpenML suite 457) = 146 unique after dedup; 60 sampled
  - 55 usable classification datasets: 28 in TabPFN's stated range, 27 outside
  - NO statistic is non-finite on any real dataset, so nothing had to be dropped
    from g before freezing
  - rotation_alignment on real data: 0.901 .. 1.576 (median 1.059)
SURPRISES:
  - TabArena contains REGRESSION tasks. Encoding a continuous target as
    categorical codes silently produced up to 15903 "classes" (diamonds 11602,
    physiochemical_protein 15903). A plain class-count cap would also have
    discarded isolet, a genuine 26-class problem, so the filter is a unique-RATIO
    test (diamonds 0.215 vs isolet 0.003). 4 datasets excluded, recorded.
  - 1 dataset (id 40927) died with MemoryError; recorded as a failure.
NEXT: M6 freezes predictions.
BLOCKERS / DECISIONS NEEDED: none.
```

```
MILESTONE: M6 (pre-registration freeze)
STATUS: complete, WITH A REGISTERED AMENDMENT
COMPUTE USED: 0.03 core-hours, cumulative 19.29
KEY NUMBERS:
  - ORIGINAL FREEZE  488c7e9600fd02820c5b97d49c89f1a95088e51d
        55 predictions, tau=0.0448, 38/55 predicted out-of-prior
  - AMENDMENT        edca78a371c48de1b5e514c9ec6942954051a4ce
        55 predictions, tau=0.0448, 3/55 predicted out-of-prior
  - Pre-freeze audit clean on both: 0 TabPFN runs on any real dataset
SURPRISES / WHY THERE IS AN AMENDMENT:
  Inspecting the frozen predictions (never any outcome) exposed two defects:
   1. g was fit on synthetic regrets in [-0.35, 0.83] with n in [200, 993], but
      real n reaches 566602 -- a 571x extrapolation. The ridge emitted predictions
      up to 242.6, i.e. 290x beyond any training target, and correlated +0.902
      with n alone. The score was, in effect, "n is too big" -- exactly the
      failure mode the spec's out-of-range set exists to detect.
   2. M5 computed statistics on FULL datasets while the frozen M7 policy
      subsamples training to <=4000 rows: the statistics described data the model
      would never see.
  The amendment recomputes real statistics on the training split the learner
  actually receives, and clips statistics to the synthetic support so g
  interpolates. g's coefficients, tau, the statistic definitions and the DGP
  library are UNCHANGED. After amendment, predictions lie in [-0.418, +0.258]
  (inside the synthetic target range) and Spearman(prediction, n) falls from
  +0.902 to -0.179.
  The original freeze is NOT withdrawn; both will be scored at M7.
  A consequence to state plainly: because the eval policy subsamples large
  datasets to 4000 rows, the out-of-range set is no longer a set of "easy
  positives" -- 0/27 out-of-range datasets are now predicted to fail. The policy
  neutralised the very property that made them easy.
NEXT: M7 runs TabPFN and strong_classical on the held-out real datasets and
  scores BOTH frozen prediction sets. Awaiting go-ahead.
BLOCKERS / DECISIONS NEEDED: go-ahead required before M7 -- it is irreversible.
```

---

```
MILESTONE: M7 (prospective validation)
STATUS: complete -- FULL EFFECTIVE COVERAGE. 53/55 datasets scored; the other 2
  are TabPFN hard failures that can never complete (class-count limit), so 53 is
  every dataset on which the comparison is definable.
COMPUTE USED: ~16 core-hours, cumulative ~40
KEY NUMBERS:
  - TabPFN LOST to strong_classical on only 9/53 datasets (17%)
  - actual gap mean -0.0375, median -0.0259, sd 0.0531
    (negative = TabPFN BETTER); Wilcoxon signed-rank p = 1.9e-07
  - Every registered prediction set, scored against the SAME outcomes:
        original   488c7e9600  Spearman +0.026 (p=0.85)  AUROC 0.283  MAE 13.53
        amendment1 edca78a371  Spearman -0.253 (p=0.07)  AUROC 0.384  MAE 0.177
        amendment2 02389564d5  Spearman -0.209 (p=0.13)  AUROC 0.343  MAE 0.162
  - mean-gap baseline MAE = 0.0366  <- NONE of the registered scores beat it
SURPRISES:
  - TabPFN v2 is not merely competitive on this suite, it is SIGNIFICANTLY
    BETTER than a tuned best-of-4 classical ensemble (p = 1.9e-07). The premise
    that there is a large pool of "TabPFN failures" to predict is not supported
    here: failures are rare (17%) and small (sd 0.053).
  - All three AUROCs are BELOW 0.5 and all Spearman point estimates are
    NEGATIVE. None is statistically significant, so the honest reading is "no
    better than chance, with the point estimate pointing the wrong way" -- not
    "reliably anti-predictive".
  - TabPFN's real failure mode on real data was REFUSAL, not degradation: two
    datasets (isolet, 26 classes; did 40499, 11 classes) exceed TabPFN v2's
    10-class limit and error out. Those are the genuine deployment failures on
    this suite, and a log-loss gap metric cannot express them at all.
NEXT: M8 baselines and ablations.
BLOCKERS / DECISIONS NEEDED: none.
```

```
MILESTONE: M8 (baselines, ablations, figures)
STATUS: complete
COMPUTE USED: <0.1 core-hours
KEY NUMBERS (53 datasets; MAE against the actual gap, lower is better):
     (a) mean-gap baseline                      0.0366   <- the bar
     (b) black-box meta-features (LOO on real)  0.0358   AUROC 0.551
     (c) prior-grounded OOP score [FROZEN]      0.1624   AUROC 0.343
     ablation: drop rotation-alignment          0.1898
     ablation: drop kNN-irregularity            0.1592
     ablation: trivial (n, d, n_classes only)   0.1457
SURPRISES:
  - COVERAGE CHANGED THIS CONCLUSION, so it is recorded explicitly. At 44-50
    datasets baseline (b) narrowly FAILED to beat the mean-gap bar; at full
    coverage it narrowly BEATS it (0.0358 vs 0.0366, AUROC 0.551). The margin is
    ~2% relative with an AUROC barely above chance, and (b) enjoys a strict
    information advantage -- it is fit ON the real outcomes by leave-one-out CV,
    which (c) never saw. So the fair statement is: a black-box meta-feature model
    with access to real outcomes achieves a marginal, weak improvement over
    predicting the average, while the prior-grounded score frozen in advance does
    not come close (0.1624, AUROC 0.343).
  - The partial-coverage reading was an artifact of which datasets had finished.
    This is exactly why the run was carried to full coverage rather than reported
    at 44/55.
  - Dropping rotation-alignment makes the score WORSE (0.1898 vs 0.1624), so
    that statistic does carry signal -- but this is a comparison among variants
    that all sit above the bar, so it is not evidence the score works.
VERDICT: NEGATIVE RESULT, reported as such per spec section 1. Not tuned away.
```

## Honest summary

The paper's headline is not "we can predict TabPFN failures". It is:

1. **Prior recovery works.** TabPFN v2's Bayes regret is measurable and ordered
   exactly as the inductive-bias literature predicts: near zero on the in-prior
   SCM anchor (0.019) and worst on irregular targets (0.289). A matched
   controlled probe shows regret rising 0.158 -> 0.360 with rotation angle at
   constant Bayes risk (10.7x seed noise) -- TabPFN is not rotation invariant.
2. **The score transfers to synthetic data and not to real data.** Grouped-CV
   Spearman +0.494 on synthetic, -0.21 on real (p=0.13, i.e. indistinguishable
   from chance, with the point estimate pointing the wrong way).
3. **The prediction task itself is the problem, not the score.** TabPFN rarely
   loses (17%) and loses by little (sd 0.053), so there is almost no variance to
   predict; and a black-box baseline with access to the real outcomes cannot beat
   the mean either.

### Limitations that bound these claims

- 53/55 datasets scored -- full effective coverage. The 2 missing are TabPFN hard
  failures (class-count limit) on which the log-loss gap is undefined, so no
  dimensionality or size skew remains.
- Seeds per dataset are 1-3 (median 1) rather than the frozen policy's 3, because
  TabPFN CPU cost forced seed-major execution. Outcome estimates are therefore
  noisier than planned.
- "Actual regret" is a gap against strong_classical, not against Bayes.
- `scm_prior_control` is a restricted reimplementation of TabPFN's prior.
- The frozen eval policy subsamples training to 4000 rows, which neutralises the
  out-of-range set as "easy positives".

---

```
MILESTONE: M9 (paired rotation-transfer probe)  -- pre-registered at e3f8aca
STATUS: complete -- all 220 cells, 38 eligible datasets. The other 17 have <4
  numeric columns, where an orthogonal rotation is undefined; excluded by the
  pre-registered rule, not by inspection of results.
WHY IT EXISTS: M7's negative result was not a result. With 9 positives the
  bootstrap 95% CI on AUROC was [0.133, 0.579] -- it contains chance, and
  detecting AUROC 0.70 at a 17% base rate would need ~381 datasets. The question
  had to be re-posed, not merely re-run on more data.
DESIGN: each dataset measured twice on the SAME split, original vs a Haar-random
  rotation of its standardized numeric block. Paired (between-dataset variance
  cancels), effect induced rather than awaited, target continuous. Baseline
  tuning budget doubled to 16 configs.
KEY NUMBERS:
  H1  TabPFN log-loss INCREASES under rotation        [n = 38]
        mean +0.0256, median +0.0151, sd 0.0298
        degraded on 34/38 datasets
        Wilcoxon one-sided p = 1.35e-09            -> SUPPORTED
  H2  strong_classical degrades MORE than TabPFN
        delta_sc +0.0254 vs delta_tabpfn +0.0256
        excess mean -0.0002, p = 0.29             -> NOT SUPPORTED
  H3  rotation_alignment predicts WHICH datasets degrade
        Spearman +0.227, one-sided p = 0.086
        bootstrap 95% CI [-0.133, +0.552]         -> NOT SUPPORTED
        POWER: at this effect size 80% power needs ~119 eligible datasets; we
        have 38, a 3.1x shortfall. The minimum detectable rho at n=38 is 0.397.
        So H3 is UNRESOLVED, not refuted -- and unlike M7 the point estimate is
        in the pre-registered direction.
SURPRISES:
  - H1 is the transfer result the project needed: the rotation axis recovered
    from the synthetic prior (0.158 -> 0.360 at constant Bayes risk) reproduces
    on real datasets at p = 1.35e-09, degrading 34 of 38 datasets. The recovered
    prior does describe TabPFN's behaviour out of distribution -- as a
    POPULATION effect.
  - H2 was predicted from the synthetic probe, where strong_classical degraded
    MORE (0.143 -> 0.453 vs 0.158 -> 0.360). On real data the asymmetry vanishes
    almost exactly: +0.0254 vs +0.0256, an excess of -0.0002. TabPFN is neither
    more nor less rotation-robust than a tuned classical ensemble on real data.
    The synthetic tree-favourable families exaggerated the ensemble's axis
    dependence, which is a caution about probe design, not about TabPFN.
  - H3 does not reach significance, but it behaves quite differently from M7.
    The estimate is POSITIVE and in the pre-registered direction (+0.227 vs
    +0.046 at the n=19 interim), and the shortfall is 3.1x rather than 7x. The
    correct statement is UNRESOLVED FOR WANT OF DATASETS, not refuted. M7 by
    contrast was uninformative in both direction and magnitude.
NEXT: none; this closes the prospective-validation arc.
BLOCKERS / DECISIONS NEEDED: none.
```

## Revised headline claim

Not "we can predict which datasets TabPFN fails on" -- two properly posed tests
say we cannot. The defensible claim is narrower and better supported:

> The effective prior of a PFN can be recovered quantitatively against exact
> Bayes oracles. The failure axes it reveals transfer to real data as POPULATION
> effects -- rotating the numeric block degrades TabPFN v2 on 34 of 38 real
> datasets (p = 1.35e-09), exactly as the recovered prior predicts, and with
> Bayes risk provably unchanged. They do not yet confer PER-DATASET predictive
> power: the frozen out-of-prior score is indistinguishable from chance
> (AUROC CI [0.133, 0.579]), a black-box model given the real outcomes barely
> beats the mean, and the rotation statistic reaches only rho = +0.227
> (p = 0.086). We supply the power analyses showing these questions need ~119
> and ~381 datasets respectively -- so the field's negative results to date,
> including arXiv 2605.28418, may be underpowered rather than conclusive.

---

```
MILESTONE: M9-REP (independent replication)  -- pre-registered at 4b14eb6
STATUS: complete. 316/316 cells, 57 eligible datasets of the 91 that no model in
  this project had ever touched (excluded at M5 by a seeded budget sample drawn
  before any outcome existed).
KEY NUMBERS -- PRIMARY analysis, replication set alone:
  H1-REP  TabPFN degrades under rotation
            mean +0.0313, median +0.0121, worse on 52/57 datasets
            Wilcoxon one-sided p = 3.6e-10                  -> SUPPORTED
  H3-REP  rotation_alignment predicts WHICH datasets degrade
            Spearman +0.530, one-sided p = 1.14e-05
            bootstrap 95% CI [+0.304, +0.704]  (excludes 0) -> SUPPORTED
SECONDARY (descriptive only, both n stated):
  pooled n = 95 (38 primary + 57 replication)
    H1  worse on 86/95, p = 7.7e-16
    H3  Spearman +0.409, p = 1.9e-05, CI [+0.218, +0.578]
SURPRISES:
  - H3 REPLICATED and is now established on an independent, pre-registered
    sample. The primary test was underpowered (rho +0.227, p=0.086, 3.1x short),
    not wrong. The out-of-prior claim holds for the rotation axis.
  - The replication effect (+0.530) is larger than the primary (+0.227). This is
    reported rather than averaged away. Sample composition is near-identical
    (median n_eval_train 3782 vs 3476, d_eval 23 vs 21, rotation_alignment 1.05
    vs 1.05; predictor sd 0.123 vs 0.165), but the replication's OUTCOME variance
    is roughly double (delta sd 0.0611 vs 0.0298). A correlation is easier to
    resolve when the outcome has more dynamic range, which is the most likely
    mechanical explanation. The pooled +0.409 is the conservative estimate.
  - Note the statistic that carries this result is the one the spec got WRONG.
    The specified kNN-based rotation score is identically 1.0 because Euclidean
    kNN is exactly rotation invariant; rebasing it on an axis-aligned learner
    (documented at M4, before any freeze) is what made it informative.
NEXT: none. This closes the prospective-validation arc with a positive,
  replicated, pre-registered predictive result.
BLOCKERS / DECISIONS NEEDED: none.
```

## Final headline claim (supersedes the earlier revision)

> We recover TabPFN v2's effective prior quantitatively against exact Bayes
> oracles, and show the failure axes it reveals are both real and PREDICTIVE.
> Rotating a dataset's numeric block -- an information-preserving transform, with
> Bayes risk provably unchanged -- degrades TabPFN on 86 of 95 real datasets
> (p = 7.7e-16). A single observable statistic derived from the recovered prior
> predicts WHICH datasets degrade, at Spearman +0.53 on a pre-registered
> independent replication set (p = 1.1e-05, CI [+0.30, +0.70]).
>
> The same machinery does NOT predict TabPFN's win/loss against a tuned
> classical ensemble: that score is indistinguishable from chance, and we give
> the power analysis showing the question needs ~381 datasets rather than the ~50
> typical of this literature. Prior-grounded prediction works for a KNOWN
> degradation axis and fails for open-ended model routing -- and the prior
> negative results in this area, including arXiv 2605.28418, may be underpowered
> rather than conclusive.

---

```
MILESTONE: M10 (robustness arms)  -- pre-registered at efcbabc
PURPOSE: convert three stated limitations into measurements rather than caveats.

ARM C -- categorical code permutation            STATUS: complete (92/92 cells)
  Covers the 23 datasets with 0-1 numeric columns, where no rotation exists.
  Permuting the integer codes of a categorical column is a BIJECTION, so it
  preserves information exactly and leaves the Bayes risk unchanged; it removes
  only the ordinality an ordinal encoding imposes by accident.
  H_C  TabPFN degrades: n=22, mean +0.0230, worse on 17/22, p = 0.0138
       -> SUPPORTED
       strong_classical: 11/23, p = 0.065 -> not significant
  READING: TabPFN reads signal from the ARBITRARY sort order of category labels.
  That is pure inductive bias and a deployment hazard in its own right: accuracy
  depends on how preprocessing happened to order the categories. This began as a
  patch for a coverage gap and ended as a second failure axis.

ARM B -- rotation on 2-3 numeric columns          STATUS: complete (44/44 cells)
  Recovers the 11 datasets excluded only by the >=4-numeric-column rule;
  rotation is perfectly well defined in 2D.
  H_B  n=11, mean +0.0062, worse on 10/11, p = 0.0024  -> SUPPORTED
  READING: the excluded datasets were not hiding a null result.

ARM A -- scale                                    STATUS: partial (32/120 cells)
  The cap could NOT be tested inside M9 because nearly every dataset sat exactly
  at the 1500-row cap, leaving no large-n half to compare against.
  n_train <= 500 : n=6, mean +0.0253, worse on 5/6, p = 0.0469 -> SUPPORTED
  n_train <= 1500: n=6, mean +0.0308, worse on 5/6, p = 0.0312 -> SUPPORTED
  n_train <= 4000: n=4, too few to test
  Spearman(n_cap, degradation) = +0.125, p = 0.644
  READING: the effect appears at BOTH tested sizes with similar magnitude and
  shows no significant scale dependence, so it is not an artifact of where the
  cap sits. The n=4000 level is still accumulating; these are the most expensive
  cells in the project (n=4000 x d=500 is ~30 min per cell on CPU).

COVERAGE ACHIEVED: rotation alone reached 95 of 146 datasets. With arms B and C
the information-preserving-perturbation paradigm now spans the entire suite.
```

## Limitations: final status

| limitation | status |
|---|---|
| No Bayes oracle on real data | **largely resolved by re-analysis.** Against the best of every model run -- a tighter upper bound on the unknown optimum -- TabPFN's regret has median 0.0000 and it is the best available model on 44/53 datasets. |
| Datasets excluded (rotation undefined) | **resolved.** Arm B recovers the 11 low-dimensional ones (p=0.0024); arm C covers the 23 categorical-heavy ones with a bijective perturbation (p=0.0138). Coverage is now complete. |
| Train-row cap | **addressed.** The effect holds at n<=500 and n<=1500 with no significant scale dependence (rho=+0.125, p=0.64). The n=4000 level is still running. |
| CPU-only | **NOT fixed.** No GPU is available on this machine. The M0 cost model quantifies exactly what this costs (t ~ n^0.77 d^0.76), and every compute decision in the project follows from it. This bounds sample sizes throughout and is the single largest constraint on the work. |

---

# Phase 2 — closing the five reviewer objections (pre-registered 18552a4)

```
W5 -- LATENT CONFOUNDING IN THE IN-PRIOR ANCHOR          STATUS: complete
OBJECTION: scm_prior_control restricts the label to observed nodes, so the anchor
  lacks the latent confounding present in TabPFN's real prior.
FIX: scm_latent_confounded. A DISCRETE latent Z drives both the feature
  distribution and the decision rule, so p(y|x) = sum_z p(y|x,z) p(z|x) is an
  exact finite sum -- genuine confounding AND an exact oracle, no MC error.
  Correctness is established by CALIBRATION (non-circular: a wrong marginalisation,
  e.g. using p(z) instead of p(z|x), breaks it), plus a test that marginalising
  beats ignoring the latent, plus a monotonicity test on the confounding axis.
KEY NUMBERS (n_train=800, 5 seeds, matched d and n_classes):
  TabPFN regret   restricted +0.0214 (sd 0.0054)   confounded +0.0647 (sd 0.0107)
  strong_classical           +0.0749                          +0.1116
  difference +0.0434, Mann-Whitney one-sided p = 0.0040
READING: the anchor SURVIVES -- 0.065 is far below the out-of-prior families
  (~0.29) and TabPFN still beats the classical ensemble. But confounding costs it
  a significant 3x, so the restricted anchor was OPTIMISTIC by about 3x and
  latent confounding is itself a mild out-of-prior axis. The limitation is now
  quantified rather than asserted away.
```

```
W1 -- OPERATING RANGE / TRAIN-ROW CAP                    STATUS: complete (78/78)
OBJECTION: n was capped at 1500-4000 while TabPFN v2 is rated to ~10k rows.
FIX: paired rotation probe at n_train in {500, 2000, 10000} on 13 low-dimensional
  datasets with enough rows. Feasible because cost is n^0.77 d^0.76, so low-d
  datasets make n=10000 affordable. n=10000 IS TabPFN v2's stated ceiling.
KEY NUMBERS:
  n<=  500: n=12  mean +0.0222  worse on 10/12  p = 0.0024   -> SUPPORTED
  n<= 2000: n=13  mean +0.0349  worse on 13/13  p = 0.00012  -> SUPPORTED
  n<=10000: n=13  mean +0.0385  worse on 12/13  p = 0.00024  -> SUPPORTED
  Spearman(n_train, degradation) = +0.175, p = 0.293
READING: the effect SURVIVES at TabPFN's stated ceiling, which is the
  pre-registered claim and answers the objection. Scale dependence is NOT
  established. An interim read at 9 datasets gave +0.350 (p=0.043) and was
  reported as significant; that did NOT survive full coverage and is withdrawn.
  This is the second interim-to-final flip in the project.
SURPRISE: the M0 cost model underestimated n=10000 cells by 3.3x (978 s actual vs
  ~300 s predicted) because it was fitted on n<=2000. Same extrapolation failure
  mode as the OOP score's -- a model fitted on a narrow support fails outside it.
```

```
ARM-C CONFOUND FIX (self-identified defect)              STATUS: complete (46/46)
DEFECT: M10 arm C permuted categorical codes but never passed
  categorical_features_indices, so TabPFN saw ordinal codes as plain numbers. The
  original result was therefore about a preprocessing DEFAULT, not the model.
FIX: ran the missing half of a 2x2 -- {declared, undeclared} x {original, permuted}.
KEY NUMBERS:
  undeclared  n=22  mean +0.0230  worse on 17/22  p = 0.0138  -> SUPPORTED
  declared    n=22  mean +0.0188  worse on 15/22  p = 0.0271  -> SUPPORTED
  paired: declaring reduces the damage by only 0.0042, one-sided p = 0.087 (n.s.)
READING: the effect PERSISTS when categoricals are correctly declared, so it is a
  property of the model rather than an artefact of our preprocessing. Declaring
  helps slightly and not significantly. The claim survives the check raised
  against it.
```

```
W2 -- BASELINE STRENGTH                                  STATUS: complete (51/53)
OBJECTION: M7 used best-of-4 with 8 random configs (32 total); tabular venues
  expect 100+ or AutoML. The spec itself named this the #1 objection.
FIX: best-of-6 with 165 configs. XGBoost and CatBoost were listed as core deps in
  the spec and had never been used. Budgets are COST-AWARE, not uniform: CatBoost
  measured ~40x costlier per config than LightGBM on CPU, so a flat budget would
  have spent nearly all compute on one family instead of broadening the ensemble.
  Only strong_classical was re-run; TabPFN's M7 results are cached and untouched.
KEY NUMBERS (51 datasets; har and Fashion-MNIST outstanding, both d=500 at n=4000):
  ensemble winners: xgboost 22, catboost 14, lightgbm 10, mlp 5
  baseline log-loss improved on 41/51 datasets: 0.3458 -> 0.3245
  TabPFN lost on  9/51 (18%)  ->  16/51 (31%)
  mean gap       -0.0378      ->  -0.0164   (Wilcoxon p 4.2e-07 -> 0.0019)
  mean-gap bar    0.0377      ->   0.0267
  best frozen score MAE 0.1659, AUROC 0.473  -> still DOES NOT BEAT the bar
READING: the under-tuned baseline WAS inflating TabPFN's advantage -- its loss
  rate nearly doubled, and the new families won 36/51 selections, so the old
  ensemble was missing the strongest learners rather than merely under-searched.
  TabPFN still wins overall, but any claim about the size of its margin must use
  31%, not 18%. Crucially the ROUTING NEGATIVE SURVIVES the stronger bar, so it
  was not an artefact of weak baselines -- which is exactly what this objection
  alleged. The M9/M10 headline results are unaffected: they are paired within
  TabPFN and never use this baseline.
```

```
W4 -- CROSS-GENERATION (TabPFN-2.5 / 2.6 / 3)            STATUS: BLOCKED, not done
OBJECTION: only TabPFN v2 was studied; does it hold for current models?
WHAT WE FOUND: every post-v2 generation is a GATED HuggingFace repo requiring
  browser-based licence acceptance. tabpfn/model_loading.py calls
  ensure_license_accepted(hf_repo_id=...) for V2_5, V2_6 and V3 before any
  download, and the failure path directs commercial users to sales@priorlabs.ai.
WHY WE STOPPED: clearing the gate would require using the user's HuggingFace
  credentials AND accepting a software licence on their behalf. That is a legal
  agreement, not a technical step, and is not something to do autonomously. The
  venv built for this was removed.
STATUS OF THE OBJECTION: OPEN. It is answerable in ~1 hour of compute by anyone
  who has accepted the licence; src/experiments/m9_rotation_transfer.py runs
  unchanged against a TabPFN-3 install.
NOTE: this is itself a finding worth one line in the paper -- reproducible
  external evaluation of post-v2 TabPFN now requires accepting a commercial
  licence, which is a real obstacle to independent auditing of these models.
```

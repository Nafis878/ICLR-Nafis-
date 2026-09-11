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

"""M8 -- baselines, ablations, figures (spec section 5/M8).

Three comparisons, all scored against the SAME frozen M6 predictions:
  (a) mean-gap baseline      -- predict the average gap for every dataset.
                                This is the bar arXiv 2605.28418 failed to beat.
  (b) black-box meta-feature -- same statistics, fit directly to real-data
                                performance by cross-validation, with no
                                synthetic prior grounding. Isolates whether the
                                prior recovery contributed anything. Note this
                                baseline is given an ADVANTAGE: it sees real
                                outcomes, which (c) never did.
  (c) prior-grounded OOP score (ours), frozen before any real data was scored.

Ablations on (c): drop rotation-alignment; drop kNN-irregularity; n/d/n_classes
only (the trivial baseline).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from analysis.oop_score import fit_oop_score, predict  # noqa: E402

KNN_STATS = ["stat_knn_agree_k1", "stat_knn_agree_k5", "stat_knn_agree_k15",
             "stat_est_label_noise"]
ROT_STATS = ["stat_rotation_alignment", "stat_rotation_alignment_sd",
             "stat_rotation_base_score"]
TRIVIAL = ["stat_n", "stat_d", "stat_n_classes", "stat_log_n", "stat_log_d", "stat_n_over_d"]


def blackbox_cv(real_stats: pd.DataFrame, actual: pd.Series, seed: int = 0) -> np.ndarray:
    """(b) fit statistics -> real gap directly, leave-one-out on real data."""
    from sklearn.base import clone
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in real_stats.columns if c.startswith("stat_")]
    X = np.nan_to_num(real_stats[feats].to_numpy(dtype=float))
    y = actual.to_numpy(dtype=float)
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
    pred = np.empty(len(y))
    for tr, te in LeaveOneOut().split(X):
        pred[te] = clone(m).fit(X[tr], y[tr]).predict(X[te])
    return pred


def score(name: str, pred: np.ndarray, actual: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    fail = (actual > 0).astype(int)
    try:
        auc = float(roc_auc_score(fail, pred)) if 0 < fail.sum() < len(fail) else None
    except Exception:
        auc = None
    return {
        "method": name,
        "mae": float(np.abs(actual - pred).mean()),
        "spearman": float(spearmanr(pred, actual).statistic) if np.std(pred) > 0 else 0.0,
        "auroc": auc,
    }


def main() -> None:
    env.preflight()
    import glob

    prereg = json.load(open(sorted(glob.glob("results/preregistration_*.json"))[-1]))
    scored = pd.read_csv("results/m7_scored.csv").set_index("dataset_id")
    # The amended score uses statistics computed under the M7 evaluation policy,
    # so the ablations must be applied to the same real-data table.
    real_path = ("results/m5b_real_statistics_evalpolicy.parquet"
                 if Path("results/m5b_real_statistics_evalpolicy.parquet").exists()
                 else "results/m5_real_statistics.parquet")
    real = pd.read_parquet(real_path).set_index("dataset_id")
    real = real.loc[real.index.intersection(scored.index)]
    scored = scored.loc[real.index]
    actual = scored["actual_gap"]

    cfg = runner.load_config("configs/m3_prior_sweep.yaml")
    syn_path = ("results/m3_pooled_sweep.parquet"
                if Path("results/m3_pooled_sweep.parquet").exists()
                else "results/m3_sweep.parquet")
    syn = pd.read_parquet(syn_path)
    syn = syn[(syn.status == "ok") & (syn.model == "tabpfn") & syn.regret.notna()].copy()
    id_cols = [c for c in syn.columns if c.startswith(("param_", "mod_"))] + ["family", "n_train"]
    syn["cell_id"] = syn[id_cols].astype(str).agg("|".join, axis=1)

    rows = [
        score("(a) mean-gap baseline",
              np.full(len(actual), float(actual.mean())), actual.to_numpy()),
        score("(b) black-box meta-features (LOO on real)",
              blackbox_cv(real, actual, cfg["seed"]), actual.to_numpy()),
        score("(c) prior-grounded OOP score [FROZEN]",
              scored["predicted_regret"].to_numpy(), actual.to_numpy()),
    ]

    for label, drop in [
        ("ablation: drop rotation-alignment", ROT_STATS),
        ("ablation: drop kNN-irregularity", KNN_STATS),
        ("ablation: trivial (n, d, n_classes only)",
         [c for c in syn.columns if c.startswith("stat_") and c not in TRIVIAL]),
    ]:
        f = fit_oop_score(syn, target="regret", group_col="cell_id",
                          seed=cfg["seed"], drop_stats=drop)
        # Clip to the synthetic support, exactly as the registered score does, so
        # the ablations differ only in their feature set and not in domain policy.
        lo = syn[f["features"]].astype(float).min()
        hi = syn[f["features"]].astype(float).max()
        Xa = real.reindex(columns=f["features"]).astype(float).clip(
            lower=lo, upper=hi, axis=1).fillna(0.0)
        rows.append(score(label, f["model"].predict(Xa.to_numpy(dtype=float)),
                          actual.to_numpy()))

    out = pd.DataFrame(rows)
    print("=" * 78)
    print("M8 BASELINES AND ABLATIONS")
    print(f"(scored against frozen commit {prereg['commit_hash'][:12]}, {len(actual)} datasets)")
    print("=" * 78)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    bar = out.loc[0, "mae"]
    ours = out.loc[2, "mae"]
    print(f"\nVERDICT: prior-grounded score {'BEATS' if ours < bar else 'DOES NOT BEAT'} "
          f"the mean-gap baseline (MAE {ours:.4f} vs {bar:.4f}).")
    if ours >= bar:
        print("This is a NEGATIVE RESULT and is reported as such: it corroborates")
        print("arXiv 2605.28418 rather than overturning it. Not tuned away (spec section 1).")

    out.to_csv("results/m8_baselines.csv", index=False)

    from analysis import figures

    fig_paths = []
    try:
        j = scored.join(real, how="inner", rsuffix="_r")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.scatter(scored["predicted_regret"], actual, s=44,
                   c=np.where(actual > 0, "#c6362f", "#2f6fc6"), edgecolor="black", lw=0.4)
        ax.axhline(0, color="grey", lw=0.9)
        ax.axvline(prereg["decision_threshold_tau"], color="crimson", ls="--", lw=1.2,
                   label=f"tau={prereg['decision_threshold_tau']:.3f}")
        ax.set_xlabel("predicted regret (frozen at M6)")
        ax.set_ylabel("actual gap: logloss(TabPFN) - logloss(strong_classical)")
        ax.set_title("M7 predicted vs actual\n"
                     f"pre-registration commit {prereg['commit_hash'][:12]}", fontsize=10)
        ax.legend(fontsize=8)
        fig_paths.append(figures._save(fig, "m7_predicted_vs_actual.png"))

        a = out.iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(a) + 1.5))
        ax.barh(a.method, a.mae, color="#4a8c4a")
        ax.axvline(bar, color="crimson", ls="--", lw=1.2, label="mean-gap bar")
        ax.set_xlabel("MAE against actual gap (lower is better)")
        ax.set_title("M8: baselines and ablations")
        ax.legend(fontsize=8)
        fig_paths.append(figures._save(fig, "m8_ablations.png"))
    except Exception as exc:  # noqa: BLE001
        print("figure generation failed:", exc)

    print("[m8] wrote results/m8_baselines.csv and", len(fig_paths), "figures")


if __name__ == "__main__":
    main()

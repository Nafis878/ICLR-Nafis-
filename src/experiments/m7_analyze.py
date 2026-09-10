"""Score the frozen predictions against whatever M7 outcomes exist in the cache.

Separated from the M7 runner so results can be read at any point. Because the M7
run order is seed-major with a seeded dataset SHUFFLE, a partial run is an
unbiased sample of the suite rather than "all of CC-18 and none of TabArena", so
an interim read is interpretable -- with the coverage stated.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.m7_validate import all_preregs  # noqa: E402


def load_outcomes() -> tuple[pd.DataFrame, pd.DataFrame]:
    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        c = r.get("config", {})
        if c.get("milestone") != "m7":
            continue
        recs.append({
            "dataset_id": c["dataset_id"], "model": c["model"], "seed": c["seed"],
            "status": r.get("status"), "logloss": r.get("logloss"),
            "accuracy": r.get("accuracy"), "name": r.get("name"),
            "n_train": r.get("n_train"), "n_test": r.get("n_test"),
            "error": str(r.get("error"))[:160],
        })
    df = pd.DataFrame(recs)
    ok = df[df.status == "ok"]
    piv = ok.pivot_table(index="dataset_id", columns="model", values="logloss", aggfunc="mean")
    piv = piv.dropna(subset=["tabpfn", "strong_classical"])
    piv["actual_gap"] = piv["tabpfn"] - piv["strong_classical"]
    seeds = ok.groupby("dataset_id").seed.nunique().rename("n_seeds")
    names = ok.groupby("dataset_id").name.first()
    piv = piv.join(seeds).join(names)
    return df, piv


def main() -> None:
    df, piv = load_outcomes()
    regs = all_preregs()
    fails = df[df.status != "ok"]

    from scipy.stats import spearmanr, wilcoxon
    from sklearn.metrics import roc_auc_score

    fail = (piv.actual_gap > 0).astype(int)
    mean_gap = float(piv.actual_gap.mean())
    print("=" * 80)
    print("M7 PROSPECTIVE VALIDATION")
    print("=" * 80)
    print(f"datasets with both arms: {len(piv)}/55   (seeds per dataset: "
          f"{piv.n_seeds.min()}-{piv.n_seeds.max()}, median {int(piv.n_seeds.median())})")
    print(f"failed cells recorded (not dropped): {len(fails)}")
    for _, r in fails.head(5).iterrows():
        print(f"    did={r.dataset_id} {r.model}: {r.error[:90]}")
    print(f"\nTabPFN LOST to strong_classical on {int(fail.sum())}/{len(piv)} "
          f"({fail.mean():.0%})")
    print(f"actual gap  mean {mean_gap:+.4f}   median {piv.actual_gap.median():+.4f}   "
          f"sd {piv.actual_gap.std():.4f}")
    try:
        w = wilcoxon(piv.actual_gap)
        print(f"Wilcoxon signed-rank vs 0: p={w.pvalue:.3g} "
              f"({'TabPFN significantly better' if mean_gap < 0 else 'TabPFN significantly worse'} "
              f"overall)" if w.pvalue < 0.05 else
              f"Wilcoxon signed-rank vs 0: p={w.pvalue:.3g} (no overall difference)")
    except Exception:
        pass
    print("\nMETRIC SUBSTITUTION: real data has no Bayes oracle, so 'actual' is")
    print("logloss(TabPFN) - logloss(strong_classical). Positive = TabPFN lost.")

    rows = []
    for p, d in regs:
        pred = pd.DataFrame(d["predictions"]).set_index("dataset_id")
        j = piv.join(pred, how="inner", rsuffix="_pred")
        if len(j) < 5:
            continue
        rho = spearmanr(j.predicted_regret, j.actual_gap)
        f = (j.actual_gap > 0).astype(int)
        try:
            auc = float(roc_auc_score(f, j.predicted_regret)) if 0 < f.sum() < len(f) else None
        except Exception:
            auc = None
        rows.append({
            "registered": Path(p).stem.replace("preregistration_", "")[:26],
            "commit": d["commit_hash"][:10],
            "n": len(j),
            "spearman": float(rho.statistic),
            "p": float(rho.pvalue),
            "auroc": auc,
            "mae": float(np.abs(j.actual_gap - j.predicted_regret).mean()),
        })
        j.to_csv(f"results/m7_scored_{Path(p).stem}.csv")
        if p == regs[-1][0]:
            j.to_csv("results/m7_scored.csv")

    s = pd.DataFrame(rows)
    mae_mean = float(np.abs(piv.actual_gap - mean_gap).mean())
    print("\n--- each registered prediction set vs the SAME outcomes ---")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n(a) mean-gap baseline MAE = {mae_mean:.4f}   <- the bar (arXiv 2605.28418)")
    if len(s):
        best = s.loc[s.mae.idxmin()]
        print(f"best registered score MAE = {best.mae:.4f} ({best.commit})  -> "
              f"{'BEATS' if best.mae < mae_mean else 'DOES NOT BEAT'} the bar")

    print("\n--- 6 datasets where TabPFN lost most ---")
    print(piv.nlargest(6, "actual_gap")[["name", "tabpfn", "strong_classical", "actual_gap"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n--- 6 datasets where TabPFN won most ---")
    print(piv.nsmallest(6, "actual_gap")[["name", "tabpfn", "strong_classical", "actual_gap"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))

    piv.to_csv("results/m7_outcomes.csv")
    json.dump({"n_datasets": int(len(piv)), "coverage_of_55": len(piv) / 55,
               "mean_gap": mean_gap, "mae_mean_gap": mae_mean,
               "tabpfn_loss_rate": float(fail.mean()),
               "n_failed_cells": int(len(fails)), "registered": rows},
              open("results/m7_summary.json", "w"), indent=2)
    print("\n[m7] wrote results/m7_outcomes.csv, m7_scored*.csv, m7_summary.json")


if __name__ == "__main__":
    main()

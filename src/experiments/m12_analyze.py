"""W2 -- re-score everything against the STRONGER classical reference.

M7 used best-of-4 with 8 random configs (32 total). This scores the same frozen
predictions against best-of-6 with 165 configs, changing exactly one side of the
comparison: TabPFN's results are the cached M7 ones, untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402
import collections  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load_new_baseline() -> pd.DataFrame:
    recs, winners = [], collections.Counter()
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") != "m12base" or r.get("status") != "ok":
            continue
        recs.append({"dataset_id": c["dataset_id"], "sc_strong": r.get("logloss"),
                     "winner": r.get("winner"), "name": r.get("name")})
        winners[r.get("winner")] += 1
    return pd.DataFrame(recs), winners


def main() -> None:
    from scipy.stats import spearmanr, wilcoxon
    from sklearn.metrics import roc_auc_score

    from experiments.m7_validate import all_preregs

    new, winners = load_new_baseline()
    old = pd.read_csv("results/m7_outcomes.csv")
    j = old.merge(new[["dataset_id", "sc_strong"]], on="dataset_id", how="inner")
    j["gap_old"] = j["tabpfn"] - j["strong_classical"]
    j["gap_new"] = j["tabpfn"] - j["sc_strong"]

    print("=" * 80)
    print("W2 -- STRONGER CLASSICAL REFERENCE (best-of-6, 165 configs vs best-of-4, 32)")
    print("=" * 80)
    print(f"datasets re-scored: {len(j)}")
    print(f"\nwhich family won the ensemble selection: {dict(winners.most_common())}")
    improved = (j.sc_strong < j.strong_classical).sum()
    print(f"\nthe stronger baseline improved on {improved}/{len(j)} datasets "
          f"(mean log-loss {j.strong_classical.mean():.4f} -> {j.sc_strong.mean():.4f})")

    for lab, g in (("OLD (best-of-4, 32 cfg)", j.gap_old), ("NEW (best-of-6, 165 cfg)", j.gap_new)):
        lost = int((g > 0).sum())
        w = wilcoxon(g)
        print(f"\n{lab}:  TabPFN lost on {lost}/{len(g)} ({lost/len(g):.0%})   "
              f"mean gap {g.mean():+.4f}   Wilcoxon p={w.pvalue:.3g}")

    print("\n--- every frozen prediction set, re-scored against the STRONGER bar ---")
    rows = []
    for p, d in all_preregs():
        # Later pre-registrations (M9/M10/M11) register HYPOTHESES, not per-dataset
        # predictions, so they have no "predictions" key and are not scoreable here.
        if not d.get("predictions"):
            continue
        pred = pd.DataFrame(d["predictions"]).set_index("dataset_id")
        m = j.set_index("dataset_id").join(pred, how="inner", rsuffix="_p")
        if len(m) < 5:
            continue
        fail = (m.gap_new > 0).astype(int)
        try:
            auc = (float(roc_auc_score(fail, m.predicted_regret))
                   if 0 < fail.sum() < len(fail) else None)
        except Exception:
            auc = None
        rho = spearmanr(m.predicted_regret, m.gap_new)
        rows.append({"registered": Path(p).stem.replace("preregistration_", "")[:24],
                     "commit": d["commit_hash"][:10], "n": len(m),
                     "spearman": float(rho.statistic), "p": float(rho.pvalue),
                     "auroc": auc,
                     "mae": float(np.abs(m.gap_new - m.predicted_regret).mean())})
    s = pd.DataFrame(rows)
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    bar_new = float(np.abs(j.gap_new - j.gap_new.mean()).mean())
    bar_old = float(np.abs(j.gap_old - j.gap_old.mean()).mean())
    print(f"\nmean-gap bar: old {bar_old:.4f} -> new {bar_new:.4f}")
    if len(s):
        b = s.loc[s.mae.idxmin()]
        print(f"best registered score MAE {b.mae:.4f} ({b.commit}) -> "
              f"{'BEATS' if b.mae < bar_new else 'DOES NOT BEAT'} the stronger bar")

    j.to_csv("results/m12_rescored.csv", index=False)
    json.dump({"n": int(len(j)), "winners": dict(winners),
               "loss_rate_old": float((j.gap_old > 0).mean()),
               "loss_rate_new": float((j.gap_new > 0).mean()),
               "mean_gap_old": float(j.gap_old.mean()),
               "mean_gap_new": float(j.gap_new.mean()),
               "bar_old": bar_old, "bar_new": bar_new,
               "registered": rows}, open("results/m12_summary.json", "w"), indent=2)
    print("\n[w2] wrote results/m12_rescored.csv, results/m12_summary.json")


if __name__ == "__main__":
    main()

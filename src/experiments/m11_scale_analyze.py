"""Score the W1 scale ladder against its pre-registration (18552a4)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> None:
    from scipy.stats import spearmanr, wilcoxon

    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") != "m11scale" or r.get("status") != "ok":
            continue
        recs.append({"dataset_id": c["dataset_id"], "n_cap": c["n_train_cap"],
                     "condition": c["condition"], "logloss": r.get("logloss"),
                     "name": r.get("name"), "n_train": r.get("n_train")})
    d = pd.DataFrame(recs)
    p = d.pivot_table(index=["dataset_id", "n_cap"], columns="condition", values="logloss")
    p = p.dropna(subset=["original", "rotated"])
    p["delta"] = p["rotated"] - p["original"]
    p = p.reset_index()

    print("=" * 78)
    print("W1 SCALE LADDER -- rotation effect up to TabPFN v2's stated 10k ceiling")
    print("(pre-registered 18552a4)")
    print("=" * 78)
    rows = []
    for cap, g in p.groupby("n_cap"):
        if len(g) < 5:
            print(f"  n_train<={cap}: n={len(g)} -- too few to test yet")
            continue
        w = wilcoxon(g.delta, alternative="greater")
        rows.append({"n_cap": cap, "n": len(g), "mean": g.delta.mean(),
                     "median": g.delta.median(), "worse": int((g.delta > 0).sum()),
                     "p": float(w.pvalue)})
        print(f"  n_train<={cap:5d}: n={len(g):2d}  mean {g.delta.mean():+.4f}  "
              f"median {g.delta.median():+.4f}  worse on {int((g.delta>0).sum())}/{len(g)}  "
              f"p={w.pvalue:.4g}  -> {'SUPPORTED' if w.pvalue<0.05 else 'not significant'}")
    if p.n_cap.nunique() > 1:
        r = spearmanr(p.n_cap, p.delta)
        print(f"\n  Spearman(n_train, degradation) = {r.statistic:+.3f}  p={r.pvalue:.3g}")
        if r.statistic > 0:
            print("  -> POSITIVE: the earlier 1500-row cap UNDERSTATED the effect, so the")
            print("     published magnitude is CONSERVATIVE, not inflated.")
        else:
            print("  -> negative: the earlier cap overstated the effect.")
    p.to_csv("results/m11_scale_ladder.csv", index=False)
    json.dump({"levels": rows}, open("results/m11_scale_summary.json", "w"), indent=2)
    print("\n[w1] wrote results/m11_scale_ladder.csv, results/m11_scale_summary.json")


if __name__ == "__main__":
    main()

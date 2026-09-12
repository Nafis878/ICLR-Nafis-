"""Cross-generation comparison: is the rotation axis present in BOTH v1 and v2?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load(milestones, model_filter=None) -> pd.DataFrame:
    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") not in milestones or r.get("status") != "ok":
            continue
        if model_filter and c.get("model") != model_filter:
            continue
        recs.append({"dataset_id": c["dataset_id"], "condition": c["condition"],
                     "logloss": r.get("logloss"), "name": r.get("name")})
    d = pd.DataFrame(recs)
    if d.empty:
        return d
    p = d.pivot_table(index="dataset_id", columns="condition", values="logloss")
    p = p.dropna(subset=["original", "rotated"])
    p["delta"] = p["rotated"] - p["original"]
    return p.join(d.groupby("dataset_id").name.first())


def main() -> None:
    from scipy.stats import spearmanr, wilcoxon

    v1 = load({"m14v1"})
    v2 = load({"m9", "m9rep"}, model_filter="tabpfn")

    print("=" * 78)
    print("M14 CROSS-GENERATION -- rotation axis in TabPFN v1 vs v2 (prereg d2ef887)")
    print("=" * 78)

    out = {}
    for tag, p in (("v1 (ICLR 2023)", v1), ("v2 (Nature 2025)", v2)):
        if len(p) < 5:
            print(f"  {tag}: n={len(p)} -- too few")
            continue
        w = wilcoxon(p.delta, alternative="greater")
        print(f"  {tag:18s} n={len(p):3d}  mean {p.delta.mean():+.4f}  "
              f"median {p.delta.median():+.4f}  worse on {int((p.delta>0).sum())}/{len(p)}  "
              f"p={w.pvalue:.4g}  -> {'SUPPORTED' if w.pvalue<0.05 else 'not significant'}")
        out[tag] = {"n": int(len(p)), "mean": float(p.delta.mean()),
                    "worse": int((p.delta > 0).sum()), "p": float(w.pvalue)}

    common = v1.index.intersection(v2.index)
    if len(common) >= 5:
        a, b = v1.loc[common, "delta"], v2.loc[common, "delta"]
        r = spearmanr(a, b)
        w = wilcoxon(a, b)
        print(f"\n  PAIRED on {len(common)} datasets measured in BOTH generations:")
        print(f"    v1 degradation {a.mean():+.4f}   v2 degradation {b.mean():+.4f}")
        print(f"    Spearman(v1 delta, v2 delta) = {r.statistic:+.3f}  p={r.pvalue:.3g}")
        print(f"    Wilcoxon v1 vs v2 magnitude: p={w.pvalue:.3g}")
        out["paired"] = {"n": int(len(common)), "v1_mean": float(a.mean()),
                         "v2_mean": float(b.mean()), "spearman": float(r.statistic),
                         "spearman_p": float(r.pvalue)}
        print("\n  READING:")
        if out.get("v1 (ICLR 2023)", {}).get("p", 1) < 0.05:
            print("    The rotation axis is present in BOTH generations, so it is a")
            print("    persistent property of the TabPFN prior/architecture rather than a")
            print("    quirk of the v2 checkpoint.")
            if r.pvalue < 0.05 and r.statistic > 0:
                print("    Moreover the SAME datasets suffer in both, so the axis is not")
                print("    just present but transfers dataset-by-dataset across generations.")
        else:
            print("    v1 shows NO significant rotation effect -> the axis is v2-specific")
            print("    and the generality claim must be narrowed to that generation.")

    json.dump(out, open("results/m14_cross_generation.json", "w"), indent=2)
    if len(v1):
        v1.to_csv("results/m14_v1_rotation.csv")
    print("\n[m14] wrote results/m14_cross_generation.json")


if __name__ == "__main__":
    main()

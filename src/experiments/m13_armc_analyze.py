"""Arm-C 2x2: is the code-permutation effect a model property or preprocessing?"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def load():
    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") not in ("m10C", "m13armc") or r.get("status") != "ok":
            continue
        if c.get("model", "tabpfn") != "tabpfn":
            continue
        recs.append({"dataset_id": c["dataset_id"], "condition": c["condition"],
                     "declared": bool(c.get("declared", False)),
                     "logloss": r.get("logloss"), "name": r.get("name")})
    return pd.DataFrame(recs)


def main() -> None:
    from scipy.stats import wilcoxon

    d = load()
    out = {}
    print("=" * 78)
    print("ARM-C 2x2 -- categorical code permutation, declared vs undeclared")
    print("(pre-registered 18552a4; fixes a defect in the original M10 arm C)")
    print("=" * 78)

    for declared in (False, True):
        s = d[d.declared == declared]
        p = s.pivot_table(index="dataset_id", columns="condition", values="logloss")
        p = p.dropna(subset=["original", "perturbed"])
        if len(p) < 5:
            print(f"  declared={declared}: n={len(p)} -- too few")
            continue
        delta = (p["perturbed"] - p["original"]).to_numpy()
        w = wilcoxon(delta, alternative="greater")
        tag = "DECLARED (model told which are categorical)" if declared else \
              "UNDECLARED (ordinal codes passed as plain numbers)"
        print(f"\n  {tag}")
        print(f"    n={len(p)}  mean {delta.mean():+.4f}  median {np.median(delta):+.4f}  "
              f"worse on {(delta > 0).sum()}/{len(delta)}  p={w.pvalue:.4g}  "
              f"-> {'SUPPORTED' if w.pvalue < 0.05 else 'not significant'}")
        out[f"declared_{declared}"] = {"n": int(len(p)), "mean": float(delta.mean()),
                                       "p": float(w.pvalue),
                                       "worse": int((delta > 0).sum())}

    # paired on the datasets present in both halves
    a = d[~d.declared].pivot_table(index="dataset_id", columns="condition", values="logloss")
    b = d[d.declared].pivot_table(index="dataset_id", columns="condition", values="logloss")
    for t in (a, b):
        t.dropna(subset=["original", "perturbed"], inplace=True)
    common = a.index.intersection(b.index)
    if len(common) >= 5:
        da = (a.loc[common, "perturbed"] - a.loc[common, "original"]).to_numpy()
        db = (b.loc[common, "perturbed"] - b.loc[common, "original"]).to_numpy()
        w = wilcoxon(da - db, alternative="greater")
        print(f"\n  PAIRED on {len(common)} datasets in both halves:")
        print(f"    undeclared degradation {da.mean():+.4f}   declared {db.mean():+.4f}")
        print(f"    declaring reduces the damage by {da.mean()-db.mean():+.4f}, "
              f"one-sided p={w.pvalue:.4g}")
        out["paired"] = {"n": int(len(common)), "undeclared": float(da.mean()),
                         "declared": float(db.mean()), "p": float(w.pvalue)}
        print("\n  VERDICT:")
        if out.get("declared_True", {}).get("p", 1) < 0.05:
            print("    The effect PERSISTS when categoricals are declared -> it is a")
            print("    property of the model, not merely a preprocessing artefact.")
        else:
            print("    The effect VANISHES once categoricals are declared -> the original")
            print("    arm C measured the cost of NOT declaring them. That is a sharp")
            print("    preprocessing warning, and the earlier claim is restated accordingly.")

    json.dump(out, open("results/m13_armc_2x2.json", "w"), indent=2)
    print("\n[armc] wrote results/m13_armc_2x2.json")


if __name__ == "__main__":
    main()

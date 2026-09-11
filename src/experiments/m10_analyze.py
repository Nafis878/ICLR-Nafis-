"""Score the M10 robustness arms against their pre-registration (efcbabc).

Each arm is reported on its own n. No arm is pooled into the M9 primary or
replication results.
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


def load(ms: str) -> pd.DataFrame:
    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") != ms or r.get("status") != "ok":
            continue
        recs.append({"dataset_id": c["dataset_id"], "condition": c["condition"],
                     "model": c["model"], "n_cap": c.get("n_train_cap"),
                     "logloss": r.get("logloss"), "name": r.get("name"),
                     "n_train": r.get("n_train"), "n_numeric": r.get("n_numeric")})
    return pd.DataFrame(recs)


def paired(df: pd.DataFrame, model="tabpfn", by=None) -> pd.DataFrame:
    idx = ["dataset_id"] + ([by] if by else [])
    d = df[df.model == model]
    p = d.pivot_table(index=idx, columns="condition", values="logloss", aggfunc="mean")
    p = p.dropna(subset=["original", "perturbed"])
    p["delta"] = p["perturbed"] - p["original"]
    return p


def report(name, delta, extra=""):
    from scipy.stats import wilcoxon

    if len(delta) < 5:
        print(f"  {name}: n={len(delta)} -- too few to test yet")
        return None
    w = wilcoxon(delta, alternative="greater")
    print(f"  {name}: n={len(delta)}  mean {delta.mean():+.4f}  median {np.median(delta):+.4f}  "
          f"worse on {(delta > 0).sum()}/{len(delta)}  p={w.pvalue:.3g}  "
          f"-> {'SUPPORTED' if w.pvalue < 0.05 else 'not significant'}{extra}")
    return float(w.pvalue)


def main() -> None:
    out = {}
    print("=" * 78)
    print("M10 ROBUSTNESS ARMS  (pre-registered efcbabc)")
    print("=" * 78)

    # ---- Arm C: categorical code permutation -------------------------------
    c = load("m10C")
    if len(c):
        pc = paired(c)
        print("\nARM C -- categorical code permutation (information-preserving bijection)")
        print("  Tests whether TabPFN exploits the ARBITRARY ordinality of ordinal-encoded")
        print("  categoricals. Bayes risk is unchanged by a relabelling.")
        out["C_tabpfn_p"] = report("H_C TabPFN", pc.delta.to_numpy())
        ps = paired(c, model="strong_classical")
        out["C_sc_p"] = report("    strong_classical", ps.delta.to_numpy())
        if len(pc) >= 5 and len(ps) >= 5:
            j = pc[["delta"]].join(ps[["delta"]], rsuffix="_sc", how="inner")
            print(f"    excess (sc - tabpfn) mean {(j.delta_sc - j.delta).mean():+.4f}")
        pc.to_csv("results/m10C_categorical.csv")

    # ---- Arm B: low-dimensional rotation -----------------------------------
    b = load("m10B")
    if len(b):
        pb = paired(b)
        print("\nARM B -- rotation on datasets with only 2-3 numeric columns")
        out["B_p"] = report("H_B TabPFN", pb.delta.to_numpy())
        pb.to_csv("results/m10B_lowdim.csv")

    # ---- Arm A: scale ------------------------------------------------------
    a = load("m10A")
    if len(a):
        print("\nARM A -- does the rotation effect survive at every training-set size?")
        pa = paired(a, by="n_cap")
        for cap, g in pa.groupby(level="n_cap"):
            report(f"n_train<={int(cap)}", g.delta.to_numpy())
        from scipy.stats import spearmanr

        flat = pa.reset_index()
        if flat.n_cap.nunique() > 1:
            r = spearmanr(flat.n_cap, flat.delta)
            print(f"  Spearman(n_cap, degradation) = {r.statistic:+.3f} p={r.pvalue:.3g}")
            print("    positive => the 1500-row cap UNDERSTATES the effect;")
            print("    negative => it overstates it.")
            out["A_scale_rho"] = float(r.statistic)
        pa.to_csv("results/m10A_scale.csv")

    json.dump(out, open("results/m10_summary.json", "w"), indent=2)
    print("\n[m10] wrote results/m10*.csv, results/m10_summary.json")


if __name__ == "__main__":
    main()

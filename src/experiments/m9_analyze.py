"""Score M9 against its pre-registered hypotheses (commit e3f8aca).

Runs exactly the three tests fixed in results/preregistration_m9_*.json, in the
directions stated there. No test is added, dropped or re-directed after seeing
the data; anything exploratory is labelled as such and is not part of the claim.
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


def load() -> pd.DataFrame:
    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        c = r.get("config", {})
        if c.get("milestone") != "m9":
            continue
        recs.append({"dataset_id": c["dataset_id"], "condition": c["condition"],
                     "model": c["model"], "status": r.get("status"),
                     "logloss": r.get("logloss"), "accuracy": r.get("accuracy"),
                     "name": r.get("name"), "n_numeric": r.get("n_numeric"),
                     "d_total": r.get("d_total"), "error": str(r.get("error"))[:120]})
    return pd.DataFrame(recs)


def boot_spearman(x, y, n=5000, seed=0):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.choice(len(x), len(x), replace=True)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            out.append(spearmanr(x[i], y[i]).statistic)
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    from scipy.stats import spearmanr, wilcoxon

    df = load()
    ok = df[df.status == "ok"]
    fails = df[df.status != "ok"]
    piv = ok.pivot_table(index="dataset_id", columns=["model", "condition"],
                         values="logloss", aggfunc="mean")
    need = [("tabpfn", "original"), ("tabpfn", "rotated"),
            ("strong_classical", "original"), ("strong_classical", "rotated")]
    piv = piv.dropna(subset=[c for c in need if c in piv.columns])
    piv = piv[[c for c in need if c in piv.columns]]
    piv.columns = ["tab_orig", "tab_rot", "sc_orig", "sc_rot"]
    piv["delta_tabpfn"] = piv.tab_rot - piv.tab_orig
    piv["delta_sc"] = piv.sc_rot - piv.sc_orig
    piv["excess"] = piv.delta_sc - piv.delta_tabpfn
    names = ok.groupby("dataset_id").name.first()
    nnum = ok.groupby("dataset_id").n_numeric.first()
    piv = piv.join(names).join(nnum)

    real = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet").set_index("dataset_id")
    piv = piv.join(real[["stat_rotation_alignment", "n_eval_train", "d_eval"]], how="left")

    print("=" * 80)
    print("M9 PAIRED ROTATION-TRANSFER PROBE  (pre-registered: commit e3f8aca)")
    print("=" * 80)
    print(f"datasets with all 4 cells: {len(piv)}   failed cells: {len(fails)}")
    for _, r in fails.drop_duplicates('dataset_id').head(4).iterrows():
        print(f"    did={r.dataset_id}: {r.error[:88]}")

    # ---------------- H1 ----------------
    d = piv.delta_tabpfn.to_numpy()
    w1 = wilcoxon(d, alternative="greater")
    print("\nH1  TabPFN log-loss increases under rotation")
    print(f"    delta mean {d.mean():+.4f}  median {np.median(d):+.4f}  sd {d.std(ddof=1):.4f}")
    print(f"    degraded on {int((d > 0).sum())}/{len(d)} datasets")
    print(f"    Wilcoxon one-sided p = {w1.pvalue:.4g}  -> "
          f"{'SUPPORTED' if w1.pvalue < 0.05 else 'NOT SUPPORTED'}")

    # ---------------- H2 ----------------
    e = piv.excess.to_numpy()
    w2 = wilcoxon(e, alternative="greater")
    print("\nH2  strong_classical degrades MORE than TabPFN")
    print(f"    delta_sc mean {piv.delta_sc.mean():+.4f}   delta_tabpfn mean {d.mean():+.4f}")
    print(f"    excess (sc - tabpfn) mean {e.mean():+.4f}  median {np.median(e):+.4f}")
    print(f"    Wilcoxon one-sided p = {w2.pvalue:.4g}  -> "
          f"{'SUPPORTED' if w2.pvalue < 0.05 else 'NOT SUPPORTED'}")

    # ---------------- H3 ----------------
    m = piv.stat_rotation_alignment.notna()
    x = piv.loc[m, "stat_rotation_alignment"].to_numpy()
    yv = piv.loc[m, "delta_tabpfn"].to_numpy()
    rho = spearmanr(x, yv)
    lo, hi = boot_spearman(x, yv)
    p1 = rho.pvalue / 2 if rho.statistic > 0 else 1 - rho.pvalue / 2
    print("\nH3  rotation_alignment predicts TabPFN's degradation")
    print(f"    n={m.sum()}  Spearman {rho.statistic:+.3f}  one-sided p={p1:.4g}")
    print(f"    bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"(contains 0: {lo <= 0 <= hi})")
    print(f"    -> {'SUPPORTED' if (p1 < 0.05 and lo > 0) else 'NOT SUPPORTED'}")

    print("\n--- datasets where rotation hurt TabPFN most ---")
    print(piv.nlargest(6, "delta_tabpfn")[
        ["name", "n_numeric", "tab_orig", "tab_rot", "delta_tabpfn", "delta_sc"]
    ].to_string(float_format=lambda v: f"{v:.4f}"))

    piv.to_csv("results/m9_rotation_transfer.csv")
    json.dump({
        "n_datasets": int(len(piv)), "n_failed_cells": int(len(fails)),
        "H1": {"claim": "TabPFN degrades under rotation", "mean_delta": float(d.mean()),
               "median_delta": float(np.median(d)), "n_degraded": int((d > 0).sum()),
               "p_one_sided": float(w1.pvalue), "supported": bool(w1.pvalue < 0.05)},
        "H2": {"claim": "strong_classical degrades more", "mean_excess": float(e.mean()),
               "p_one_sided": float(w2.pvalue), "supported": bool(w2.pvalue < 0.05)},
        "H3": {"claim": "rotation_alignment predicts degradation",
               "spearman": float(rho.statistic), "p_one_sided": float(p1),
               "ci95": [float(lo), float(hi)],
               "supported": bool(p1 < 0.05 and lo > 0)},
    }, open("results/m9_summary.json", "w"), indent=2)
    print("\n[m9] wrote results/m9_rotation_transfer.csv, results/m9_summary.json")


if __name__ == "__main__":
    main()

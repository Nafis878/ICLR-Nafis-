"""M7 -- prospective validation, and M8 -- baselines/ablations (spec 5/M7, 5/M8).

RUNS ONLY AFTER the M6 pre-registration commit exists. Refuses to start
otherwise. The frozen predictions are read from the pre-registration JSON and
are never recomputed here.

METRIC SUBSTITUTION, stated explicitly because it is a real limitation: real
datasets have no Bayes oracle, so the "actual regret" scored here is the proxy
    gap = logloss(TabPFN) - logloss(strong_classical)
which is a comparison against a strong baseline, not against the true optimum.
A positive gap means TabPFN lost, not that it is far from Bayes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from models.wrappers import fit_model, strong_classical  # noqa: E402
from oracles.metrics import evaluate  # noqa: E402


def all_preregs() -> list[tuple[str, dict]]:
    """Every registered prediction set, each verified to be a real commit.

    All of them were registered before any real outcome existed, so all are
    scored. Reporting only the last one would hide the progression.
    """
    out = []
    for p in sorted(glob.glob("results/preregistration_*.json")):
        d = json.load(open(p))
        if not d.get("commit_hash"):
            raise SystemExit(f"REFUSING: {p} has no commit hash; the freeze is not committed.")
        got = subprocess.run(["git", "cat-file", "-t", d["commit_hash"]],
                             capture_output=True, text=True)
        if got.stdout.strip() != "commit":
            raise SystemExit(f"REFUSING: commit {d['commit_hash']} not found in this repo.")
        out.append((p, d))
    if not out:
        raise SystemExit("REFUSING: no pre-registration file. Run M6 first.")
    return out


def latest_prereg() -> tuple[str, dict]:
    return all_preregs()[-1]


def prepare_split(X, y, pol, seed):
    """Fixed, pre-registered evaluation policy (frozen in configs/m5 before M6)."""
    from sklearn.model_selection import train_test_split

    n, d = X.shape
    notes = []
    if d > pol["max_features"]:
        keep = np.argsort(-np.nanvar(X, axis=0))[: pol["max_features"]]
        X = X[:, np.sort(keep)]
        notes.append(f"features capped {d}->{pol['max_features']} by variance")

    counts = np.bincount(y)
    strat = y if (pol["stratify"] and counts[counts > 0].min() >= 2) else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=pol["test_size"], random_state=seed, stratify=strat
    )
    if len(Xtr) > pol["max_train_rows"]:
        idx = np.random.default_rng(seed).choice(len(Xtr), pol["max_train_rows"], replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
        notes.append(f"train subsampled to {pol['max_train_rows']}")
    if len(Xte) > pol["max_test_rows"]:
        idx = np.random.default_rng(seed + 1).choice(len(Xte), pol["max_test_rows"], replace=False)
        Xte, yte = Xte[idx], yte[idx]
    return Xtr, ytr, Xte, yte, notes


def run_one(cfg: dict) -> dict:
    from experiments.m5_real_datasets import load_dataset

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    pol = cfg["m5"]["eval_policy"]
    Xtr, ytr, Xte, yte, notes = prepare_split(X, y, pol, cfg["seed"])
    K = int(len(np.unique(y)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cfg["model"] == "strong_classical":
            proba, info = strong_classical(Xtr, ytr, Xte, K, cfg["model_cfgs"], cfg["seed"])
        else:
            proba, info = fit_model(cfg["model"], Xtr, ytr, Xte, K,
                                    cfg["model_cfgs"][cfg["model"]], cfg["seed"])
    m = evaluate(proba, yte, K, None)
    return {**m, "dataset_id": cfg["dataset_id"], "name": name,
            "notes": notes, "info": info, "n_train": len(Xtr), "n_test": len(Xte)}


def main() -> None:
    env.preflight()
    regs = all_preregs()
    print(f"[m7] {len(regs)} registered prediction sets will be scored:")
    for p, d in regs:
        print(f"      {Path(p).name}  commit {d['commit_hash'][:12]}  "
              f"{d['n_predicted_out_of_prior']}/{d['n_datasets']} predicted out-of-prior")
    prereg_path, prereg = regs[-1]

    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    base = runner.load_config("configs/m2_baselines.yaml")
    mc = base["models"]
    for nme in ("lightgbm", "random_forest", "knn", "mlp"):
        mc[nme] = {**mc[nme], "n_random_configs": 8, "cv_folds": 3}

    dsets = sorted({p["dataset_id"] for _, d in regs for p in d["predictions"]})
    cells = [
        {"milestone": "m7", "dataset_id": did, "model": model, "seed": s,
         "m5": m5cfg, "model_cfgs": mc}
        for did in dsets
        for model in ("tabpfn", "strong_classical")
        for s in range(m5cfg["eval_policy"]["n_seeds"])
    ]
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)

    recs = []
    for r in rows:
        c = r["config"]
        recs.append({"dataset_id": c["dataset_id"], "model": c["model"], "seed": c["seed"],
                     "status": r["status"], "logloss": r.get("logloss"),
                     "accuracy": r.get("accuracy"), "error": str(r.get("error"))[:200]})
    df = pd.DataFrame(recs)
    df.to_parquet("results/m7_real_results.parquet", index=False)

    piv = (df[df.status == "ok"]
           .pivot_table(index="dataset_id", columns="model", values="logloss", aggfunc="mean"))
    piv = piv.dropna()
    piv["actual_gap"] = piv["tabpfn"] - piv["strong_classical"]

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    actual_fail = (piv.actual_gap > 0).astype(int)
    mean_gap = float(piv.actual_gap.mean())
    print("\n" + "=" * 78)
    print("M7 PROSPECTIVE VALIDATION")
    print("=" * 78)
    print(f"datasets scored: {len(piv)} | TabPFN lost to strong_classical on "
          f"{int(actual_fail.sum())} ({actual_fail.mean():.0%})")
    print(f"actual gap: mean {mean_gap:+.4f}  median {piv.actual_gap.median():+.4f}  "
          f"sd {piv.actual_gap.std():.4f}")
    print("\nMetric substitution: no Bayes oracle exists on real data, so 'actual' is")
    print("logloss(TabPFN) - logloss(strong_classical). Positive means TabPFN lost.")

    summaries = []
    for p, d in regs:
        pred = pd.DataFrame(d["predictions"]).set_index("dataset_id")
        j = piv.join(pred, how="inner")
        if j.empty:
            continue
        rho = spearmanr(j.predicted_regret, j.actual_gap)
        fail = (j.actual_gap > 0).astype(int)
        try:
            auc = (float(roc_auc_score(fail, j.predicted_regret))
                   if 0 < fail.sum() < len(fail) else None)
        except Exception:
            auc = None
        summaries.append({
            "prereg": Path(p).stem.replace("preregistration_", ""),
            "commit": d["commit_hash"][:12],
            "n": int(len(j)),
            "spearman": float(rho.statistic),
            "p": float(rho.pvalue),
            "auroc": auc,
            "mae": float(np.abs(j.actual_gap - j.predicted_regret).mean()),
            "pred_min": float(j.predicted_regret.min()),
            "pred_max": float(j.predicted_regret.max()),
        })
        j.to_csv(f"results/m7_scored_{Path(p).stem}.csv")
        if p == prereg_path:
            j.to_csv("results/m7_scored.csv")

    s = pd.DataFrame(summaries)
    print("\n--- every registered prediction set, scored against the SAME outcomes ---")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    mae_mean = float(np.abs(piv.actual_gap - mean_gap).mean())
    best = s.loc[s.mae.idxmin()]
    print(f"\n(a) mean-gap baseline MAE = {mae_mean:.4f}   <- the bar from arXiv 2605.28418")
    print(f"best registered score MAE = {best.mae:.4f}  ({best.commit})")
    print(f"  -> {'BEATS' if best.mae < mae_mean else 'DOES NOT BEAT'} the mean-gap baseline")

    json.dump({"n_datasets": int(len(piv)), "mean_gap": mean_gap,
               "mae_mean_gap": mae_mean,
               "tabpfn_loss_rate": float(actual_fail.mean()),
               "registered": summaries},
              open("results/m7_summary.json", "w"), indent=2)
    print("\n[m7] wrote results/m7_scored*.csv, results/m7_summary.json")


if __name__ == "__main__":
    main()

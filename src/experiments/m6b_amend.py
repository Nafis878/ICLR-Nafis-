"""M6b -- documented PRE-REGISTRATION AMENDMENT (registered before any outcome).

The original freeze (commit 488c7e9) stands and will be scored as pre-registered.
This amendment is registered ALONGSIDE it, before TabPFN has touched any real
dataset, because two defects in the application of g were found by inspecting the
predictions themselves -- never by looking at outcomes:

DEFECT 1 -- statistics described data the model will never see.
  M5 computed the statistic vector on the FULL real dataset, but the frozen M7
  evaluation policy subsamples training to <=4000 rows and caps features at 500.
  M3 computed its statistics on the TRAINING split. The amendment recomputes real
  statistics under the M7 policy, so synthetic and real vectors describe the same
  kind of object: the data the learner is actually given.

DEFECT 2 -- unbounded extrapolation.
  g was fit on synthetic regrets in [-0.35, 0.83] with n in [200, 993]. Real n
  reaches 566602, a 571x extrapolation, and the ridge produced predictions up to
  242.6 -- 290x beyond any training target, and Spearman +0.90 with n alone. The
  amendment CLIPS every statistic to the synthetic support before applying g, so
  the score interpolates within the region it was estimated on and cannot be a
  linear restatement of "n is too big".

Neither change touches the DGP library, the statistic definitions, the fitted
coefficients of g, or tau. Only the DOMAIN on which g is evaluated changes.
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
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from stats.features import compute_statistics  # noqa: E402

CANONICAL_SPLIT_SEED = 0  # M7 runs 3 seeds; statistics use the first, stated here.


def run_one(cfg: dict) -> dict:
    """Statistics under the frozen M7 evaluation policy. Still no TabPFN."""
    from experiments.m5_real_datasets import in_operating_range, load_dataset
    from experiments.m7_validate import prepare_split

    X, y, name = load_dataset(cfg["dataset_id"], cfg["m5"])
    n_full, d_full = X.shape
    k = int(len(np.unique(y)))
    Xtr, ytr, Xte, yte, notes = prepare_split(
        X, y, cfg["m5"]["eval_policy"], CANONICAL_SPLIT_SEED
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = compute_statistics(Xtr, ytr)
    return {
        "dataset_id": cfg["dataset_id"], "name": name,
        "n_full": n_full, "d_full": d_full, "n_classes": k,
        "n_eval_train": int(len(Xtr)), "d_eval": int(Xtr.shape[1]),
        "policy_notes": notes,
        "in_range": bool(in_operating_range(n_full, d_full, k,
                                            cfg["m5"]["tabpfn_operating_range"])),
        "stats": stats,
    }


def main() -> None:
    env.preflight()
    from experiments.m6_preregister import audit_no_tabpfn_on_real, sha256_file

    audit = audit_no_tabpfn_on_real()
    if not audit["clean"]:
        raise SystemExit("REFUSING: TabPFN has already touched real data.")
    print("[m6b] pre-freeze audit clean: 0 TabPFN runs on real data")

    m5cfg = runner.load_config("configs/m5_real_datasets.yaml")
    keep = set(pd.read_parquet("results/m5_real_statistics.parquet").dataset_id.tolist())
    cells = [{"milestone": "m5b", "dataset_id": int(d), "m5": m5cfg} for d in sorted(keep)]
    rows = runner.run_many(cells, run_one, n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
                           verbose=5)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"[m6b] statistics under eval policy: {len(ok)} ok, {len(bad)} failed (recorded)")
    recs = []
    for r in ok:
        rec = {kk: r[kk] for kk in ("dataset_id", "name", "n_full", "d_full", "n_classes",
                                    "n_eval_train", "d_eval", "in_range")}
        rec.update({f"stat_{k}": v for k, v in r["stats"].items()})
        recs.append(rec)
    real = pd.DataFrame(recs).sort_values("dataset_id").reset_index(drop=True)
    real.to_parquet("results/m5b_real_statistics_evalpolicy.parquet", index=False)

    # ---- clip to synthetic support -----------------------------------------
    import joblib

    fit = joblib.load("results/m4_oop_score.joblib")
    meta = json.load(open("results/m4_oop_score.json"))
    syn = pd.read_parquet("results/m3_sweep.parquet")
    syn = syn[(syn.status == "ok") & (syn.model == "tabpfn") & syn.regret.notna()]

    feats = fit["features"]
    lo = syn[feats].astype(float).quantile(0.00)
    hi = syn[feats].astype(float).quantile(1.00)
    Xr = real.reindex(columns=feats).astype(float)
    n_clipped = int((Xr.lt(lo, axis=1) | Xr.gt(hi, axis=1)).sum().sum())
    Xc = Xr.clip(lower=lo, upper=hi, axis=1)
    Xc = Xc.fillna(0.0)

    pred = fit["model"].predict(Xc.to_numpy(dtype=float))
    tau = float(meta["tau"])
    spread = float(np.std(pred)) or 1.0
    conf = np.clip(np.abs(pred - tau) / (2 * spread), 0.0, 1.0)

    from scipy.stats import spearmanr

    rho_n = float(spearmanr(pred, real.n_full).statistic)
    print(f"\n[m6b] clipped {n_clipped} statistic values into the synthetic support")
    print(f"[m6b] amended predictions: min {pred.min():+.3f} max {pred.max():+.3f} "
          f"(synthetic target range {syn.regret.min():+.3f}..{syn.regret.max():+.3f})")
    print(f"[m6b] Spearman(prediction, n) = {rho_n:+.3f} "
          f"(original frozen score: +0.902)")

    records = []
    for i, row in real.iterrows():
        records.append({
            "dataset_id": int(row["dataset_id"]), "name": str(row["name"]),
            "n_full": int(row["n_full"]), "n_eval_train": int(row["n_eval_train"]),
            "d_eval": int(row["d_eval"]), "n_classes": int(row["n_classes"]),
            "in_operating_range": bool(row["in_range"]),
            "predicted_regret": float(pred[i]),
            "predicted_out_of_prior": bool(pred[i] > tau),
            "predicted_loss_vs_strong_classical": bool(pred[i] > tau),
            "confidence": float(conf[i]),
        })

    orig = json.load(open(sorted(glob.glob("results/preregistration_2*.json"))[-1]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"results/preregistration_amended_{stamp}.json"
    payload = {
        "created_utc": stamp,
        "type": "AMENDMENT registered before any real outcome was observed",
        "supersedes": None,
        "amends_commit": orig["commit_hash"],
        "original_still_scored": True,
        "amendment_rationale": {
            "defect_1": "M5 statistics described full datasets; the frozen M7 policy "
                        "subsamples train to <=4000 rows and caps d at 500. Real statistics "
                        "are recomputed on the training split the learner actually receives, "
                        "matching how M3 computed them.",
            "defect_2": "g was fit on synthetic n in [200, 993]; real n reaches 566602 "
                        "(571x). Unclipped ridge extrapolation produced predictions up to "
                        "242.6 against a training target range of [-0.35, 0.83], and "
                        "correlated +0.90 with n alone. Statistics are now clipped to the "
                        "synthetic support so g interpolates.",
            "unchanged": ["DGP library", "statistic definitions", "g coefficients", "tau"],
            "outcome_information_used": "none; no TabPFN run has touched a real dataset",
        },
        "canonical_split_seed": CANONICAL_SPLIT_SEED,
        "decision_threshold_tau": tau,
        "threshold_rule": meta["threshold_rule"],
        "g": meta["g"],
        "clip_bounds": {"lower": {k: float(v) for k, v in lo.items()},
                        "upper": {k: float(v) for k, v in hi.items()}},
        "n_statistic_values_clipped": n_clipped,
        "spearman_prediction_vs_n": rho_n,
        "statistic_code_sha256": {
            "src/stats/features.py": sha256_file("src/stats/features.py"),
            "src/experiments/m6b_amend.py": sha256_file("src/experiments/m6b_amend.py"),
        },
        "n_datasets": len(records),
        "n_predicted_out_of_prior": int(sum(r["predicted_out_of_prior"] for r in records)),
        "predictions": records,
        "commit_hash": None,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))

    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m",
        f"M6b pre-registration AMENDMENT ({stamp})\n\n"
        f"Amends {orig['commit_hash'][:12]}, which remains scored as pre-registered.\n"
        f"Fixes two defects found by inspecting predictions, not outcomes:\n"
        f"  1. real statistics now computed on the training split the learner receives\n"
        f"  2. statistics clipped to synthetic support (was extrapolating 571x on n)\n"
        f"g coefficients, tau, statistic definitions and DGP library are unchanged.\n"
        f"No TabPFN run has touched a real dataset.\n\n"
        f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"], check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    payload["commit_hash"] = commit
    Path(out_path).write_text(json.dumps(payload, indent=2))
    subprocess.run(["git", "add", out_path], check=True)
    subprocess.run(["git", "commit", "-m",
        f"Echo amendment commit hash {commit[:12]} into {Path(out_path).name}\n\n"
        f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"], check=True)

    print("\n" + "=" * 72)
    print(f"ORIGINAL PRE-REGISTRATION (still scored): {orig['commit_hash']}")
    print(f"AMENDMENT COMMIT:                        {commit}")
    print(f"FILE:                                    {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

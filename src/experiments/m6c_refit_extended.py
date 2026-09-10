"""M6c -- refit g on the EXTENDED synthetic support and register it (pre-outcome).

This is the root-cause fix for the defect that broke the first freeze. Amendment 1
(m6b) treated the symptom by clipping real statistics into a synthetic support
that was too small. Here the support itself is enlarged: the M3-EXT sweep covers
n_train in [1000, 4000], which contains the range the frozen M7 policy actually
trains on, so g INTERPOLATES rather than extrapolating.

Registered before TabPFN has touched any real dataset. The earlier freezes are not
withdrawn; M7 scores all of them, which is what makes the progression auditable:
  488c7e9  original      -- unbounded extrapolation, predictions to 242.6
  edca78a  amendment 1   -- domain-restricted (clipped) on the old support
  this     amendment 2   -- refit on a support that contains the real data
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from analysis import collect  # noqa: E402
from analysis.oop_score import feature_importances, fit_oop_score, serialize  # noqa: E402


def pooled_frame() -> pd.DataFrame:
    a = collect.load_milestone("m3")
    b = collect.load_milestone("m3ext")
    df = pd.concat([a, b], ignore_index=True)
    id_cols = [c for c in collect.cell_id_columns(df) if c in df.columns]
    df["cell_id"] = df[id_cols].astype(str).agg("|".join, axis=1)
    return df


def main() -> None:
    env.preflight()
    from experiments.m6_preregister import audit_no_tabpfn_on_real, sha256_file

    if not audit_no_tabpfn_on_real()["clean"]:
        raise SystemExit("REFUSING: TabPFN has already touched real data.")
    print("[m6c] pre-freeze audit clean: 0 TabPFN runs on real data")

    df = pooled_frame()
    collect.save(df, "results/m3_pooled_sweep.parquet")
    ok = df[(df.status == "ok") & (df.model == "tabpfn") & df.regret.notna()]
    print(f"[m6c] pooled TabPFN rows: {len(ok)} "
          f"(m3 {int((ok.n_train <= 1000).sum())}, m3ext {int((ok.n_train > 1000).sum())})")
    print(f"[m6c] synthetic n_train support now [{int(ok.n_train.min())}, {int(ok.n_train.max())}]")

    cfg = runner.load_config("configs/m3_prior_sweep.yaml")
    anchor = ok[ok.family == cfg["threshold_rule"]["reference_family"]]
    tau = float(np.percentile(anchor.regret, cfg["threshold_rule"]["percentile"]))
    print(f"[m6c] tau recomputed on the pooled anchor = {tau:.4f}")

    fit = fit_oop_score(ok, target="regret", group_col="cell_id", seed=cfg["seed"])
    print(f"[m6c] g refit: {fit['n_rows']} rows / {fit['n_groups']} groups")
    for name, m in fit["cv"].items():
        print(f"    {name:12s} R2={m['cv_r2']:+.3f}  Spearman={m['cv_spearman']:+.3f}  "
              f"MAE={m['cv_mae']:.4f}")
    print(f"    selected: {fit['model_name']}")
    joblib.dump(fit, "results/m6c_oop_score_extended.joblib")
    imp = feature_importances(fit)
    imp.to_csv("results/m6c_feature_importances.csv", index=False)
    print("\n[m6c] top statistics:")
    print(imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- apply to real data (eval-policy statistics) -----------------------
    real = pd.read_parquet("results/m5b_real_statistics_evalpolicy.parquet")
    feats = fit["features"]
    lo = ok[feats].astype(float).min()
    hi = ok[feats].astype(float).max()
    Xr = real.reindex(columns=feats).astype(float)
    n_clipped = int((Xr.lt(lo, axis=1) | Xr.gt(hi, axis=1)).sum().sum())
    Xc = Xr.clip(lower=lo, upper=hi, axis=1).fillna(0.0)
    pred = fit["model"].predict(Xc.to_numpy(dtype=float))

    from scipy.stats import spearmanr

    rho_n = float(spearmanr(pred, real.n_full).statistic)
    rho_ne = float(spearmanr(pred, real.n_eval_train).statistic)
    print(f"\n[m6c] clipped {n_clipped} values (was 539 on the old support)")
    print(f"[m6c] predictions {pred.min():+.3f} .. {pred.max():+.3f} "
          f"| synthetic targets {ok.regret.min():+.3f} .. {ok.regret.max():+.3f}")
    print(f"[m6c] Spearman(prediction, n_full)={rho_n:+.3f}  "
          f"(prediction, n_eval_train)={rho_ne:+.3f}")

    spread = float(np.std(pred)) or 1.0
    conf = np.clip(np.abs(pred - tau) / (2 * spread), 0.0, 1.0)
    records = [{
        "dataset_id": int(r.dataset_id), "name": str(r.name_),
        "n_full": int(r.n_full), "n_eval_train": int(r.n_eval_train),
        "d_eval": int(r.d_eval), "n_classes": int(r.n_classes),
        "in_operating_range": bool(r.in_range),
        "predicted_regret": float(pred[i]),
        "predicted_out_of_prior": bool(pred[i] > tau),
        "predicted_loss_vs_strong_classical": bool(pred[i] > tau),
        "confidence": float(conf[i]),
    } for i, r in enumerate(real.rename(columns={"name": "name_"}).itertuples())]

    prev = [json.load(open(p)) for p in sorted(glob.glob("results/preregistration_*.json"))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = f"results/preregistration_extended_{stamp}.json"
    payload = {
        "created_utc": stamp,
        "type": "AMENDMENT 2 (root-cause fix), registered before any real outcome",
        "amends_commits": [p.get("commit_hash") for p in prev if p.get("commit_hash")],
        "earlier_freezes_still_scored": True,
        "rationale": (
            "Amendment 1 clipped real statistics into a synthetic support that was "
            "too small. Here the support is enlarged instead: M3-EXT adds 28 LHS "
            "points with n_train in [1000, 4000], covering the range the frozen M7 "
            "policy trains on, so g interpolates. tau is recomputed on the pooled "
            "in-prior anchor by the same stated rule."
        ),
        "synthetic_n_train_support": [int(ok.n_train.min()), int(ok.n_train.max())],
        "decision_threshold_tau": tau,
        "threshold_rule": cfg["threshold_rule"],
        "g": serialize(fit),
        "g_cv_on_synthetic": fit["cv"],
        "n_statistic_values_clipped": n_clipped,
        "spearman_prediction_vs_n_full": rho_n,
        "spearman_prediction_vs_n_eval_train": rho_ne,
        "statistic_code_sha256": {
            "src/stats/features.py": sha256_file("src/stats/features.py"),
            "src/experiments/m6c_refit_extended.py": sha256_file(
                "src/experiments/m6c_refit_extended.py"),
        },
        "n_datasets": len(records),
        "n_predicted_out_of_prior": int(sum(r["predicted_out_of_prior"] for r in records)),
        "predictions": records,
        "commit_hash": None,
    }
    Path(out).write_text(json.dumps(payload, indent=2))

    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m",
        f"M6c pre-registration AMENDMENT 2 -- refit on extended support ({stamp})\n\n"
        f"Root-cause fix: M3-EXT extends the synthetic sweep to n_train<=4000 so g\n"
        f"interpolates over the range the frozen M7 policy actually trains on.\n"
        f"tau recomputed on the pooled in-prior anchor by the same stated rule.\n"
        f"Earlier freezes remain scored. No TabPFN run has touched a real dataset.\n\n"
        f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"], check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    payload["commit_hash"] = commit
    Path(out).write_text(json.dumps(payload, indent=2))
    subprocess.run(["git", "add", out], check=True)
    subprocess.run(["git", "commit", "-m",
        f"Echo amendment-2 commit {commit[:12]} into {Path(out).name}\n\n"
        f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"], check=True)
    print(f"\n{'='*72}\nAMENDMENT 2 COMMIT: {commit}\nFILE: {out}\n{'='*72}")


if __name__ == "__main__":
    main()

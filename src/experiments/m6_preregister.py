"""M6 -- PRE-REGISTRATION FREEZE (spec section 5/M6).

The milestone that makes the paper credible. Writes predicted regret, predicted
win/loss vs strong_classical, and a confidence for every held-out real dataset,
together with the decision threshold, the exact coefficients of g, and a hash of
the statistic-computation code. Then commits, and echoes the commit hash back
into the JSON.

HARD PRECONDITION, verified from the result cache rather than from memory:
TabPFN must not have been run on any real dataset before this commit exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_no_tabpfn_on_real() -> dict:
    """Prove from the cache that no real dataset has been touched by TabPFN."""
    offenders = []
    n_m5 = 0
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        cfg = r.get("config", {})
        if cfg.get("milestone") == "m5":
            n_m5 += 1
        if cfg.get("milestone") in ("m7", "m8") or (
            cfg.get("dataset_id") is not None and cfg.get("model") == "tabpfn"
        ):
            offenders.append({"file": f, "config": {k: cfg.get(k) for k in ("milestone", "dataset_id", "model")}})
    return {"clean": not offenders, "offenders": offenders, "m5_cells": n_m5}


def main() -> None:
    env.preflight()

    audit = audit_no_tabpfn_on_real()
    if not audit["clean"]:
        print("REFUSING TO FREEZE: TabPFN has already touched real data.")
        print(json.dumps(audit["offenders"][:5], indent=2))
        raise SystemExit(1)
    print(f"[m6] pre-freeze audit clean ({audit['m5_cells']} m5 statistic cells, 0 TabPFN runs on real data)")

    fit_meta = json.load(open("results/m4_oop_score.json"))
    real = pd.read_parquet("results/m5_real_statistics.parquet")

    import joblib

    from analysis.oop_score import predict

    fit = joblib.load("results/m4_oop_score.joblib")

    missing = [f for f in fit["features"] if f not in real.columns]
    if missing:
        raise SystemExit(f"REFUSING TO FREEZE: statistics missing on real data: {missing}")

    pred = predict(fit, real)
    tau = float(fit_meta["tau"])

    # Confidence: distance from the threshold, scaled by the spread of predictions.
    spread = float(np.std(pred)) or 1.0
    conf = np.clip(np.abs(pred - tau) / (2 * spread), 0.0, 1.0)

    records = []
    for i, row in real.reset_index(drop=True).iterrows():
        records.append({
            "dataset_id": int(row["dataset_id"]),
            "name": str(row["name"]),
            "n": int(row["n"]),
            "d": int(row["d"]),
            "n_classes": int(row["n_classes"]),
            "in_operating_range": bool(row["in_range"]),
            "suites": str(row["suites"]),
            "predicted_regret": float(pred[i]),
            "predicted_out_of_prior": bool(pred[i] > tau),
            "predicted_loss_vs_strong_classical": bool(pred[i] > tau),
            "confidence": float(conf[i]),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"results/preregistration_{stamp}.json"
    payload = {
        "created_utc": stamp,
        "spec": "RESEARCH_SPEC.md M6 pre-registration freeze",
        "subject_model": fit_meta["subject_model"],
        "decision_threshold_tau": tau,
        "threshold_rule": fit_meta["threshold_rule"],
        "g": fit_meta["g"],
        "g_cv_on_synthetic": fit_meta["cv"],
        "statistic_code_sha256": {
            "src/stats/features.py": sha256_file("src/stats/features.py"),
            "src/dgp/registry.py": sha256_file("src/dgp/registry.py"),
            "src/oracles/metrics.py": sha256_file("src/oracles/metrics.py"),
        },
        "n_datasets": len(records),
        "n_predicted_out_of_prior": int(sum(r["predicted_out_of_prior"] for r in records)),
        "pre_freeze_audit": {"clean": True, "m5_cells": audit["m5_cells"]},
        "predictions": records,
        "commit_hash": None,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"[m6] wrote {out_path} with {len(records)} predictions")

    subprocess.run(["git", "add", "-A"], check=True)
    msg = (
        f"M6 pre-registration freeze ({stamp})\n\n"
        f"Frozen predictions for {len(records)} held-out real datasets.\n"
        f"tau={tau:.6f}; g={fit_meta['g']['model_name']}; "
        f"{payload['n_predicted_out_of_prior']} predicted out-of-prior.\n"
        f"No TabPFN run touched a real dataset before this commit.\n\n"
        f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
    )
    subprocess.run(["git", "commit", "-m", msg], check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    payload["commit_hash"] = commit
    Path(out_path).write_text(json.dumps(payload, indent=2))
    subprocess.run(["git", "add", out_path], check=True)
    subprocess.run(
        ["git", "commit", "-m",
         f"Echo pre-registration commit hash {commit[:12]} into {Path(out_path).name}\n\n"
         f"Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"],
        check=True,
    )
    final = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    print("\n" + "=" * 72)
    print(f"PRE-REGISTRATION COMMIT: {commit}")
    print(f"ECHO COMMIT:             {final}")
    print(f"FILE:                    {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()

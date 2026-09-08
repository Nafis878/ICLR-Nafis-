"""M3 analysis + M4 -- fit the OOP score (spec sections 5/M3 and 5/M4)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from analysis import collect  # noqa: E402
from analysis.oop_score import feature_importances, fit_oop_score, serialize  # noqa: E402


def build_frame() -> pd.DataFrame:
    df = collect.load_milestone("m3")
    if df.empty:
        raise SystemExit("no M3 rows in the cache; run src/experiments/m3_sweep.py first")
    id_cols = [c for c in collect.cell_id_columns(df) if c in df.columns]
    df["cell_id"] = df[id_cols].astype(str).agg("|".join, axis=1)
    return df


def regret_surface(df: pd.DataFrame, cfg: dict) -> dict:
    """Per-family regret summary and the tau level set (spec section 5/M3)."""
    ok = df[(df.status == "ok") & df.regret.notna()]
    tab = ok[ok.model == "tabpfn"]
    anchor = tab[tab.family == cfg["threshold_rule"]["reference_family"]]
    if anchor.empty:
        raise SystemExit("no scm_prior_control rows; cannot set tau by the stated rule")
    tau = float(np.percentile(anchor.regret, cfg["threshold_rule"]["percentile"]))

    per_family = (
        tab.groupby("family")
        .agg(
            n_cells=("regret", "size"),
            regret_mean=("regret", "mean"),
            regret_median=("regret", "median"),
            regret_p90=("regret", lambda s: float(np.percentile(s, 90))),
            regret_sd=("regret", "std"),
            frac_above_tau=("regret", lambda s: float((s > tau).mean())),
            frac_significant=("regret_significant", "mean"),
        )
        .sort_values("regret_median", ascending=False)
    )
    return {"tau": tau, "per_family": per_family, "tabpfn": tab, "all_ok": ok}


def degradation_axes(tab: pd.DataFrame) -> pd.DataFrame:
    """Which DGP axes move TabPFN's regret most (Spearman |rho|)."""
    from scipy.stats import spearmanr

    rows = []
    for col in [c for c in tab.columns if c.startswith(("param_", "mod_"))]:
        s = pd.to_numeric(tab[col], errors="coerce")
        m = s.notna() & tab.regret.notna()
        if m.sum() < 30 or s[m].nunique() < 5:
            continue
        r = spearmanr(s[m], tab.regret[m])
        rows.append({"axis": col, "spearman": float(r.statistic), "p": float(r.pvalue),
                     "n": int(m.sum())})
    return (
        pd.DataFrame(rows)
        .assign(abs_rho=lambda d: d.spearman.abs())
        .sort_values("abs_rho", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    env.preflight()
    cfg = runner.load_config("configs/m3_prior_sweep.yaml")
    df = build_frame()
    collect.save(df, "results/m3_sweep.parquet")

    n_fail = int((df.status != "ok").sum())
    print(f"[m3] rows={len(df)}  failures={n_fail} (recorded, not dropped)")
    if n_fail:
        print(df[df.status != "ok"].error.astype(str).str[:90].value_counts().head(5).to_string())

    surf = regret_surface(df, cfg)
    tau = surf["tau"]
    print(f"\n=== M3 regret surface | tau = {tau:.4f} "
          f"({cfg['threshold_rule']['percentile']}th pct of "
          f"{cfg['threshold_rule']['reference_family']}) ===")
    print(surf["per_family"].to_string(float_format=lambda v: f"{v:.4f}"))

    axes = degradation_axes(surf["tabpfn"])
    print("\n=== Sharpest degradation axes (Spearman rho vs TabPFN regret) ===")
    print(axes.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- M4: fit g on observable statistics only --------------------------
    tab = surf["tabpfn"].copy()
    fit = fit_oop_score(tab, target="regret", group_col="cell_id", seed=cfg["seed"])
    print(f"\n=== M4 OOP score g | {fit['n_rows']} rows, {fit['n_groups']} LHS groups, "
          f"{len(fit['features'])} statistics ===")
    for name, m in fit["cv"].items():
        print(f"  {name:12s} grouped-CV  R2={m['cv_r2']:+.3f}  "
              f"Spearman={m['cv_spearman']:+.3f}  MAE={m['cv_mae']:.4f}")
    print(f"  selected: {fit['model_name']}  (synthetic CV only -- not yet evidence about real data)")

    imp = feature_importances(fit)
    print("\n=== g feature importances (top 12) ===")
    print(imp.head(12).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    import joblib

    joblib.dump(fit, "results/m4_oop_score.joblib")
    Path("results").mkdir(exist_ok=True)
    Path("results/m4_oop_score.json").write_text(json.dumps({
        "subject_model": "TabPFN v2 (tabpfn==2.2.1, Prior-Labs/TabPFN-v2-clf, CPU)",
        "tau": tau,
        "threshold_rule": cfg["threshold_rule"],
        "g": serialize(fit),
        "cv": fit["cv"],
        "n_rows": fit["n_rows"],
        "n_groups": fit["n_groups"],
        "per_family_regret": json.loads(surf["per_family"].to_json(orient="index")),
        "degradation_axes": json.loads(axes.head(10).to_json(orient="records")),
    }, indent=2, default=float))
    imp.to_csv("results/m4_feature_importances.csv", index=False)
    axes.to_csv("results/m3_degradation_axes.csv", index=False)
    print("\n[m4] wrote results/m4_oop_score.{joblib,json}, m4_feature_importances.csv")


if __name__ == "__main__":
    main()

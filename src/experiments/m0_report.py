"""M0 checkpoint report: cost table, cost model, budget numbers (spec 5/M0)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import glob  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402


def load_rows() -> pd.DataFrame:
    recs = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        c = r.get("config", {})
        if c.get("milestone") != "m0":
            continue
        for row in r.get("rows", []):
            recs.append({
                "model": c["model"], "n": c["n_train"], "d": c["d"], "k": c["n_classes"],
                "rep": row.get("rep"), "status": row["status"],
                "t": row.get("fit_predict_s"), "wall": row.get("wall_s"),
                "rss_mb": row.get("peak_rss_mb"), "censored": row.get("censored", False),
            })
    return pd.DataFrame(recs)


def fit_cost_model(df: pd.DataFrame, model: str):
    """log t = a + b log n + c log d + e log k, fitted on UNCENSORED cells only."""
    s = df[(df.model == model) & (df.status == "ok") & df.t.notna()]
    if len(s) < 8:
        return None
    A = np.column_stack([
        np.ones(len(s)), np.log(s.n), np.log(s.d), np.log(s.k),
    ])
    yv = np.log(s.t.to_numpy(dtype=float))
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    pred = A @ coef
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {
        "coef": {"intercept": float(coef[0]), "log_n": float(coef[1]),
                 "log_d": float(coef[2]), "log_k": float(coef[3])},
        "r2_log": 1.0 - ss_res / max(ss_tot, 1e-12),
        "n_cells": int(len(s)),
        "median_s": float(s.t.median()),
        "predict": lambda n, d, k: float(
            np.exp(coef[0] + coef[1] * np.log(n) + coef[2] * np.log(d) + coef[3] * np.log(k))
        ),
    }


def main() -> None:
    cfg = runner.load_config("configs/m0_budget.yaml")
    df = load_rows()
    if df.empty:
        raise SystemExit("no M0 rows yet")

    print("=" * 78)
    print("M0 COMPUTE BUDGET PROBE")
    print("=" * 78)
    import psutil

    print(f"Cores: {env.n_physical_cores()} physical / {psutil.cpu_count(True)} logical")
    print(f"RAM:   {psutil.virtual_memory().total/1e9:.1f} GB")
    print(f"Cells: {len(df)}  |  " + "  ".join(
        f"{k}={v}" for k, v in df.status.value_counts().items()))

    print("\n--- median fit+predict seconds by (model, n_train) x d ---")
    for model in sorted(df.model.unique()):
        s = df[(df.model == model) & (df.status == "ok")]
        if s.empty:
            continue
        piv = s.pivot_table(index="n", columns="d", values="t", aggfunc="median")
        print(f"\n[{model}]")
        print(piv.to_string(float_format=lambda v: f"{v:7.1f}"))
        cen = df[(df.model == model) & df.censored]
        if len(cen):
            grid = sorted({(int(r.n), int(r.d)) for r in cen.itertuples()})
            print(f"  censored at >{cfg['cell_timeout_s']}s: {len(cen)} cells; "
                  f"smallest censored (n,d) = {grid[0] if grid else None}")

    print("\n--- peak RSS (MB) by n_train ---")
    rss = df[df.status == "ok"].pivot_table(index="n", columns="model", values="rss_mb",
                                            aggfunc="median")
    print(rss.to_string(float_format=lambda v: f"{v:7.0f}"))

    print("\n--- cost model  log t = a + b log n + c log d + e log k ---")
    models = {}
    for model in sorted(df.model.unique()):
        m = fit_cost_model(df, model)
        if not m:
            continue
        models[model] = m
        c = m["coef"]
        print(f"[{model}] R^2(log)={m['r2_log']:.3f} on {m['n_cells']} uncensored cells | "
              f"a={c['intercept']:+.2f} b_n={c['log_n']:+.2f} "
              f"c_d={c['log_d']:+.2f} e_k={c['log_k']:+.2f}")

    print("\n--- budget: max probe cells that fit ---")
    print("A 'probe cell' = one TabPFN fit+predict plus one tuned-LightGBM fit+predict.")
    rows = []
    for n, d, k in [(500, 10, 2), (1000, 20, 2), (1000, 20, 5)]:
        per = sum(m["predict"](n, d, k) for m in models.values())
        row = {"n": n, "d": d, "k": k, "sec_per_cell": per}
        for ch in cfg["budget_targets_core_hours"]:
            row[f"cells@{ch}ch"] = int(ch * 3600 / per)
        rows.append(row)
    bud = pd.DataFrame(rows)
    print(bud.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    Path("results").mkdir(exist_ok=True)
    df.to_parquet("results/m0_costs.parquet", index=False)
    Path("results/m0_cost_model.json").write_text(json.dumps({
        "cores_physical": env.n_physical_cores(),
        "ram_gb": psutil.virtual_memory().total / 1e9,
        "cell_timeout_s": cfg["cell_timeout_s"],
        "status_counts": {k: int(v) for k, v in df.status.value_counts().items()},
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "predict"}
                   for k, v in models.items()},
        "budget_table": json.loads(bud.to_json(orient="records")),
    }, indent=2))
    print("\n[m0] wrote results/m0_costs.parquet, results/m0_cost_model.json")


if __name__ == "__main__":
    main()

"""Turn the JSON result cache into tidy parquet frames (spec section 5/M3)."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd


def load_milestone(milestone: str) -> pd.DataFrame:
    """One row per (family, params, seed, model), flattened, failures INCLUDED.

    Failed cells are kept with status != 'ok' so that downstream code can report
    them rather than silently analysing a biased subset (spec 3.4).
    """
    records = []
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            continue
        cfg = r.get("config", {})
        if cfg.get("milestone") != milestone:
            continue
        rec = {
            "cell_key": r.get("cell_key"),
            "status": r.get("status"),
            "family": cfg.get("family"),
            "model": cfg.get("model"),
            "seed": cfg.get("seed"),
            "n_train": cfg.get("n_train"),
            "n_test": cfg.get("n_test"),
            "wall_s": r.get("wall_s"),
            "error": r.get("error"),
        }
        for k, v in (cfg.get("params") or {}).items():
            rec[f"param_{k}"] = v
        for mname, mp in (cfg.get("modifiers") or {}).items():
            for k, v in (mp or {}).items():
                rec[f"mod_{mname}_{k}"] = v
        for k in ("logloss", "logloss_se", "accuracy", "auc", "bayes_logloss",
                  "regret", "regret_se", "regret_significant", "exact_bayes", "n_eval"):
            if k in r:
                rec[k] = r[k]
        for k, v in (r.get("stats") or {}).items():
            rec[f"stat_{k}"] = v
        info = r.get("info") or {}
        rec["winner"] = info.get("winner")
        records.append(rec)
    return pd.DataFrame(records)


def cell_id_columns(df: pd.DataFrame) -> list[str]:
    """Columns identifying an LHS point (everything but seed/model/metrics)."""
    return [c for c in df.columns if c.startswith(("param_", "mod_"))] + ["family", "n_train"]


def save(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[collect] wrote {path}  rows={len(df)}  cols={len(df.columns)}")

"""M5 -- real dataset inventory and statistics (spec section 5/M5).

CRITICAL: this milestone computes the observable statistic vector on real data.
It does NOT run TabPFN on any real dataset -- that is forbidden until the M6
pre-registration commit exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import runner  # noqa: E402
from stats.features import compute_statistics  # noqa: E402


def _setup_openml(cache_dir: str):
    import openml

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    openml.config.cache_directory = str(Path(cache_dir).resolve())
    openml.config.retry_policy = "robust"
    return openml


def resolve_suites(cfg: dict) -> tuple[dict[str, list[int]], list[str]]:
    """Return {suite_name: [dataset ids]} plus any substitution notes."""
    openml = _setup_openml(cfg["cache_dir"])
    out, notes = {}, []
    for name, sid in cfg["suites"].items():
        try:
            suite = openml.study.get_suite(sid)
            out[name] = list(suite.data)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"suite {name} (id {sid}) unavailable: {type(exc).__name__}: {exc}")

    found = False
    for sid in cfg["tabarena_candidates"]:
        try:
            suite = openml.study.get_suite(sid)
            name = (getattr(suite, "name", "") or "").lower()
            if "tabarena" in name or "tab arena" in name:
                out["tabarena"] = list(suite.data)
                notes.append(f"TabArena resolved to OpenML suite {sid} ({suite.name})")
                found = True
                break
        except Exception:
            continue
    if not found:
        notes.append(
            "TabArena could not be resolved to an OpenML suite id from candidates "
            f"{cfg['tabarena_candidates']}. Proceeding with CC-18 + Grinsztajn only. "
            "This is a REPORTED SUBSTITUTION, not a silent drop: the real-dataset "
            "suite is therefore narrower than the spec asked for."
        )
    return out, notes


def load_dataset(did: int, cfg: dict):
    """Load one OpenML dataset as a numeric matrix plus an integer target."""
    openml = _setup_openml(cfg["cache_dir"])
    ds = openml.datasets.get_dataset(did, download_data=True,
                                     download_qualities=False,
                                     download_features_meta_data=True)
    X, y, cat_mask, names = ds.get_data(target=ds.default_target_attribute)
    if y is None:
        raise ValueError("no default target")
    X = pd.DataFrame(X).copy()

    pp = cfg["preprocessing"]
    for j, col in enumerate(X.columns):
        s = X[col]
        if pp["ordinal_encode_categoricals"] and (
            str(s.dtype) in ("category", "object", "bool") or (cat_mask and cat_mask[j])
        ):
            X[col] = pd.Categorical(s).codes.astype(float)
            X.loc[s.isna(), col] = np.nan
        else:
            X[col] = pd.to_numeric(s, errors="coerce")

    Xv = X.to_numpy(dtype=float)
    if pp["drop_constant_columns"]:
        keep = np.array([np.nanstd(Xv[:, j]) > 0 for j in range(Xv.shape[1])])
        if keep.any():
            Xv = Xv[:, keep]
    yv = pd.Categorical(y).codes.astype(int)

    good = yv >= 0
    Xv, yv = Xv[good], yv[good]
    all_nan = np.isnan(Xv).all(axis=1)
    Xv, yv = Xv[~all_nan], yv[~all_nan]

    if len(Xv) < pp["min_rows"]:
        raise ValueError(f"too few rows after cleaning: {len(Xv)}")
    if len(np.unique(yv)) < pp["min_classes"]:
        raise ValueError("fewer than 2 classes after cleaning")
    return Xv, yv, ds.name


def in_operating_range(n, d, k, rng_cfg) -> bool:
    return n <= rng_cfg["max_n"] and d <= rng_cfg["max_d"] and k <= rng_cfg["max_classes"]


def run_one(cfg: dict) -> dict:
    """Statistics for one real dataset. No TabPFN here, by design."""
    did = cfg["dataset_id"]
    X, y, name = load_dataset(did, cfg["m5"])
    n, d = X.shape
    k = int(len(np.unique(y)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = compute_statistics(X, y)
    return {
        "dataset_id": did,
        "name": name,
        "n": n,
        "d": d,
        "n_classes": k,
        "in_range": bool(in_operating_range(n, d, k, cfg["m5"]["tabpfn_operating_range"])),
        "suites": cfg["suites_of"],
        "stats": stats,
    }


def main() -> None:
    env.preflight()
    cfg = runner.load_config("configs/m5_real_datasets.yaml")
    suites, notes = resolve_suites(cfg)
    for nte in notes:
        print("[m5] NOTE:", nte)

    membership: dict[int, list[str]] = {}
    for sname, dids in suites.items():
        for did in dids:
            membership.setdefault(int(did), []).append(sname)
    all_ids = sorted(membership)
    print(f"[m5] {len(all_ids)} unique datasets across {len(suites)} suites "
          f"(deduplicated from {sum(len(v) for v in suites.values())})")

    rng = np.random.default_rng(cfg["seed"])
    if len(all_ids) > cfg["max_datasets"]:
        all_ids = sorted(rng.choice(all_ids, size=cfg["max_datasets"], replace=False).tolist())
        print(f"[m5] sampled {len(all_ids)} datasets to fit the budget (seeded, recorded)")

    cells = [
        {"milestone": "m5", "dataset_id": int(did), "suites_of": membership[int(did)], "m5": cfg}
        for did in all_ids
    ]
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rows = runner.run_many(cells, run_one, n_jobs=n_jobs, verbose=5)

    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    print(f"\n[m5] usable {len(ok)} / attempted {len(rows)}; {len(bad)} failed (recorded, not dropped)")
    for r in bad[:10]:
        print(f"    did={r['config']['dataset_id']} {str(r.get('error'))[:110]}")

    recs = []
    for r in ok:
        rec = {kk: r[kk] for kk in ("dataset_id", "name", "n", "d", "n_classes", "in_range")}
        rec["suites"] = ",".join(r["suites"])
        rec.update({f"stat_{k}": v for k, v in r["stats"].items()})
        recs.append(rec)
    df = pd.DataFrame(recs).sort_values("dataset_id")
    Path("results").mkdir(exist_ok=True)
    df.to_parquet("results/m5_real_statistics.parquet", index=False)
    Path("results/m5_suite_notes.json").write_text(json.dumps(
        {"notes": notes, "suites": {k: len(v) for k, v in suites.items()},
         "n_usable": len(ok), "n_failed": len(bad)}, indent=2))
    print(f"[m5] wrote results/m5_real_statistics.parquet ({len(df)} datasets)")
    print(f"[m5] in-range {int(df['in_range'].sum())} / out-of-range {int((~df['in_range']).sum())}")


if __name__ == "__main__":
    main()

"""Cached, parallel, resumable experiment loop (spec sections 2 and 3).

Contract:
  * every cell is keyed by a sha256 of its canonical-JSON config;
  * completed cells are skipped on restart;
  * failures are RECORDED as rows, never dropped (spec 3.4);
  * every cell records its seed and its wall-clock cost.
"""
from __future__ import annotations

import env  # noqa: F401  -- must be first; sets threads/CPU/telemetry

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

CACHE_DIR = Path("results/cache")


def config_hash(config: dict) -> str:
    """Stable key for a cell. sort_keys makes it order-independent."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def run_cell(config: dict, fn: Callable[[dict], dict], *, force: bool = False) -> dict:
    """Execute one experiment cell, or return its cached result.

    `fn` receives the config and returns a dict of metrics. Any exception is
    caught and stored as a failure row -- silent exclusion is how fake results
    happen (spec 3.4).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = config_hash(config)
    path = _cache_path(key)

    if path.exists() and not force:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass  # truncated by an interrupted write; recompute below

    t0 = time.perf_counter()
    peak_rss_mb = None
    try:
        import psutil

        proc = psutil.Process()
        rss_before = proc.memory_info().rss
    except Exception:
        proc, rss_before = None, 0

    try:
        metrics = fn(config)
        row = {"status": "ok", **metrics}
    except MemoryError:
        row = {"status": "oom", "error": "MemoryError", "traceback": traceback.format_exc()}
    except Exception as exc:  # noqa: BLE001 -- deliberate: all failures are data
        row = {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    if proc is not None:
        try:
            peak_rss_mb = max(0.0, (proc.memory_info().rss - rss_before) / 1e6)
        except Exception:
            pass

    row.update(
        {
            "cell_key": key,
            "config": config,
            "wall_s": time.perf_counter() - t0,
            "peak_rss_delta_mb": peak_rss_mb,
        }
    )

    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, default=str))
    tmp.replace(path)  # atomic: a killed run never leaves a half-written cache hit
    return row


def run_many(
    configs: Iterable[dict],
    fn: Callable[[dict], dict],
    *,
    n_jobs: int = 1,
    verbose: int = 10,
) -> list[dict]:
    """Run cells in parallel at the experiment level (workers stay single-threaded)."""
    from joblib import Parallel, delayed

    configs = list(configs)
    todo = [c for c in configs if not _cache_path(config_hash(c)).exists()]
    print(f"[runner] {len(configs)} cells, {len(configs) - len(todo)} cached, {len(todo)} to run")

    if todo:
        Parallel(n_jobs=n_jobs, verbose=verbose, backend="loky")(
            delayed(run_cell)(c, fn) for c in todo
        )
    return [run_cell(c, fn) for c in configs]  # all cache hits now


def load_config(path: str | Path) -> dict:
    """Every experiment reads a YAML config; no hardcoded hyperparameters (spec 3.5)."""
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh)


def safe_n_jobs(peak_rss_mb: float, *, reserve_gb: float = 2.5) -> int:
    """Worker count bounded by RAM headroom, not just core count.

    13.9 GB total on this machine, so a TabPFN cell that peaks at ~1.5 GB caps
    us well below the 8 physical cores.
    """
    import psutil

    avail_gb = psutil.virtual_memory().total / 1e9 - reserve_gb
    by_ram = max(1, int(avail_gb * 1000 / max(peak_rss_mb, 1.0)))
    return max(1, min(env.n_physical_cores(), by_ram))

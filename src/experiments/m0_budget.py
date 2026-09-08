"""M0 -- compute budget probe. Gates the whole project (spec section 5/M0)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import env  # noqa: F401,E402

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import runner  # noqa: E402

WORKER = str(Path(__file__).resolve().parents[1] / "m0_worker.py")
PYEXE = sys.executable


def run_subprocess_cell(cfg: dict, timeout_s: float) -> dict:
    """Hard-timeout a single cell. A timeout is a right-censored datum, not a gap."""
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [PYEXE, "-u", WORKER],
            input=json.dumps(cfg),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(Path(WORKER).parent),
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "censored": True,
            "fit_predict_s": None,
            "timeout_s": timeout_s,
            "wall_s": time.perf_counter() - t0,
        }
    if proc.returncode != 0:
        return {
            "status": "error",
            "censored": False,
            "fit_predict_s": None,
            "error": (proc.stderr or "")[-1500:],
            "wall_s": time.perf_counter() - t0,
        }
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "censored": False,
            "fit_predict_s": None,
            "error": f"unparseable worker output: {exc} :: {proc.stdout[-500:]}",
            "wall_s": time.perf_counter() - t0,
        }
    out["censored"] = False
    out["wall_s"] = time.perf_counter() - t0
    return out


def run_group(group_cfg: dict) -> dict:
    """All repeats of one (model, n, d, c) configuration.

    Early-stop rule: if repeat 0 times out, later repeats are recorded as
    censored-by-inference rather than spending budget to re-confirm.
    """
    rows, timed_out = [], False
    for rep in range(group_cfg["repeats"]):
        cell = {k: v for k, v in group_cfg.items() if k != "repeats"}
        cell["rep"] = rep
        cell["seed"] = group_cfg["seed"] * 1000 + rep
        if timed_out:
            rows.append(
                {
                    "status": "skipped_after_timeout",
                    "censored": True,
                    "fit_predict_s": None,
                    "rep": rep,
                    "reason": "repeat 0 of this configuration exceeded the cap",
                }
            )
            continue
        out = run_subprocess_cell(cell, group_cfg["cell_timeout_s"])
        out["rep"] = rep
        rows.append(out)
        if out["status"] == "timeout":
            timed_out = True
    return {"rows": rows}


def build_groups(cfg: dict) -> list[dict]:
    g = cfg["grid"]
    groups = []
    for model, mp in cfg["models"].items():
        for n in g["n_train"]:
            for d in g["d"]:
                for c in g["n_classes"]:
                    groups.append(
                        {
                            "milestone": "m0",
                            "model": model,
                            "model_params": mp,
                            "n_train": n,
                            "d": d,
                            "n_classes": c,
                            "n_test": cfg["n_test"],
                            "repeats": g["repeats"],
                            "seed": cfg["seed"],
                            "cell_timeout_s": cfg["cell_timeout_s"],
                        }
                    )
    return groups


def main() -> None:
    env.preflight()
    cfg = runner.load_config("configs/m0_budget.yaml")
    groups = build_groups(cfg)
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"[m0] {len(groups)} configurations x {cfg['grid']['repeats']} repeats, n_jobs={n_jobs}")
    runner.run_many(groups, run_group, n_jobs=n_jobs, verbose=5)
    print("[m0] done")


if __name__ == "__main__":
    main()

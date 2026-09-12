"""Wait for the last strong-baseline cell (Fashion-MNIST, did=40996) to land.

Liveness is checked with psutil, NOT `kill -0`: under Git Bash the shell's PID
namespace is not the Windows one, so `kill -0 <winpid>` reports a live process
as dead. That false alarm cost a watcher restart once already.
"""
import glob, json, sys, time
import psutil

WORKER_PID, DID = 15424, 40996


def landed():
    for f in glob.glob("results/cache/*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r.get("config", {})
        if c.get("milestone") == "m12base" and c.get("dataset_id") == DID:
            return r.get("status")
    return None


while True:
    st = landed()
    if st is not None:
        print(f"FASHION_MNIST_CELL_LANDED status={st}", flush=True)
        sys.exit(0)
    if not psutil.pid_exists(WORKER_PID):
        print("WORKER_DIED_WITHOUT_RESULT", flush=True)
        sys.exit(2)
    time.sleep(120)

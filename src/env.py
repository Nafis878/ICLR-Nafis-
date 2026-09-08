"""Import-time environment guard. Every entrypoint must import this FIRST.

Enforces spec section 2: CPU only, single-threaded BLAS inside workers
(parallelism happens at the experiment level via joblib), no telemetry.
"""
from __future__ import annotations

import os

# Must be set before numpy/torch/sklearn import, hence module top-level.
for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_v] = "1"

# Force CPU at the framework level. CUDA_VISIBLE_DEVICES="" makes any accidental
# .cuda() call fail loudly rather than silently selecting a device.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Telemetry opt-out (mechanism VERIFIED against tabpfn_common_utils 0.2.23) ---
# That package gates telemetry on two independent conditions:
#   1. ProductTelemetry.PROJECT_API_KEY = os.getenv("PRIOR_POSTHOG_PROJECT_TOKEN");
#      no credential ships in the package, and with no token it is a hard no-op.
#   2. telemetry/core/config.download_config() GETs a remote JSON config and
#      fail-closes to {"enabled": False} on any network/parse error.
# Pointing the config URL at an unroutable address makes (2) deterministic and
# also stops the outbound request itself, rather than relying on Prior Labs'
# server-side default. Earlier guesses such as TABPFN_DISABLE_TELEMETRY are NOT
# read by this version and were removed rather than left in as false comfort.
os.environ["TABPFN_TELEMETRY_CONFIG_URL"] = "http://127.0.0.1:9/telemetry-disabled.json"
os.environ.pop("PRIOR_POSTHOG_PROJECT_TOKEN", None)
os.environ["DO_NOT_TRACK"] = "1"

# TabPFN v2 hard-refuses CPU inference above 1000 training rows ("not allowed by
# default due to slow performance"). This project is CPU-only by mandate (spec
# section 2), so the documented override is required rather than optional. The
# refusal threshold is itself an M0 finding about the compute budget.
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

DEVICE = "cpu"


def assert_cpu() -> None:
    """Fail loudly if any GPU path is reachable (spec section 6)."""
    import torch

    assert not torch.cuda.is_available(), "CUDA is visible; spec forbids any GPU code path"
    if hasattr(torch.backends, "mps"):
        assert not torch.backends.mps.is_available(), "MPS is visible; spec forbids it"
    assert torch.empty(1).device.type == "cpu", "default torch device is not CPU"


def n_physical_cores() -> int:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or 1
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


def assert_no_telemetry() -> None:
    """Verify the telemetry opt-out actually took effect (see notes above)."""
    from tabpfn_common_utils.telemetry.core.config import download_config
    from tabpfn_common_utils.telemetry.core.service import ProductTelemetry

    assert download_config().get("enabled") is False, "telemetry config did not fail closed"
    assert not os.environ.get("PRIOR_POSTHOG_PROJECT_TOKEN"), "a PostHog token is present"
    # ProductTelemetry is wrapped by @singleton, so it is a factory function here,
    # not the class; the instance is what carries the client.
    assert ProductTelemetry()._posthog_client is None, "a PostHog client was constructed"


def preflight() -> None:
    assert_cpu()
    assert_no_telemetry()

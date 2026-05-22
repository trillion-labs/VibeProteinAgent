"""Phase 20 validation matrix probe.

For every registered adapter, attempt a cold-warm load (instantiate the
adapter class) and capture:

  - warm_ok (bool): adapter constructor returned without raising
  - warm_time_s (float | None): wall-clock of the constructor
  - healthy (bool | None): adapter.healthcheck() result (or None if construct failed)
  - reason (str | None): if not warm_ok or not healthy, a human-readable note
  - support_status (str): descriptor-level support tier
  - available (bool): true only when warm_ok, healthy, and descriptor status is verified
  - verdict (str): machine-readable combined row state

This is NOT a full benchmark — we are answering the binary "does the warm
path succeed on this host?" question for the registered adapters in one script.
Results feed docs/validation-matrix.md.

Run with:
    python scripts/_phase20_matrix.py > /tmp/phase20_matrix.json
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from vibeproteinagent.adapters import ToolRegistry  # noqa: E402
from vibeproteinagent.adapters.availability import descriptor_availability  # noqa: E402


@contextlib.contextmanager
def _redirect_process_stdout_to_stderr():
    """Route native/subprocess stdout to stderr while preserving JSON stdout.

    Some upstream warm paths compile native extensions and write directly to
    file descriptor 1, bypassing ``contextlib.redirect_stdout``. The validation
    matrix must keep stdout machine-readable, so probes run with fd 1 duped to
    fd 2 and the final JSON is emitted after restoration.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def _probe_adapter(name: str, descriptor, cls) -> dict:
    """Try cold-warm load + healthcheck. Never raise."""
    support_status = getattr(descriptor.support_status, "value", descriptor.support_status)
    availability = descriptor_availability(descriptor)
    record: dict = {
        "name": name,
        "version": descriptor.version,
        "category": str(descriptor.category).replace("Category.", "").lower(),
        "gpu_required": bool(descriptor.gpu_required),
        "pin_warm": bool(descriptor.pin_warm),
        "support_status": support_status or "verified",
        "support_status_reason": descriptor.support_status_reason,
        "availability_probe": descriptor.availability_probe,
        "warm_ok": False,
        "warm_time_s": None,
        "healthy": None,
        "reason": None,
        "availability_reason": availability.get("availability_reason"),
        "available": False,
        "verdict": "warm_failed",
    }

    # Probe the package for an availability_skip_reason() short-circuit;
    # gated adapters use this hook to signal a known operator-side gap
    # without spawning subprocesses.
    if availability.get("availability_reason"):
        record["reason"] = availability["availability_reason"]

    module_name = getattr(cls, "__module__", None)
    if module_name:
        pkg_name = module_name.rsplit(".", 1)[0]
        pkg = sys.modules.get(pkg_name)
        probe = getattr(pkg, "availability_skip_reason", None)
        if callable(probe):
            try:
                reason = probe()
                if isinstance(reason, str) and reason:
                    record["availability_reason"] = reason
                    record["reason"] = reason
            except Exception as exc:  # pragma: no cover - defensive
                record["reason"] = f"availability_skip_reason raised: {exc}"

    device = torch.device("cuda" if torch.cuda.is_available()
                          and descriptor.gpu_required else "cpu")
    t0 = time.time()
    try:
        adapter = cls(config={}, weights_dir=Path("/tmp"), device=device)
        record["warm_ok"] = True
        record["warm_time_s"] = round(time.time() - t0, 2)
        try:
            record["healthy"] = bool(adapter.healthcheck())
        except Exception as exc:
            record["healthy"] = False
            record["reason"] = (record["reason"] or "") + \
                f" healthcheck raised {type(exc).__name__}: {exc}"
        # Best-effort unload to free VRAM between probes.
        try:
            adapter.unload()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        record["warm_time_s"] = round(time.time() - t0, 2)
        record["warm_ok"] = False
        tb = traceback.format_exc(limit=3)
        # Use the existing reason (gate message) when present, else the exc.
        if record["reason"] is None:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        record["traceback_tail"] = tb.splitlines()[-3:]
    if (
        record["warm_ok"]
        and not record["healthy"]
        and record["reason"] is None
        and cls.__name__.endswith("UnavailableAdapter")
    ):
        record["reason"] = (
            "adapter import failed during registry discovery; descriptor-only "
            "placeholder is not executable"
        )
    record["available"] = (
        bool(record["warm_ok"])
        and bool(record["healthy"])
        and record["support_status"] == "verified"
    )
    if not record["warm_ok"]:
        record["verdict"] = "warm_failed"
    elif record["support_status"] != "verified":
        record["verdict"] = record["support_status"]
    elif record["healthy"]:
        record["verdict"] = "verified"
    else:
        record["verdict"] = "unhealthy"
    return record


def main() -> None:
    reg = ToolRegistry()
    reg.discover()
    out: list[dict] = []
    for descriptor in sorted(reg.list(), key=lambda d: (str(d.category), d.name)):
        cls = reg.get(descriptor.name, "*")[1]
        with contextlib.redirect_stdout(sys.stderr), _redirect_process_stdout_to_stderr():
            rec = _probe_adapter(descriptor.name, descriptor, cls)
        out.append(rec)
        # One-line progress to stderr so we can watch the loop.
        verdict = (
            "AVAILABLE" if rec["available"]
            else rec["verdict"].upper() if rec["warm_ok"]
            else "WARM_FAILED"
        )
        health = (
            "WARM_OK_HEALTHY" if rec["warm_ok"] and rec["healthy"]
            else "WARM_OK_UNHEALTHY" if rec["warm_ok"]
            else "WARM_FAILED"
        )
        print(
            f"[probe] {descriptor.name:34s} {verdict:18s} {health:18s} "
            f"warm={rec['warm_time_s']}s healthy={rec['healthy']}",
            file=sys.stderr,
        )
    json.dump(out, sys.stdout, indent=2, default=str)
    print(file=sys.stdout)


if __name__ == "__main__":
    main()

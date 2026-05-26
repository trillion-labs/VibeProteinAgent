from __future__ import annotations

import importlib
from typing import Any

from vibeproteinagent.adapters.runtime import AvailabilityReport, RuntimeContract


def descriptor_availability(
    descriptor: Any,
    adapter_cls: type[Any] | None = None,
) -> dict[str, Any]:
    """Return live, descriptor-level availability metadata.

    This is intentionally shallow: it runs only the adapter's declared
    availability probe, not a constructor warm or model load. Heavy warm/run
    checks remain daemon and validation-matrix responsibilities.
    """
    contract = RuntimeContract.from_descriptor(descriptor)
    support_status = contract.support_status
    reason = contract.support_reason
    probe_ref = contract.availability_probe
    checked = False
    details: dict[str, Any] = {
        "tool": contract.tool,
        "execution_isolation": contract.execution_isolation,
        "environment_name": contract.environment_name,
        "environment_reason": contract.environment_reason,
    }

    if isinstance(probe_ref, str) and ":" in probe_ref:
        checked = True
        module_name, func_name = probe_ref.split(":", 1)
        try:
            probe = getattr(importlib.import_module(module_name), func_name)
            probe_reason = probe()
            if isinstance(probe_reason, dict):
                details.update(probe_reason.get("checks") or {})
                if probe_reason.get("reason"):
                    reason = str(probe_reason["reason"])
                elif probe_reason.get("available") is True:
                    reason = None
            elif isinstance(probe_reason, str) and probe_reason:
                reason = probe_reason
        except Exception as exc:
            reason = f"{probe_ref} raised {type(exc).__name__}: {exc}"
            details["probe_error_type"] = type(exc).__name__

    import_error_details = (
        getattr(adapter_cls, "_vpa_unavailable_details", None)
        if adapter_cls is not None
        else None
    )
    if isinstance(import_error_details, dict):
        checked = True
        reason = str(import_error_details.get("import_error") or "adapter import failed")
        details["adapter_import"] = dict(import_error_details)

    report = AvailabilityReport.from_skip_reason(
        reason,
        support_status=support_status,
        details=details,
    )
    out = {
        "support_status": support_status,
        "availability_checked": checked,
        "availability_reason": reason,
        "available": report.available,
    }
    report_payload = report.to_dict()
    # CLI/MCP JSON commands are expected to be deterministic across equivalent
    # invocations. Keep checked_at on the internal report object, but do not
    # attach a wall-clock timestamp to descriptor listings.
    report_payload.pop("availability_checked_at", None)
    out.update(report_payload)
    return out


def attach_availability(
    payload: dict[str, Any],
    descriptor: Any,
    adapter_cls: type[Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``payload`` with live availability fields attached."""
    out = dict(payload)
    out.update(descriptor_availability(descriptor, adapter_cls))
    return out


__all__ = ["attach_availability", "descriptor_availability"]

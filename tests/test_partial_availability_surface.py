"""Compatibility-island adapters export typed availability probes.

The probes are still useful after promotion to VERIFIED: daemon warm and
validation-matrix reporting can surface concrete operator bootstrap messages
when a host lacks an optional sidecar runtime.
"""
from __future__ import annotations

import importlib

import pytest

# Adapters whose runtime readiness depends on optional local sidecar assets/envs.
SIDECAR_PROBED_ADAPTERS = [
    "bindcraft",
]


@pytest.mark.parametrize("name", SIDECAR_PROBED_ADAPTERS)
def test_sidecar_adapter_exports_availability_skip_reason(name: str) -> None:
    mod = importlib.import_module(f"vibeproteinagent.adapters.{name}")
    probe = getattr(mod, "availability_skip_reason", None)
    assert callable(probe), (
        f"adapter package {name!r} must expose availability_skip_reason() "
        f"so worker healthcheck warnings carry a concrete bootstrap message"
    )
    result = probe()
    assert result is None or isinstance(result, str), (
        f"{name}.availability_skip_reason() returned {type(result).__name__}; "
        f"expected str or None"
    )

from __future__ import annotations

import os
import sys
from pathlib import Path

from vibeproteinagent.adapters.runtime import (
    AvailabilityReport,
    EnvLocator,
    RuntimeContract,
    sanitized_python_env,
    smoke_import,
)


class _Status:
    value = "verified"


class _Descriptor:
    name = "unit"
    support_status = _Status()
    support_status_reason = None
    execution_isolation = "sidecar"
    environment_name = "unit-env"
    environment_reason = "test"
    availability_probe = None


def test_availability_report_preserves_legacy_shape():
    report = AvailabilityReport.from_skip_reason(None, support_status="verified")
    payload = report.to_dict()
    assert payload["available"] is True
    assert payload["availability_status"] == "available"
    assert report.skip_reason() is None


def test_runtime_contract_from_descriptor():
    contract = RuntimeContract.from_descriptor(_Descriptor())
    assert contract.tool == "unit"
    assert contract.support_status == "verified"
    assert contract.execution_isolation == "sidecar"
    assert contract.environment_name == "unit-env"


def test_sanitized_python_env_strips_parent_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/leaky/path")
    env = sanitized_python_env(seed=7, vendor_paths=[Path("/vendor")])
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0].endswith("/src")
    assert "/vendor" in parts
    assert "/leaky/path" not in parts
    assert env["PYTHONHASHSEED"] == "7"
    assert env["PYTHONNOUSERSITE"] == "1"


def test_smoke_import_success_uses_current_python():
    reason = smoke_import(
        label="unit",
        python=Path(sys.executable),
        code="import json; print('ok')",
        timeout_s=10,
    )
    assert reason is None


def test_env_locator_python_in_env():
    assert EnvLocator.python_in_env(Path("/tmp/env")) == Path("/tmp/env/bin/python")

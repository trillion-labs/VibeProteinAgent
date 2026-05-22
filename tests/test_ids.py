"""Unit tests for the deterministic job-id derivation (Architecture v1 — Errata 2).

Pre-Phase-10 the framework used UUIDv7 for ``job_id``, which produced
time-sortable but **non-deterministic** ids. The original brief and
Codex round 4 RC1 require *deterministic naming*: the same
``(tool, version, params, workspace, workflow, salt)`` always resolves
to the same ``job_id``.
"""

from __future__ import annotations

import re

import pytest

from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace.ids import (
    derive_job_id,
    is_job_id,
    parse_id,
    validate_call_id,
)

JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{32}$")


def _sample(**overrides) -> dict:
    base = {
        "tool": "noop", "version": "1.0.0",
        "params": {"payload": "x"}, "workspace": "default",
        "workflow_id": None, "workflow_node_id": None, "salt": None,
    }
    base.update(overrides)
    return base


def test_derive_job_id_format() -> None:
    jid = derive_job_id(**_sample())
    assert JOB_ID_PATTERN.match(jid), jid
    assert is_job_id(jid)


def test_derive_job_id_is_deterministic() -> None:
    a = derive_job_id(**_sample())
    b = derive_job_id(**_sample())
    assert a == b


def test_derive_job_id_changes_with_params() -> None:
    a = derive_job_id(**_sample(params={"payload": "x"}))
    b = derive_job_id(**_sample(params={"payload": "y"}))
    assert a != b


def test_derive_job_id_changes_with_version() -> None:
    a = derive_job_id(**_sample(version="1.0.0"))
    b = derive_job_id(**_sample(version="1.0.1"))
    assert a != b


def test_derive_job_id_changes_with_workspace() -> None:
    a = derive_job_id(**_sample(workspace="alpha"))
    b = derive_job_id(**_sample(workspace="beta"))
    assert a != b


def test_derive_job_id_changes_with_workflow_binding() -> None:
    a = derive_job_id(**_sample(workflow_id="wf_a", workflow_node_id="n1"))
    b = derive_job_id(**_sample(workflow_id="wf_b", workflow_node_id="n1"))
    c = derive_job_id(**_sample(workflow_id="wf_a", workflow_node_id="n2"))
    assert len({a, b, c}) == 3


def test_derive_job_id_salt_breaks_collision() -> None:
    """``--force-new`` passes a random salt — the resulting id MUST differ."""
    base = derive_job_id(**_sample())
    salted = derive_job_id(**_sample(salt="rerun-001"))
    assert base != salted


def test_derive_job_id_key_order_independent() -> None:
    """Params with different key insertion order canonicalise to the same id."""
    a = derive_job_id(**_sample(params={"a": 1, "b": 2}))
    b = derive_job_id(**_sample(params={"b": 2, "a": 1}))
    assert a == b


def test_validate_call_id_accepts_valid() -> None:
    for s in ["abc-001", "my.call_id", "A1", "x" * 128, "user.name-1.0"]:
        assert validate_call_id(s) == s


def test_validate_call_id_rejects_invalid() -> None:
    bad = ["", "x" * 129, "has spaces", "slash/inside", "tab\t", "comma,inside"]
    for s in bad:
        with pytest.raises(InvalidParamsError):
            validate_call_id(s)


def test_parse_id_classifies() -> None:
    jid = derive_job_id(**_sample())
    assert parse_id(jid) == ("job_id", jid)
    assert parse_id("my-call-id") == ("call_id", "my-call-id")
    with pytest.raises(InvalidParamsError):
        parse_id("has spaces")
    with pytest.raises(InvalidParamsError):
        parse_id("")


def test_is_job_id_accepts_legacy_uuid7_shape() -> None:
    """Backward-compat: a 32-hex string after ``job_`` matches whether it's a
    legacy UUIDv7-derived id or the new sha256-derived id.
    """
    # Synthetic legacy id — typical UUIDv7 timestamp prefix.
    legacy = "job_019e2f48a0b478178e99a7e274182b59"
    assert is_job_id(legacy)
    # And the new sha256-derived one.
    assert is_job_id(derive_job_id(**_sample()))

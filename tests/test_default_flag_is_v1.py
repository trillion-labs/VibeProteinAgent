from __future__ import annotations

import pytest

from vibeproteinagent.agents.config import resolve_planner_ir_mode


def test_default_planner_ir_mode_is_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPA_PLANNER_IR", raising=False)
    assert resolve_planner_ir_mode() == "v1"


def test_explicit_planner_ir_off_keeps_legacy_escape_hatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    assert resolve_planner_ir_mode() == "off"


def test_explicit_planner_ir_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "v1")
    assert resolve_planner_ir_mode() == "v1"


def test_invalid_planner_ir_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "legacy")
    with pytest.raises(ValueError, match="Invalid VPA_PLANNER_IR"):
        resolve_planner_ir_mode()

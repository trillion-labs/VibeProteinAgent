from __future__ import annotations

import os

from vibeproteinagent.core.compat import compat_entries, ensure_esm3_shared_hf_cache


def test_compat_registry_entries_are_named_and_actionable():
    entries = {entry.name: entry for entry in compat_entries()}
    assert "torch_full_load" in entries
    assert "dgl_graphbolt_stub" in entries
    for entry in entries.values():
        assert entry.affected_adapter
        assert entry.why
        assert entry.activation_condition
        assert entry.removal_condition
        assert entry.test.startswith("tests/")


def test_esm3_cache_default_is_non_destructive(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    ensure_esm3_shared_hf_cache("/tmp/hf-a")
    assert os.environ["HF_HOME"] == "/tmp/hf-a"
    ensure_esm3_shared_hf_cache("/tmp/hf-b")
    assert os.environ["HF_HOME"] == "/tmp/hf-a"

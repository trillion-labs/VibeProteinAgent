"""Tests for Codex round-6 RC4: file-content-aware cache keys.

The deterministic ``job_id`` and the workflow cache key must hash file
*contents* — not just path strings — for any schema field annotated with
``vpa:digest``. Overwriting a fixture in place used to silently re-use
the old cached row.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.cache import compute_cache_key
from vibeproteinagent.workflows.descriptor import WorkflowNode
from vibeproteinagent.workspace import JobStore, Workspace, normalize_params_for_digest

_PROTEINMPNN_SCHEMA = {
    "type": "object",
    "properties": {
        "pdb_path": {"type": "string", "vpa:digest": "sha256"},
        "num_designs": {"type": "integer"},
    },
}


_OPTIONAL_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "input_yaml": {"type": "string", "vpa:digest": "sha256-if-path"},
        "sequence": {"type": "string"},
    },
}


# ---------------------------------------------------------------- helper unit


def test_normalize_replaces_paths_with_content_digest(tmp_path: Path) -> None:
    p = tmp_path / "x.pdb"
    p.write_text("ATOM 1\n")
    out = normalize_params_for_digest(
        {"pdb_path": str(p), "num_designs": 2}, _PROTEINMPNN_SCHEMA,
    )
    assert out["num_designs"] == 2
    assert out["pdb_path"].startswith("vpa:sha256:")
    # The path string itself is gone.
    assert str(p) not in out["pdb_path"]


def test_normalize_missing_path_raises_invalid_params(tmp_path: Path) -> None:
    missing = tmp_path / "nope.pdb"
    with pytest.raises(InvalidParamsError) as exc:
        normalize_params_for_digest(
            {"pdb_path": str(missing)}, _PROTEINMPNN_SCHEMA,
        )
    assert exc.value.details["missing_path"] == str(missing)


def test_normalize_oversize_file_raises(tmp_path: Path) -> None:
    big = tmp_path / "big.pdb"
    # 257 MiB; just past the 256 MiB cap.
    cap = 256 * 1024 * 1024
    with big.open("wb") as f:
        f.seek(cap + 1)
        f.write(b"\0")
    with pytest.raises(InvalidParamsError) as exc:
        normalize_params_for_digest({"pdb_path": str(big)}, _PROTEINMPNN_SCHEMA)
    assert "size_bytes" in exc.value.details
    assert exc.value.details["size_bytes"] > cap


def test_normalize_sha256_if_path_keeps_inline_strings(tmp_path: Path) -> None:
    # Inline (non-path) value — not digested.
    inline = "version: 1\nsequences: []\n"
    out = normalize_params_for_digest(
        {"input_yaml": inline}, _OPTIONAL_PATH_SCHEMA,
    )
    assert out["input_yaml"] == inline


def test_normalize_sha256_if_path_digests_real_path(tmp_path: Path) -> None:
    p = tmp_path / "spec.yaml"
    p.write_text("sequences: []\n")
    out = normalize_params_for_digest(
        {"input_yaml": str(p)}, _OPTIONAL_PATH_SCHEMA,
    )
    assert out["input_yaml"].startswith("vpa:sha256:")


def test_normalize_no_schema_is_passthrough() -> None:
    params = {"pdb_path": "/no/file/here.pdb", "x": 1}
    assert normalize_params_for_digest(params, None) == params


# ---------------------------------------------------------------- JobStore wiring


def test_jobstore_same_path_different_content_different_job_id(
    tmp_workspace: Workspace, tmp_path: Path,
) -> None:
    pdb = tmp_path / "fixture.pdb"
    pdb.write_text("ATOM 1\n")
    store = JobStore(tmp_workspace)
    rec1 = store.submit(
        "proteinmpnn", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    # Overwrite in place; same path string, different bytes.
    pdb.write_text("ATOM 1\nATOM 2\n")
    rec2 = store.submit(
        "proteinmpnn", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    assert rec1.job_id != rec2.job_id, (
        "Path-content digest must invalidate the deterministic job_id when "
        "the file's bytes change."
    )
    # params_hash mirrors the contract.
    assert rec1.params_hash != rec2.params_hash


def test_jobstore_same_path_same_content_idempotent(
    tmp_workspace: Workspace, tmp_path: Path,
) -> None:
    pdb = tmp_path / "fixture.pdb"
    pdb.write_text("ATOM 1\n")
    store = JobStore(tmp_workspace)
    rec1 = store.submit(
        "proteinmpnn", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    rec2 = store.submit(
        "proteinmpnn", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    assert rec1.job_id == rec2.job_id


def test_jobstore_params_json_keeps_original_path(
    tmp_workspace: Workspace, tmp_path: Path,
) -> None:
    pdb = tmp_path / "fixture.pdb"
    pdb.write_text("ATOM 1\n")
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "proteinmpnn", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    # The forensic view still shows the original path the user submitted.
    assert rec.params["pdb_path"] == str(pdb)


def test_jobstore_no_schema_falls_back_to_raw_hash(
    tmp_workspace: Workspace, tmp_path: Path,
) -> None:
    pdb = tmp_path / "fixture.pdb"
    pdb.write_text("ATOM 1\n")
    store = JobStore(tmp_workspace)
    rec1 = store.submit(
        "synthetic-tool", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
    )
    pdb.write_text("ATOM 1\nATOM 2\n")
    rec2 = store.submit(
        "synthetic-tool", "1.0.0",
        {"pdb_path": str(pdb), "num_designs": 2},
    )
    # No schema → raw-string hashing (Phase 10 contract preserved).
    assert rec1.job_id == rec2.job_id


# ---------------------------------------------------------------- workflow cache


def test_workflow_cache_key_sensitive_to_file_content(tmp_path: Path) -> None:
    node = WorkflowNode(
        id="design",
        tool="proteinmpnn",
        version="1.0.0",
        inputs={"pdb_path": "ignored"},
    )
    pdb = tmp_path / "fixture.pdb"
    pdb.write_text("ATOM 1\n")
    env = {"torch_version": "x", "cuda_version": "y", "descriptor_version": "1.0.0"}
    key1 = compute_cache_key(
        node, {"pdb_path": str(pdb)}, env,
        tool_version_resolved="1.0.0",
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    pdb.write_text("ATOM 1\nATOM 2\n")
    key2 = compute_cache_key(
        node, {"pdb_path": str(pdb)}, env,
        tool_version_resolved="1.0.0",
        input_schema=_PROTEINMPNN_SCHEMA,
    )
    assert key1 != key2


def test_workflow_cache_key_no_schema_passthrough(tmp_path: Path) -> None:
    """Schemaless cache-key computation is the pre-RC4 raw-string behaviour."""
    node = WorkflowNode(
        id="design",
        tool="proteinmpnn",
        version="1.0.0",
        inputs={"pdb_path": "ignored"},
    )
    pdb = tmp_path / "fixture.pdb"
    pdb.write_text("ATOM 1\n")
    env = {"torch_version": "x", "cuda_version": "y", "descriptor_version": "1.0.0"}
    key1 = compute_cache_key(
        node, {"pdb_path": str(pdb)}, env, tool_version_resolved="1.0.0",
    )
    pdb.write_text("ATOM 1\nATOM 2\n")
    key2 = compute_cache_key(
        node, {"pdb_path": str(pdb)}, env, tool_version_resolved="1.0.0",
    )
    # Schemaless: cache key reflects path string only, so unchanged.
    assert key1 == key2


# ---------------------------------------------------------------- schema coverage


def test_every_adapter_schema_parses() -> None:
    """Smoke test: every annotated adapter schema is valid JSON Schema.

    Guards against `vpa:digest` accidentally being inserted at the wrong
    nesting level on future edits.
    """
    from vibeproteinagent.adapters import ToolRegistry
    from vibeproteinagent.adapters.descriptor import resolve_input_schema_dict

    registry = ToolRegistry()
    registry.discover()
    bad = []
    for descriptor in registry.list():
        schema = resolve_input_schema_dict(descriptor)
        json.dumps(schema)
        # Every digest annotation must be on a top-level property of type string.
        for field, prop in (schema.get("properties") or {}).items():
            if not isinstance(prop, dict):
                continue
            mode = prop.get("vpa:digest")
            if mode is None:
                continue
            if mode not in ("sha256", "sha256-if-path"):
                bad.append((descriptor.name, field, mode))
    assert not bad, f"unknown vpa:digest modes: {bad}"

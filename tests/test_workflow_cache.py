from __future__ import annotations

from vibeproteinagent.workflows import compute_cache_key, env_fingerprint
from vibeproteinagent.workflows.descriptor import WorkflowNode


def _node() -> WorkflowNode:
    return WorkflowNode(id="a", tool="noop", version="*", inputs={})


def test_cache_key_stable_under_dict_order() -> None:
    n = _node()
    env = env_fingerprint()
    k1 = compute_cache_key(n, {"a": 1, "b": 2}, env)
    k2 = compute_cache_key(n, {"b": 2, "a": 1}, env)
    assert k1 == k2


def test_cache_key_changes_with_tool_version() -> None:
    n = _node()
    env = env_fingerprint()
    k1 = compute_cache_key(n, {"x": 1}, env, tool_version_resolved="1.0.0")
    k2 = compute_cache_key(n, {"x": 1}, env, tool_version_resolved="1.0.1")
    assert k1 != k2


def test_cache_key_changes_with_env() -> None:
    n = _node()
    e1 = {"torch_version": "2.4.0", "cuda_version": "12.2", "descriptor_version": "1"}
    e2 = {"torch_version": "2.5.0", "cuda_version": "12.2", "descriptor_version": "1"}
    assert compute_cache_key(n, {}, e1) != compute_cache_key(n, {}, e2)


def test_cache_key_changes_with_inputs() -> None:
    n = _node()
    env = env_fingerprint()
    k1 = compute_cache_key(n, {"x": 1}, env)
    k2 = compute_cache_key(n, {"x": 2}, env)
    assert k1 != k2


def test_env_fingerprint_has_expected_keys() -> None:
    e = env_fingerprint()
    assert set(e) == {"torch_version", "cuda_version", "descriptor_version"}

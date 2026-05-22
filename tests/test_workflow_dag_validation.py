"""Codex round-3 RC2: workflow loader rejects missing upstreams + cycles.

Before this fix, ``load_workflow`` only checked duplicate node IDs.
Missing upstreams and cycles were accepted silently and surfaced as a
runtime timeout (or as a generic "node never became ready" stall in the
executor). The right behaviour is to fail fast at submit time with
``E_INVALID_PARAMS`` and a structured ``details`` payload naming what's
wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.descriptor import Workflow, WorkflowNode
from vibeproteinagent.workflows.loader import load_workflow

# ---------------------------------------------------------------- missing upstream


def test_missing_upstream_raises_invalid_params(tmp_path: Path) -> None:
    """A node that references an undeclared upstream must fail at load."""
    yaml = (
        "name: missing_upstream\n"
        'version: "1.0.0"\n'
        "nodes:\n"
        "  - id: a\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "x"\n'
        "  - id: b\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "${outputs.nonexistent.x}"\n'
    )
    p = tmp_path / "wf.yaml"
    p.write_text(yaml)
    with pytest.raises(InvalidParamsError) as exc_info:
        load_workflow(p)
    err = exc_info.value
    assert err.code == "E_INVALID_PARAMS"
    assert err.details["missing"] == ["nonexistent"]


# ---------------------------------------------------------------- cycle detection


def test_three_node_cycle_raises_invalid_params(tmp_path: Path) -> None:
    """A → B → C → A (via ${outputs} refs) — must be rejected at load time."""
    yaml = (
        "name: cycle_three\n"
        'version: "1.0.0"\n'
        "nodes:\n"
        "  - id: a\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "${outputs.c.payload}"\n'
        "  - id: b\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "${outputs.a.payload}"\n'
        "  - id: c\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "${outputs.b.payload}"\n'
    )
    p = tmp_path / "wf.yaml"
    p.write_text(yaml)
    with pytest.raises(InvalidParamsError) as exc_info:
        load_workflow(p)
    err = exc_info.value
    assert err.code == "E_INVALID_PARAMS"
    cycle = err.details["cycle"]
    assert isinstance(cycle, list) and len(cycle) >= 2
    # The cycle list closes on itself (first == last) and references
    # only declared nodes.
    assert cycle[0] == cycle[-1]
    assert set(cycle).issubset({"a", "b", "c"})


def test_self_reference_is_a_cycle(tmp_path: Path) -> None:
    yaml = (
        "name: self_loop\n"
        'version: "1.0.0"\n'
        "nodes:\n"
        "  - id: a\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "${outputs.a.payload}"\n'
    )
    p = tmp_path / "wf.yaml"
    p.write_text(yaml)
    with pytest.raises(InvalidParamsError) as exc_info:
        load_workflow(p)
    err = exc_info.value
    assert err.code == "E_INVALID_PARAMS"


# ---------------------------------------------------------------- happy paths


def test_valid_chain_passes(tmp_path: Path) -> None:
    yaml = (
        "name: valid_chain\n"
        'version: "1.0.0"\n'
        "nodes:\n"
        "  - id: a\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "hi"\n'
        "  - id: b\n"
        "    tool: noop\n"
        "    inputs:\n"
        '      payload: "${outputs.a.payload}"\n'
    )
    p = tmp_path / "wf.yaml"
    p.write_text(yaml)
    wf = load_workflow(p)
    assert [n.id for n in wf.nodes] == ["a", "b"]


def test_foreach_reference_must_resolve(tmp_path: Path) -> None:
    """A ``foreach: ${outputs.X.field}`` whose X is undeclared must fail."""
    yaml = (
        "name: foreach_missing\n"
        'version: "1.0.0"\n'
        "nodes:\n"
        "  - id: fan\n"
        "    tool: noop\n"
        '    foreach: "${outputs.unknown.items}"\n'
        "    inputs:\n"
        '      payload: "${item}"\n'
    )
    p = tmp_path / "wf.yaml"
    p.write_text(yaml)
    with pytest.raises(InvalidParamsError) as exc_info:
        load_workflow(p)
    err = exc_info.value
    assert "unknown" in err.details["missing"]


# ---------------------------------------------------------------- python builder


def test_python_builder_validate_dag_directly() -> None:
    """The Python-builder API exposes :meth:`Workflow.validate_dag` too."""
    wf = Workflow(
        name="cycle_py", version="1.0.0",
        nodes=[
            WorkflowNode(id="a", tool="noop",
                         inputs={"payload": "${outputs.b.payload}"}),
            WorkflowNode(id="b", tool="noop",
                         inputs={"payload": "${outputs.a.payload}"}),
        ],
    )
    with pytest.raises(InvalidParamsError):
        wf.validate_dag()


def test_python_builder_valid_dag_passes() -> None:
    wf = Workflow(
        name="good", version="1.0.0",
        nodes=[
            WorkflowNode(id="a", tool="noop", inputs={"payload": "x"}),
            WorkflowNode(id="b", tool="noop",
                         inputs={"payload": "${outputs.a.payload}"}),
        ],
    )
    wf.validate_dag()  # no raise

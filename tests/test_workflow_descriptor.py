from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows import (
    Workflow,
    WorkflowNode,
    load_workflow,
    serialize_workflow,
)


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "wf.yaml"
    p.write_text(body)
    return p


def test_load_valid_yaml(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
name: tiny
version: "1.0.0"
params:
  msg: {type: str, default: hi}
nodes:
  - id: a
    tool: noop
    inputs:
      payload: ${params.msg}
""",
    )
    wf = load_workflow(p)
    assert wf.name == "tiny"
    assert wf.version == "1.0.0"
    assert len(wf.nodes) == 1
    assert wf.nodes[0].tool == "noop"
    assert wf.params["msg"].default == "hi"


def test_load_invalid_yaml_missing_required(tmp_path: Path) -> None:
    p = _write(tmp_path, "name: x\nversion: '1'\nnodes:\n  - tool: noop")
    with pytest.raises(InvalidParamsError):
        load_workflow(p)


def test_load_invalid_yaml_duplicate_node(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
name: dup
version: "1"
nodes:
  - id: a
    tool: noop
  - id: a
    tool: noop
""",
    )
    with pytest.raises(InvalidParamsError):
        load_workflow(p)


def test_round_trip_python_builder_to_yaml(tmp_path: Path) -> None:
    wf = Workflow(
        name="rt",
        version="0.1.0",
        user_query="Design fixed-backbone sequences for chain A.",
        objective="Find low-risk sequences that satisfy the recorded thresholds.",
    )
    wf.node("a", tool="noop", inputs={"payload": "${params.msg}"})
    wf.node("b", tool="noop", inputs={"payload": "${outputs.a.payload}-b"})

    out = tmp_path / "rt.yaml"
    wf.save(out)
    loaded = load_workflow(out)

    assert loaded.name == wf.name
    assert loaded.version == wf.version
    assert loaded.user_query == wf.user_query
    assert loaded.objective == wf.objective
    assert [n.id for n in loaded.nodes] == ["a", "b"]
    assert loaded.nodes[1].inputs == {"payload": "${outputs.a.payload}-b"}


def test_serialize_round_trip_with_foreach() -> None:
    wf = Workflow(name="fan", version="1.0.0")
    wf.fanout("each", tool="noop", over="${params.items}", inputs={"payload": "${item}"})
    yaml = serialize_workflow(wf)
    assert "foreach" in yaml
    assert "each" in yaml


def test_serialize_round_trip_with_foreach_max_parallel(tmp_path: Path) -> None:
    wf = Workflow(name="fan", version="1.0.0")
    wf.fanout(
        "each",
        tool="noop",
        over="${params.items}",
        inputs={"payload": "${item}"},
        max_parallel=2,
    )

    yaml = serialize_workflow(wf)
    assert "max_parallel: 2" in yaml

    p = tmp_path / "fan.yaml"
    p.write_text(yaml)
    loaded = load_workflow(p)
    assert loaded.nodes[0].max_parallel == 2


def test_node_id_cannot_contain_brackets() -> None:
    with pytest.raises(ValidationError):
        WorkflowNode(id="x[0]", tool="noop")


def test_task_specific_bundled_examples_are_not_public_surface() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "vibeproteinagent" / "workflows" / "examples"
    for name in (
        "rfdiffusion_backbone_design_round.yaml",
        "mutation_repair_round.yaml",
        "semantic_constraint_to_family_sequences.yaml",
        "target_sequence_binder_design_round.yaml",
        "template_guided_design_prep.yaml",
        "mpnn_then_refold.yaml",
    ):
        assert not (root / name).exists()


def test_user_provided_workflow_yaml_still_loads_and_validates(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
name: user_qc
version: "1.0.0"
params:
  sequence: {type: str, required: true}
nodes:
  - id: qc
    tool: protein_qc
    inputs:
      sequence: ${params.sequence}
""",
    )
    registry = ToolRegistry()
    registry.discover()
    tools = {descriptor.name for descriptor in registry.list()}

    wf = load_workflow(p)
    wf.validate_dag()
    assert {node.tool for node in wf.nodes} <= tools
    assert wf.nodes[0].tool == "protein_qc"

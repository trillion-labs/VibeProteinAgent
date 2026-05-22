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
    wf = Workflow(name="rt", version="0.1.0")
    wf.node("a", tool="noop", inputs={"payload": "${params.msg}"})
    wf.node("b", tool="noop", inputs={"payload": "${outputs.a.payload}-b"})

    out = tmp_path / "rt.yaml"
    wf.save(out)
    loaded = load_workflow(out)

    assert loaded.name == wf.name
    assert loaded.version == wf.version
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


def test_iterative_protein_examples_are_loadable_and_registered() -> None:
    root = Path(__file__).resolve().parents[1] / "workflows" / "examples"
    registry = ToolRegistry()
    registry.discover()
    tools = {descriptor.name for descriptor in registry.list()}
    for name in (
        "rfdiffusion_backbone_design_round.yaml",
        "mutation_repair_round.yaml",
        "semantic_constraint_to_family_sequences.yaml",
        "template_guided_design_prep.yaml",
    ):
        wf = load_workflow(root / name)
        wf.validate_dag()
        assert {node.tool for node in wf.nodes} <= tools

    rfdiffusion = load_workflow(root / "rfdiffusion_backbone_design_round.yaml")
    assert [node.tool for node in rfdiffusion.nodes] == [
        "rfdiffusion",
        "proteinmpnn",
        "mpnn_to_protenix_input",
        "protenix_v2",
        "structure_analysis",
        "protein_qc",
        "rank_designs",
    ]
    assert "rfdiffusion_num_designs" not in rfdiffusion.params
    assert rfdiffusion.nodes[1].inputs["pdb_path"] == "${outputs.backbone.designs[0].pdb_path}"
    repair = load_workflow(root / "mutation_repair_round.yaml")
    assert [node.tool for node in repair.nodes] == [
        "structure_analysis",
        "sequence_edit",
        "sequence_to_protenix_input",
        "protenix_v2",
        "structure_analysis",
        "rank_designs",
    ]

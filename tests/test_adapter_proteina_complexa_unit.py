from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters import ToolRegistry, load_descriptor
from vibeproteinagent.adapters.proteina_complexa import (
    ProteinaComplexaDesign,
    ProteinaComplexaInput,
    ProteinaComplexaOutput,
)
from vibeproteinagent.adapters import _proteina_complexa_common as common
from vibeproteinagent.adapters.proteina_complexa import adapter as adapter_mod
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.mcp.util import resolve_input_schema, validate_against

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vibeproteinagent"
    / "adapters"
    / "proteina_complexa"
)
_DESCRIPTOR = _ADAPTER_DIR / "descriptor.yaml"
_INPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "proteina_complexa.input.json"
_OUTPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "proteina_complexa.output.json"


def test_descriptor_is_single_product_tool() -> None:
    descriptor = load_descriptor(_DESCRIPTOR)
    assert descriptor.name == "proteina_complexa"
    assert descriptor.category.value == "protein_design"
    assert set(descriptor.supported_tasks) == {
        "binder_design",
        "ligand_binder_design",
        "motif_scaffolding",
    }


def test_registry_exposes_only_unified_tool_name() -> None:
    registry = ToolRegistry()
    registry.discover(refresh=True)
    names = [descriptor.name for descriptor in registry.list()]
    assert [name for name in names if "proteina" in name] == ["proteina_complexa"]


def test_input_schema_mode_contract() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    assert raw["properties"]["mode"]["enum"] == [
        "protein_binder",
        "ligand_binder",
        "motif",
    ]
    assert "steps" not in raw["properties"]
    descriptor = load_descriptor(_DESCRIPTOR)
    schema = resolve_input_schema(descriptor)
    validate_against(schema, {"mode": "protein_binder", "target_pdb": "/tmp/t.pdb"})
    validate_against(
        schema,
        {
            "mode": "ligand_binder",
            "target_pdb": "/tmp/l.pdb",
            "target_pocket_residues": [12],
            "ligand_smiles": "CCO",
        },
    )
    validate_against(
        schema,
        {
            "mode": "motif",
            "motif_pdb": "/tmp/m.pdb",
            "motif_residues": [1, 2],
            "total_length": 80,
        },
    )
    with pytest.raises(InvalidParamsError):
        validate_against(schema, {"mode": "motif", "motif_residues": [], "total_length": 80})


def test_output_schema_and_pydantic_round_trip() -> None:
    raw = json.loads(_OUTPUT_SCHEMA.read_text())
    assert raw["required"] == [
        "mode",
        "designs",
        "runtime_ms",
        "num_designs_returned",
        "seeds_used",
    ]
    output = ProteinaComplexaOutput(
        mode="motif",
        designs=[ProteinaComplexaDesign(pdb_path="/tmp/design.pdb", length=80)],
        runtime_ms=10,
        num_designs_returned=1,
        seeds_used=[101],
    )
    assert ProteinaComplexaOutput.model_validate(output.model_dump()).mode == "motif"


def test_pydantic_requires_mode_specific_inputs() -> None:
    with pytest.raises(ValidationError):
        ProteinaComplexaInput(mode="protein_binder")
    with pytest.raises(ValidationError):
        ProteinaComplexaInput(mode="ligand_binder", target_pdb="/tmp/l.pdb")
    with pytest.raises(ValidationError):
        ProteinaComplexaInput(mode="motif", motif_pdb="/tmp/m.pdb", total_length=80)


def test_mode_specific_overrides_keep_expected_upstream_configs(tmp_path: Path) -> None:
    target_pdb = tmp_path / "target.pdb"
    target_pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00 10.00           C\n"
        "END\n"
    )
    ligand_pdb = tmp_path / "ligand.pdb"
    ligand_pdb.write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00 10.00           C\n"
        "END\n"
    )
    motif_pdb = tmp_path / "motif.pdb"
    motif_pdb.write_text(
        "ATOM      1  CA  ALA A   5       0.000   0.000   0.000  1.00 10.00           C\n"
        "END\n"
    )
    protein = ProteinaComplexaInput(
        mode="protein_binder",
        target_pdb=str(target_pdb),
        target_chain="B",
        binder_length=64,
        num_designs=2,
        seed=7,
        search_algorithm="single-pass",
        reward_model="none",
    )
    ligand = ProteinaComplexaInput(
        mode="ligand_binder",
        target_pdb=str(ligand_pdb),
        target_pocket_residues=[1, 2],
        ligand_smiles="CC",
    )
    motif = ProteinaComplexaInput(
        mode="motif",
        motif_pdb=str(motif_pdb),
        motif_residues=[5, 6],
        total_length=90,
    )
    protein_overrides = adapter_mod._build_overrides(
        params=protein,
        run_name="run",
        prepared_input=target_pdb,
    )
    assert any("target_input=" in item for item in protein_overrides)
    assert "++generation.search.algorithm=single-pass" in protein_overrides
    assert "++generation.reward_model=null" in protein_overrides
    assert any("SMILES='CC'" in item for item in adapter_mod._build_overrides(
        params=ligand,
        run_name="run",
        prepared_input=ligand_pdb,
    ))
    assert any("contig_string=" in item for item in adapter_mod._build_overrides(
        params=motif,
        run_name="run",
        prepared_input=motif_pdb,
    ))


def test_protein_chain_input_spec_preserves_discontinuous_residue_ranges(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "discontinuous.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   2       0.000   0.000   0.000  1.00 10.00           C\n"
        "ATOM      2  CA  ALA A   3       0.000   0.000   0.000  1.00 10.00           C\n"
        "ATOM      3  CA  ALA A   8       0.000   0.000   0.000  1.00 10.00           C\n"
        "ATOM      4  CA  ALA A  10       0.000   0.000   0.000  1.00 10.00           C\n"
        "END\n"
    )
    assert common.protein_chain_input_spec(pdb, "A") == "A2-3/A8/A10"


def test_proteina_complexa_steps_are_fixed_to_generate_evaluate() -> None:
    protein = ProteinaComplexaInput(mode="protein_binder", target_pdb="/tmp/t.pdb")
    motif = ProteinaComplexaInput(
        mode="motif",
        motif_pdb="/tmp/m.pdb",
        motif_residues=[1],
        total_length=80,
    )
    assert adapter_mod._resolve_steps(protein) == ["generate", "evaluate"]
    assert adapter_mod._resolve_steps(motif) == ["generate", "evaluate"]


def test_build_design_cmd_separates_steps_from_hydra_overrides(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    venv = tmp_path / "venv"
    cmd = common.build_design_cmd(
        config_path=config,
        overrides=[
            "++run_name=coverage_proteina_complexa_seed301",
            "++generation.reward_model=null",
        ],
        steps=["generate", "evaluate"],
        venv=venv,
    )

    assert cmd == [
        str(venv / "bin" / "complexa"),
        "design",
        "--steps",
        "generate",
        "evaluate",
        "--",
        str(config),
        "++run_name=coverage_proteina_complexa_seed301",
        "++generation.reward_model=null",
    ]

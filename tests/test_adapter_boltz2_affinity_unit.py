from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vibeproteinagent.adapters.boltz2_affinity.adapter import (
    Boltz2AffinityInput,
    _collect_outputs,
    _stage_input,
)
from vibeproteinagent.adapters.msa_policy import (
    should_use_remote_msa,
    yaml_file_has_prepared_msa,
)

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vibeproteinagent"
    / "adapters"
    / "boltz2_affinity"
)
_INPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "boltz2_affinity.input.json"
_OUTPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "boltz2_affinity.output.json"


def test_boltz2_affinity_schema_declares_msa_policy_defaults() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    props = raw["properties"]
    assert props["use_msa"]["default"] is False
    assert props["msa_backend"]["default"] == "auto"
    assert sorted(props["msa_backend"]["enum"]) == ["auto", "none", "remote"]


def test_boltz2_affinity_output_schema_exposes_msa_artifacts() -> None:
    raw = json.loads(_OUTPUT_SCHEMA.read_text())
    props = raw["properties"]
    assert props["msa_paths"]["type"] == "array"
    assert props["msa_paths"]["default"] == []
    assert props["msa_index_csv"]["type"] == ["string", "null"]


def test_boltz2_affinity_msa_defaults_and_validation() -> None:
    p = Boltz2AffinityInput(
        sequences=[
            {"type": "protein", "id": "A", "sequence": "ACDE"},
            {"type": "ligand", "id": "L", "smiles": "C"},
        ],
        binder_chain="L",
    )
    assert p.use_msa is False
    assert p.msa_backend == "auto"
    with pytest.raises(ValidationError):
        Boltz2AffinityInput(
            sequences=[{"type": "protein", "id": "A", "sequence": "ACDE"}],
            binder_chain="A",
            msa_backend="local",
        )


def test_boltz_inline_msa_suppresses_remote_server_in_auto(tmp_path) -> None:
    params = Boltz2AffinityInput(
        sequences=[
            {"type": "protein", "id": "A", "sequence": "ACDE", "msa": "A.a3m"},
            {"type": "ligand", "id": "L", "smiles": "C"},
        ],
        binder_chain="L",
        use_msa=True,
        msa_backend="auto",
    )
    staged = _stage_input(params, tmp_path)
    assert yaml_file_has_prepared_msa(staged)
    assert should_use_remote_msa(
        use_msa=params.use_msa,
        msa_backend=params.msa_backend,
        has_prepared_msa=yaml_file_has_prepared_msa(staged),
    ) is False


def test_boltz_inline_msa_is_stripped_when_disabled_or_remote(tmp_path) -> None:
    base_sequences = [
        {"type": "protein", "id": "A", "sequence": "ACDE", "msa": "A.a3m"},
        {"type": "ligand", "id": "L", "smiles": "C"},
    ]
    disabled_dir = tmp_path / "disabled"
    disabled_dir.mkdir()
    disabled = _stage_input(
        Boltz2AffinityInput(
            sequences=base_sequences,
            binder_chain="L",
            use_msa=False,
        ),
        disabled_dir,
    )
    assert not yaml_file_has_prepared_msa(disabled)

    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    remote = _stage_input(
        Boltz2AffinityInput(
            sequences=base_sequences,
            binder_chain="L",
            use_msa=True,
            msa_backend="remote",
        ),
        remote_dir,
    )
    assert not yaml_file_has_prepared_msa(remote)


def test_boltz_yaml_msa_empty_is_stripped_for_remote_backend(tmp_path) -> None:
    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "ACDE", "msa": "empty"}},
                    {"ligand": {"id": "L", "smiles": "C"}},
                ],
                "properties": [{"affinity": {"binder": "L"}}],
            },
            sort_keys=False,
        )
    )
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    staged = _stage_input(
        Boltz2AffinityInput(
            input_yaml=str(input_yaml),
            use_msa=True,
            msa_backend="remote",
        ),
        staged_dir,
    )
    raw = yaml.safe_load(staged.read_text())
    protein = raw["sequences"][0]["protein"]
    assert "msa" not in protein
    assert not yaml_file_has_prepared_msa(staged)


def test_boltz_yaml_without_msa_falls_back_to_remote_in_auto(tmp_path) -> None:
    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "ACDE"}},
                    {"ligand": {"id": "L", "smiles": "C"}},
                ],
                "properties": [{"affinity": {"binder": "L"}}],
            },
            sort_keys=False,
        )
    )
    params = Boltz2AffinityInput(
        input_yaml=str(input_yaml),
        use_msa=True,
        msa_backend="auto",
    )
    staged_dir = tmp_path / "staged"
    staged_dir.mkdir()
    staged = _stage_input(params, staged_dir)
    assert not yaml_file_has_prepared_msa(staged)
    assert should_use_remote_msa(
        use_msa=params.use_msa,
        msa_backend=params.msa_backend,
        has_prepared_msa=yaml_file_has_prepared_msa(staged),
    ) is True


def test_collect_outputs_surfaces_remote_msa_files(tmp_path) -> None:
    record = "input"
    pred_dir = tmp_path / f"boltz_results_{record}" / "predictions" / record
    pred_dir.mkdir(parents=True)
    cif = pred_dir / f"{record}_model_0.cif"
    cif.write_text("data_input\n_atom_site.\nATOM\n")
    (pred_dir / f"affinity_{record}.json").write_text(
        json.dumps(
            {
                "affinity_pred_value": 1.0,
                "affinity_probability_binary": 0.25,
            }
        )
    )
    (pred_dir / f"confidence_{record}_model_0.json").write_text(
        json.dumps({"confidence_score": 0.75})
    )
    msa_dir = tmp_path / f"boltz_results_{record}" / "msa" / "input_unpaired_tmp_env"
    msa_dir.mkdir(parents=True)
    a3m = msa_dir / "uniref.a3m"
    a3m.write_text(">query\nACDE\n>hit\nAC-E\n")
    index_csv = tmp_path / f"boltz_results_{record}" / "msa" / f"{record}_0.csv"
    index_csv.write_text("entity_id,path\nA,uniref.a3m\n")

    out = _collect_outputs(tmp_path, record)

    assert out["predicted_cif"] == cif
    assert out["msa_paths"] == [str(a3m.resolve())]
    assert out["msa_index_csv"] == str(index_csv)


def test_rdkit_descriptors_compat_exposes_boltz_expected_alias() -> None:
    from rdkit.Chem import AllChem, Descriptors

    from vibeproteinagent.adapters.boltz2_affinity.adapter import (
        _ensure_rdkit_descriptors_compat,
    )

    _ensure_rdkit_descriptors_compat()

    assert AllChem.Descriptors is Descriptors

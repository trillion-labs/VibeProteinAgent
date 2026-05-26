"""Fast unit tests for the Protenix-v2 adapter shell.

These tests do NOT load the Protenix model — they only validate the descriptor,
the pydantic I/O shapes, the env-var resolution, and the package-level export
surface. Heavy GPU work lives in ``test_adapter_protenix_v2_real.py`` /
``..._e2e.py`` (both marked ``slow``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vibeproteinagent.adapters import load_descriptor
from vibeproteinagent.adapters.protenix_v2 import (
    PROTENIX_CKPT_DIR,
    PROTENIX_TREE,
    ProtenixV2Adapter,
    ProtenixV2Input,
    ProtenixV2Metrics,
    ProtenixV2Output,
    ProtenixV2PredictionRecord,
)
from vibeproteinagent.adapters.protenix_v2 import adapter as adapter_mod
from vibeproteinagent.core.errors import VPAError

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vibeproteinagent"
    / "adapters"
    / "protenix_v2"
)
_DESCRIPTOR = _ADAPTER_DIR / "descriptor.yaml"
_INPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "protenix_v2.input.json"
_OUTPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "protenix_v2.output.json"
_FIXTURE = Path(__file__).resolve().parents[0] / "fixtures" / "protenix_v2_input_tiny.json"


def test_descriptor_loads_and_matches_spec() -> None:
    d = load_descriptor(_DESCRIPTOR)
    assert d.name == "protenix_v2"
    assert d.version == "1.0.0"
    assert d.category.value == "structure_prediction"
    assert d.runtime.value == "python"
    assert d.backend.value == "pytorch"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.vram_min_gib == 16
    assert d.startup_cost.value == "high"
    assert d.warm_load_capable is True
    assert d.pin_warm is True
    assert d.supported_tasks == ["structure_prediction"]
    assert [p.value for p in d.supported_precisions] == ["bf16"]
    assert d.supports_batching is False
    assert d.entry_point == (
        "vibeproteinagent.adapters.protenix_v2.adapter:ProtenixV2Adapter"
    )


def test_descriptor_input_schema_is_valid_json() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    assert raw["$id"].endswith("protenix_v2.input.json")
    assert {"required": ["input_json"]} in raw["anyOf"]
    assert {"required": ["candidates"]} in raw["anyOf"]
    props = raw["properties"]
    assert "seeds" in props and props["seeds"]["default"] == [101]
    assert "candidates" in props
    assert props["use_msa"]["default"] is False
    assert props["msa_backend"]["default"] == "auto"
    assert sorted(props["msa_backend"]["enum"]) == ["auto", "none", "remote"]
    assert props["use_template"]["default"] is False
    assert "additionalProperties" in raw and raw["additionalProperties"] is False


def test_descriptor_output_schema_is_valid_json() -> None:
    raw = json.loads(_OUTPUT_SCHEMA.read_text())
    assert raw["$id"].endswith("protenix_v2.output.json")
    for required in ("predicted_cif", "metrics", "runtime_ms", "seeds_used"):
        assert required in raw["required"]
    props = raw["properties"]
    assert props["pae_path"]["type"] == ["string", "null"]
    prediction_props = props["all_predictions"]["items"]["properties"]
    assert prediction_props["pae_path"]["type"] == ["string", "null"]
    assert prediction_props["full_confidence_json"]["type"] == ["string", "null"]
    metrics_required = raw["properties"]["metrics"]["required"]
    for k in ("plddt", "iptm", "ptm", "ranking_score"):
        assert k in metrics_required


def test_descriptor_yaml_round_trips() -> None:
    yml = yaml.safe_load(_DESCRIPTOR.read_text())
    # entry_point colon-form is the contract the registry consumes.
    assert ":" in yml["entry_point"]


def test_input_pydantic_defaults() -> None:
    p = ProtenixV2Input(input_json="/tmp/x.json")
    assert p.seeds == [101]
    assert p.use_msa is False
    assert p.msa_backend == "auto"
    assert p.use_template is False
    assert p.use_rna_msa is False
    assert p.enable_cache is True
    assert p.enable_tf32 is True
    assert p.n_sample == 5
    assert p.n_step == 200
    assert p.n_cycle == 10


def test_input_pydantic_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        ProtenixV2Input(input_json="/tmp/x.json", n_sample=0)
    with pytest.raises(ValidationError):
        ProtenixV2Input(input_json="/tmp/x.json", n_step=5)
    with pytest.raises(ValidationError):
        ProtenixV2Input(input_json="/tmp/x.json", seeds=[])
    with pytest.raises(ValidationError):
        ProtenixV2Input(input_json="/tmp/x.json", msa_backend="local")


def test_output_pydantic_round_trip() -> None:
    m = ProtenixV2Metrics(plddt=85.2, iptm=0.62, ptm=0.71, ranking_score=0.64)
    rec = ProtenixV2PredictionRecord(
        seed=101,
        rank=0,
        cif_path="/tmp/a.cif",
        summary_json="/tmp/a.json",
        pae_path="/tmp/a_confidences_sample_0.json",
        full_confidence_json="/tmp/a_full_data_sample_0.json",
        ranking_score=0.64,
        plddt=85.2,
        iptm=0.62,
        ptm=0.71,
    )
    out = ProtenixV2Output(
        predicted_cif="/tmp/a.cif",
        pae_path="/tmp/a_confidences_sample_0.json",
        metrics=m,
        all_predictions=[rec],
        runtime_ms=12345,
        seeds_used=[101],
    )
    serialized = out.model_dump()
    again = ProtenixV2Output.model_validate(serialized)
    assert again.metrics.plddt == pytest.approx(85.2)
    assert again.runtime_ms == 12345
    assert again.predicted_pdb is None
    assert again.pae_path == "/tmp/a_confidences_sample_0.json"
    assert again.all_predictions[0].full_confidence_json == "/tmp/a_full_data_sample_0.json"


def test_env_var_overrides_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """``VPA_PROTENIX_TREE`` / ``VPA_PROTENIX_CKPT`` override the defaults.

    We re-import the module under the overridden env vars and verify the
    module-level constants reflect them.
    """
    monkeypatch.setenv("VPA_PROTENIX_TREE", "/tmp/__vpa_tree_test")
    monkeypatch.setenv("VPA_PROTENIX_CKPT", "/tmp/__vpa_ckpt_test")
    import importlib

    reloaded = importlib.reload(adapter_mod)
    try:
        assert str(reloaded.PROTENIX_TREE) == "/tmp/__vpa_tree_test"
        assert str(reloaded.PROTENIX_CKPT_DIR) == "/tmp/__vpa_ckpt_test"
    finally:
        # Restore the module to its default-path state for any downstream test.
        monkeypatch.delenv("VPA_PROTENIX_TREE")
        monkeypatch.delenv("VPA_PROTENIX_CKPT")
        importlib.reload(adapter_mod)


def test_resolve_checkpoint_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_mod, "PROTENIX_CKPT_DIR", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        adapter_mod._resolve_checkpoint()


def test_ensure_protenix_importable_raises_for_bad_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_mod, "PROTENIX_TREE", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="source tree not found"):
        adapter_mod._ensure_protenix_importable()


def test_ensure_protenix_importable_rejects_non_protenix_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "fake_protenix"
    fake.mkdir()
    (fake / "README.md").write_text("nothing useful")
    monkeypatch.setattr(adapter_mod, "PROTENIX_TREE", fake)
    with pytest.raises(FileNotFoundError, match="does not look like a Protenix"):
        adapter_mod._ensure_protenix_importable()


def test_registry_finds_protenix_v2() -> None:
    from vibeproteinagent.adapters import ToolAdapter, ToolRegistry

    r = ToolRegistry()
    r.discover()
    names = {d.name for d in r.list()}
    assert "protenix_v2" in names
    desc, cls = r.get("protenix_v2")
    assert desc.name == "protenix_v2"
    # Compare by name + base class instead of identity: the env-var override
    # test reloads the adapter module, so `cls` may be a freshly-decorated
    # twin of the original class.
    assert cls.__name__ == "ProtenixV2Adapter"
    assert issubclass(cls, ToolAdapter)


def test_collect_outputs_picks_highest_ranking_score(tmp_path: Path) -> None:
    """_collect_outputs walks the DataDumper layout and picks the best rank."""
    sample_name = "tiny_monomer_44aa"
    seed_dir = tmp_path / sample_name / "seed_101" / "predictions"
    seed_dir.mkdir(parents=True)
    # Two ranks; rank 0 has the best score.
    for rank, ranking_score, plddt in [(0, 0.83, 92.1), (1, 0.55, 80.5)]:
        cif = seed_dir / f"{sample_name}_seed_101_sample_{rank}.cif"
        cif.write_text(f"# fake cif sample {rank}\n")
        summary = (
            seed_dir / f"{sample_name}_seed_101_summary_confidence_sample_{rank}.json"
        )
        summary.write_text(
            json.dumps(
                {
                    "ranking_score": ranking_score,
                    "plddt": plddt,
                    "iptm": 0.5 + rank * 0.01,
                    "ptm": 0.6 + rank * 0.01,
                }
            )
        )
        full_data = seed_dir / f"{sample_name}_full_data_sample_{rank}.json"
        full_data.write_text(
            json.dumps(
                {
                    "token_pair_pae": [[0.0, float(rank + 1)], [float(rank + 2), 0.0]],
                    "atom_plddt": [0.91, 0.82],
                }
            )
        )

    records, best = ProtenixV2Adapter._collect_outputs(tmp_path, [101])
    assert len(records) == 2
    assert best is not None
    assert best.rank == 0
    assert best.ranking_score == pytest.approx(0.83)
    assert best.plddt == pytest.approx(92.1)
    assert best.pae_path is not None
    pae_payload = json.loads(Path(best.pae_path).read_text())
    assert pae_payload["pae"] == [[0.0, 1.0], [2.0, 0.0]]
    assert pae_payload["atom_plddts"] == [91.0, 82.0]
    assert best.full_confidence_json is not None
    assert Path(best.full_confidence_json).name == (
        f"{sample_name}_full_data_sample_0.json"
    )
    paired_summary = seed_dir / f"{sample_name}_summary_confidences_sample_0.json"
    assert paired_summary.is_file()


def test_collect_outputs_empty_returns_none(tmp_path: Path) -> None:
    records, best = ProtenixV2Adapter._collect_outputs(tmp_path, [101])
    assert records == []
    assert best is None


def test_tiny_fixture_is_well_formed() -> None:
    raw = json.loads(_FIXTURE.read_text())
    assert isinstance(raw, list)
    assert len(raw) == 1
    record = raw[0]
    assert "sequences" in record
    assert "name" in record
    # Protenix expects a top-level proteinChain wrapper inside sequences.
    chain = record["sequences"][0]["proteinChain"]
    assert chain["sequence"].isalpha()
    assert 30 <= len(chain["sequence"]) <= 150  # tiny monomer for fast inference


def test_module_exports() -> None:
    """Public symbols are importable from the package root."""
    import vibeproteinagent.adapters.protenix_v2 as pkg

    for name in (
        "ProtenixV2Adapter",
        "ProtenixV2Input",
        "ProtenixV2Output",
        "ProtenixV2Metrics",
        "ProtenixV2PredictionRecord",
        "PROTENIX_TREE",
        "PROTENIX_CKPT_DIR",
    ):
        assert hasattr(pkg, name), f"missing export: {name}"


def test_descriptor_default_pin_warm_is_true() -> None:
    """Heavy structure-prediction adapter — pin so eviction doesn't churn weights."""
    d = load_descriptor(_DESCRIPTOR)
    assert d.pin_warm is True


def test_vpaerror_unrelated_to_descriptor() -> None:
    # Sanity: imports we do don't accidentally fail at module level.
    assert VPAError is not None


def test_paths_are_pathlib() -> None:
    assert isinstance(PROTENIX_TREE, Path)
    assert isinstance(PROTENIX_CKPT_DIR, Path)

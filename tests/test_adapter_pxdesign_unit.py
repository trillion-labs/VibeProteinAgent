"""Fast unit tests for the PXDesign adapter shell.

These tests do NOT load the PXDesign model — they only validate descriptor,
pydantic I/O shapes, env-var resolution, package-level exports, and the
``_collect_outputs`` glob path against a synthetic dump layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from vibeproteinagent.adapters import load_descriptor
from vibeproteinagent.adapters.pxdesign import (
    PROTENIX_PXD_VENDOR,
    PXDBENCH_VENDOR,
    PXDESIGN_CKPT_DIR,
    PXDESIGN_VENDOR,
    PXDesignAdapter,
    PXDesignDesign,
    PXDesignInput,
    PXDesignOutput,
)
from vibeproteinagent.adapters.pxdesign import adapter as adapter_mod

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vibeproteinagent"
    / "adapters"
    / "pxdesign"
)
_DESCRIPTOR = _ADAPTER_DIR / "descriptor.yaml"
_INPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "pxdesign.input.json"
_OUTPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "pxdesign.output.json"


def test_descriptor_loads_and_matches_spec() -> None:
    d = load_descriptor(_DESCRIPTOR)
    assert d.name == "pxdesign"
    assert d.version == "1.0.0"
    assert d.category.value == "protein_design"
    assert d.runtime.value == "python"
    assert d.backend.value == "pytorch"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.vram_min_gib == 24
    assert d.startup_cost.value == "high"
    assert d.warm_load_capable is True
    assert d.pin_warm is False
    assert d.supported_tasks == ["binder_design"]
    assert [p.value for p in d.supported_precisions] == ["bf16"]
    assert d.incompatible_with == ["protenix_v2"]
    assert d.support_status.value == "verified"
    assert d.availability_probe == (
        "vibeproteinagent.adapters.pxdesign:availability_skip_reason"
    )
    assert d.support_status_reason is None
    assert d.entry_point == (
        "vibeproteinagent.adapters.pxdesign.adapter:PXDesignAdapter"
    )


def test_multi_gpu_alias_descriptor_declares_distributed_entrypoint() -> None:
    p = _ADAPTER_DIR.parent / "pxdesign_pipeline_multi_gpu" / "descriptor.yaml"
    d = load_descriptor(p)
    assert d.name == "pxdesign_pipeline_multi_gpu"
    assert d.gpus_required == 2
    assert d.warm_load_capable is False
    assert d.distributed_entrypoint == (
        "vibeproteinagent.adapters.pxdesign.adapter:run_distributed"
    )
    assert d.execution_isolation.value == "subprocess"
    assert d.input_schema == "schemas/pxdesign_pipeline_multi_gpu.input.json"


def test_multi_gpu_alias_schema_requires_pipeline_mode() -> None:
    p = (
        _ADAPTER_DIR.parent
        / "pxdesign_pipeline_multi_gpu"
        / "schemas"
        / "pxdesign_pipeline_multi_gpu.input.json"
    )
    raw = json.loads(p.read_text())
    assert "mode" in raw["required"]
    assert raw["properties"]["mode"]["const"] == "pipeline"


def test_run_distributed_threads_rendezvous_env(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "target.pdb"
    target.write_text("HEADER target\n")
    captured: dict[str, object] = {}

    class FakeSidecar:
        def __init__(self, *, python, env, name, startup_timeout_s):
            del python, startup_timeout_s
            captured["env"] = env
            captured["name"] = name

        def call(self, target, kwargs, *, timeout_s):
            captured["target"] = target
            captured["kwargs"] = kwargs
            captured["timeout_s"] = timeout_s
            return {"ok": True}

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: None)
    monkeypatch.setattr(adapter_mod, "pipeline_python", lambda: Path("/bin/python"))
    monkeypatch.setattr(adapter_mod, "PersistentPythonSidecar", FakeSidecar)
    monkeypatch.setattr(
        adapter_mod.PXDesignAdapter,
        "_collect_pipeline_outputs",
        staticmethod(
            lambda *_args: [
                adapter_mod.PXDesignDesign(
                    pdb_path=None,
                    cif_path="/tmp/out.cif",
                    score=None,
                    sequence="A" * 30,
                    binder_length=30,
                )
            ]
        ),
    )

    params = adapter_mod.PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=30,
        mode="pipeline",
        num_samples=2,
        top_k_designs=1,
    )
    ctx = SimpleNamespace(
        job_dir=tmp_path / "job",
        rank=0,
        world_size=2,
        local_world_size=2,
        master_addr="127.0.0.1",
        master_port=31234,
    )

    out = adapter_mod.run_distributed(params, ctx)

    assert isinstance(out, adapter_mod.PXDesignOutput)
    env = captured["env"]
    assert env["MASTER_ADDR"] == "127.0.0.1"
    assert env["MASTER_PORT"] == "31234"
    assert env["RANK"] == "0"
    assert env["WORLD_SIZE"] == "2"
    assert env["LOCAL_RANK"] == "0"
    assert env["LOCAL_WORLD_SIZE"] == "2"
    assert captured["target"].endswith(":run_sidecar")
    assert captured["closed"] is True


def test_descriptor_input_schema_is_valid_json() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    assert raw["$id"].endswith("pxdesign.input.json")
    for required in ("target_pdb", "target_chain", "binder_length"):
        assert required in raw["required"]
    props = raw["properties"]
    assert props["num_samples"]["default"] == 5
    assert props["num_steps"]["default"] == 400
    assert raw["additionalProperties"] is False


def test_descriptor_output_schema_is_valid_json() -> None:
    raw = json.loads(_OUTPUT_SCHEMA.read_text())
    assert raw["$id"].endswith("pxdesign.output.json")
    for required in ("designs", "runtime_ms", "num_samples"):
        assert required in raw["required"]
    design_required = raw["properties"]["designs"]["items"]["required"]
    for k in ("pdb_path", "cif_path", "score", "sequence", "binder_length"):
        assert k in design_required


def test_descriptor_yaml_round_trips() -> None:
    yml = yaml.safe_load(_DESCRIPTOR.read_text())
    assert ":" in yml["entry_point"]
    assert yml["incompatible_with"] == ["protenix_v2"]


def test_input_pydantic_defaults() -> None:
    p = PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50)
    assert p.num_samples == 5
    assert p.num_steps == 400
    assert p.seed is None
    assert p.temperature == pytest.approx(1.0)
    assert p.hotspots == []
    assert p.crop is None
    assert not hasattr(p, "use_msa")


def test_input_pydantic_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=10)
    with pytest.raises(ValidationError):
        PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50,
                      num_samples=0)
    with pytest.raises(ValidationError):
        PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50,
                      num_steps=1)


def test_output_pydantic_round_trip() -> None:
    design = PXDesignDesign(
        pdb_path=None,
        cif_path="/tmp/a.cif",
        score=0.74,
        sequence="GAGAGAGAGA",
        binder_length=10,
    )
    out = PXDesignOutput(
        designs=[design], runtime_ms=4242, num_samples=1, task_name="x", seed=101,
    )
    again = PXDesignOutput.model_validate(out.model_dump())
    assert again.designs[0].sequence == "GAGAGAGAGA"
    assert again.runtime_ms == 4242
    assert again.task_name == "x"


def test_env_var_overrides_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_PXDESIGN_VENDOR", "/tmp/__vpa_pxd_vendor")
    monkeypatch.setenv("VPA_PXDBENCH_VENDOR", "/tmp/__vpa_pxdbench_vendor")
    monkeypatch.setenv("VPA_PROTENIX_PXD_VENDOR", "/tmp/__vpa_protenix_pxd_vendor")
    monkeypatch.setenv("VPA_PXDESIGN_CKPT_DIR", "/tmp/__vpa_pxd_ckpt")
    import importlib

    reloaded = importlib.reload(adapter_mod)
    try:
        assert str(reloaded.PXDESIGN_VENDOR) == "/tmp/__vpa_pxd_vendor"
        assert str(reloaded.PXDBENCH_VENDOR) == "/tmp/__vpa_pxdbench_vendor"
        assert str(reloaded.PROTENIX_PXD_VENDOR) == "/tmp/__vpa_protenix_pxd_vendor"
        assert str(reloaded.PXDESIGN_CKPT_DIR) == "/tmp/__vpa_pxd_ckpt"
    finally:
        monkeypatch.delenv("VPA_PXDESIGN_VENDOR")
        monkeypatch.delenv("VPA_PXDBENCH_VENDOR")
        monkeypatch.delenv("VPA_PROTENIX_PXD_VENDOR")
        monkeypatch.delenv("VPA_PXDESIGN_CKPT_DIR")
        importlib.reload(adapter_mod)


def test_resolve_checkpoint_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_mod, "PXDESIGN_CKPT_DIR", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        adapter_mod._resolve_checkpoint()


def test_registry_finds_pxdesign() -> None:
    from vibeproteinagent.adapters import ToolAdapter, ToolRegistry

    r = ToolRegistry()
    r.discover()
    names = {d.name for d in r.list()}
    assert "pxdesign" in names
    desc, cls = r.get("pxdesign")
    assert desc.name == "pxdesign"
    assert desc.incompatible_with == ["protenix_v2"]
    assert cls.__name__ == "PXDesignAdapter"
    assert issubclass(cls, ToolAdapter)


def test_protenix_v2_reciprocal_incompatibility() -> None:
    """Protenix-v2 declares the same incompatibility back at pxdesign."""
    from vibeproteinagent.adapters import ToolRegistry

    r = ToolRegistry()
    r.discover()
    desc, _ = r.get("protenix_v2")
    assert "pxdesign" in desc.incompatible_with


def test_collect_outputs_finds_designs(tmp_path: Path) -> None:
    """Synthetic PXDesign dump layout — _collect_outputs returns 2 designs."""
    task = "design_5o45_A_80aa"
    seed = 42
    pred_dir = tmp_path / task / f"seed_{seed}" / "predictions"
    pred_dir.mkdir(parents=True)
    # Minimal CIF body with entity_poly_seq rows for entity 2 (binder).
    cif_text = (
        "data_test\n#\n"
        "loop_\n"
        "_entity_poly_seq.entity_id\n"
        "_entity_poly_seq.hetero\n"
        "_entity_poly_seq.mon_id\n"
        "_entity_poly_seq.num\n"
        "1 n MET 1\n"
        "1 n ALA 2\n"
        "2 n GLY 1\n"
        "2 n ALA 2\n"
        "2 n VAL 3\n"
        "#\n"
    )
    for rank in (0, 1):
        (pred_dir / f"{task}_sample_{rank}.cif").write_text(cif_text)
        (pred_dir / f"{task}_summary_confidence_sample_{rank}.json").write_text(
            json.dumps({"ranking_score": 0.7 - 0.1 * rank, "plddt": 80.0})
        )
    (pred_dir.parent / "SUCCESS_FILE").write_text(json.dumps({"prediction": True}))

    designs = PXDesignAdapter._collect_outputs(tmp_path, task, seed, 80)
    assert len(designs) == 2
    assert designs[0].cif_path.endswith("_sample_0.cif")
    # Sequence comes from entity_id == 2 rows: GAV
    assert designs[0].sequence == "GAV"
    assert designs[0].score == pytest.approx(0.7)


def test_collect_outputs_empty_returns_empty(tmp_path: Path) -> None:
    designs = PXDesignAdapter._collect_outputs(tmp_path, "no_task", 1, 50)
    assert designs == []


def test_module_exports() -> None:
    import vibeproteinagent.adapters.pxdesign as pkg

    for name in (
        "PXDesignAdapter",
        "PXDesignInput",
        "PXDesignOutput",
        "PXDesignDesign",
        "PXDESIGN_VENDOR",
        "PXDBENCH_VENDOR",
        "PROTENIX_PXD_VENDOR",
        "PXDESIGN_CKPT_DIR",
        "PXDESIGN_MODEL_NAME",
    ):
        assert hasattr(pkg, name), f"missing export: {name}"


def test_paths_are_pathlib() -> None:
    assert isinstance(PXDESIGN_VENDOR, Path)
    assert isinstance(PXDBENCH_VENDOR, Path)
    assert isinstance(PROTENIX_PXD_VENDOR, Path)
    assert isinstance(PXDESIGN_CKPT_DIR, Path)


def test_vendor_trees_exist_on_disk() -> None:
    """Phase 5.1 vendored these; the unit test checks they remain in place."""
    assert PXDESIGN_VENDOR.is_dir(), f"missing {PXDESIGN_VENDOR}"
    assert PXDBENCH_VENDOR.is_dir(), f"missing {PXDBENCH_VENDOR}"
    assert PROTENIX_PXD_VENDOR.is_dir(), f"missing {PROTENIX_PXD_VENDOR}"


def test_descriptor_default_pin_warm_is_false() -> None:
    """PXDesign opts out of pin_warm: its 24 GiB ceiling would block protenix_v2."""
    d = load_descriptor(_DESCRIPTOR)
    assert d.pin_warm is False


def test_healthcheck_includes_pipeline_availability_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = PXDesignAdapter.__new__(PXDesignAdapter)
    adapter.ckpt_path = tmp_path / "pxdesign_v0.1.0.pt"
    adapter.ckpt_path.write_text("fake")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    monkeypatch.setattr(adapter_mod, "PXDESIGN_VENDOR", vendor)

    monkeypatch.setattr(
        adapter_mod, "availability_skip_reason", lambda: "pipeline env missing",
    )
    assert adapter.healthcheck() is False

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: None)
    assert adapter.healthcheck() is True

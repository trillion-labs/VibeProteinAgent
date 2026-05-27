from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from jsonschema import Draft202012Validator

from vibeproteinagent.adapters.bindcraft import adapter as bindcraft_mod
from vibeproteinagent.adapters.bindcraft.adapter import (
    BindCraftAdapter,
    BindCraftInput,
    NoAcceptedBindCraftDesigns,
    _collect_designs,
)
from vibeproteinagent.adapters.context import null_context


REPO_ROOT = Path(__file__).resolve().parents[1]


def _configure_bindcraft_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    fake_python = tmp_path / "env" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("# fake python\n")
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "params_model_5_ptm.npz").write_text("fake")

    monkeypatch.setattr(bindcraft_mod, "_check_paths", lambda: None)
    monkeypatch.setattr(bindcraft_mod, "BINDCRAFT_PARAMS", params_dir)
    monkeypatch.setattr(bindcraft_mod, "availability_skip_reason", lambda: None)
    monkeypatch.setenv("VPA_BINDCRAFT_ENV_PATH", str(fake_python.parents[1]))
    monkeypatch.setattr(bindcraft_mod, "_find_conda_exe", lambda: None)
    monkeypatch.setattr(bindcraft_mod.torch.cuda, "is_available", lambda: True)
    return fake_python


class _NoAcceptedSidecar:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def call(
        self,
        target: str,
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, bool]:
        del timeout_s
        assert target.endswith(":run_sidecar")
        settings_json = Path((kwargs or {})["settings_json"])
        settings = json.loads(settings_json.read_text())
        output_dir = Path(settings["design_path"])
        relaxed = output_dir / "Trajectory" / "Relaxed"
        clashing = output_dir / "Trajectory" / "Clashing"
        low_confidence = output_dir / "Trajectory" / "LowConfidence"
        relaxed.mkdir(parents=True, exist_ok=True)
        clashing.mkdir(parents=True, exist_ok=True)
        low_confidence.mkdir(parents=True, exist_ok=True)
        (relaxed / "target_l60_s1.pdb").write_text("END\n")
        (clashing / "target_l60_s2.pdb").write_text("END\n")
        (low_confidence / "target_l60_s3.pdb").write_text("END\n")
        (output_dir / "failure_csv.csv").write_text(
            "Trajectory_Clashes,Trajectory_final_pLDDT\n1,2\n"
        )
        return {"ok": True}

    def close(self) -> None:
        self.closed = True


def test_collect_designs_uses_specific_no_accepted_exception(tmp_path: Path) -> None:
    with pytest.raises(NoAcceptedBindCraftDesigns, match="no Accepted/ directory"):
        _collect_designs(design_path=tmp_path, binder_length=30)

    (tmp_path / "Accepted").mkdir()
    with pytest.raises(NoAcceptedBindCraftDesigns, match="contains no PDBs"):
        _collect_designs(design_path=tmp_path, binder_length=30)


def test_bindcraft_run_returns_structured_empty_result_when_no_designs_accepted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_python = _configure_bindcraft_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        bindcraft_mod,
        "_bindcraft_python",
        lambda *_args: (_ for _ in ()).throw(AssertionError("conda lookup skipped")),
    )
    monkeypatch.setattr(bindcraft_mod, "PersistentPythonSidecar", _NoAcceptedSidecar)

    target = tmp_path / "target.pdb"
    target.write_text("HEADER target\nEND\n")
    adapter = BindCraftAdapter(config={}, weights_dir=tmp_path, device=torch.device("cuda"))

    out = adapter.run(
        BindCraftInput(
            target_pdb=str(target),
            binder_length=60,
            max_attempted_trajectories=4,
        ),
        null_context(tmp_path / "job"),
    )

    assert fake_python.is_file()
    assert out.status == "no_acceptable_candidate"
    assert out.designs == []
    assert out.candidates == []
    assert out.num_designs_returned == 0
    assert out.failure_summary["attempted"] == 3
    assert out.failure_summary["accepted"] == 0
    assert out.failure_summary["failure_events"] == 3
    assert out.run_metadata["status"] == "no_acceptable_candidate"
    assert out.run_metadata["accepted_design_count"] == 0
    assert any("accepted 0 designs" in warning for warning in out.warnings)


def test_bindcraft_cuda_infrastructure_error_still_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(bindcraft_mod, "_check_paths", lambda: None)
    monkeypatch.setattr(bindcraft_mod.torch.cuda, "is_available", lambda: False)

    adapter = BindCraftAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    with pytest.raises(RuntimeError, match="requires CUDA"):
        adapter.run(
            BindCraftInput(target_pdb="/tmp/nonexistent.pdb"),
            null_context(tmp_path / "job"),
        )


def test_bindcraft_output_schema_accepts_empty_result_shape() -> None:
    schema = json.loads(
        (
            REPO_ROOT
            / "src/vibeproteinagent/adapters/bindcraft/schemas/bindcraft.output.json"
        ).read_text()
    )
    Draft202012Validator(schema).validate(
        {
            "status": "no_acceptable_candidate",
            "designs": [],
            "candidates": [],
            "runtime_ms": 123,
            "num_designs_returned": 0,
            "seeds_used": [],
            "position_mappings": {},
            "warnings": ["BindCraft attempted 3 trajectories and accepted 0 designs."],
            "failure_summary": {
                "attempted": 3,
                "accepted": 0,
                "relaxed": 1,
                "clashing": 1,
                "low_confidence": 1,
                "failure_events": 3,
            },
            "run_metadata": {
                "status": "no_acceptable_candidate",
                "accepted_design_count": 0,
            },
        }
    )

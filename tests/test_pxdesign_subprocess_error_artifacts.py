from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from vibeproteinagent.core.errors import VPAError
from vibeproteinagent.adapters.pxdesign import adapter as pxdesign_adapter


def test_pxdesign_subprocess_failure_writes_debug_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "pxdesign_v0.1.0.pt"
    ckpt.write_text("weights")
    monkeypatch.setattr(pxdesign_adapter, "pipeline_python", lambda: Path("/bin/false"))
    monkeypatch.setattr(pxdesign_adapter, "availability_skip_reason", lambda: None)

    with pytest.raises(VPAError) as exc_info:
        pxdesign_adapter._run_preview_pipeline_until_artifacts(
            argv=["--example"],
            cwd=tmp_path,
            job_outputs=tmp_path,
            task_name="debug_case",
            seed=7,
            mpnn_temperature=0.1,
            ckpt_path=ckpt,
            timeout_s=30,
        )

    debug_path = tmp_path / "pxdesign_failure_debug.json"
    payload = json.loads(debug_path.read_text())
    assert payload["command"][0] == "/bin/false"
    assert payload["cwd"] == str(tmp_path)
    assert payload["returncode"] != 0
    assert Path(payload["stdout_path"]).is_file()
    assert Path(payload["stderr_path"]).is_file()
    assert payload["checkpoint_state"]["pxdesign_checkpoint_exists"] is True
    assert payload["raw_artifact_state"]["raw_cif_exists"] is False
    assert payload["preview_fallback_chain_metadata"] is None
    assert payload["preview_fallback_chain_metadata_error"] is None
    assert "env_summary" in payload
    assert "preflight_result" in payload
    assert exc_info.value.details["command"] == payload["command"]
    assert exc_info.value.details["stdout_path"] == payload["stdout_path"]


def test_pxdesign_preview_fallback_failure_writes_debug_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt = tmp_path / "pxdesign_v0.1.0.pt"
    ckpt.write_text("weights")
    raw_dir = tmp_path / "global_run_0" / "debug_case" / "seed_101" / "predictions"
    raw_dir.mkdir(parents=True)
    (raw_dir.parent / "SUCCESS_FILE").write_text("")
    raw_cif = raw_dir / "debug_case_sample_0.cif"
    raw_cif.write_text("data_raw\n")

    def fake_run(*args, **kwargs):
        stdout = kwargs["stdout"]
        stderr = kwargs["stderr"]
        stdout.write("fallback stdout\n")
        stderr.write("fallback stderr\n")
        return SimpleNamespace(returncode=5)

    monkeypatch.setattr(pxdesign_adapter, "pipeline_python", lambda: Path("/bin/python"))
    monkeypatch.setattr(pxdesign_adapter, "availability_skip_reason", lambda: None)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(VPAError) as exc_info:
        pxdesign_adapter._materialize_preview_sequence_outputs(
            raw_cif=raw_cif,
            job_outputs=tmp_path,
            task_name="debug_case",
            seed=101,
            mpnn_temperature=0.1,
            debug_path=tmp_path / "pxdesign_failure_debug.json",
            ckpt_path=ckpt,
        )

    payload = json.loads((tmp_path / "pxdesign_failure_debug.json").read_text())
    assert payload["returncode"] == 5
    assert payload["raw_artifact_state"]["raw_cif_exists"] is True
    assert payload["preview_fallback_chain_metadata"] is None
    assert payload["preview_fallback_chain_metadata_error"] == "metadata_not_written"
    assert Path(payload["stdout_path"]).name == "pipeline_preview_fallback.stdout.log"
    assert Path(payload["stderr_path"]).name == "pipeline_preview_fallback.stderr.log"
    assert exc_info.value.details["command"] == payload["command"]

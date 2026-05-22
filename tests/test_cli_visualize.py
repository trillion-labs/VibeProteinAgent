from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.workspace import ArtifactStore, JobState, JobStore, Workspace

PDB_TEXT = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 80.00           N
ATOM      2  CA  ALA A   1      12.560  13.233   9.447  1.00 80.00           C
ATOM      3  C   ALA A   1      13.048  14.665   9.447  1.00 80.00           C
ATOM      4  O   ALA A   1      12.250  15.590   9.447  1.00 80.00           O
ATOM      5  N   GLY A   2      14.360  14.850   9.447  1.00 82.00           N
ATOM      6  CA  GLY A   2      14.940  16.180   9.447  1.00 82.00           C
ATOM      7  C   GLY A   2      16.450  16.120   9.447  1.00 82.00           C
ATOM      8  O   GLY A   2      17.100  17.150   9.447  1.00 82.00           O
TER
END
"""


def test_visualize_structure_file_writes_standalone_html(
    runner: CliRunner,
    isolated_vpa_home: Path,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text(PDB_TEXT)
    out = tmp_path / "viewer.html"

    res = runner.invoke(app, ["visualize", str(pdb), "--out", str(out), "--json"])

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["viewer_html"] == str(out.resolve())
    assert data["structure_count"] == 1
    assert data["viewer_mode"] == "online"
    assert data["structures"][0]["metadata"]["residue_count"] == 2
    html = out.read_text()
    # Online viewer uses the 3Dmol CDN with a vendored fallback. ``$3Dmol`` is
    # the global entry point exposed by 3Dmol.js, so it's the most stable
    # contract to assert across either source path.
    assert "https://3Dmol.org/build/3Dmol-min.js" in html
    assert "$3Dmol" in html
    assert "model.pdb" in html
    assert "ATOM      1  N   ALA" in html


def test_visualize_structure_file_writes_offline_html(
    runner: CliRunner,
    isolated_vpa_home: Path,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text(PDB_TEXT)
    out = tmp_path / "viewer_offline.html"

    res = runner.invoke(
        app,
        ["visualize", str(pdb), "--viewer", "offline", "--out", str(out), "--json"],
    )

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["viewer_html"] == str(out.resolve())
    assert data["viewer_mode"] == "offline"
    assert data["structure_count"] == 1
    html = out.read_text()
    assert "const VPA_OFFLINE_VIEWER = true" in html
    assert "https://3Dmol.org/build/3Dmol-min.js" not in html
    assert "ATOM      1  N   ALA" in html


def test_visualize_embeds_structure_text_script_safely(
    runner: CliRunner,
    isolated_vpa_home: Path,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text("REMARK </script><script>alert(1)</script>\n" + PDB_TEXT)
    out = tmp_path / "viewer_offline.html"

    res = runner.invoke(
        app,
        ["visualize", str(pdb), "--viewer", "offline", "--out", str(out), "--json"],
    )

    assert res.exit_code == 0, res.stdout
    html = out.read_text()
    assert "REMARK </script>" not in html
    assert "REMARK \\u003c/script\\u003e" in html


def test_visualize_job_uses_registered_structure_artifacts(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "design.pdb"
    pdb.write_text(PDB_TEXT)
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "protenix_v2",
        "1.0.0",
        {"input": "x"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(
        rec.job_id,
        JobState.COMPLETED,
        output={"predicted_pdb": str(pdb), "metrics": {"plddt": 81.0}},
    )
    artifact = ArtifactStore(tmp_workspace).register(
        rec.job_id,
        "predicted_pdb",
        pdb,
        kind="pdb",
        mime="chemical/x-pdb",
    )
    out = tmp_path / "job_viewer.html"

    res = runner.invoke(
        app,
        [
            "visualize",
            "--job-id",
            rec.job_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(out),
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["source"]["kind"] == "job"
    assert data["structures"][0]["source"]["artifact_id"] == artifact.artifact_id
    assert data["structures"][0]["label"] == "predicted_pdb"
    html = out.read_text()
    assert rec.job_id in html
    assert "protenix_v2" in html
    assert "predicted_pdb" in html


def test_visualize_requires_exactly_one_source(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["visualize", "--json"])

    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["error"]["code"] == "E_INVALID_PARAMS"


def test_visualize_rejects_unknown_viewer_mode(
    runner: CliRunner,
    isolated_vpa_home: Path,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text(PDB_TEXT)

    res = runner.invoke(app, ["visualize", str(pdb), "--viewer", "vr", "--json"])

    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["error"]["code"] == "E_INVALID_PARAMS"
    assert payload["error"]["details"]["allowed"] == ["online", "offline"]

"""Codex review B1: unit test the worker's artifact-introspection helper.

The worker doesn't own SQLite; it summarises what the adapter wrote and
hands the list to the supervisor for registration. This test pins the
contract for what counts as an "artifact" without spinning up a daemon.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.workers.worker import _introspect_artifacts


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "job_001" / "outputs"
    d.mkdir(parents=True)
    return tmp_path / "job_001"


def test_top_level_string_path_is_admitted(job_dir: Path) -> None:
    cif = job_dir / "outputs" / "predicted.cif"
    cif.write_text("ATOM ...\n")
    out = _introspect_artifacts({"predicted_cif": str(cif)}, job_dir)
    assert len(out) == 1
    assert out[0]["field"] == "predicted_cif"
    assert out[0]["kind"] == "cif"
    assert out[0]["path"] == str(cif)


def test_list_of_paths_expands_to_indexed_fields(job_dir: Path) -> None:
    p0 = job_dir / "outputs" / "a.json"
    p1 = job_dir / "outputs" / "b.json"
    p0.write_text("{}")
    p1.write_text("{}")
    out = _introspect_artifacts(
        {"input_jsons": [str(p0), str(p1)]}, job_dir,
    )
    fields = sorted(a["field"] for a in out)
    assert fields == ["input_jsons[0]", "input_jsons[1]"]
    assert all(a["kind"] == "json" for a in out)


def test_non_path_strings_are_ignored(job_dir: Path) -> None:
    out = _introspect_artifacts(
        {"payload": "hi", "score": 0.5, "names": ["design_1"]}, job_dir,
    )
    assert out == []


def test_paths_outside_job_dir_are_ignored(tmp_path: Path, job_dir: Path) -> None:
    elsewhere = tmp_path / "rogue.txt"
    elsewhere.write_text("nope")
    out = _introspect_artifacts({"sneak": str(elsewhere)}, job_dir)
    assert out == []


def test_paths_to_nonexistent_files_are_ignored(job_dir: Path) -> None:
    fake = job_dir / "outputs" / "missing.fa"
    # NOT written — Path.is_file() returns False.
    out = _introspect_artifacts({"designs_fasta": str(fake)}, job_dir)
    assert out == []


def test_kind_mapping_per_suffix(job_dir: Path) -> None:
    files = {
        "designs_fasta": ("designs.fa", "fasta"),
        "predicted_cif": ("model.cif", "cif"),
        "scores_npz": ("scores.npz", "npz"),
        "report_json": ("report.json", "json"),
        "weights_pt": ("ckpt.pt", "pt"),
        "weird": ("unknown.xyz", "file"),
    }
    out_dict: dict[str, object] = {}
    for field, (name, _kind) in files.items():
        p = job_dir / "outputs" / name
        p.write_text("x")
        out_dict[field] = str(p)
    out = _introspect_artifacts(out_dict, job_dir)
    got = {a["field"]: a["kind"] for a in out}
    for field, (_name, kind) in files.items():
        assert got[field] == kind

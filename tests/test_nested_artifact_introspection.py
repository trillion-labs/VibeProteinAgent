"""Codex round-2 RC2: nested artifact introspection.

Phase 7's helper only walked top-level strings and ``list[str]`` fields. PXDesign
returns ``{"designs": [{"cif_path": "...", "sequence": "...", "score": 0.8}]}``
— the cif paths sit *inside* the list of dicts, so the previous walk missed
them. The recursive rewrite (Phase 8) admits them as ``"designs[0].cif_path"``,
``"designs[1].cif_path"``, etc.
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


def test_nested_list_of_dicts_admits_each_cif_path(job_dir: Path) -> None:
    """PXDesign-style ``designs: list[{cif_path: <path>}]`` shape."""
    p0 = job_dir / "outputs" / "d0.cif"
    p1 = job_dir / "outputs" / "d1.cif"
    p0.write_text("ATOM ...\n")
    p1.write_text("ATOM ...\n")
    out = _introspect_artifacts(
        {
            "designs": [
                {"cif_path": str(p0), "sequence": "ACDEFG", "score": 0.8},
                {"cif_path": str(p1), "sequence": "GHIKLM", "score": 0.7},
            ],
        },
        job_dir,
    )
    fields = sorted(a["field"] for a in out)
    assert fields == ["designs[0].cif_path", "designs[1].cif_path"]
    assert all(a["kind"] == "cif" for a in out)


def test_doubly_nested_dicts_unroll(job_dir: Path) -> None:
    inner = job_dir / "outputs" / "inner.json"
    inner.write_text("{}")
    out = _introspect_artifacts(
        {"top": {"middle": {"leaf_path": str(inner)}}},
        job_dir,
    )
    assert len(out) == 1
    assert out[0]["field"] == "top.middle.leaf_path"


def test_mixed_lists_and_dicts(job_dir: Path) -> None:
    a = job_dir / "outputs" / "a.fasta"
    b = job_dir / "outputs" / "b.fasta"
    a.write_text(">x\nA\n")
    b.write_text(">y\nB\n")
    out = _introspect_artifacts(
        {
            "groups": [
                {"name": "g0", "files": [str(a), str(b)]},
            ],
        },
        job_dir,
    )
    fields = sorted(a["field"] for a in out)
    assert fields == ["groups[0].files[0]", "groups[0].files[1]"]


def test_top_level_strings_still_work(job_dir: Path) -> None:
    """Phase 7 contract — recursion must not regress top-level scalars."""
    p = job_dir / "outputs" / "predicted.cif"
    p.write_text("ATOM ...\n")
    out = _introspect_artifacts({"predicted_cif": str(p)}, job_dir)
    assert [a["field"] for a in out] == ["predicted_cif"]


def test_non_path_scalars_in_nested_dicts_ignored(job_dir: Path) -> None:
    out = _introspect_artifacts(
        {"designs": [{"score": 0.5, "name": "d0"}]},
        job_dir,
    )
    assert out == []

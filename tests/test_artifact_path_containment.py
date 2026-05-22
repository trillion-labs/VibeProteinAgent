"""Codex round-3 C1: artifact introspection uses symlink-safe containment.

Before this fix, ``_introspect_artifacts`` admitted any path whose
string representation started with ``str(job_dir)`` — a symlink under
``job_dir`` pointing OUTSIDE ``job_dir`` would slip through and be
registered as an artifact (potential exfil of host data into the
content-addressed pool). The fix resolves both sides and uses
:meth:`pathlib.Path.is_relative_to` to check containment after symlink
resolution.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from vibeproteinagent.workers.worker import _introspect_artifacts


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    d = tmp_path / "job_xyz"
    (d / "outputs").mkdir(parents=True)
    return d


def test_legitimate_file_under_outputs_is_admitted(job_dir: Path) -> None:
    p = job_dir / "outputs" / "legit.txt"
    p.write_text("ok")
    out = _introspect_artifacts({"y": str(p)}, job_dir)
    fields = [a["field"] for a in out]
    assert fields == ["y"]


def test_absolute_path_outside_jobdir_is_rejected(
    tmp_path: Path, job_dir: Path,
) -> None:
    """``/etc/passwd``-style paths must not be admitted."""
    out = _introspect_artifacts({"x": "/etc/passwd"}, job_dir)
    assert out == []


def test_symlink_escaping_jobdir_is_rejected(
    tmp_path: Path, job_dir: Path,
) -> None:
    """A symlink *inside* job_dir pointing OUTSIDE must not be admitted.

    Naive ``startswith`` would pass: the symlink's literal path string
    starts with ``str(job_dir)``. ``Path.resolve()`` collapses the
    link and the post-resolve containment check catches the escape.
    """
    outside = tmp_path / "outside_target.txt"
    outside.write_text("secret")
    sneaky_link = job_dir / "outputs" / "sneaky.txt"
    os.symlink(outside, sneaky_link)
    # Sanity: the link IS a file, but resolves outside the job_dir tree.
    assert sneaky_link.is_file()
    out = _introspect_artifacts({"sneak": str(sneaky_link)}, job_dir)
    assert out == []


def test_mixed_legit_and_outside_only_legit_admitted(
    tmp_path: Path, job_dir: Path,
) -> None:
    outside = tmp_path / "rogue.cif"
    outside.write_text("nope")
    legit = job_dir / "outputs" / "good.cif"
    legit.write_text("ATOM ...")
    out = _introspect_artifacts(
        {"x": "/etc/passwd", "y": str(legit), "z": str(outside)}, job_dir,
    )
    fields = [a["field"] for a in out]
    assert fields == ["y"]


def test_symlink_inside_to_inside_is_admitted(
    job_dir: Path,
) -> None:
    """An in-tree symlink to an in-tree file is fine and stays registered."""
    real = job_dir / "outputs" / "real.json"
    real.write_text("{}")
    link = job_dir / "outputs" / "link.json"
    os.symlink(real, link)
    out = _introspect_artifacts({"a": str(link)}, job_dir)
    # Resolves to the real file, still under job_dir.
    assert len(out) == 1
    assert out[0]["field"] == "a"

from __future__ import annotations

import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import ArtifactStore, JobStore, Workspace


@pytest.fixture
def stores(tmp_workspace: Workspace) -> tuple[JobStore, ArtifactStore]:
    return JobStore(tmp_workspace), ArtifactStore(tmp_workspace)


def _make_file(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_register_and_dedup(
    tmp_path: Path, stores: tuple[JobStore, ArtifactStore],
) -> None:
    job_store, art_store = stores
    job = job_store.submit("noop", "1.0.0", {}, workspace_name=job_store.workspace.name)

    a_path = _make_file(tmp_path, "a.txt", b"hello world\n")
    b_path = _make_file(tmp_path, "b.txt", b"hello world\n")  # same content -> same sha
    c_path = _make_file(tmp_path, "c.txt", b"different content\n")

    ra = art_store.register(job.job_id, "designs_fasta", a_path, kind="text", mime="text/plain")
    rb = art_store.register(job.job_id, "alt_fasta", b_path, kind="text")
    rc = art_store.register(job.job_id, "logs", c_path, kind="text")

    assert ra.sha256 == rb.sha256
    assert ra.artifact_id == rb.artifact_id  # dedup -> same row
    assert rc.sha256 != ra.sha256
    assert rc.artifact_id != ra.artifact_id
    assert Path(ra.pool_path).exists()
    assert Path(rc.pool_path).exists()


def test_for_job_returns_by_field(
    tmp_path: Path, stores: tuple[JobStore, ArtifactStore],
) -> None:
    job_store, art_store = stores
    job = job_store.submit("noop", "1.0.0", {}, workspace_name=job_store.workspace.name)
    f1 = _make_file(tmp_path, "designs.fa", b"FASTA1\n")
    f2 = _make_file(tmp_path, "logs.npz", b"BINARYBLOB\n")

    art_store.register(job.job_id, "designs", f1, kind="fasta")
    art_store.register(job.job_id, "log_prob", f2, kind="npz")

    out = art_store.for_job(job.job_id)
    assert set(out) == {"designs", "log_prob"}
    assert out["designs"].kind == "fasta"
    assert out["log_prob"].kind == "npz"


def test_prune_respects_pinned(
    tmp_path: Path, stores: tuple[JobStore, ArtifactStore],
) -> None:
    job_store, art_store = stores
    job = job_store.submit("noop", "1.0.0", {}, workspace_name=job_store.workspace.name)

    pinned_file = _make_file(tmp_path, "pin.txt", b"pinned-content\n")
    free_file = _make_file(tmp_path, "free.txt", b"free-content\n")
    art_store.register(job.job_id, "pin_field", pinned_file, kind="text", pinned=True)
    art_store.register(job.job_id, "free_field", free_file, kind="text", pinned=False)

    # Wait a hair, then prune with cutoff in the past (catches both rows by age 0s).
    time.sleep(0.01)
    victims = art_store.prune(older_than_s=0)
    victim_fields = {v.kind for v in victims}
    # Only the unpinned one is in victims.
    assert len(victims) == 1
    assert "text" in victim_fields  # both rows have kind text, ensure one was selected

    remaining = art_store.for_job(job.job_id)
    assert "pin_field" in remaining
    assert "free_field" not in remaining


def test_prune_dry_run_leaves_pool_alone(
    tmp_path: Path, stores: tuple[JobStore, ArtifactStore],
) -> None:
    job_store, art_store = stores
    job = job_store.submit("noop", "1.0.0", {}, workspace_name=job_store.workspace.name)
    f = _make_file(tmp_path, "x.txt", b"hello\n")
    art = art_store.register(job.job_id, "x", f, kind="text")
    pool = Path(art.pool_path)
    assert pool.exists()
    time.sleep(0.01)
    victims = art_store.prune(older_than_s=0, dry_run=True)
    assert len(victims) == 1
    # File still on disk + row still readable.
    assert pool.exists()
    assert art_store.get(art.artifact_id).artifact_id == art.artifact_id

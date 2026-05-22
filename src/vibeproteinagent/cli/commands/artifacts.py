from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import typer

from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_err,
    print_envelope_ok,
)
from vibeproteinagent.core.errors import ErrorCode
from vibeproteinagent.workspace import ArtifactStore, JobStore, Workspace


def _resolve(workspace: str | None) -> tuple[Workspace, JobStore, ArtifactStore]:
    ws = Workspace.resolve(workspace)
    return ws, JobStore(ws), ArtifactStore(ws)


def artifacts(
    id: str = typer.Argument(..., help="Job id."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List artifacts produced by a job."""
    try:
        _, store, art = _resolve(workspace)
        rec = store.get(id)
        records = art.for_job(rec.job_id)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {
            "job_id": rec.job_id,
            "artifacts": [
                {
                    "field": k,
                    "id": v.artifact_id,
                    "kind": v.kind,
                    "mime": v.mime,
                    "size_bytes": v.size_bytes,
                    "sha256": v.sha256,
                    "path": v.pool_path,
                }
                for k, v in records.items()
            ],
        },
        json_mode=json_,
    )


def download(
    id: str = typer.Argument(..., help="Job id."),
    out: str | None = typer.Option(None, "--out", "-o", help="Destination directory."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Copy a job's artifacts into ``--out`` (default: ./<job_id>/)."""
    try:
        _, store, art = _resolve(workspace)
        rec = store.get(id)
        records = art.for_job(rec.job_id)
        out_dir = Path(out) if out else Path.cwd() / rec.job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for field, a in records.items():
            src = Path(a.pool_path)
            if not src.is_file():
                continue
            dest_name = f"{field}{src.suffix}" if src.suffix else field
            dest = out_dir / dest_name
            shutil.copyfile(src, dest)
            files.append(
                {
                    "field": field,
                    "id": a.artifact_id,
                    "src": str(src),
                    "dest": str(dest),
                    "size_bytes": a.size_bytes,
                }
            )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"job_id": rec.job_id, "out_dir": str(out_dir), "files": files},
        json_mode=json_,
    )


def export(
    id: str = typer.Argument(..., help="Job id."),
    format: str = typer.Option("directory", "--format", help="tar | zip | directory."),
    out: str | None = typer.Option(None, "--out", "-o", help="Destination path."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Export a job's per-job directory as a tarball, zip, or copied directory."""
    try:
        if format not in ("tar", "zip", "directory"):
            print_envelope_err(
                ErrorCode.INVALID_PARAMS,
                f"unknown export format {format!r}",
                {"got": format, "allowed": ["tar", "zip", "directory"]},
                json_mode=json_,
            )
        ws, store, _ = _resolve(workspace)
        rec = store.get(id)
        job_dir = ws.job_dir(rec.job_id)
        suffix = format if format != "directory" else "d"
        out_path = Path(out) if out else Path.cwd() / f"{rec.job_id}.{suffix}"
        if format == "directory":
            if out_path.exists():
                shutil.rmtree(out_path)
            shutil.copytree(job_dir, out_path)
        elif format == "tar":
            with tarfile.open(out_path, "w") as tf:
                tf.add(job_dir, arcname=rec.job_id)
        elif format == "zip":
            with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in job_dir.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=str(Path(rec.job_id) / p.relative_to(job_dir)))
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"job_id": rec.job_id, "out_path": str(out_path), "format": format},
        json_mode=json_,
    )


__all__ = ["artifacts", "download", "export"]

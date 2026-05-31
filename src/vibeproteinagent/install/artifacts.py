from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibeproteinagent.core.errors import DependencyError


@dataclass(frozen=True)
class ArtifactRecord:
    ref: str
    kind: str
    tool: str | None = None
    url: str | None = None
    path: str | None = None
    hf_repo: str | None = None
    hf_filename: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    unpack: str | None = None
    cuda_profile: str | None = None
    version: str | None = None


def load_artifact_index(url_or_path: str | None = None) -> dict[str, ArtifactRecord]:
    """Load a small external artifact index used by prebuilt-first installs."""
    raw_ref = os.environ.get("VPA_ARTIFACT_INDEX") or url_or_path
    if not raw_ref:
        return {}
    raw, base_dir = _read_json(raw_ref)
    records = raw.get("artifacts") if isinstance(raw, dict) else None
    if not isinstance(records, dict):
        raise DependencyError(
            "artifact index must contain an 'artifacts' mapping",
            details={"index": raw_ref},
        )
    out: dict[str, ArtifactRecord] = {}
    for ref, value in records.items():
        if not isinstance(value, dict):
            raise DependencyError(
                "artifact index entries must be mappings",
                details={"index": raw_ref, "ref": ref},
            )
        path = str(value["path"]) if value.get("path") else None
        if path and base_dir is not None:
            path_obj = Path(path).expanduser()
            if not path_obj.is_absolute():
                path = str(base_dir / path_obj)
        out[str(ref)] = ArtifactRecord(
            ref=str(ref),
            kind=str(value.get("kind") or "file"),
            tool=str(value["tool"]) if value.get("tool") else None,
            url=str(value["url"]) if value.get("url") else None,
            path=path,
            hf_repo=str(value["hf_repo"]) if value.get("hf_repo") else None,
            hf_filename=str(value["hf_filename"]) if value.get("hf_filename") else None,
            sha256=str(value["sha256"]) if value.get("sha256") else None,
            size_bytes=int(value["size_bytes"]) if value.get("size_bytes") is not None else None,
            unpack=str(value["unpack"]) if value.get("unpack") else None,
            cuda_profile=str(value["cuda_profile"]) if value.get("cuda_profile") else None,
            version=str(value["version"]) if value.get("version") else None,
        )
    return out


def materialize_artifact(record: ArtifactRecord, target: Path) -> Path:
    """Copy/download one artifact into target atomically and verify checksum."""
    source = _record_source(record)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.startswith(("http://", "https://")):
        tmp = target.parent / f".{target.name}.download"
        _download_url(source, tmp)
        return _finish_materialized_file(record, tmp, target)
    else:
        src = Path(source).expanduser()
        if not src.exists():
            raise DependencyError(
                "artifact source does not exist",
                details={"ref": record.ref, "source": source},
            )
        if src.is_dir():
            tmp_dir = target.parent / f".{target.name}.artifact-tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            _copy_directory(src, tmp_dir)
            _verify_dir_nonempty(tmp_dir, record)
            _replace_path(tmp_dir, target)
            return target
        tmp = target.parent / f".{target.name}.download"
        shutil.copy2(src, tmp)
        return _finish_materialized_file(record, tmp, target)


def _finish_materialized_file(
    record: ArtifactRecord,
    tmp: Path,
    target: Path,
) -> Path:
    if record.sha256 and _sha256(tmp) != record.sha256:
        tmp.unlink(missing_ok=True)
        raise DependencyError(
            "artifact checksum verification failed",
            details={"ref": record.ref, "target": str(target)},
        )
    if record.size_bytes is not None and tmp.stat().st_size != record.size_bytes:
        tmp.unlink(missing_ok=True)
        raise DependencyError(
            "artifact size verification failed",
            details={
                "ref": record.ref,
                "expected": record.size_bytes,
                "actual": tmp.stat().st_size,
            },
        )
    if record.unpack:
        tmp_dir = target.parent / f".{target.name}.artifact-tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(tmp, tmp_dir, record.unpack)
        tmp.unlink(missing_ok=True)
        _verify_dir_nonempty(tmp_dir, record)
        _replace_path(tmp_dir, target)
        return target
    tmp.replace(target)
    return target


def _download_url(url: str, tmp: Path) -> None:
    headers = _download_headers()
    resume_at = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers=headers)
    if resume_at:
        req.add_header("Range", f"bytes={resume_at}-")
    with urllib.request.urlopen(req) as response:
        if resume_at and response.status != 206:
            tmp.unlink(missing_ok=True)
            resume_at = 0
        mode = "ab" if resume_at else "wb"
        with tmp.open(mode) as fh:
            shutil.copyfileobj(response, fh)


def _read_json(ref: str) -> tuple[dict[str, Any], Path | None]:
    if ref.startswith(("http://", "https://")):
        req = urllib.request.Request(ref, headers=_download_headers())
        with urllib.request.urlopen(req) as response:
            return dict(json.loads(response.read().decode("utf-8"))), None
    path = Path(ref).expanduser()
    return dict(json.loads(path.read_text())), path.parent


def _record_source(record: ArtifactRecord) -> str:
    mirror = os.environ.get("VPA_ARTIFACT_MIRROR")
    if mirror:
        candidate = Path(mirror).expanduser() / record.ref
        if candidate.exists():
            return str(candidate)
    if record.path:
        return record.path
    if record.url:
        return record.url
    raise DependencyError(
        "artifact record has no url/path",
        details={"ref": record.ref, "kind": record.kind},
    )


def _replace_path(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    source.replace(target)


def _copy_directory(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    cp = shutil.which("cp")
    if cp is not None:
        proc = subprocess.run(
            [cp, "-a", "--reflink=auto", f"{source}/.", str(target)],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return
    shutil.rmtree(target)
    shutil.copytree(source, target, symlinks=True)


def _download_headers() -> dict[str, str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("VPA_HF_CREDENTIAL")
    )
    return {"Authorization": f"Bearer {token}"} if token else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_dir_nonempty(path: Path, record: ArtifactRecord) -> None:
    if not path.is_dir() or not any(path.iterdir()):
        raise DependencyError(
            "artifact directory copy produced an empty target",
            details={"ref": record.ref, "target": str(path)},
        )


def _extract_archive(path: Path, target: Path, kind: str) -> None:
    if kind == "tar":
        with tarfile.open(path) as archive:
            _safe_extract_tar(archive, target)
        return
    if kind == "zip":
        with zipfile.ZipFile(path) as archive:
            _safe_extract_zip(archive, target)
        return
    raise DependencyError(
        "unsupported artifact unpack format",
        details={"ref": path.name, "unpack": kind, "allowed": ["tar", "zip"]},
    )


def _safe_extract_tar(archive: tarfile.TarFile, target: Path) -> None:
    root = target.resolve()
    for member in archive.getmembers():
        candidate = (target / member.name).resolve()
        if not candidate.is_relative_to(root):
            raise DependencyError(
                "refusing to extract archive member outside destination",
                details={"member": member.name, "destination": str(target)},
            )
    archive.extractall(target, filter="data")


def _safe_extract_zip(archive: zipfile.ZipFile, target: Path) -> None:
    root = target.resolve()
    for member in archive.infolist():
        candidate = (target / member.filename).resolve()
        if not candidate.is_relative_to(root):
            raise DependencyError(
                "refusing to extract archive member outside destination",
                details={"member": member.filename, "destination": str(target)},
            )
    archive.extractall(target)


__all__ = ["ArtifactRecord", "load_artifact_index", "materialize_artifact"]

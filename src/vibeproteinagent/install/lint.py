from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from vibeproteinagent.install.manifest import InstallManifest


def lint_manifest(manifest: InstallManifest) -> dict[str, Any]:
    """Return install-manifest contract issues without mutating the machine."""
    issues: list[dict[str, Any]] = []
    for tool in manifest.tools.values():
        for source in tool.sources:
            has_remote = bool(source.repo or source.artifact_ref)
            if source.source_path and not has_remote and tool.artifact_policy != "user_provided":
                issues.append(
                    _issue(
                        "source_path_only",
                        tool.name,
                        f"source:{source.name}",
                        "source_path is a local fallback, not a release install contract",
                        {"source": asdict(source)},
                    )
                )
            if (
                _developer_absolute(source.source_path)
                and not has_remote
                and tool.artifact_policy != "user_provided"
            ):
                issues.append(
                    _issue(
                        "developer_local_source_path",
                        tool.name,
                        f"source:{source.name}",
                        "absolute developer-local source_path needs a URL/artifact_ref "
                        "or user_provided policy",
                        {"source_path": source.source_path},
                    )
                )
        for weight in tool.weights:
            has_remote = bool(weight.url or weight.hf_repo_id or weight.artifact_ref)
            if weight.source_path and not has_remote and tool.artifact_policy != "user_provided":
                issues.append(
                    _issue(
                        "source_path_only",
                        tool.name,
                        f"weight:{weight.name}",
                        "source_path is a local fallback, not a release install contract",
                        {"weight": asdict(weight)},
                    )
                )
            if (
                _developer_absolute(weight.source_path)
                and not has_remote
                and tool.artifact_policy != "user_provided"
            ):
                issues.append(
                    _issue(
                        "developer_local_source_path",
                        tool.name,
                        f"weight:{weight.name}",
                        "absolute developer-local source_path needs a URL/artifact_ref "
                        "or user_provided policy",
                        {"source_path": weight.source_path},
                    )
                )
            if has_remote and not (weight.sha256 or weight.hf_repo_id):
                issues.append(
                    _issue(
                        "missing_checksum",
                        tool.name,
                        f"weight:{weight.name}",
                        "downloaded non-HF artifacts should declare sha256",
                        {"weight": weight.name},
                        severity="warning",
                    )
                )
        if tool.sidecar and not (tool.sidecar.prebuilt_ref or tool.sidecar.install_script):
            issues.append(
                _issue(
                    "sidecar_without_install_contract",
                    tool.name,
                    "sidecar",
                    "sidecar tools need a prebuilt_ref and/or install_script",
                    {"sidecar": asdict(tool.sidecar)},
                )
            )
    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "issues": issues,
    }


def _issue(
    code: str,
    tool: str,
    unit: str,
    message: str,
    details: dict[str, Any],
    *,
    severity: str = "error",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "tool": tool,
        "unit": unit,
        "message": message,
        "details": details,
    }


def _developer_absolute(value: str | None) -> bool:
    if not value:
        return False
    path = Path(value)
    return path.is_absolute() and str(path).startswith(("/opt/", "/mnt/"))


__all__ = ["lint_manifest"]

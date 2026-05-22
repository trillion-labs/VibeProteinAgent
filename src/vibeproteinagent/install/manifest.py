from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from vibeproteinagent.core.errors import InvalidParamsError


@dataclass(frozen=True)
class InstallWeight:
    name: str
    path: str
    env_var: str | None = None
    env_value: str | None = None
    url: str | None = None
    sha256: str | None = None
    source_path: str | None = None
    required: bool = True
    modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class InstallSidecar:
    env_name: str
    path: str
    install_script: str | None = None


@dataclass(frozen=True)
class InstallTool:
    name: str
    runtime: str
    required: bool = True
    depends_on: tuple[str, ...] = ()
    weights: tuple[InstallWeight, ...] = ()
    sidecar: InstallSidecar | None = None
    binary: str | None = None
    smoke_test: str | None = None
    disk_bytes: int = 0


@dataclass(frozen=True)
class InstallProfile:
    name: str
    description: str = ""
    tools: tuple[str, ...] = ()
    extends: tuple[str, ...] = ()


@dataclass(frozen=True)
class ManifestDefaults:
    install_root: str
    shared_env: str
    sidecars_root: str
    cache_dir: str
    state_db: str


@dataclass(frozen=True)
class InstallManifest:
    schema_version: int
    package: str
    defaults: ManifestDefaults
    profiles: dict[str, InstallProfile] = field(default_factory=dict)
    tools: dict[str, InstallTool] = field(default_factory=dict)

    @property
    def tool_names(self) -> set[str]:
        return set(self.tools)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise InvalidParamsError(
                f"unsupported install manifest schema_version {self.schema_version}",
                details={"schema_version": self.schema_version},
            )
        for profile in self.profiles.values():
            for parent in profile.extends:
                if parent not in self.profiles:
                    raise InvalidParamsError(
                        f"profile '{profile.name}' extends unknown profile '{parent}'",
                        details={"profile": profile.name, "extends": parent},
                    )
            for tool in profile.tools:
                if tool == "@all-tools":
                    continue
                if tool not in self.tools:
                    raise InvalidParamsError(
                        f"profile '{profile.name}' references unknown tool '{tool}'",
                        details={"profile": profile.name, "tool": tool},
                    )
        for tool in self.tools.values():
            for dep in tool.depends_on:
                if dep not in self.tools:
                    raise InvalidParamsError(
                        f"tool '{tool.name}' depends on unknown tool '{dep}'",
                        details={"tool": tool.name, "dependency": dep},
                    )
            for weight in tool.weights:
                if weight.env_value not in (None, "path", "parent"):
                    raise InvalidParamsError(
                        f"tool '{tool.name}' weight '{weight.name}' has invalid env_value",
                        details={
                            "tool": tool.name,
                            "weight": weight.name,
                            "env_value": weight.env_value,
                            "allowed": ["path", "parent"],
                        },
                    )

    def validate_registry_coverage(self, registry_tool_names: set[str]) -> dict[str, list[str]]:
        manifest_names = set(self.tools)
        return {
            "missing_from_manifest": sorted(registry_tool_names - manifest_names),
            "missing_from_registry": sorted(manifest_names - registry_tool_names),
        }


def load_manifest(path: Path | None = None) -> InstallManifest:
    if path is None:
        payload = resources.files("vibeproteinagent.install").joinpath("manifest.yaml").read_text()
        raw = yaml.safe_load(payload)
    else:
        raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise InvalidParamsError("install manifest must be a mapping")
    manifest = _parse_manifest(raw)
    manifest.validate()
    return manifest


def _parse_manifest(raw: dict[str, Any]) -> InstallManifest:
    defaults_raw = _mapping(raw.get("defaults"), "defaults")
    defaults = ManifestDefaults(
        install_root=str(defaults_raw.get("install_root", "~/.vibeproteinagent")),
        shared_env=str(defaults_raw.get("shared_env", "envs/vpa")),
        sidecars_root=str(defaults_raw.get("sidecars_root", "sidecars")),
        cache_dir=str(defaults_raw.get("cache_dir", "cache")),
        state_db=str(defaults_raw.get("state_db", "install/state.json")),
    )
    tool_defaults = _mapping(raw.get("tool_defaults", {}), "tool_defaults")
    tools = {
        name: _parse_tool(name, value, tool_defaults)
        for name, value in _mapping(raw.get("tools"), "tools").items()
    }
    profiles = {
        name: _parse_profile(name, value)
        for name, value in _mapping(raw.get("profiles"), "profiles").items()
    }
    return InstallManifest(
        schema_version=int(raw.get("schema_version", 0)),
        package=str(raw.get("package", "vibeproteinagent")),
        defaults=defaults,
        profiles=profiles,
        tools=tools,
    )


def _parse_profile(name: str, value: Any) -> InstallProfile:
    raw = _mapping(value, f"profiles.{name}")
    extends_raw = raw.get("extends", ())
    if isinstance(extends_raw, str):
        extends = (extends_raw,)
    else:
        extends = tuple(str(item) for item in (extends_raw or ()))
    return InstallProfile(
        name=name,
        description=str(raw.get("description", "")),
        tools=tuple(str(item) for item in (raw.get("tools") or ())),
        extends=extends,
    )


def _parse_tool(name: str, value: Any, defaults: dict[str, Any]) -> InstallTool:
    raw = dict(defaults)
    raw.update(_mapping(value, f"tools.{name}"))
    sidecar_raw = raw.get("sidecar")
    sidecar = None
    if sidecar_raw:
        side = _mapping(sidecar_raw, f"tools.{name}.sidecar")
        sidecar = InstallSidecar(
            env_name=str(side["env_name"]),
            path=str(side["path"]),
            install_script=str(side["install_script"]) if side.get("install_script") else None,
        )
    weights = tuple(_parse_weight(name, item) for item in (raw.get("weights") or ()))
    return InstallTool(
        name=name,
        runtime=str(raw.get("runtime", "shared")),
        required=bool(raw.get("required", True)),
        depends_on=tuple(str(item) for item in (raw.get("depends_on") or ())),
        weights=weights,
        sidecar=sidecar,
        binary=str(raw["binary"]) if raw.get("binary") else None,
        smoke_test=str(raw["smoke_test"]) if raw.get("smoke_test") else None,
        disk_bytes=int(raw.get("disk_bytes") or 0),
    )


def _parse_weight(tool_name: str, value: Any) -> InstallWeight:
    raw = _mapping(value, f"tools.{tool_name}.weights[]")
    return InstallWeight(
        name=str(raw["name"]),
        path=str(raw["path"]),
        env_var=str(raw["env_var"]) if raw.get("env_var") else None,
        env_value=str(raw["env_value"]) if raw.get("env_value") else None,
        url=str(raw["url"]) if raw.get("url") else None,
        sha256=str(raw["sha256"]) if raw.get("sha256") else None,
        source_path=str(raw["source_path"]) if raw.get("source_path") else None,
        required=bool(raw.get("required", True)),
        modes=tuple(str(item) for item in (raw.get("modes") or ())),
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvalidParamsError(f"install manifest field '{label}' must be a mapping")
    return value

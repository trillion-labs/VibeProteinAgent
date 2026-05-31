from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches

from vibeproteinagent.core.errors import InvalidParamsError, UnknownToolError
from vibeproteinagent.install.manifest import InstallManifest, InstallTool


@dataclass(frozen=True)
class InstallSelection:
    profile: str | None
    requested_tools: tuple[str, ...]
    tools: tuple[InstallTool, ...]
    registry_coverage: dict[str, list[str]]

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


def resolve_selection(
    manifest: InstallManifest,
    *,
    profile: str | None,
    tools: tuple[str, ...],
    registry_tool_names: set[str] | None = None,
) -> InstallSelection:
    if profile and tools:
        raise InvalidParamsError("choose either --profile or --tools, not both")
    selected_profile = profile or (manifest.defaults.default_profile if not tools else None)
    if selected_profile and selected_profile not in manifest.profiles:
        raise InvalidParamsError(
            f"unknown install profile '{selected_profile}'",
            details={"profile": selected_profile, "available_profiles": sorted(manifest.profiles)},
        )
    registry_names = registry_tool_names or set(manifest.tools)
    coverage = manifest.validate_registry_coverage(registry_names)
    requested = (
        _profile_tools(manifest, selected_profile, registry_names)
        if selected_profile
        else _parse_tool_list(tools)
    )
    ordered = _dependency_order(manifest, requested)
    return InstallSelection(
        profile=selected_profile,
        requested_tools=tuple(requested),
        tools=tuple(manifest.tools[name] for name in ordered),
        registry_coverage=coverage,
    )


def _parse_tool_list(tools: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in tools:
        for part in item.split(","):
            name = part.strip()
            if name and name not in out:
                out.append(name)
    if not out:
        raise InvalidParamsError("--tools must name at least one tool")
    return out


def _profile_tools(
    manifest: InstallManifest,
    profile_name: str | None,
    registry_tool_names: set[str],
) -> list[str]:
    if profile_name is None:
        return []
    profile = manifest.profiles[profile_name]
    out: list[str] = []
    for parent in profile.extends:
        out.extend(_profile_tools(manifest, parent, registry_tool_names))
    for tool in profile.tools:
        names = sorted(registry_tool_names) if tool == "@all-tools" else [tool]
        for name in names:
            if name not in out:
                out.append(name)
    return out


def _dependency_order(manifest: InstallManifest, requested: list[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name not in manifest.tools:
            suggestions = get_close_matches(name, sorted(manifest.tools), n=3)
            raise UnknownToolError(
                f"unknown install tool '{name}'",
                details={"tool": name, "suggestions": suggestions},
            )
        if name in seen:
            return
        if name in visiting:
            raise InvalidParamsError(
                f"cycle in install tool dependencies at '{name}'",
                details={"tool": name},
            )
        visiting.add(name)
        for dep in manifest.tools[name].depends_on:
            visit(dep)
        visiting.remove(name)
        seen.add(name)
        ordered.append(name)

    for name in requested:
        visit(name)
    return ordered

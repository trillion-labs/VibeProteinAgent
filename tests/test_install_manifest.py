from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.core.errors import InvalidParamsError, UnknownToolError
from vibeproteinagent.install.installer import InstallOptions, ManifestInstaller
from vibeproteinagent.install.manifest import load_manifest
from vibeproteinagent.install.planner import resolve_selection


def _registry_tool_names() -> set[str]:
    registry = ToolRegistry()
    registry.discover()
    return {descriptor.name for descriptor in registry.list()}


def test_install_manifest_covers_registered_tools(isolated_vpa_home: Path) -> None:
    manifest = load_manifest()
    coverage = manifest.validate_registry_coverage(_registry_tool_names())
    assert coverage == {"missing_from_manifest": [], "missing_from_registry": []}


def test_full_profile_resolves_supported_first_class_tools(isolated_vpa_home: Path) -> None:
    manifest = load_manifest()
    registry_names = _registry_tool_names()
    selection = resolve_selection(
        manifest,
        profile="full",
        tools=(),
        registry_tool_names=registry_names,
    )
    assert "rfdiffusion" in selection.tool_names
    assert "bindcraft" in selection.tool_names
    assert "esm3" in selection.tool_names
    assert "proteina_complexa" in selection.tool_names
    assert "pxdesign" in selection.tool_names
    assert not any(
        name.startswith("proteina_complexa_") for name in selection.tool_names
    )
    assert "noop" not in selection.tool_names
    assert "mpnn_to_protenix_input" not in selection.tool_names
    assert "sequence_to_protenix_input" not in selection.tool_names
    assert "pxdesign_pipeline_multi_gpu" not in selection.tool_names


def test_retired_and_gated_tools_do_not_block_default_doctor(
    isolated_vpa_home: Path,
) -> None:
    manifest = load_manifest()
    assert manifest.tools["esm3"].required is False
    assert manifest.tools["proteina_complexa"].required is False
    assert manifest.tools["pxdesign"].sidecar is not None


def test_selective_tools_resolve_only_requested_runtime_closure(
    isolated_vpa_home: Path,
) -> None:
    manifest = load_manifest()
    selection = resolve_selection(
        manifest,
        profile=None,
        tools=("rfdiffusion,proteinmpnn",),
        registry_tool_names=_registry_tool_names(),
    )
    assert selection.tool_names == ("rfdiffusion", "proteinmpnn")


def test_profile_and_tools_are_mutually_exclusive(isolated_vpa_home: Path) -> None:
    with pytest.raises(InvalidParamsError):
        resolve_selection(
            load_manifest(),
            profile="full",
            tools=("rfdiffusion",),
            registry_tool_names=_registry_tool_names(),
        )


def test_unknown_tool_reports_suggestions(isolated_vpa_home: Path) -> None:
    with pytest.raises(UnknownToolError) as exc:
        resolve_selection(
            load_manifest(),
            profile=None,
            tools=("rfdifusion",),
            registry_tool_names=_registry_tool_names(),
        )
    assert "rfdiffusion" in exc.value.details["suggestions"]


def test_install_dry_run_is_deterministic_and_non_mutating(
    isolated_vpa_home: Path,
) -> None:
    installer = ManifestInstaller(load_manifest())
    options = InstallOptions(
        profile=None,
        tools=("rfdiffusion,proteinmpnn",),
        install_root=isolated_vpa_home,
        dry_run=True,
    )
    first = installer.install(options)
    second = installer.install(options)
    assert first == second
    assert first["dry_run"] is True
    assert first["selected_tools"] == ["rfdiffusion", "proteinmpnn"]
    assert first["env_exports"]["VPA_RFDIFFUSION_WEIGHTS"].endswith("/cache/rfdiffusion")
    assert not (isolated_vpa_home / "install" / "state.json").exists()


def test_manifest_can_export_exact_weight_file_path(isolated_vpa_home: Path) -> None:
    installer = ManifestInstaller(load_manifest())
    options = InstallOptions(
        profile=None,
        tools=("rfdiffusion3",),
        install_root=isolated_vpa_home,
        dry_run=True,
    )
    plan = installer.install(options)

    assert plan["env_exports"]["VPA_RFDIFFUSION3_CKPT"].endswith(
        "/cache/rfdiffusion3/rfd3_latest.ckpt"
    )

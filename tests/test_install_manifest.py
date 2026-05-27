from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.core.config import load_vpa_env_file
from vibeproteinagent.core.errors import InvalidParamsError, UnknownToolError
from vibeproteinagent.install import installer as installer_module
from vibeproteinagent.install.cuda import resolve_cuda_stack
from vibeproteinagent.install.installer import (
    InstallOptions,
    InstallPaths,
    InstallState,
    ManifestInstaller,
    _BOLTZGEN_RUNTIME_DEPS,
    _ensure_weight,
    _env_exports,
    _link_prebuilt_sidecar,
    _manifest_readiness,
    _torch_install_matches_stack,
)
from vibeproteinagent.install.manifest import InstallTool, InstallWeight, load_manifest
from vibeproteinagent.install.planner import resolve_selection


def _registry_tool_names() -> set[str]:
    registry = ToolRegistry()
    registry.discover()
    return {descriptor.name for descriptor in registry.list()}


def test_install_manifest_covers_registered_tools(isolated_vpa_home: Path) -> None:
    manifest = load_manifest()
    coverage = manifest.validate_registry_coverage(_registry_tool_names())
    assert coverage == {"missing_from_manifest": [], "missing_from_registry": []}


def test_resolver_tools_are_shared_install_tools(isolated_vpa_home: Path) -> None:
    manifest = load_manifest()
    resolver_tools = {
        "function_term_resolver",
        "go_term_search",
        "ontology_lookup",
        "interpro_entry_search",
        "interpro_entry_lookup",
    }

    assert resolver_tools <= set(manifest.tools)
    for name in resolver_tools:
        assert manifest.tools[name].runtime == "shared"

    readiness = installer_module.install_readiness(
        tuple(sorted(resolver_tools)),
        runtime_probe=False,
    )
    assert readiness["install_ready"] is True
    assert {tool["name"] for tool in readiness["tools"]} == resolver_tools


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
    assert first["env_exports"]["VPA_CUDA_PROFILE"] == first["cuda_profile"]
    assert not (isolated_vpa_home / "install" / "state.json").exists()


def test_boltz_tools_dry_run_includes_sources_and_caches(
    isolated_vpa_home: Path,
) -> None:
    installer = ManifestInstaller(load_manifest())
    options = InstallOptions(
        profile=None,
        tools=("boltz2_affinity,boltzgen",),
        install_root=isolated_vpa_home,
        dry_run=True,
    )
    plan = installer.install(options)
    units = {step["unit"] for step in plan["steps"]}

    assert "source:boltz2_affinity:boltz_source" in units
    assert "source:boltzgen:boltzgen_source" in units
    assert "weight:boltz2_affinity:boltz2_conf" in units
    assert "weight:boltzgen:boltzgen_model_cache" in units
    assert plan["env_exports"]["VPA_BOLTZ_TREE"].endswith("/sidecars/boltz")
    assert plan["env_exports"]["VPA_BOLTZ_CACHE"].endswith("/cache/boltz")
    assert plan["env_exports"]["VPA_BOLTZGEN_TREE"].endswith("/sidecars/boltzgen")
    assert plan["env_exports"]["VPA_BOLTZGEN_CACHE"].endswith("/cache/boltzgen")


def test_boltzgen_shared_deps_include_analyze_stack_requirements() -> None:
    assert "hydride" in _BOLTZGEN_RUNTIME_DEPS
    assert "biopython" in _BOLTZGEN_RUNTIME_DEPS


def test_manifest_readiness_reports_missing_required_source(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPA_BOLTZGEN_TREE", raising=False)
    monkeypatch.delenv("VPA_BOLTZGEN_CACHE", raising=False)
    manifest = load_manifest()
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )

    readiness = _manifest_readiness(
        "boltzgen",
        manifest.tools["boltzgen"],
        paths,
        required=True,
    )

    missing_units = {item["unit"] for item in readiness["missing"]}
    assert readiness["ready"] is False
    assert "source:boltzgen:boltzgen_source" in missing_units


def test_env_exports_prefer_ready_source_override(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest()
    source = isolated_vpa_home / "external" / "boltzgen"
    (source / "src" / "boltzgen" / "cli").mkdir(parents=True)
    (source / "src" / "boltzgen" / "cli" / "boltzgen.py").write_text("")
    monkeypatch.setenv("VPA_BOLTZGEN_TREE", str(source))
    selection = resolve_selection(
        manifest,
        profile=None,
        tools=("boltzgen",),
        registry_tool_names=_registry_tool_names(),
    )
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )

    exports = _env_exports(selection, paths, InstallOptions(install_root=isolated_vpa_home))

    assert exports["VPA_BOLTZGEN_TREE"] == str(source)


def test_archive_weight_download_extracts_safely(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    (payload / "mols").mkdir(parents=True)
    (payload / "mols" / "ligand.sdf").write_text("mock")
    archive = tmp_path / "mols.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(payload / "mols", arcname="mols")
    paths = InstallPaths(
        root=tmp_path / "install",
        shared_env=tmp_path / "install" / "envs" / "vpa",
        sidecars_root=tmp_path / "install" / "sidecars",
        cache_dir=tmp_path / "install" / "cache",
        state_db=tmp_path / "install" / "state.json",
    )
    weight = InstallWeight(
        name="mols",
        path="cache/boltz/mols",
        url=archive.as_uri(),
        unpack="tar",
    )
    tool = InstallTool(name="boltz2_affinity", runtime="shared")
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _ensure_weight(
        tool,
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        state,
        results,
        required=True,
    )

    assert (paths.cache_dir / "boltz" / "mols" / "ligand.sdf").is_file()
    assert results[-1]["status"] == "downloaded"


def test_vpa_env_file_loader_expands_installer_exports(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPA_BOLTZ_TREE", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    (isolated_vpa_home / "env.sh").write_text(
        "\n".join(
            [
                f"export VPA_HOME={isolated_vpa_home}",
                'export VPA_BOLTZ_TREE="$VPA_HOME/sidecars/boltz"',
                'export PATH="$VPA_HOME/envs/vpa/bin:$PATH"',
                "",
            ]
        )
    )

    load_vpa_env_file()

    assert Path(os.environ["VPA_BOLTZ_TREE"]) == isolated_vpa_home / "sidecars" / "boltz"
    assert os.environ["PATH"].startswith(str(isolated_vpa_home / "envs" / "vpa" / "bin"))


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


def test_cuda_profile_parameterizes_install_plan(isolated_vpa_home: Path) -> None:
    installer = ManifestInstaller(load_manifest())
    options = InstallOptions(
        profile=None,
        tools=("rfdiffusion3",),
        install_root=isolated_vpa_home,
        dry_run=True,
        cuda_profile="cu126",
    )
    plan = installer.install(options)

    assert plan["cuda_profile"] == "cu126"
    assert plan["cuda_version"] == "12.6"
    assert plan["env_exports"]["VPA_TORCH_SPEC"] == "torch==2.7.1+cu126"
    assert plan["env_exports"]["VPA_TORCH_INDEX_URL"].endswith("/cu126")


def test_torch_probe_must_match_selected_cuda_profile() -> None:
    cu128 = resolve_cuda_stack("cu128")
    cu126 = resolve_cuda_stack("cu126")
    cpu = resolve_cuda_stack("cpu")

    assert _torch_install_matches_stack({"version": "2.7.1+cu128", "cuda": "12.8"}, cu128)
    assert not _torch_install_matches_stack(
        {"version": "2.7.1+cu128", "cuda": "12.8"},
        cu126,
    )
    assert _torch_install_matches_stack({"version": "2.7.1+cpu", "cuda": None}, cpu)
    assert not _torch_install_matches_stack(
        {"version": "2.7.1+cu128", "cuda": "12.8"},
        cpu,
    )


def test_prebuilt_sidecar_link_replaces_empty_stale_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "prebuilt"
    (source / "bin").mkdir(parents=True)
    (source / "bin" / "python").write_text("")
    target = tmp_path / "install" / "sidecars" / "rfdiffusion3"
    target.mkdir(parents=True)
    monkeypatch.setenv("VPA_RFDIFFUSION3_ENV", str(source))

    linked = _link_prebuilt_sidecar(
        "rfdiffusion3",
        target,
        InstallOptions(use_prebuilt_sidecars=True),
    )

    assert linked == source
    assert target.is_symlink()
    assert (target / "bin" / "python").is_file()


def test_prebuilt_sidecar_link_does_not_claim_nonempty_stale_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "prebuilt"
    (source / "bin").mkdir(parents=True)
    (source / "bin" / "python").write_text("")
    target = tmp_path / "install" / "sidecars" / "rfdiffusion3"
    target.mkdir(parents=True)
    (target / "partial.log").write_text("failed")
    monkeypatch.setenv("VPA_RFDIFFUSION3_ENV", str(source))

    linked = _link_prebuilt_sidecar(
        "rfdiffusion3",
        target,
        InstallOptions(use_prebuilt_sidecars=True),
    )

    assert linked is None
    assert not target.is_symlink()


def test_manifest_readiness_accepts_prebuilt_sidecar_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_manifest()
    root = tmp_path / "install"
    source = tmp_path / "rfd3-env"
    (source / "bin").mkdir(parents=True)
    (source / "bin" / "python").write_text("")
    monkeypatch.setenv("VPA_RFDIFFUSION3_ENV", str(source))
    paths = InstallPaths(
        root=root,
        shared_env=root / "envs" / "vpa",
        sidecars_root=root / "sidecars",
        cache_dir=root / "cache",
        state_db=root / "install" / "state.json",
    )

    readiness = _manifest_readiness(
        "rfdiffusion3",
        manifest.tools["rfdiffusion3"],
        paths,
        required=True,
    )

    assert readiness["ready"] is True
    assert readiness["env_override_ready"] is True


def test_install_readiness_can_skip_duplicate_runtime_probe(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_probe(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("runtime probe should be skipped")

    monkeypatch.setattr(installer_module, "_runtime_tool_readiness", fail_probe)

    readiness = installer_module.install_readiness(
        ("go_sequences",),
        runtime_probe=False,
    )

    assert readiness["install_ready"] is True
    assert readiness["tools"][0]["ready"] is True
    assert readiness["tools"][0]["availability"]["availability_details"][
        "runtime_probe"
    ] == "skipped"

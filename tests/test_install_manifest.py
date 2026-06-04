from __future__ import annotations

import gzip
import io
import os
import json
import shlex
import shutil
import subprocess
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
    _FOLDSEEK_BINARY_ARCHIVE_MEMBER,
    _MMSEQS_BINARY_ARCHIVE_MEMBER,
    _TMALIGN_BINARY_GZIP_URL,
    _VINA_BINARY_URL,
    _capability_summary,
    _ensure_weight,
    _env_exports,
    _has_unresolved_artifact_refs,
    _link_prebuilt_sidecar,
    _manifest_readiness,
    _repair_sidecar_python_links,
    _torch_install_matches_stack,
    _write_env_exports,
)
from vibeproteinagent.install.lint import lint_manifest
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


def test_default_install_profile_is_full(isolated_vpa_home: Path) -> None:
    manifest = load_manifest()
    selection = resolve_selection(
        manifest,
        profile=None,
        tools=(),
        registry_tool_names=_registry_tool_names(),
    )

    assert selection.profile == "full"
    assert "protenix_v2" in selection.tool_names
    assert "pxdesign" in selection.tool_names


def test_manifest_lint_rejects_no_error_contracts(isolated_vpa_home: Path) -> None:
    result = lint_manifest(load_manifest())

    assert result["ok"] is True


def test_declared_sidecar_install_scripts_exist(isolated_vpa_home: Path) -> None:
    manifest = load_manifest()
    source_root = Path(__file__).resolve().parents[1]
    missing: list[str] = []
    for tool in manifest.tools.values():
        if tool.sidecar is None or not tool.sidecar.install_script:
            continue
        argv = shlex.split(tool.sidecar.install_script)
        if len(argv) < 2 or argv[0] != "bash":
            continue
        if argv[1].startswith("-"):
            continue
        script = source_root / argv[1]
        if not script.is_file():
            missing.append(f"{tool.name}: {argv[1]}")

    assert missing == []


def test_protenix_v2_manifest_has_source_and_checkpoint_contract(
    isolated_vpa_home: Path,
) -> None:
    tool = load_manifest().tools["protenix_v2"]

    assert any(source.env_var == "VPA_PROTENIX_TREE" for source in tool.sources)
    assert any(source.repo and "bytedance/Protenix" in source.repo for source in tool.sources)
    checkpoint = next(weight for weight in tool.weights if weight.name == "protenix_checkpoint")
    assert checkpoint.artifact_ref
    assert checkpoint.url
    assert checkpoint.source_path is None
    assert checkpoint.env_var == "VPA_PROTENIX_CKPT"
    ccd_cache = next(weight for weight in tool.weights if weight.name == "protenix_ccd_cache")
    assert ccd_cache.artifact_ref
    assert ccd_cache.url
    assert ccd_cache.unpack == "tar"
    assert ccd_cache.source_path is None


def test_remote_fallback_artifacts_do_not_force_prebuilt_skip(
    isolated_vpa_home: Path,
) -> None:
    manifest = load_manifest()

    assert _has_unresolved_artifact_refs(
        manifest.tools["bindcraft"],
        {},
        cuda_profile="cu128",
    )
    assert not _has_unresolved_artifact_refs(
        manifest.tools["boltz2_affinity"],
        {},
        cuda_profile="cu128",
    )
    assert not _has_unresolved_artifact_refs(
        manifest.tools["protenix_v2"],
        {},
        cuda_profile="cu128",
    )


def test_retired_and_optional_tools_do_not_block_default_doctor(
    isolated_vpa_home: Path,
) -> None:
    manifest = load_manifest()
    assert manifest.tools["esm3"].required is False
    assert manifest.tools["proteina_complexa"].required is False
    assert manifest.tools["pyrosetta"].artifact_policy == "redistributable"
    assert "license_key" not in manifest.tools["pyrosetta"].access_requirements
    assert manifest.tools["pxdesign"].sidecar is not None


def test_pyrosetta_install_is_not_license_or_gated_policy_blocked(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPA_PYROSETTA_LICENSE_KEY", raising=False)
    monkeypatch.delenv("PYROSETTA_LICENSE_KEY", raising=False)
    installer = ManifestInstaller(load_manifest())

    plan = installer.install(
        InstallOptions(
            tools=("pyrosetta",),
            install_root=isolated_vpa_home,
            dry_run=True,
        )
    )

    assert plan["capabilities"]["needs_credential"] == []
    assert plan["capabilities"]["ready"] == [
        {
            "tool": "pyrosetta",
            "policy": "redistributable",
            "install_method": "shared_runtime",
        }
    ]
    assert plan["license_notices"] == [
        {
            "tool": "pyrosetta",
            "notice": (
                "Installs without a VPA license key; follow upstream "
                "PyRosetta/Rosetta Commons terms. Upstream states commercial "
                "use requires a separate license."
            ),
            "url": "https://www.pyrosetta.org/downloads",
        }
    ]


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


def test_artifact_index_resolves_local_weight_contract(
    isolated_vpa_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pt"
    source.write_text("mock-checkpoint")
    index = tmp_path / "artifacts.json"
    index.write_text(
        json.dumps(
            {"artifacts": {"mock/protenix.pt": {"kind": "file", "path": str(source)}}}
        )
    )
    monkeypatch.setenv("VPA_ARTIFACT_INDEX", str(index))
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )
    weight = InstallWeight(
        name="checkpoint",
        path="cache/protenix/protenix_base_default_v1.0.0.pt",
        artifact_ref="mock/protenix.pt",
    )
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _ensure_weight(
        InstallTool(name="protenix_v2", runtime="shared"),
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        state,
        results,
        required=True,
        artifact_index=installer_module.load_artifact_index(),
    )

    assert (
        paths.cache_dir / "protenix" / "protenix_base_default_v1.0.0.pt"
    ).read_text() == "mock-checkpoint"
    assert results[-1]["status"] == "installed"


def test_foldseek_weight_downloads_public_binary_archive(
    isolated_vpa_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "payload"
    binary = payload / _FOLDSEEK_BINARY_ARCHIVE_MEMBER
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho foldseek\n")
    archive = tmp_path / "foldseek-linux-gpu.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload / "foldseek", arcname="foldseek")

    def fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, object]:
        shutil.copy2(archive, filename)
        return str(filename), None

    monkeypatch.setattr(installer_module.urllib.request, "urlretrieve", fake_urlretrieve)
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )
    weight = InstallWeight(
        name="foldseek_binary",
        path="envs/vpa/bin/foldseek",
        artifact_ref="external_binaries/foldseek",
        source_path="/missing/foldseek",
    )
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _ensure_weight(
        InstallTool(name="foldseek", runtime="external_binary", binary="foldseek"),
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        state,
        results,
        required=True,
        artifact_index={},
    )

    installed = paths.shared_env / "bin" / "foldseek"
    assert installed.read_text() == "#!/bin/sh\necho foldseek\n"
    assert installed.stat().st_mode & 0o111
    assert results[-1]["status"] == "downloaded"


def test_mmseqs_weight_downloads_public_binary_archive(
    isolated_vpa_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = tmp_path / "payload"
    binary = payload / _MMSEQS_BINARY_ARCHIVE_MEMBER
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho mmseqs\n")
    archive = tmp_path / "mmseqs-linux-gpu.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(payload / "mmseqs", arcname="mmseqs")

    def fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, object]:
        shutil.copy2(archive, filename)
        return str(filename), None

    monkeypatch.setattr(installer_module.urllib.request, "urlretrieve", fake_urlretrieve)
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )
    weight = InstallWeight(
        name="mmseqs_binary",
        path="envs/vpa/bin/mmseqs",
        artifact_ref="external_binaries/mmseqs",
        source_path="/missing/mmseqs",
    )
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _ensure_weight(
        InstallTool(name="mmseqs2_gpu", runtime="external_binary", binary="mmseqs"),
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        state,
        results,
        required=True,
        artifact_index={},
    )

    installed = paths.shared_env / "bin" / "mmseqs"
    assert installed.read_text() == "#!/bin/sh\necho mmseqs\n"
    assert installed.stat().st_mode & 0o111
    assert results[-1]["status"] == "downloaded"


def test_tmalign_weight_downloads_public_gzip_binary(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = gzip.compress(b"#!/bin/sh\necho TMalign\n")

    def fake_open_download_url(url: str) -> io.BytesIO:
        assert url == _TMALIGN_BINARY_GZIP_URL
        return io.BytesIO(payload)

    monkeypatch.setattr(installer_module, "_open_download_url", fake_open_download_url)
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )
    weight = InstallWeight(
        name="tmalign_binary",
        path="envs/vpa/bin/TMalign",
        artifact_ref="external_binaries/TMalign",
        source_path="/missing/TMalign",
    )
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _ensure_weight(
        InstallTool(name="tmalign", runtime="external_binary", binary="TMalign"),
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        state,
        results,
        required=True,
        artifact_index={},
    )

    installed = paths.shared_env / "bin" / "TMalign"
    assert installed.read_text() == "#!/bin/sh\necho TMalign\n"
    assert installed.stat().st_mode & 0o111
    assert results[-1]["status"] == "downloaded"


def test_vina_weight_downloads_public_release_binary(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"#!/bin/sh\necho vina\n"

    def fake_open_download_url(url: str) -> io.BytesIO:
        assert url == _VINA_BINARY_URL
        return io.BytesIO(payload)

    monkeypatch.setattr(installer_module, "_open_download_url", fake_open_download_url)
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )
    weight = InstallWeight(
        name="vina_binary",
        path="envs/vpa/bin/vina",
        artifact_ref="external_binaries/vina",
        source_path="/missing/vina",
    )
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _ensure_weight(
        InstallTool(name="vina", runtime="external_binary", binary="vina"),
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        state,
        results,
        required=True,
        artifact_index={},
    )

    installed = paths.shared_env / "bin" / "vina"
    assert installed.read_text() == "#!/bin/sh\necho vina\n"
    assert installed.stat().st_mode & 0o111
    assert results[-1]["status"] == "downloaded"


def test_pxdesign_weight_downloads_public_cache_fallback(
    isolated_vpa_home: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "alphafold_params_2022-12-06.tar"
    payload = tmp_path / "af2"
    payload.mkdir()
    for name in ("params_model_1.npz", "params_model_1_ptm.npz"):
        (payload / name).write_text(name)
    with tarfile.open(archive, "w") as tf:
        for file in payload.iterdir():
            tf.add(file, arcname=file.name)

    def fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, object]:
        target = Path(filename)
        if "alphafold_params_2022-12-06.tar" in url:
            shutil.copy2(archive, target)
        else:
            target.write_bytes(f"mock from {url}".encode())
        return str(target), None

    monkeypatch.setattr(installer_module.urllib.request, "urlretrieve", fake_urlretrieve)
    paths = InstallPaths(
        root=isolated_vpa_home,
        shared_env=isolated_vpa_home / "envs" / "vpa",
        sidecars_root=isolated_vpa_home / "sidecars",
        cache_dir=isolated_vpa_home / "cache",
        state_db=isolated_vpa_home / "install" / "state.json",
    )
    weight = InstallWeight(
        name="pxdesign_weights",
        path="cache/pxdesign",
        artifact_ref="pxdesign/checkpoints",
        source_path="/missing/pxdesign",
    )
    results: list[dict[str, object]] = []

    _ensure_weight(
        InstallTool(name="pxdesign", runtime="sidecar"),
        weight,
        paths,
        InstallOptions(install_root=paths.root),
        InstallState(paths.state_db),
        results,
        required=True,
        artifact_index={},
    )

    cache = paths.cache_dir / "pxdesign"
    assert (cache / "pxdesign_v0.1.0.pt").is_file()
    assert (cache / "protenix_base_default_v0.5.0.pt").is_file()
    assert (cache / "tool_weights" / "af2" / "params_model_1.npz").is_file()
    assert (cache / "tool_weights" / "af2" / "params_model_1_ptm.npz").is_file()
    assert results[-1]["status"] == "downloaded"


def test_esm3_dependency_install_preserves_existing_torch(
    isolated_vpa_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probes = [
        subprocess.CompletedProcess(["probe"], 1, stdout="", stderr="missing esm"),
        subprocess.CompletedProcess(["verify"], 0, stdout="ok\n", stderr=""),
    ]
    install_commands: list[list[str]] = []

    def fake_subprocess_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return probes.pop(0)

    def fake_run(cmd: list[str], *, cwd: Path | None = None) -> None:
        install_commands.append(cmd)

    monkeypatch.setattr(installer_module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(installer_module, "_run", fake_run)

    state = InstallState(isolated_vpa_home / "install" / "state.json")
    results: list[dict[str, object]] = []
    installer_module._ensure_esm3_shared_deps(
        isolated_vpa_home / "envs" / "vpa" / "bin" / "python",
        InstallOptions(install_root=isolated_vpa_home),
        state,
        results,
    )

    esm_commands = [cmd for cmd in install_commands if "esm==3.2.1.post1" in cmd]
    assert len(esm_commands) == 1
    assert "--no-deps" in esm_commands[0]
    assert not any(arg.startswith("torch") for cmd in install_commands for arg in cmd)
    assert results[-1]["status"] == "installed"


def test_dry_run_reports_source_fallback_as_method_not_terminal_state(
    isolated_vpa_home: Path,
) -> None:
    installer = ManifestInstaller(load_manifest())

    plan = installer.install(
        InstallOptions(
            tools=("bindcraft",),
            install_root=isolated_vpa_home,
            dry_run=True,
        )
    )

    assert "will_build_from_source" not in plan["capabilities"]
    assert plan["capabilities"]["ready"] == [
        {
            "tool": "bindcraft",
            "policy": "redistributable",
            "install_method": "source_fallback",
        }
    ]


def test_capability_summary_classifies_external_access_skip_without_source_build_state(
    isolated_vpa_home: Path,
) -> None:
    manifest = load_manifest()
    selection = resolve_selection(
        manifest,
        profile=None,
        tools=("esm3",),
        registry_tool_names=_registry_tool_names(),
    )
    results: list[dict[str, object]] = [
        {
            "unit": "tool:esm3",
            "status": "skipped",
            "runtime": "shared",
            "required": False,
            "reason": "tool requires HF_TOKEN or VPA_HF_CREDENTIAL",
        },
    ]

    summary = _capability_summary(results, selection)

    assert "will_build_from_source" not in summary
    assert summary["needs_credential"] == [
        {"tool": "esm3", "requirements": ["hf_token"]}
    ]
    assert summary["failed"] == []


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


def test_generated_env_file_puts_installed_cli_on_path(tmp_path: Path) -> None:
    manifest = load_manifest()
    paths = InstallPaths(
        root=tmp_path / "install",
        shared_env=tmp_path / "install" / "envs" / "vpa",
        sidecars_root=tmp_path / "install" / "sidecars",
        cache_dir=tmp_path / "install" / "cache",
        state_db=tmp_path / "install" / "state.json",
    )
    shared_bin = paths.shared_env / "bin"
    shared_bin.mkdir(parents=True)
    (shared_bin / "python").write_text("")
    vpa_bin = shared_bin / "vpa"
    vpa_bin.write_text("#!/usr/bin/env bash\n")
    vpa_bin.chmod(0o755)
    selection = resolve_selection(manifest, profile=None, tools=("protein_qc",))
    state = InstallState(paths.state_db)
    results: list[dict[str, object]] = []

    _write_env_exports(selection, paths, InstallOptions(install_root=paths.root), state, results)

    probe = subprocess.run(
        [
            "bash",
            "-c",
            f"PATH=/usr/bin; source {paths.root / 'env.sh'}; command -v vpa",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert probe.stdout.strip() == str(vpa_bin)


def test_sidecar_python_repair_removes_external_absolute_symlink(tmp_path: Path) -> None:
    paths = InstallPaths(
        root=tmp_path / "install",
        shared_env=tmp_path / "install" / "envs" / "vpa",
        sidecars_root=tmp_path / "install" / "sidecars",
        cache_dir=tmp_path / "install" / "cache",
        state_db=tmp_path / "install" / "state.json",
    )
    shared_bin = paths.shared_env / "bin"
    shared_bin.mkdir(parents=True)
    (shared_bin / "python").write_text("")
    sidecar_bin = paths.sidecars_root / "tool" / ".venv" / "bin"
    sidecar_bin.mkdir(parents=True)
    external = tmp_path / "external" / "python"
    external.parent.mkdir()
    external.write_text("")
    python_link = sidecar_bin / "python"
    python_link.symlink_to(external)
    pyvenv = sidecar_bin.parent / "pyvenv.cfg"
    pyvenv.write_text(f"home = {external.parent}\ninclude-system-site-packages = false\n")
    results: list[dict[str, object]] = []

    _repair_sidecar_python_links(paths, paths.sidecars_root / "tool", results)

    assert python_link.is_symlink()
    assert not os.readlink(python_link).startswith("/")
    assert python_link.resolve() == (shared_bin / "python").resolve()
    assert f"home = {shared_bin}" in pyvenv.read_text()
    assert results[-1]["status"] == "repaired"


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

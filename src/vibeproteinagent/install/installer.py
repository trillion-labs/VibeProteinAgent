from __future__ import annotations

import hashlib
import json
import os
import os.path
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vibeproteinagent.adapters.visibility import is_retired_registry_tool_name
from vibeproteinagent.core import config as cfg
from vibeproteinagent.core.errors import DependencyError
from vibeproteinagent.install.cuda import (
    GPU_HEAVY_TOOLS,
    RF_DGL_TOOLS,
    CudaStack,
    preflight_report,
    resolve_cuda_stack,
)
from vibeproteinagent.install.manifest import (
    InstallManifest,
    InstallSource,
    InstallTool,
    InstallWeight,
    load_manifest,
)
from vibeproteinagent.install.planner import InstallSelection, resolve_selection
from vibeproteinagent.install.state import InstallPaths, InstallState, install_lock


@dataclass(frozen=True)
class InstallOptions:
    profile: str | None = None
    tools: tuple[str, ...] = ()
    install_root: Path | None = None
    cache_dir: Path | None = None
    dry_run: bool = False
    yes: bool = False
    force: bool = False
    repair: bool = False
    offline: bool = False
    source_root: Path | None = None
    cuda_profile: str | None = None
    use_prebuilt_sidecars: bool = False


class ManifestInstaller:
    def __init__(self, manifest: InstallManifest) -> None:
        self.manifest = manifest

    def resolve(self, options: InstallOptions) -> tuple[InstallPaths, InstallSelection]:
        registry_names = _registry_tool_names()
        selection = resolve_selection(
            self.manifest,
            profile=options.profile,
            tools=options.tools,
            registry_tool_names=registry_names,
        )
        paths = _resolve_paths(self.manifest, options)
        return paths, selection

    def plan(self, options: InstallOptions) -> dict[str, Any]:
        paths, selection = self.resolve(options)
        return _plan_payload(paths, selection, options)

    def install(self, options: InstallOptions) -> dict[str, Any]:
        paths, selection = self.resolve(options)
        if options.dry_run:
            payload = _plan_payload(paths, selection, options)
            payload["dry_run"] = True
            return payload

        paths.root.mkdir(parents=True, exist_ok=True)
        with install_lock(paths.root):
            paths.shared_env.parent.mkdir(parents=True, exist_ok=True)
            paths.sidecars_root.mkdir(parents=True, exist_ok=True)
            paths.cache_dir.mkdir(parents=True, exist_ok=True)
            state = InstallState(paths.state_db)
            results: list[dict[str, Any]] = []
            _ensure_shared_env(paths, options, state, results)
            _ensure_vpa_install(paths, options, state, results)
            _ensure_shared_runtime_deps(paths, selection, options, state, results)
            _write_env_exports(selection, paths, options, state, results)
            exports = _env_exports(selection, paths, options)
            old_env = _apply_install_env(paths, exports)
            try:
                for tool in selection.tools:
                    _ensure_tool_units(
                        tool,
                        paths,
                        options,
                        state,
                        results,
                        required=_is_strict_tool(tool, options),
                    )
                for tool in selection.tools:
                    _ensure_tool_readiness(
                        tool,
                        paths,
                        state,
                        results,
                        required=_is_strict_tool(tool, options),
                    )
            finally:
                _restore_env(old_env)
            state.save()
        status = _overall_status(results)
        return {
            "status": status,
            "profile": selection.profile,
            "cuda_profile": resolve_cuda_stack(options.cuda_profile).name,
            "selected_tools": list(selection.tool_names),
            "install_root": str(paths.root),
            "shared_env": str(paths.shared_env),
            "sidecars_root": str(paths.sidecars_root),
            "cache_dir": str(paths.cache_dir),
            "results": results,
            "next_command": _next_doctor_command(paths, selection),
        }


def install_readiness(
    tool_names: tuple[str, ...] = (),
    *,
    runtime_probe: bool = True,
) -> dict[str, Any]:
    manifest = load_manifest()
    paths = _resolve_paths(manifest, InstallOptions())
    shared_python = paths.shared_env / "bin" / "python"
    runtime_python = shared_python if shared_python.is_file() else None
    selection = resolve_selection(
        manifest,
        profile=None,
        tools=tool_names,
        registry_tool_names=_registry_tool_names(),
    ) if tool_names else None
    names = (
        selection.tool_names
        if selection
        else tuple(sorted(manifest.tool_names))
    )
    tools: list[dict[str, Any]] = []
    export_selection = selection or InstallSelection(
        profile=None,
        requested_tools=(),
        tools=tuple(manifest.tools[name] for name in sorted(manifest.tools)),
        registry_coverage={},
    )
    readiness_exports = _env_exports(export_selection, paths)
    for key in tuple(readiness_exports):
        if os.environ.get(key):
            readiness_exports[key] = os.environ[key]
    old_env = _apply_install_env(paths, readiness_exports)
    try:
        for name in names:
            spec = manifest.tools.get(name)
            effective_required = bool(spec.required if spec else True)
            if tool_names and spec is not None:
                effective_required = True
            runtime_status = (
                _runtime_tool_readiness(name, python=runtime_python)
                if runtime_probe
                else {
                    "ready": True,
                    "reason": None,
                    "availability": {
                        "available": True,
                        "availability_checked": False,
                        "availability_reason": None,
                        "availability_details": {
                            "runtime_probe": "skipped",
                            "reason": (
                                "caller already checked descriptor availability"
                            ),
                        },
                    },
                }
            )
            availability = runtime_status.get("availability") or {}
            manifest_status = _manifest_readiness(name, spec, paths, required=effective_required)
            runtime_ready = bool(runtime_status.get("ready"))
            ready = runtime_ready and manifest_status["ready"]
            reason = _combined_reason(
                runtime_status.get("reason"),
                None if ready else manifest_status.get("reason"),
            )
            checks = availability.get("availability_details") or availability.get("details") or {}
            entry = {
                "name": name,
                "ready": ready,
                "reason": reason,
                "runtime": spec.runtime if spec else None,
                "required": effective_required,
                "availability": availability,
                "manifest": manifest_status,
            }
            if name == "rfdiffusion":
                entry["rfdiffusion_modes"] = {
                    "full_mode_set_available": checks.get("full_mode_set_available"),
                    "missing_mode_checkpoints": checks.get("missing_mode_checkpoints", []),
                }
            tools.append(entry)
    finally:
        _restore_env(old_env)
    return {
        "install_ready": all(item.get("ready") or item.get("required") is False for item in tools),
        "tools": tools,
    }


def install_preflight(
    *,
    profile: str | None = None,
    tools: tuple[str, ...] = (),
    install_root: Path | None = None,
    cache_dir: Path | None = None,
    cuda_profile: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest()
    installer = ManifestInstaller(manifest)
    paths, selection = installer.resolve(
        InstallOptions(
            profile=profile,
            tools=tools,
            install_root=install_root,
            cache_dir=cache_dir,
            cuda_profile=cuda_profile,
        )
    )
    data = preflight_report(
        requested_profile=cuda_profile,
        selected_tools=tuple(selection.tool_names),
    )
    data.update(
        {
            "profile": selection.profile,
            "install_root": str(paths.root),
            "shared_env": str(paths.shared_env),
            "sidecars_root": str(paths.sidecars_root),
            "cache_dir": str(paths.cache_dir),
        }
    )
    return data


def _registry_tool_names() -> set[str]:
    names: set[str] = set()
    adapters_root = Path(__file__).resolve().parents[1] / "adapters"
    for child in adapters_root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        descriptor = child / "descriptor.yaml"
        if not descriptor.is_file():
            continue
        try:
            raw = yaml.safe_load(descriptor.read_text())
        except Exception:
            continue
        if isinstance(raw, dict) and raw.get("name"):
            name = str(raw["name"])
            if not is_retired_registry_tool_name(name):
                names.add(name)
    return names


def _resolve_paths(manifest: InstallManifest, options: InstallOptions) -> InstallPaths:
    root = options.install_root or Path(
        os.environ.get(cfg.VPA_HOME_ENV, manifest.defaults.install_root)
    )
    root = root.expanduser()
    cache = (
        options.cache_dir.expanduser()
        if options.cache_dir
        else root / manifest.defaults.cache_dir
    )
    return InstallPaths(
        root=root,
        shared_env=root / manifest.defaults.shared_env,
        sidecars_root=root / manifest.defaults.sidecars_root,
        cache_dir=cache,
        state_db=root / manifest.defaults.state_db,
    )


def _plan_payload(
    paths: InstallPaths,
    selection: InstallSelection,
    options: InstallOptions,
) -> dict[str, Any]:
    cuda_stack = resolve_cuda_stack(options.cuda_profile)
    steps = [
        {"unit": "bootstrap:micromamba", "action": "ensure micromamba is available"},
        {"unit": "env:vpa", "action": f"create shared env at {paths.shared_env}"},
        {"unit": "package:vibeproteinagent", "action": "install VPA into shared env"},
        {
            "unit": "cuda:profile",
            "action": f"use {cuda_stack.name} ({cuda_stack.summary})",
        },
    ]
    for tool in selection.tools:
        steps.extend(_tool_steps(tool, paths))
    exports = _env_exports(selection, paths, options)
    steps.append(
        {
            "unit": "env:vpa-activation",
            "action": (
                "write VPA_HOME and adapter cache exports into the shared env "
                "activation hook"
            ),
        }
    )
    warnings = []
    coverage = selection.registry_coverage
    if coverage["missing_from_manifest"] or coverage["missing_from_registry"]:
        warnings.append({"type": "registry_manifest_drift", **coverage})
    warnings.extend(_credential_warnings(paths, selection))
    warnings.extend(_cuda_profile_warnings(selection, cuda_stack))
    return {
        "dry_run": options.dry_run,
        "profile": selection.profile,
        "requested_cuda_profile": (
            options.cuda_profile or os.environ.get("VPA_CUDA_PROFILE") or "auto"
        ),
        "cuda_profile": cuda_stack.name,
        "cuda_version": cuda_stack.cuda_version,
        "requested_tools": list(selection.requested_tools),
        "selected_tools": list(selection.tool_names),
        "install_root": str(paths.root),
        "shared_env": str(paths.shared_env),
        "sidecars_root": str(paths.sidecars_root),
        "cache_dir": str(paths.cache_dir),
        "state_db": str(paths.state_db),
        "disk_estimate_bytes": sum(tool.disk_bytes for tool in selection.tools),
        "steps": steps,
        "warnings": warnings,
        "env_exports": exports,
        "next_command": _next_doctor_command(paths, selection),
    }


def _cuda_profile_warnings(
    selection: InstallSelection,
    cuda_stack: CudaStack,
) -> list[dict[str, Any]]:
    selected = set(selection.tool_names)
    warnings: list[dict[str, Any]] = []
    if cuda_stack.name == "cpu" and selected.intersection(GPU_HEAVY_TOOLS):
        warnings.append(
            {
                "type": "cpu_cuda_profile_with_gpu_tools",
                "cuda_profile": cuda_stack.name,
                "message": "GPU-heavy tools are selected under the CPU CUDA profile.",
            }
        )
    if selected.intersection(RF_DGL_TOOLS):
        warnings.append(
            {
                "type": "rfdiffusion_dgl_best_effort",
                "cuda_profile": cuda_stack.name,
                "message": (
                    "RFdiffusion/RFpeptides use a best-effort DGL CUDA wheel "
                    "because DGL does not publish a first-class cu128 wheel for "
                    "the primary VPA stack."
                ),
            }
        )
    return warnings


def _credential_warnings(
    paths: InstallPaths,
    selection: InstallSelection,
) -> list[dict[str, Any]]:
    selected = set(selection.tool_names)
    warnings: list[dict[str, Any]] = []
    if "esm3" in selected and not _hf_credential_present() and not _esm3_open_cache_present(paths):
        warnings.append(
            {
                "type": "credential_optional_for_esm3",
                "tool": "esm3",
                "env": ["HF_TOKEN", "VPA_HF_CREDENTIAL"],
                "message": (
                    "ESM3 is selected. Install can proceed, but first-run ESM3 downloads "
                    "or gated variants may need HF_TOKEN (or VPA_HF_CREDENTIAL) plus "
                    "accepted Hugging Face model terms."
                ),
            }
        )
    if selection.profile == "agent":
        warnings.append(
            {
                "type": "agent_provider_credentials_runtime_only",
                "env": ["OPENROUTER_API_KEY", "VPA_OPENROUTER_CREDENTIAL"],
                "message": (
                    "The agent profile does not require OpenRouter credentials to install. "
                    "Set OPENROUTER_API_KEY (or VPA_OPENROUTER_CREDENTIAL) only when "
                    "running with --provider openrouter."
                ),
            }
        )
    return warnings


def _hf_credential_present() -> bool:
    return any(
        os.environ.get(name)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VPA_HF_CREDENTIAL")
    )


def _esm3_open_cache_present(paths: InstallPaths) -> bool:
    cache = (
        paths.cache_dir
        / "huggingface"
        / "hub"
        / "models--EvolutionaryScale--esm3-sm-open-v1"
        / "snapshots"
    )
    return cache.is_dir() and any(cache.iterdir())


def _tool_steps(tool: InstallTool, paths: InstallPaths) -> list[dict[str, str]]:
    steps = [{"unit": f"tool:{tool.name}", "action": f"check {tool.runtime} readiness"}]
    if tool.name == "boltz2_affinity":
        steps.append(
            {
                "unit": "shared-dep:boltz2_affinity",
                "action": "ensure Boltz-2 runtime Python dependencies in the shared env",
            }
        )
    if tool.name == "boltzgen":
        steps.append(
            {
                "unit": "shared-dep:boltzgen",
                "action": "ensure BoltzGen runtime Python dependencies in the shared env",
            }
        )
    for source in tool.sources:
        steps.append(
            {
                "unit": f"source:{tool.name}:{source.name}",
                "action": f"ensure source checkout {_source_path(paths, source)}",
            }
        )
    if tool.sidecar:
        steps.append(
            {
                "unit": f"sidecar:{tool.name}",
                "action": f"ensure sidecar env {paths.root / tool.sidecar.path}",
            }
        )
    for weight in tool.weights:
        steps.append(
            {
                "unit": f"weight:{tool.name}:{weight.name}",
                "action": f"ensure weight/cache {_weight_path(paths, weight)}",
            }
        )
    if tool.binary:
        steps.append({"unit": f"binary:{tool.name}", "action": f"check binary {tool.binary}"})
    return steps


def _next_doctor_command(paths: InstallPaths, selection: InstallSelection) -> str:
    tools = ",".join(selection.tool_names)
    return f"{paths.shared_env / 'bin' / 'vpa'} install doctor --tools {shlex.quote(tools)} --json"


def _ensure_shared_env(
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    python = paths.shared_env / "bin" / "python"
    if python.is_file() and not options.force:
        state.mark("env:vpa", "ready", {"python": str(python)})
        results.append({"unit": "env:vpa", "status": "ready", "path": str(paths.shared_env)})
        return
    micromamba = _find_micromamba()
    if micromamba is None:
        raise DependencyError(
            "micromamba is required to create the VPA shared env",
            details={
                "env": str(paths.shared_env),
                "hint": "set VPA_MICROMAMBA or install micromamba",
            },
        )
    _run(
        [
            str(micromamba),
            "create",
            "-y",
            "-p",
            str(paths.shared_env),
            "python=3.12",
            "pip",
        ],
        cwd=None,
    )
    state.mark("env:vpa", "ready", {"python": str(python)})
    results.append({"unit": "env:vpa", "status": "installed", "path": str(paths.shared_env)})


def _ensure_vpa_install(
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    python = paths.shared_env / "bin" / "python"
    source_root = options.source_root or _source_root()
    if not (source_root / "pyproject.toml").is_file():
        raise DependencyError(
            "could not locate VPA source root for editable install",
            details={"source_root": str(source_root)},
        )
    _run(
        [str(python), "-m", "pip", "install", "--no-user", "-e", str(source_root)],
        cwd=source_root,
    )
    state.mark("package:vibeproteinagent", "ready", {"source_root": str(source_root)})
    results.append(
        {"unit": "package:vibeproteinagent", "status": "installed", "source_root": str(source_root)}
    )


_TORCH_TOOLS = frozenset(
    {
        "bindcraft",
        "boltz2_affinity",
        "boltzgen",
        "colabdesign_utils",
        "esm2",
        "esm3",
        "go_sequences",
        "interpro_sequences",
        "ipsae",
        "ligandmpnn",
        "literature_search",
        "prodigy",
        "proteinmpnn",
        "proteina_complexa",
        "protenix_v2",
        "pxdesign",
        "pxmeter",
        "pyrosetta",
        "rfdiffusion",
        "rfdiffusion3",
        "rfpeptides",
        "tmalign",
        "vina",
    }
)


def _ensure_shared_runtime_deps(
    paths: InstallPaths,
    selection: InstallSelection,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    selected_torch_tools = sorted(set(selection.tool_names).intersection(_TORCH_TOOLS))
    if not selected_torch_tools:
        return
    cuda_stack = resolve_cuda_stack(options.cuda_profile)
    python = paths.shared_env / "bin" / "python"
    probe = _probe_torch_install(python)
    if probe.get("ok") and not options.force and _torch_install_matches_stack(probe, cuda_stack):
        version = str(probe["version"])
        results.append(
            {
                "unit": "shared-dep:torch",
                "status": "ready",
                "version": version,
                "cuda_profile": cuda_stack.name,
            }
        )
        state.mark(
            "shared-dep:torch",
            "ready",
            {"version": version, "cuda_profile": cuda_stack.name},
        )
        _ensure_selected_shared_deps(python, selection, options, state, results)
        return
    if options.offline:
        raise DependencyError(
            "torch is required for selected tools but is not installed and --offline was set",
            details={"tools": selected_torch_tools},
        )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            cuda_stack.torch_spec,
            "--index-url",
            cuda_stack.torch_index_url,
        ],
        cwd=None,
    )
    verify = _probe_torch_install(python)
    if not verify.get("ok"):
        raise DependencyError(
            "torch installation did not become importable in the shared VPA env",
            details={"stderr": verify.get("stderr"), "stdout": verify.get("stdout")},
        )
    if not _torch_install_matches_stack(verify, cuda_stack):
        raise DependencyError(
            "torch installation does not match the selected CUDA profile",
            details={
                "installed": verify,
                "cuda_profile": cuda_stack.name,
                "expected_torch": cuda_stack.torch_spec,
            },
        )
    version = str(verify["version"])
    results.append(
        {
            "unit": "shared-dep:torch",
            "status": "installed",
            "version": version,
            "cuda_profile": cuda_stack.name,
        }
    )
    state.mark("shared-dep:torch", "ready", {"version": version, "cuda_profile": cuda_stack.name})
    _ensure_selected_shared_deps(python, selection, options, state, results)


def _probe_torch_install(python: Path) -> dict[str, Any]:
    code = (
        "import json, torch; "
        "print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda}))"
    )
    probe = subprocess.run(
        [str(python), "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        return {
            "ok": False,
            "stdout": probe.stdout[-2000:],
            "stderr": probe.stderr[-2000:],
        }
    try:
        info = json.loads(probe.stdout.strip())
    except json.JSONDecodeError:
        return {
            "ok": False,
            "stdout": probe.stdout[-2000:],
            "stderr": "torch probe did not emit JSON",
        }
    return {"ok": True, **info}


def _torch_install_matches_stack(probe: dict[str, Any], stack: CudaStack) -> bool:
    version = str(probe.get("version") or "")
    cuda = probe.get("cuda")
    if stack.name == "cpu":
        return version.startswith("2.7.1") and not cuda
    expected_suffix = f"+{stack.name}"
    return version == stack.torch_spec.removeprefix("torch==") and version.endswith(
        expected_suffix
    )


def _ensure_selected_shared_deps(
    python: Path,
    selection: InstallSelection,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    if "boltz2_affinity" in selection.tool_names:
        _ensure_boltz2_affinity_shared_deps(python, options, state, results)
    if "boltzgen" in selection.tool_names:
        _ensure_boltzgen_shared_deps(python, options, state, results)
    if "esm3" in selection.tool_names:
        _ensure_esm3_shared_deps(python, options, state, results)
    if "ligandmpnn" in selection.tool_names:
        _ensure_ligandmpnn_shared_deps(python, options, state, results)
    if "prodigy" in selection.tool_names:
        _ensure_prodigy_shared_deps(python, options, state, results)
    if "proteina_complexa" in selection.tool_names:
        _ensure_proteina_complexa_shared_deps(python, options, state, results)
    if "pyrosetta" in selection.tool_names:
        _ensure_pyrosetta_shared_deps(python, options, state, results)
    if {"rfdiffusion", "rfpeptides"}.intersection(selection.tool_names):
        _ensure_rfdiffusion_shared_deps(python, options, state, results)


def _ensure_boltz2_affinity_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe_code = (
        "import hydra, pytorch_lightning, tree, einx, fairscale, mashumaro, "
        "modelcif, wandb, numba, sklearn, chembl_structure_pipeline; "
        "from rdkit import Chem; "
        "import numpy as np; "
        "assert int(np.__version__.split('.')[0]) < 2; "
        "print('ok')"
    )
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:boltz2_affinity", "status": "ready"})
        state.mark("shared-dep:boltz2_affinity", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "Boltz-2 affinity dependencies are required but are not installed and "
            "--offline was set",
            details={
                "packages": _BOLTZ2_AFFINITY_SHARED_DEPS,
                "stderr": probe.stderr[-2000:],
                "stdout": probe.stdout[-2000:],
            },
        )
    # Boltz is pinned to NumPy 1.x. If the shared env previously had NumPy 2.x,
    # pip can leave stale dist-info/extension state behind when resolving the
    # larger dependency set in one pass. Repair the ABI-sensitive pair first,
    # then install the remaining pure/runtime deps.
    subprocess.run(
        [str(python), "-m", "pip", "uninstall", "-y", "numpy", "scipy"],
        text=True,
        capture_output=True,
        check=False,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            "--no-cache-dir",
            "--force-reinstall",
            *_BOLTZ2_AFFINITY_NUMERIC_DEPS,
        ],
        cwd=None,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            *_BOLTZ2_AFFINITY_RUNTIME_DEPS,
        ],
        cwd=None,
    )
    verify = subprocess.run(
        [str(python), "-c", probe_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "Boltz-2 affinity shared dependency installation did not make Boltz "
            "runtime imports available",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:boltz2_affinity", "status": "installed"})
    state.mark("shared-dep:boltz2_affinity", "ready", {})


_BOLTZ2_AFFINITY_NUMERIC_DEPS = (
    "numpy>=1.26,<2.0",
    "scipy==1.13.1",
)

_BOLTZ2_AFFINITY_RUNTIME_DEPS = (
    "hydra-core==1.3.2",
    "pytorch-lightning==2.5.0",
    "rdkit>=2024.3.2",
    "dm-tree==0.1.8",
    "einx==0.3.0",
    "fairscale==0.4.13",
    "mashumaro==3.14",
    "modelcif==1.2",
    "wandb==0.18.7",
    "numba==0.61.0",
    "scikit-learn==1.6.1",
    "chembl_structure_pipeline==1.2.2",
)

_BOLTZ2_AFFINITY_SHARED_DEPS = (
    *_BOLTZ2_AFFINITY_NUMERIC_DEPS,
    *_BOLTZ2_AFFINITY_RUNTIME_DEPS,
)


def _ensure_boltzgen_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe_code = (
        "import biotite, edit_distance, gemmi, huggingface_hub, hydra, hydride, "
        "logomaker, matplotlib, pandas, pdbeccdutils, pydssp, pytorch_lightning, "
        "sklearn; "
        "print('ok')"
    )
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:boltzgen", "status": "ready"})
        state.mark("shared-dep:boltzgen", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "BoltzGen dependencies are required but are not installed and --offline was set",
            details={"stderr": probe.stderr[-2000:], "stdout": probe.stdout[-2000:]},
        )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            *_BOLTZGEN_RUNTIME_DEPS,
        ],
        cwd=None,
    )
    verify = subprocess.run(
        [str(python), "-c", probe_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "BoltzGen dependency installation did not make runtime imports available",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:boltzgen", "status": "installed"})
    state.mark("shared-dep:boltzgen", "ready", {})


_BOLTZGEN_RUNTIME_DEPS = (
    "matplotlib",
    "hydride",
    "biotite",
    "pydssp",
    "logomaker",
    "gemmi>=0.6.5",
    "scikit-learn",
    "hydra-core",
    "edit_distance",
    "pytorch-lightning",
    "pandas",
    "pdbeccdutils",
    "rdkit>=2024.3.2",
    "einx",
    "einops",
    "mashumaro",
    "nvidia-ml-py>=12.535.133",
    "huggingface_hub",
)


def _ensure_esm3_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import esm; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:esm3", "status": "ready"})
        state.mark("shared-dep:esm3", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "esm is required for esm3 but is not installed and --offline was set",
            details={"package": "esm"},
        )
    source = Path("/opt/vpa-data/esm")
    if not source.is_dir():
        raise DependencyError(
            "could not locate local ESM3 source checkout",
            details={"source": str(source)},
        )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            "anarcii",
            "zstandard",
            "cloudpathlib",
            "msgpack-numpy",
            "pygtrie",
            "tenacity",
            "pydssp",
            "dna_features_viewer",
            "py3dmol",
            "zstd",
            "transformers",
            "tokenizers",
            "einops",
            "biotite",
        ],
        cwd=None,
    )
    _run(
        [str(python), "-m", "pip", "install", "--no-user", "--no-deps", str(source)],
        cwd=source,
    )
    verify = subprocess.run(
        [str(python), "-c", "import esm; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "ESM3 package installation did not become importable",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:esm3", "status": "installed"})
    state.mark("shared-dep:esm3", "ready", {})


def _ensure_ligandmpnn_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import prody; print(prody.__version__)"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        version = probe.stdout.strip()
        results.append(
            {"unit": "shared-dep:ligandmpnn", "status": "ready", "version": version}
        )
        state.mark("shared-dep:ligandmpnn", "ready", {"version": version})
        return
    if options.offline:
        raise DependencyError(
            "prody is required for ligandmpnn but is not installed and --offline was set",
            details={"package": "prody"},
        )
    _run([str(python), "-m", "pip", "install", "--no-user", "prody"], cwd=None)
    verify = subprocess.run(
        [str(python), "-c", "import prody; print(prody.__version__)"],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "LigandMPNN shared dependency installation did not make prody importable",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    version = verify.stdout.strip()
    results.append(
        {"unit": "shared-dep:ligandmpnn", "status": "installed", "version": version}
    )
    state.mark("shared-dep:ligandmpnn", "ready", {"version": version})


def _ensure_prodigy_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import freesasa; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:prodigy", "status": "ready"})
        state.mark("shared-dep:prodigy", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "freesasa is required for prodigy but is not installed and --offline was set",
            details={"package": "freesasa"},
        )
    _run([str(python), "-m", "pip", "install", "--no-user", "freesasa"], cwd=None)
    verify = subprocess.run(
        [str(python), "-c", "import freesasa; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "Prodigy shared dependency installation did not make freesasa importable",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:prodigy", "status": "installed"})
    state.mark("shared-dep:prodigy", "ready", {})


def _ensure_proteina_complexa_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe = subprocess.run(
        [str(python), "-c", "from rdkit import Chem; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:proteina-complexa", "status": "ready"})
        state.mark("shared-dep:proteina-complexa", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "rdkit is required for proteina_complexa ligand_binder mode but is not installed "
            "and --offline was set",
            details={"package": "rdkit"},
        )
    _run([str(python), "-m", "pip", "install", "--no-user", "rdkit"], cwd=None)
    verify = subprocess.run(
        [str(python), "-c", "from rdkit import Chem; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "Proteina-Complexa shared dependency installation did not make rdkit importable",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:proteina-complexa", "status": "installed"})
    state.mark("shared-dep:proteina-complexa", "ready", {})


def _ensure_pyrosetta_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    probe = subprocess.run(
        [str(python), "-c", "import pyrosetta; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:pyrosetta", "status": "ready"})
        state.mark("shared-dep:pyrosetta", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "pyrosetta is required for the pyrosetta tool but is not installed and "
            "--offline was set",
            details={"package": "pyrosetta"},
        )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            "--no-deps",
            "pyrosetta",
            "--find-links",
            "https://west.rosettacommons.org/pyrosetta/quarterly/release",
        ],
        cwd=None,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            "--no-deps",
            "torchdata==0.7.1",
        ],
        cwd=None,
    )
    verify = subprocess.run(
        [str(python), "-c", "import pyrosetta; print('ok')"],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "PyRosetta shared dependency installation did not make pyrosetta importable",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:pyrosetta", "status": "installed"})
    state.mark("shared-dep:pyrosetta", "ready", {})


def _ensure_rfdiffusion_shared_deps(
    python: Path,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    cuda_stack = resolve_cuda_stack(options.cuda_profile)
    probe_code = (
        "from vibeproteinagent.core.compat import import_dgl_with_graphbolt_guard; "
        "import_dgl_with_graphbolt_guard(); "
        "print('ok')"
    )
    probe = subprocess.run(
        [str(python), "-c", probe_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and not options.force:
        results.append({"unit": "shared-dep:rfdiffusion-dgl", "status": "ready"})
        state.mark("shared-dep:rfdiffusion-dgl", "ready", {})
        return
    if options.offline:
        raise DependencyError(
            "dgl is required for rfdiffusion/rfpeptides but is not installed and "
            "--offline was set",
            details={"package": "dgl"},
        )
    if not cuda_stack.dgl_package or not cuda_stack.dgl_wheel_url:
        raise DependencyError(
            "DGL CUDA support is not available for the selected CPU CUDA profile",
            details={"cuda_profile": cuda_stack.name},
        )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            "--no-deps",
            "--force-reinstall",
            cuda_stack.dgl_package,
            "-f",
            cuda_stack.dgl_wheel_url,
        ],
        cwd=None,
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-user",
            "decorator==5.1.0",
            "e3nn==0.3.3",
            "hydra-core==1.3.2",
            "networkx",
            "opt_einsum",
            "pandas",
            "psutil",
            "pynvml==11.0.0",
            "pyrsistent==0.19.3",
            "requests",
            "scipy",
            "torchdata==0.7.1",
            "tqdm",
        ],
        cwd=None,
    )
    verify = subprocess.run(
        [str(python), "-c", probe_code],
        text=True,
        capture_output=True,
        check=False,
    )
    if verify.returncode != 0:
        raise DependencyError(
            "RFdiffusion shared dependency installation did not make DGL importable",
            details={"stderr": verify.stderr[-2000:], "stdout": verify.stdout[-2000:]},
        )
    results.append({"unit": "shared-dep:rfdiffusion-dgl", "status": "installed"})
    state.mark("shared-dep:rfdiffusion-dgl", "ready", {})


def _ensure_tool_units(
    tool: InstallTool,
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    for source in tool.sources:
        _ensure_source(tool, source, paths, options, state, results, required=required)
    if tool.sidecar:
        _ensure_sidecar(tool, paths, options, state, results, required=required)
    for weight in tool.weights:
        _ensure_weight(tool, weight, paths, options, state, results, required=required)
    if tool.binary:
        status = "ready" if shutil.which(tool.binary) else "missing"
        if status == "missing" and required:
            raise DependencyError(
                f"required binary for tool '{tool.name}' is missing",
                details={"tool": tool.name, "binary": tool.binary},
            )
        results.append({"unit": f"binary:{tool.name}", "status": status, "binary": tool.binary})
        state.mark(f"binary:{tool.name}", status, {"binary": tool.binary})


def _ensure_source(
    tool: InstallTool,
    source: InstallSource,
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    path = _source_path(paths, source)
    ready_path = _source_ready_path(paths, source)
    if ready_path is not None and not options.force and not options.repair:
        results.append(
            {
                "unit": f"source:{tool.name}:{source.name}",
                "status": "ready",
                "path": str(ready_path),
            }
        )
        state.mark(
            f"source:{tool.name}:{source.name}",
            "ready",
            {"path": str(ready_path), "repo": source.repo, "ref": source.ref},
        )
        return
    if not source.required and not required:
        results.append(
            {
                "unit": f"source:{tool.name}:{source.name}",
                "status": "skipped",
                "path": str(path),
            }
        )
        state.mark(f"source:{tool.name}:{source.name}", "skipped", {"path": str(path)})
        return
    if options.offline:
        raise DependencyError(
            (
                f"source checkout '{source.name}' for tool '{tool.name}' "
                "is missing and --offline was set"
            ),
            details={"tool": tool.name, "source": source.name, "path": str(path)},
        )
    git = shutil.which("git")
    if git is None:
        raise DependencyError(
            "git is required to fetch source checkouts",
            details={"tool": tool.name, "source": source.name, "repo": source.repo},
        )
    if path.exists():
        if path.is_symlink() or path.is_file():
            if options.force or options.repair:
                path.unlink()
            else:
                raise DependencyError(
                    f"source target exists but is not a directory: {path}",
                    details={"tool": tool.name, "source": source.name, "path": str(path)},
                )
        elif any(path.iterdir()):
            if options.force or options.repair:
                shutil.rmtree(path)
            else:
                raise DependencyError(
                    f"source target exists but is not a valid checkout: {path}",
                    details={
                        "tool": tool.name,
                        "source": source.name,
                        "path": str(path),
                        "hint": "rerun with --repair or set the matching VPA_*_TREE env var",
                    },
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    _run([git, "clone", source.repo, str(path)], cwd=None)
    if source.ref:
        _run([git, "-C", str(path), "checkout", source.ref], cwd=None)
    if not _source_ready(path, source):
        raise DependencyError(
            f"source checkout '{source.name}' for tool '{tool.name}' is not usable",
            details={
                "tool": tool.name,
                "source": source.name,
                "path": str(path),
                "markers": list(source.markers),
            },
        )
    results.append(
        {
            "unit": f"source:{tool.name}:{source.name}",
            "status": "cloned",
            "path": str(path),
            "repo": source.repo,
            "ref": source.ref,
        }
    )
    state.mark(
        f"source:{tool.name}:{source.name}",
        "ready",
        {"path": str(path), "repo": source.repo, "ref": source.ref},
    )


def _ensure_tool_readiness(
    tool: InstallTool,
    paths: InstallPaths,
    state: InstallState,
    results: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    readiness = _runtime_tool_readiness(tool.name, python=paths.shared_env / "bin" / "python")
    status = "ready" if readiness.get("ready") else ("failed" if required else "skipped")
    if status == "failed":
        raise DependencyError(
            f"tool '{tool.name}' is not runtime-ready after installation",
            details={"tool": tool.name, "readiness": readiness},
        )
    state.mark(f"tool:{tool.name}", status, {"runtime": tool.runtime, "readiness": readiness})
    results.append(
        {
            "unit": f"tool:{tool.name}",
            "status": status,
            "runtime": tool.runtime,
            "required": required,
            "readiness": readiness,
        }
    )


def _write_env_exports(
    selection: InstallSelection,
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
) -> None:
    exports = _env_exports(selection, paths, options)
    lines = [
        "# Generated by vibeproteinagent install. Safe to regenerate.",
        f"export {cfg.VPA_HOME_ENV}={shlex.quote(str(paths.root))}",
    ]
    lines.extend(f"export {key}={shlex.quote(value)}" for key, value in sorted(exports.items()))
    content = "\n".join(lines) + "\n"
    env_file = paths.root / "env.sh"
    env_file.write_text(content)
    activate_dir = paths.shared_env / "etc" / "conda" / "activate.d"
    if (paths.shared_env / "bin" / "python").exists():
        activate_dir.mkdir(parents=True, exist_ok=True)
        (activate_dir / "vibeproteinagent.sh").write_text(content)
    state.mark("env:vpa-activation", "ready", {"env_file": str(env_file), "exports": exports})
    results.append(
        {
            "unit": "env:vpa-activation",
            "status": "ready",
            "env_file": str(env_file),
            "exports": exports,
        }
    )


def _apply_install_env(paths: InstallPaths, exports: dict[str, str]) -> dict[str, str | None]:
    values = {cfg.VPA_HOME_ENV: str(paths.root), **exports}
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    old["PATH"] = os.environ.get("PATH")
    shared_bin = str(paths.shared_env / "bin")
    os.environ["PATH"] = (
        f"{shared_bin}{os.pathsep}{old['PATH']}" if old["PATH"] else shared_bin
    )
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _env_exports(
    selection: InstallSelection,
    paths: InstallPaths,
    options: InstallOptions | None = None,
) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for tool in selection.tools:
        for source in tool.sources:
            if not source.env_var:
                continue
            env_path = _env_source_path(source)
            if env_path is not None and _source_ready(env_path, source):
                export_path = env_path
            else:
                export_path = _source_path(paths, source)
            grouped.setdefault(source.env_var, []).append(str(export_path))
    for tool in selection.tools:
        for weight in tool.weights:
            if not weight.env_var:
                continue
            weight_path = _weight_path(paths, weight)
            if weight.env_value == "path":
                export_path = weight_path
            else:
                export_path = weight_path if weight_path.suffix == "" else weight_path.parent
            grouped.setdefault(weight.env_var, []).append(str(export_path))
    exports: dict[str, str] = {}
    for key, values in grouped.items():
        unique = sorted(set(values))
        exports[key] = unique[0] if len(unique) == 1 else os.path.commonpath(unique)
    for tool in selection.tools:
        if not tool.sidecar:
            continue
        sidecar_path = paths.root / tool.sidecar.path
        if tool.name == "bindcraft":
            exports.setdefault(
                "VPA_BINDCRAFT_ENV_PATH",
                _prebuilt_export_or_default(
                    tool.name,
                    "VPA_BINDCRAFT_ENV_PATH",
                    sidecar_path,
                    options,
                ),
            )
        elif tool.name == "proteina_complexa":
            exports.setdefault(
                "VPA_PROTEINA_COMPLEXA_VENV",
                _prebuilt_export_or_default(
                    tool.name,
                    "VPA_PROTEINA_COMPLEXA_VENV",
                    sidecar_path / ".venv",
                    options,
                ),
            )
        elif tool.name in {"pxdesign", "pxdesign_pipeline_multi_gpu"}:
            exports.setdefault(
                "VPA_PXDESIGN_PIPELINE_ENV",
                _prebuilt_export_or_default(
                    tool.name,
                    "VPA_PXDESIGN_PIPELINE_ENV",
                    sidecar_path,
                    options,
                ),
            )
        elif tool.name == "rfdiffusion3":
            exports.setdefault(
                "VPA_RFDIFFUSION3_ENV",
                _prebuilt_export_or_default(
                    tool.name,
                    "VPA_RFDIFFUSION3_ENV",
                    sidecar_path,
                    options,
                ),
            )
    if options is not None:
        exports.update(_cuda_env_exports(resolve_cuda_stack(options.cuda_profile)))
    return exports


def _prebuilt_export_or_default(
    tool_name: str,
    env_var: str,
    default_path: Path,
    options: InstallOptions | None,
) -> str:
    if options is None or not options.use_prebuilt_sidecars:
        return str(default_path)
    candidate = os.environ.get(env_var)
    if candidate and _prebuilt_python_path(tool_name, Path(candidate).expanduser()).is_file():
        return str(Path(candidate).expanduser())
    return str(default_path)


def _cuda_env_exports(stack: CudaStack) -> dict[str, str]:
    exports = {
        "VPA_CUDA_PROFILE": stack.name,
        "VPA_CUDA_TAG": stack.name,
        "VPA_TORCH_SPEC": stack.torch_spec,
        "VPA_TORCH_INDEX_URL": stack.torch_index_url,
    }
    if stack.cuda_version:
        exports["VPA_CUDA_VERSION"] = stack.cuda_version
    if stack.pyg_wheel_url:
        exports["VPA_PYG_WHEEL_URL"] = stack.pyg_wheel_url
    if stack.jax_extra:
        exports["VPA_JAX_CUDA_EXTRA"] = stack.jax_extra
    if stack.bindcraft_cuda:
        exports["VPA_BINDCRAFT_CUDA"] = stack.bindcraft_cuda
    if stack.dgl_package:
        exports["VPA_DGL_PACKAGE"] = stack.dgl_package
    if stack.dgl_wheel_url:
        exports["VPA_DGL_WHEEL_URL"] = stack.dgl_wheel_url
    return exports


def _ensure_sidecar(
    tool: InstallTool,
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    assert tool.sidecar is not None
    path = paths.root / tool.sidecar.path
    python = _sidecar_python(path)
    if python.is_file() and not options.force and not options.repair:
        results.append({"unit": f"sidecar:{tool.name}", "status": "ready", "path": str(path)})
        state.mark(f"sidecar:{tool.name}", "ready", {"path": str(path), "python": str(python)})
        return
    prebuilt = _link_prebuilt_sidecar(tool.name, path, options)
    if prebuilt is not None and not options.force and not options.repair:
        results.append(
            {
                "unit": f"sidecar:{tool.name}",
                "status": "linked",
                "path": str(path),
                "source": str(prebuilt),
            }
        )
        state.mark(
            f"sidecar:{tool.name}",
            "ready",
            {"path": str(path), "source": str(prebuilt)},
        )
        return
    if not tool.sidecar.install_script:
        if required:
            raise DependencyError(
                f"sidecar tool '{tool.name}' does not declare an install script",
                details={"tool": tool.name, "path": str(path)},
            )
        results.append({"unit": f"sidecar:{tool.name}", "status": "skipped", "path": str(path)})
        state.mark(f"sidecar:{tool.name}", "skipped", {"path": str(path)})
        return
    source_root = options.source_root or _source_root()
    command = tool.sidecar.install_script.format(
        install_root=str(paths.root),
        sidecar_path=str(path),
        repo_root=str(source_root),
        **_cuda_format_values(resolve_cuda_stack(options.cuda_profile)),
    )
    _run(shlex.split(command), cwd=source_root)
    results.append({"unit": f"sidecar:{tool.name}", "status": "installed", "path": str(path)})
    state.mark(f"sidecar:{tool.name}", "ready", {"path": str(path)})


def _cuda_format_values(stack: CudaStack) -> dict[str, str]:
    return {
        "cuda_profile": stack.name,
        "cuda_version": stack.cuda_version or "",
        "cuda_tag": stack.name,
        "torch_spec": stack.torch_spec,
        "torch_index_url": stack.torch_index_url,
        "pyg_wheel_url": stack.pyg_wheel_url or "",
        "jax_extra": stack.jax_extra or "",
        "bindcraft_cuda": stack.bindcraft_cuda or "12.8",
        "dgl_package": stack.dgl_package or "",
        "dgl_wheel_url": stack.dgl_wheel_url or "",
    }


def _link_prebuilt_sidecar(
    tool_name: str,
    target: Path,
    options: InstallOptions,
) -> Path | None:
    if not options.use_prebuilt_sidecars:
        return None
    env_var = _sidecar_env_var(tool_name)
    if not env_var:
        return None
    raw = os.environ.get(env_var)
    if not raw:
        return None
    source = Path(raw).expanduser()
    if not _prebuilt_python_path(tool_name, source).is_file():
        return None
    if tool_name == "proteina_complexa" and source.name == ".venv":
        target.mkdir(parents=True, exist_ok=True)
        link = target / ".venv"
        if not _prepare_sidecar_link(link):
            return None
        link.symlink_to(source, target_is_directory=True)
        return source if _sidecar_python(target).is_file() else None
    target.parent.mkdir(parents=True, exist_ok=True)
    if not _prepare_sidecar_link(target):
        return None
    target.symlink_to(source, target_is_directory=True)
    return source if _sidecar_python(target).is_file() else None


def _sidecar_env_var(tool_name: str) -> str | None:
    return {
        "bindcraft": "VPA_BINDCRAFT_ENV_PATH",
        "proteina_complexa": "VPA_PROTEINA_COMPLEXA_VENV",
        "pxdesign": "VPA_PXDESIGN_PIPELINE_ENV",
        "pxdesign_pipeline_multi_gpu": "VPA_PXDESIGN_PIPELINE_ENV",
        "rfdiffusion3": "VPA_RFDIFFUSION3_ENV",
    }.get(tool_name)


def _sidecar_ready_from_manifest_or_env(
    tool_name: str,
    manifest_path: Path,
) -> tuple[bool, Path, bool]:
    env_var = _sidecar_env_var(tool_name)
    if env_var:
        raw = os.environ.get(env_var)
        if raw:
            env_path = Path(raw).expanduser()
            if _prebuilt_python_path(tool_name, env_path).is_file():
                return True, env_path, True
    return _sidecar_python(manifest_path).is_file(), manifest_path, False


def _prepare_sidecar_link(path: Path) -> bool:
    if path.is_symlink():
        path.unlink()
        return True
    if not path.exists():
        return True
    if path.is_dir():
        try:
            next(path.iterdir())
        except StopIteration:
            path.rmdir()
            return True
    return False


def _prebuilt_python_path(tool_name: str, source: Path) -> Path:
    if tool_name == "proteina_complexa" and source.name == ".venv":
        return source / "bin" / "python"
    return _sidecar_python(source)


def _ensure_weight(
    tool: InstallTool,
    weight: InstallWeight,
    paths: InstallPaths,
    options: InstallOptions,
    state: InstallState,
    results: list[dict[str, Any]],
    *,
    required: bool,
) -> None:
    path = _weight_path(paths, weight)
    if _weight_ready(path, weight):
        results.append(
            {"unit": f"weight:{tool.name}:{weight.name}", "status": "ready", "path": str(path)}
        )
        state.mark(f"weight:{tool.name}:{weight.name}", "ready", {"path": str(path)})
        return
    source = _source_weight_path(weight, options)
    if source is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if path.exists() and not path.is_dir():
                path.unlink()
            shutil.copytree(source, path, dirs_exist_ok=True)
        else:
            shutil.copy2(source, path)
        if not _weight_ready(path, weight):
            raise DependencyError(
                f"checksum verification failed for copied weight '{weight.name}'",
                details={"tool": tool.name, "weight": weight.name, "path": str(path)},
            )
        results.append(
            {
                "unit": f"weight:{tool.name}:{weight.name}",
                "status": "copied",
                "path": str(path),
                "source": str(source),
            }
        )
        state.mark(f"weight:{tool.name}:{weight.name}", "ready", {"path": str(path)})
        return
    if weight.hf_repo_id:
        if options.offline:
            raise DependencyError(
                f"required Hugging Face cache '{weight.name}' is missing and --offline was set",
                details={
                    "tool": tool.name,
                    "weight": weight.name,
                    "repo_id": weight.hf_repo_id,
                    "path": str(path),
                },
            )
        _download_hf_snapshot(paths.shared_env / "bin" / "python", weight, path)
        if not _weight_ready(path, weight):
            raise DependencyError(
                f"Hugging Face cache '{weight.name}' did not become ready",
                details={"tool": tool.name, "weight": weight.name, "path": str(path)},
            )
        results.append(
            {
                "unit": f"weight:{tool.name}:{weight.name}",
                "status": "downloaded",
                "path": str(path),
                "repo_id": weight.hf_repo_id,
            }
        )
        state.mark(f"weight:{tool.name}:{weight.name}", "ready", {"path": str(path)})
        return
    if not weight.url:
        status = "missing"
        if weight.required and required:
            raise DependencyError(
                f"required weight '{weight.name}' for tool '{tool.name}' is missing",
                details={"tool": tool.name, "weight": weight.name, "path": str(path)},
            )
        results.append(
            {"unit": f"weight:{tool.name}:{weight.name}", "status": status, "path": str(path)}
        )
        state.mark(f"weight:{tool.name}:{weight.name}", status, {"path": str(path)})
        return
    if options.offline:
        raise DependencyError(
            f"required weight '{weight.name}' is not cached and --offline was set",
            details={"tool": tool.name, "weight": weight.name, "path": str(path)},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.download"
    urllib.request.urlretrieve(weight.url, tmp)
    if weight.unpack:
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        _extract_archive(tmp, path.parent, weight.unpack)
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(path)
    if not _weight_ready(path, weight):
        raise DependencyError(
            f"checksum verification failed for weight '{weight.name}'",
            details={"tool": tool.name, "weight": weight.name, "path": str(path)},
        )
    results.append(
        {"unit": f"weight:{tool.name}:{weight.name}", "status": "downloaded", "path": str(path)}
    )
    state.mark(f"weight:{tool.name}:{weight.name}", "ready", {"path": str(path)})


def _weight_ready(path: Path, weight: InstallWeight) -> bool:
    if not path.exists():
        return False
    if weight.unpack:
        return path.is_dir() and any(path.iterdir())
    if path.is_dir():
        return any(path.iterdir())
    if weight.sha256 and path.is_file():
        return _sha256(path) == weight.sha256
    return True


def _download_hf_snapshot(python: Path, weight: InstallWeight, path: Path) -> None:
    cache_dir = _hf_cache_dir_for_weight(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    code = """
import json
import sys
from huggingface_hub import snapshot_download

repo_id, repo_type, cache_dir = sys.argv[1:4]
snapshot_download(repo_id=repo_id, repo_type=repo_type or None, cache_dir=cache_dir)
print(json.dumps({"ok": True}))
"""
    proc = subprocess.run(
        [
            str(python),
            "-c",
            code,
            weight.hf_repo_id or "",
            weight.hf_repo_type or "",
            str(cache_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise DependencyError(
            f"failed to download Hugging Face cache '{weight.name}'",
            details={
                "repo_id": weight.hf_repo_id,
                "repo_type": weight.hf_repo_type,
                "cache_dir": str(cache_dir),
                "stderr": proc.stderr[-2000:],
                "stdout": proc.stdout[-2000:],
            },
        )


def _hf_cache_dir_for_weight(path: Path) -> Path:
    name = path.name
    if name.startswith(("models--", "datasets--", "spaces--")):
        return path.parent
    return path


def _extract_archive(archive: Path, destination: Path, kind: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if kind == "tar":
        with tarfile.open(archive) as tf:
            _safe_extract_tar(tf, destination)
        return
    if kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, destination)
        return
    raise DependencyError(
        "unsupported archive type",
        details={"archive": str(archive), "kind": kind},
    )


def _safe_extract_tar(tf: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in tf.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(root):
            raise DependencyError(
                "refusing to extract archive member outside destination",
                details={"member": member.name, "destination": str(destination)},
            )
    tf.extractall(destination, filter="data")


def _safe_extract_zip(zf: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in zf.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(root):
            raise DependencyError(
                "refusing to extract archive member outside destination",
                details={"member": member.filename, "destination": str(destination)},
            )
    zf.extractall(destination)


def _manifest_readiness(
    name: str,
    spec: InstallTool | None,
    paths: InstallPaths,
    *,
    required: bool,
) -> dict[str, Any]:
    if spec is None:
        return {"ready": False, "reason": "tool is missing from install manifest"}
    missing: list[dict[str, str]] = []
    env_override_ready = False
    for source in spec.sources:
        if not (source.required and required):
            continue
        ready_path = _source_ready_path(paths, source)
        if ready_path is not None:
            if ready_path != _source_path(paths, source):
                env_override_ready = True
            continue
        missing.append(
            {
                "unit": f"source:{name}:{source.name}",
                "path": str(_source_path(paths, source)),
            }
        )
    for weight in spec.weights:
        if not (weight.required and required):
            continue
        weight_ready, checked_paths = _weight_ready_from_manifest_or_env(paths, weight)
        if weight_ready:
            if checked_paths and checked_paths[0] != _weight_path(paths, weight):
                env_override_ready = True
            continue
        primary = _weight_path(paths, weight)
        detail = {
            "unit": f"weight:{name}:{weight.name}",
            "path": str(primary),
        }
        if len(checked_paths) > 1:
            detail["checked_paths"] = ", ".join(str(path) for path in checked_paths)
        missing.append(detail)
    if spec.sidecar:
        path = paths.root / spec.sidecar.path
        sidecar_ready, checked_path, used_env = _sidecar_ready_from_manifest_or_env(
            name, path
        )
        if used_env:
            env_override_ready = True
        if required and not sidecar_ready:
            detail = {"unit": f"sidecar:{name}", "path": str(path)}
            if checked_path != path:
                detail["checked_path"] = str(checked_path)
            missing.append(detail)
    if spec.binary and required and shutil.which(spec.binary) is None:
        missing.append({"unit": f"binary:{name}", "path": spec.binary})
    if missing:
        return {
            "ready": False,
            "reason": "required install units are missing",
            "missing": missing,
            "env_override_ready": env_override_ready,
        }
    return {"ready": True, "reason": None, "missing": [], "env_override_ready": env_override_ready}


def _combined_reason(*reasons: Any) -> str | None:
    parts: list[str] = []
    for reason in reasons:
        if not reason:
            continue
        text = str(reason)
        if text not in parts:
            parts.append(text)
    return "; ".join(parts) if parts else None


def _weight_ready_from_manifest_or_env(
    paths: InstallPaths,
    weight: InstallWeight,
) -> tuple[bool, list[Path]]:
    checked_paths = _weight_candidate_paths(paths, weight)
    return any(_weight_ready(path, weight) for path in checked_paths), checked_paths


def _weight_candidate_paths(paths: InstallPaths, weight: InstallWeight) -> list[Path]:
    candidates = [_weight_path(paths, weight)]
    env_path = _env_weight_path(weight)
    if env_path is not None and env_path not in candidates:
        candidates.insert(0, env_path)
    source_path = _source_weight_readiness_path(weight)
    if source_path is not None and source_path not in candidates:
        candidates.append(source_path)
    return candidates


def _env_weight_path(weight: InstallWeight) -> Path | None:
    if not weight.env_var:
        return None
    raw_value = os.environ.get(weight.env_var)
    if not raw_value:
        return None
    value = raw_value.split(os.pathsep, 1)[0]
    if not value:
        return None
    base = Path(value).expanduser()
    if weight.env_value == "path":
        return base
    raw = Path(weight.path)
    return base / raw.name


def _source_weight_readiness_path(weight: InstallWeight) -> Path | None:
    if not weight.source_path:
        return None
    candidate = _source_root() / weight.source_path
    return candidate if candidate.exists() else None


def _runtime_tool_readiness(name: str, *, python: Path | None = None) -> dict[str, Any]:
    if python is not None:
        encoded_name = json.dumps(name)
        code = f"""
import json

try:
    from vibeproteinagent.adapters import ToolRegistry
    from vibeproteinagent.adapters.availability import descriptor_availability

    registry = ToolRegistry()
    descriptor, adapter_cls = registry.get({encoded_name})
    availability = descriptor_availability(descriptor, adapter_cls)
    payload = {{
        "ready": bool(availability.get("available")),
        "reason": availability.get("availability_reason"),
        "availability": availability,
    }}
except Exception as exc:
    payload = {{
        "ready": False,
        "reason": type(exc).__name__ + ": " + str(exc),
    }}
print(json.dumps(payload))
"""
        proc = subprocess.run(
            [str(python), "-c", code],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "ready": False,
                "reason": f"shared-env readiness probe failed with {proc.returncode}",
                "stderr": proc.stderr[-2000:],
                "stdout": proc.stdout[-2000:],
            }
        try:
            return dict(json.loads(proc.stdout))
        except Exception as exc:
            return {
                "ready": False,
                "reason": f"could not parse shared-env readiness probe: {exc}",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
            }
    try:
        from vibeproteinagent.adapters import ToolRegistry
        from vibeproteinagent.adapters.availability import descriptor_availability

        registry = ToolRegistry()
        descriptor, adapter_cls = registry.get(name)
        availability = descriptor_availability(descriptor, adapter_cls)
        return {
            "ready": bool(availability.get("available")),
            "reason": availability.get("availability_reason"),
            "availability": availability,
        }
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _sidecar_python(path: Path) -> Path:
    for candidate in (path / "bin" / "python", path / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return path / "bin" / "python"


def _weight_path(paths: InstallPaths, weight: InstallWeight) -> Path:
    raw = Path(weight.path)
    if raw.is_absolute():
        return raw
    cache_prefix = Path("cache")
    try:
        relative_to_cache = raw.relative_to(cache_prefix)
    except ValueError:
        return paths.root / raw
    return paths.cache_dir / relative_to_cache


def _source_path(paths: InstallPaths, source: InstallSource) -> Path:
    raw = Path(source.path)
    return raw if raw.is_absolute() else paths.root / raw


def _source_ready_path(paths: InstallPaths, source: InstallSource) -> Path | None:
    env_path = _env_source_path(source)
    if env_path is not None and _source_ready(env_path, source):
        return env_path
    manifest_path = _source_path(paths, source)
    if _source_ready(manifest_path, source):
        return manifest_path
    return None


def _env_source_path(source: InstallSource) -> Path | None:
    if not source.env_var:
        return None
    raw = os.environ.get(source.env_var)
    if not raw:
        return None
    return Path(raw).expanduser()


def _source_ready(path: Path, source: InstallSource) -> bool:
    if not path.is_dir():
        return False
    return all((path / marker).exists() for marker in source.markers)


def _is_strict_tool(tool: InstallTool, options: InstallOptions) -> bool:
    if options.tools:
        return True
    return tool.required


def _source_weight_path(weight: InstallWeight, options: InstallOptions) -> Path | None:
    if not weight.source_path:
        return None
    source_root = options.source_root or _source_root()
    candidate = source_root / weight.source_path
    return candidate if candidate.exists() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_micromamba() -> Path | None:
    env = os.environ.get("VPA_MICROMAMBA")
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    found = shutil.which("micromamba")
    return Path(found) if found else None


def _source_root() -> Path:
    env = os.environ.get("VPA_SOURCE_ROOT")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[3]


def _run(argv: list[str], *, cwd: Path | None) -> None:
    proc = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise DependencyError(
            f"install command failed: {' '.join(argv)}",
            details={
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            },
        )


def _overall_status(results: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status")) for item in results}
    if statuses & {"failed", "missing"}:
        return "partial"
    return "ready"

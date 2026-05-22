from __future__ import annotations

import os
import sys
from typing import Any

import typer

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_ok,
    protect_json_stdout,
)
from vibeproteinagent.cli.render import render_doctor
from vibeproteinagent.core import config as cfg
from vibeproteinagent.workspace import Workspace


def doctor(
    json_: bool = typer.Option(False, "--json"),
    tools: str | None = typer.Option(
        None,
        "--tools",
        help="Comma-separated tool ids for install/readiness details.",
    ),
) -> None:
    """Report environment, GPU, workspace, and registry health."""
    try:
        with protect_json_stdout(json_):
            data = _gather(tools=tools)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(data, json_mode=json_, render=render_doctor)


def _gather(*, tools: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "executable": sys.executable,
    }

    info.update(_torch_info())

    ws = Workspace.resolve()
    info["active_workspace"] = ws.name
    info["workspace_root"] = str(ws.root)

    info["env_vars"] = {
        cfg.VPA_HOME_ENV: os.environ.get(cfg.VPA_HOME_ENV),
        cfg.VPA_WORKSPACE_ENV: os.environ.get(cfg.VPA_WORKSPACE_ENV),
        "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }

    info["model_cache_root"] = str(cfg.model_cache_root())

    try:
        registry = ToolRegistry()
        registry.discover()
        info["registry_tools"] = sorted(d.name for d in registry.list())
    except Exception as exc:
        info["registry_tools"] = []
        info["registry_error"] = str(exc)

    if tools:
        try:
            from vibeproteinagent.install.installer import install_readiness

            info["install_readiness"] = install_readiness((tools,))
        except Exception as exc:
            info["install_readiness"] = {
                "install_ready": False,
                "error": str(exc),
                "exception_type": type(exc).__name__,
            }
    else:
        info["install_readiness"] = {
            "install_ready": None,
            "tools": [],
            "message": "pass --tools or run `vpa install doctor` for tool readiness probes",
        }

    # Supervisor presence + total adapters loaded across all workers.
    info["daemon_running"] = False
    info["supervisor_running"] = False
    info["total_loaded_adapters_across_workers"] = 0
    try:
        from vibeproteinagent.workers.supervisor_client import (
            SupervisorClient,
        )
        client = SupervisorClient()
        if client.is_running():
            info["daemon_running"] = True
            info["supervisor_running"] = True
            try:
                wi = client.worker_info()
                workers = wi.get("workers", []) or []
                total = sum(
                    len(w.get("loaded_adapters", []) or []) for w in workers
                )
                info["total_loaded_adapters_across_workers"] = int(total)
            except Exception:
                pass
    except Exception:
        pass
    return info


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": None,
        "cuda_version": None,
        "cuda_available": False,
        "num_gpus": 0,
        "gpus": [],
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    if not torch.cuda.is_available():
        return info

    info["cuda_available"] = True
    n = torch.cuda.device_count()
    info["num_gpus"] = n
    gpus: list[dict[str, Any]] = []
    for i in range(n):
        try:
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_gib": int(props.total_memory // (1024**3)),
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                }
            )
        except Exception as exc:  # pragma: no cover - hardware-dependent
            gpus.append({"index": i, "error": str(exc)})
    info["gpus"] = gpus
    return info


__all__ = ["doctor"]

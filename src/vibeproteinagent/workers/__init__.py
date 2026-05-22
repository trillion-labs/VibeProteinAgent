from __future__ import annotations

from typing import Any

__all__ = [
    "Supervisor",
    "SupervisorClient",
    "Worker",
    "poll_cancel",
    "resolve_default_gpu_ids",
    "runtime_dir",
    "supervisor_main",
    "supervisor_pid_file",
    "supervisor_socket",
    "worker_main",
    "worker_pid_file",
    "worker_socket",
]


def __getattr__(name: str) -> Any:
    if name == "poll_cancel":
        from .cancel import poll_cancel

        return poll_cancel
    if name in {
        "runtime_dir",
        "supervisor_pid_file",
        "supervisor_socket",
        "worker_pid_file",
        "worker_socket",
    }:
        from . import paths

        return getattr(paths, name)
    if name in {"Supervisor", "resolve_default_gpu_ids", "supervisor_main"}:
        from .supervisor import Supervisor, resolve_default_gpu_ids, supervisor_main

        return {
            "Supervisor": Supervisor,
            "resolve_default_gpu_ids": resolve_default_gpu_ids,
            "supervisor_main": supervisor_main,
        }[name]
    if name == "SupervisorClient":
        from .supervisor_client import SupervisorClient

        return SupervisorClient
    if name in {"Worker", "worker_main"}:
        from .worker import Worker, worker_main

        return {"Worker": Worker, "worker_main": worker_main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

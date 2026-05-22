from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import typer

from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_err,
    print_envelope_ok,
)
from vibeproteinagent.cli.render import (
    render_daemon_start,
    render_daemon_status,
    render_daemon_stop,
    render_warm_result,
)
from vibeproteinagent.core.errors import DaemonUnavailableError, ErrorCode
from vibeproteinagent.workers import paths
from vibeproteinagent.workers.supervisor import (
    resolve_default_gpu_ids,
)
from vibeproteinagent.workers.supervisor_client import SupervisorClient
from vibeproteinagent.workspace import Workspace

daemon_app = typer.Typer(no_args_is_help=True, help="Worker daemon lifecycle.")


@daemon_app.command("start")
def daemon_start(
    workers: int | None = typer.Option(
        None,
        "--workers",
        help="Number of GPU worker procs.",
    ),
    cpu_workers: int = typer.Option(
        0,
        "--cpu-workers",
        min=0,
        help="Number of CPU-only worker procs.",
    ),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Start the supervisor + workers."""
    try:
        client = SupervisorClient()
        if client.is_running():
            print_envelope_err(
                ErrorCode.CONFLICT,
                f"supervisor already running (pid={client.pid()})",
                {"pid": client.pid(), "socket": str(client.socket_path)},
                json_mode=json_,
            )
            return
        ws = Workspace.resolve(workspace)
        if not ws.db_path.exists():
            ws = Workspace.create(ws.name)

        gpu_ids = resolve_default_gpu_ids(workers)
        if gpu_ids == [-1]:
            gpu_ids = []
            if workers is None and cpu_workers == 0:
                cpu_workers = 1
            elif workers is not None and workers > 0 and cpu_workers == 0:
                print_envelope_err(
                    ErrorCode.NO_GPU,
                    "no CUDA GPUs are available; use --cpu-workers for CPU-only jobs",
                    {"requested_gpu_workers": workers},
                    json_mode=json_,
                )
                return
        if not gpu_ids and cpu_workers == 0:
            print_envelope_err(
                ErrorCode.INVALID_PARAMS,
                "daemon start needs at least one GPU worker or CPU worker",
                {"requested_gpu_workers": workers, "cpu_workers": cpu_workers},
                json_mode=json_,
            )
            return
        gpu_arg = ",".join(str(g) for g in gpu_ids)
        runtime_dir = paths.runtime_dir()
        child_env = os.environ.copy()
        child_env["VPA_RUNTIME_DIR"] = str(runtime_dir)
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from vibeproteinagent.workers.supervisor import supervisor_main; "
                    "workspace=sys.argv[1]; "
                    "gpu_ids=[int(x) for x in sys.argv[2].split(',') if x]; "
                    "cpu_workers=int(sys.argv[3]); "
                    "supervisor_main(workspace, len(gpu_ids)+cpu_workers, gpu_ids, cpu_workers)"
                ),
                ws.name,
                gpu_arg,
                str(cpu_workers),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            env=child_env,
        )

        if not client.wait_until_running(timeout_s=45.0):
            print_envelope_err(
                ErrorCode.DAEMON_UNAVAILABLE,
                "supervisor did not come up within 45s",
                {"log": str(paths.supervisor_log_file())},
                json_mode=json_,
            )

        info = client.worker_info()
        print_envelope_ok(
            {
                "pid": client.pid(),
                "socket": str(client.socket_path),
                "workspace": ws.name,
                "gpu_workers": len(gpu_ids),
                "cpu_workers": cpu_workers,
                "workers": info.get("workers", []),
                "log": str(paths.supervisor_log_file()),
            },
            json_mode=json_,
            render=render_daemon_start,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@daemon_app.command("stop")
def daemon_stop(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Workspace name. Accepted for symmetry with daemon start/status.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Stop the supervisor (and all workers) cleanly."""
    try:
        del workspace
        client = SupervisorClient()
        if not client.is_running():
            print_envelope_err(
                ErrorCode.DAEMON_UNAVAILABLE,
                "supervisor is not running",
                {"socket": str(client.socket_path)},
                json_mode=json_,
            )
        pid = client.pid()
        try:
            client.shutdown()
        except DaemonUnavailableError:
            pass
        # Wait up to 10s for clean stop.
        if not client.wait_until_stopped(timeout_s=10.0):
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                # Cleanup leftover pidfile / socket.
                for f in (paths.supervisor_pid_file(), paths.supervisor_socket()):
                    try:
                        f.unlink()
                    except FileNotFoundError:
                        pass
        print_envelope_ok(
            {"stopped": True, "pid": pid},
            json_mode=json_,
            render=render_daemon_stop,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@daemon_app.command("status")
def daemon_status(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Workspace name. Accepted for symmetry with daemon start.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Report whether the supervisor is running, and list its workers."""
    try:
        ws = Workspace.resolve(workspace)
        client = SupervisorClient()
        running = client.is_running()
        if not running:
            print_envelope_ok(
                {"running": False, "workspace": ws.name},
                json_mode=json_,
                render=render_daemon_status,
            )
            return
        info = client.worker_info()
        print_envelope_ok(
            {
                "running": True,
                "pid": client.pid(),
                "socket": str(client.socket_path),
                "workspace": ws.name,
                "workers": info.get("workers", []),
                "log": str(paths.supervisor_log_file()),
            },
            json_mode=json_,
            render=render_daemon_status,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@daemon_app.command("warm")
def daemon_warm(
    tool: str = typer.Argument(..., help="Tool name to preload."),
    version: str = typer.Option("*", "--version", help="PEP-440 version specifier."),
    workers: str = typer.Option(
        "all",
        "--workers",
        help="'all' to warm every worker, or a comma-separated list of GPU ids.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Preload ``tool`` into the warm pool of one or more workers."""
    try:
        client = SupervisorClient()
        if not client.is_running():
            print_envelope_err(
                ErrorCode.DAEMON_UNAVAILABLE,
                "supervisor is not running",
                {"socket": str(client.socket_path)},
                json_mode=json_,
            )
        gpu_ids: list[int] | None = None
        if workers.strip().lower() != "all":
            try:
                gpu_ids = [int(x) for x in workers.split(",") if x.strip()]
            except ValueError:
                print_envelope_err(
                    ErrorCode.INVALID_PARAMS,
                    "--workers must be 'all' or a comma-separated list of gpu ids; "
                    f"got {workers!r}",
                    {"workers": workers},
                    json_mode=json_,
                )
        result = client.warm(tool, version_spec=version, gpu_ids=gpu_ids)
        print_envelope_ok(
            {
                "tool": result.get("tool", tool),
                "version": result.get("version_spec", version),
                "warmed": result.get("warmed", []),
            },
            json_mode=json_,
            render=render_warm_result,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


# A few imports referenced for side-effects (typing).
_ = time  # silence linter if not used elsewhere

__all__ = ["daemon_app"]

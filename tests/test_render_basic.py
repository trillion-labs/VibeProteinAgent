from __future__ import annotations

from io import StringIO

from rich.console import Console

from vibeproteinagent.cli.render import (
    render_agent_plan_validation,
    render_agent_result,
    render_daemon_start,
    render_daemon_status,
    render_doctor,
    render_job,
    render_job_list,
    render_tool_info,
    render_tool_list,
    render_version,
    render_warm_result,
    render_workflow_status,
    render_workspace_list,
)
from vibeproteinagent.core.log import LogStyle, Verbosity, set_cli_state


def _plain_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, no_color=True, force_terminal=False, highlight=False, width=120)
    return c, buf


def _pretty_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    c = Console(file=buf, force_terminal=True, color_system="standard", width=120,
                highlight=False)
    return c, buf


def setup_module(_module: object) -> None:
    set_cli_state(LogStyle.PLAIN, Verbosity.NORMAL)


def test_render_tool_list_plain_contains_table() -> None:
    c, buf = _plain_console()
    render_tool_list(
        {
            "category": None,
            "tools": [
                {"name": "noop", "version": "1.0.0", "category": "misc",
                 "gpu_required": False, "vram_min_gib": None,
                 "description": "no-op test adapter"},
                {"name": "proteinmpnn", "version": "1.0.1", "category": "inverse_folding",
                 "gpu_required": True, "vram_min_gib": 4,
                 "description": "ProteinMPNN inverse folding"},
            ],
        },
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "noop" in out
    assert "proteinmpnn" in out
    assert "1.0.0" in out
    # No ANSI escapes in plain output.
    assert "\x1b[" not in out


def test_render_tool_list_pretty_has_styling() -> None:
    c, buf = _pretty_console()
    render_tool_list(
        {"tools": [{"name": "noop", "version": "1.0.0", "category": "misc",
                    "gpu_required": False}]},
        style=LogStyle.PRETTY,
        console=c,
    )
    out = buf.getvalue()
    assert "noop" in out
    # Pretty mode should emit at least one ANSI escape.
    assert "\x1b[" in out


def test_render_tool_info_plain_lists_fields() -> None:
    c, buf = _plain_console()
    render_tool_info(
        {"tool": {
            "name": "proteinmpnn", "version": "1.0.1",
            "category": "inverse_folding", "backend": "pytorch",
            "runtime": "python", "gpu_required": True,
            "gpus_required": 1, "vram_min_gib": 4,
            "pin_warm": True, "warm_load_capable": True,
            "startup_cost": "med",
            "supported_tasks": ["seq_design"],
            "incompatible_with": [],
            "entry_point": "vpa.adapters.proteinmpnn:Adapter",
            "description": "MPNN.",
        }},
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "proteinmpnn" in out
    assert "category" in out and "inverse_folding" in out
    assert "vram_min_gib" in out and "4" in out
    assert "\x1b[" not in out


def test_render_job_pretty_emits_line() -> None:
    c, buf = _pretty_console()
    render_job(
        {"job": {
            "job_id": "job_abcdef1234567890" + "0" * 14,
            "tool": "noop", "version": "1.0.0",
            "state": "completed", "error_code": None,
        }},
        style=LogStyle.PRETTY,
        console=c,
    )
    out = buf.getvalue()
    assert "noop" in out
    assert "completed" in out


def test_render_job_list_plain() -> None:
    c, buf = _plain_console()
    render_job_list(
        {"jobs": [
            {"job_id": "j1", "tool": "noop", "version": "1.0.0", "state": "completed",
             "worker_id": 0, "error_code": None},
            {"job_id": "j2", "tool": "noop", "version": "1.0.0", "state": "failed",
             "worker_id": 1, "error_code": "E_INTERNAL"},
        ]},
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "j1" in out and "j2" in out
    assert "completed" in out and "failed" in out
    assert "\x1b[" not in out


def test_render_daemon_start_plain_shows_workers() -> None:
    c, buf = _plain_console()
    render_daemon_start(
        {
            "pid": 1234, "socket": "/tmp/sup.sock", "workspace": "default",
            "workers": [
                {"gpu_id": 0, "pid": 5001, "total_vram_mib": 178 * 1024},
                {"gpu_id": 1, "pid": 5002, "total_vram_mib": 178 * 1024},
            ],
        },
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "supervisor running" in out
    assert "pid=1234" in out
    assert "worker 0" in out and "worker 1" in out
    assert "\x1b[" not in out


def test_render_daemon_status_plain_when_down() -> None:
    c, buf = _plain_console()
    render_daemon_status({"running": False}, style=LogStyle.PLAIN, console=c)
    out = buf.getvalue()
    assert "not running" in out


def test_render_warm_result_summarises() -> None:
    c, buf = _plain_console()
    render_warm_result(
        {"tool": "proteinmpnn", "warmed": [
            {"gpu_id": 0, "success": True, "vram_used_mib": 14},
            {"gpu_id": 1, "success": True, "vram_used_mib": 14},
            {"gpu_id": 2, "success": False, "error_code": "E_INTERNAL",
             "error": "boom"},
        ]},
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "proteinmpnn" in out
    assert "2/3" in out
    assert "E_INTERNAL" in out


def test_render_workflow_status_plain() -> None:
    c, buf = _plain_console()
    render_workflow_status(
        {
            "workflow_id": "wf_abcdef", "name": "mpnn_then_refold",
            "version": "1.0.0", "state": "completed",
            "nodes": [
                {"node_id": "design", "tool": "proteinmpnn",
                 "tool_version": "1.0.1", "state": "completed", "job_id": "j1"},
                {"node_id": "refold", "tool": "protenix_v2",
                 "tool_version": "1.0.0", "state": "cached", "job_id": "j2"},
            ],
        },
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "mpnn_then_refold" in out
    assert "completed" in out
    assert "cached" in out
    assert "design" in out
    assert "\x1b[" not in out


def test_render_version_pretty_and_plain() -> None:
    c, buf = _pretty_console()
    render_version(
        {"version": "0.1.0", "python": "3.12.13", "torch": "2.7.1+cu128",
         "cuda": "12.8"},
        style=LogStyle.PRETTY,
        console=c,
    )
    out = buf.getvalue()
    assert "0.1.0" in out and "3.12.13" in out and "2.7.1" in out

    c2, buf2 = _plain_console()
    render_version(
        {"version": "0.1.0", "python": "3.12.13", "torch": None, "cuda": None},
        style=LogStyle.PLAIN,
        console=c2,
    )
    assert "0.1.0" in buf2.getvalue()
    assert "\x1b[" not in buf2.getvalue()


def test_render_doctor_plain() -> None:
    c, buf = _plain_console()
    render_doctor(
        {
            "python_version": "3.12.13",
            "executable": "/usr/bin/python",
            "torch_version": "2.7.1+cu128",
            "cuda_version": "12.8",
            "cuda_available": True,
            "num_gpus": 8,
            "active_workspace": "default",
            "workspace_root": "/tmp/ws",
            "model_cache_root": "/tmp/cache",
            "daemon_running": False,
            "total_loaded_adapters_across_workers": 0,
            "registry_tools": ["noop", "proteinmpnn"],
            "gpus": [
                {"index": 0, "name": "B200", "compute_capability": "10.0",
                 "total_memory_gib": 178},
            ],
        },
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "python" in out and "3.12.13" in out
    assert "B200" in out
    assert "\x1b[" not in out


def test_render_agent_result_plain_is_concise() -> None:
    c, buf = _plain_console()
    render_agent_result(
        {
            "workspace": "my_test",
            "run_id": "arun_1234567890abcdef",
            "mode": "agent_plan",
            "state": "completed",
            "workflow_ids": ["wf_abcdef1234567890"],
            "job_ids": ["job_ok", "job_fail"],
            "summary": {
                "nodes": [
                    {
                        "node_id": "design",
                        "tool": "boltzgen",
                        "state": "completed",
                        "job_id": "job_ok",
                        "output": {"large": "payload" * 100},
                    },
                    {
                        "node_id": "score",
                        "tool": "protenix_v2",
                        "state": "failed",
                        "job_id": "job_fail",
                        "error_code": "E_TOOL_FAILED",
                        "error_message": "checkpoint missing",
                    },
                ],
                "artifacts": {"job_ok": {"huge": "metadata"}},
            },
        },
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "my_test" in out
    assert "wf_abcdef1234567890" in out
    assert "boltzgen" in out
    assert "E_TOOL_FAILED: checkpoint missing" in out
    assert "payloadpayload" not in out
    assert "metadata" not in out


def test_render_agent_plan_validation_plain_summarises_nodes() -> None:
    c, buf = _plain_console()
    render_agent_plan_validation(
        {
            "workspace": "my_test",
            "valid": True,
            "plan": {
                "required_tools": ["boltzgen"],
                "workflow_plan": {
                    "name": "coverage",
                    "nodes": [
                        {
                            "id": "design",
                            "tool": "boltzgen",
                            "on_failure": "continue",
                            "max_parallel": 2,
                        }
                    ],
                },
            },
        },
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "Agent plan valid" in out
    assert "coverage" in out
    assert "boltzgen" in out
    assert "max_parallel" in out


def test_render_workspace_list_marks_active() -> None:
    c, buf = _plain_console()
    render_workspace_list(
        {"workspaces": ["default", "demo"], "active": "demo"},
        style=LogStyle.PLAIN,
        console=c,
    )
    out = buf.getvalue()
    assert "default" in out and "demo" in out


def test_quiet_verbosity_suppresses_chrome() -> None:
    set_cli_state(LogStyle.PLAIN, Verbosity.QUIET)
    try:
        c, buf = _plain_console()
        render_version(
            {"version": "0.1.0", "python": "3.12", "torch": None, "cuda": None},
            style=LogStyle.PLAIN,
            console=c,
        )
        out = buf.getvalue().strip()
        # Quiet mode → just the bare version, no chrome.
        assert out == "0.1.0"
    finally:
        set_cli_state(LogStyle.PLAIN, Verbosity.NORMAL)

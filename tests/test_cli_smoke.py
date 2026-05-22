from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.workspace import Workspace

# Each entry: (label, argv, expected_ok). expected_ok=True asserts a success
# envelope; False asserts an error envelope. argv is rewritten on the fly so that
# job-id placeholders ($JOB) point at a real submitted job in the isolated
# workspace fixture.
COMMANDS: list[tuple[str, list[str], bool]] = [
    ("version",            ["version", "--json"], True),
    ("tools",              ["tools", "--json"], True),
    ("tools --category",   ["tools", "-c", "protein_design", "--json"], True),
    ("tools --refresh",    ["tools", "--refresh", "--json"], True),
    ("tool info",          ["tool", "info", "noop", "--json"], True),
    ("tool info missing",  ["tool", "info", "no_such_tool", "--json"], False),
    ("run noop",           ["run", "noop", "payload=hi", "--json"], True),
    ("run KEY=VAL bool",   ["run", "noop", "payload=hi", "--json"], True),
    ("run --priority",     ["run", "noop", "payload=hi", "--priority", "high", "--json"], True),
    ("run bad params",     ["run", "noop", "x=1", "--json"], False),
    ("run unknown tool",   ["run", "no_such_tool", "x=1", "--json"], False),
    ("status",             ["status", "$JOB", "--json"], True),
    ("status missing",     ["status", "job_deadbeefdeadbeefdeadbeefdeadbeef", "--json"], False),
    ("wait timeout=0",     ["wait", "$JOB", "--timeout-ms", "0", "--json"], True),
    ("logs",               ["logs", "$JOB", "--json"], True),
    ("logs --tail",        ["logs", "$JOB", "--tail", "10", "--json"], True),
    ("results not done",   ["results", "$JOB", "--json"], False),
    ("cancel",             ["cancel", "$JOB", "--json"], True),
    ("artifacts",          ["artifacts", "$JOB", "--json"], True),
    ("download",           ["download", "$JOB", "--json"], True),
    (
        "export tar",
        ["export", "$JOB", "--format", "tar", "--out", "$TMP/x.tar", "--json"],
        True,
    ),
    ("workspace create",   ["workspace", "create", "ws1", "--json"], True),
    ("workspace use",      ["workspace", "use", "ws1", "--json"], True),
    ("workspace status",   ["workspace", "status", "--json"], True),
    ("workspace list",     ["workspace", "list", "--json"], True),
    ("workspace delete",   ["workspace", "delete", "ws1", "--json"], True),
    ("workspace delete missing", ["workspace", "delete", "nope", "--json"], True),
    ("workflow list",          ["workflow", "list", "--json"], True),
    ("workflow inspect missing", ["workflow", "inspect", "wf1", "--json"], False),
    ("workflow run missing",   ["workflow", "run", "wf1", "--json"], False),
    ("workflow status missing", ["workflow", "status", "wf_abc", "--json"], False),
    # daemon start is exercised by the slow e2e test; it spawns a real supervisor
    # and so is not suitable for in-process CliRunner smoke testing.
    ("daemon stop",        ["daemon", "stop", "--json"], False),
    ("daemon status",      ["daemon", "status", "--json"], True),
    # stdio mode actually starts a blocking MCP server (covered by the slow e2e test).
    # ws is documented as unimplemented and must surface E_DEPENDENCY.
    ("mcp serve ws",       ["mcp", "serve", "--transport", "ws", "--json"], False),
    ("agent run direct",   ["agent", "run", "run", "noop", "payload=hi", "--json"], True),
    ("agent runs",         ["agent", "runs", "--json"], True),
    ("doctor",             ["doctor", "--json"], True),
]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def smoke_env(
    isolated_vpa_home: Path,
    tmp_path: Path,
) -> dict[str, str]:
    """Create an isolated workspace and submit a noop job for the placeholder."""
    ws = Workspace.create("default")
    Workspace.create("ws1")  # for the "workspace use ws1" smoke entry
    from vibeproteinagent.adapters import ToolRegistry
    from vibeproteinagent.workspace import JobState, JobStore

    registry = ToolRegistry()
    registry.discover()
    desc, _ = registry.get("noop", "*")
    js = JobStore(ws)
    rec = js.submit(desc.name, desc.version, {"payload": "hi"}, workspace_name=ws.name)
    js.transition(rec.job_id, JobState.QUEUED)
    return {"JOB": rec.job_id, "TMP": str(tmp_path)}


@pytest.mark.parametrize("label,argv,expected_ok", COMMANDS, ids=[c[0] for c in COMMANDS])
def test_command_returns_envelope(
    runner: CliRunner,
    smoke_env: dict[str, str],
    label: str,
    argv: list[str],
    expected_ok: bool,
) -> None:
    rewritten = [_substitute(a, smoke_env) for a in argv]
    result = runner.invoke(app, rewritten)
    payload = json.loads(result.stdout)
    assert payload.get("ok") is expected_ok, f"{label} stdout={result.stdout!r}"
    if expected_ok:
        assert result.exit_code == 0, f"{label} exit={result.exit_code} stdout={result.stdout!r}"
        assert "data" in payload
        assert isinstance(payload["data"], dict)
    else:
        assert result.exit_code != 0, f"{label} expected non-zero exit"
        assert "error" in payload
        assert payload["error"]["code"].startswith("E_")
        assert isinstance(payload["error"]["details"], dict)


def test_command_count_matches_architecture_surface() -> None:
    """Guardrail: surface should cover the §11 commands."""
    assert len(COMMANDS) >= 30


def _substitute(s: str, env: dict[str, str]) -> str:
    out = s
    for k, v in env.items():
        out = out.replace(f"${k}", v)
    return out

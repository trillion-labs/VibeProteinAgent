from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.workspace import Workspace


def _submit_noop(runner: CliRunner) -> str:
    res = runner.invoke(app, ["run", "noop", "payload=hi", "--json"])
    assert res.exit_code == 0, res.stdout
    return json.loads(res.stdout)["data"]["job_id"]


def test_status_returns_record(runner: CliRunner, isolated_vpa_home: Path) -> None:
    Workspace.create("default")
    job_id = _submit_noop(runner)
    res = runner.invoke(app, ["status", job_id, "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    job = payload["data"]["job"]
    assert job["job_id"] == job_id
    assert job["state"] == "queued"


def test_logs_empty_for_fresh_job(runner: CliRunner, isolated_vpa_home: Path) -> None:
    Workspace.create("default")
    job_id = _submit_noop(runner)
    res = runner.invoke(app, ["logs", job_id, "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["data"]["lines"] == []


def test_logs_stream_stderr_reads_stderr_log(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    """``vpa logs <id> --stream stderr`` reads ``stderr.log``, not ``stdout.log``.

    Codex review C14: the flag exists on :class:`LogFileManager.read` but was
    not threaded through the CLI; ``vpa logs`` always read stdout.
    """
    Workspace.create("default")
    job_id = _submit_noop(runner)
    ws = Workspace.resolve("default")
    job_dir = ws.job_dir(job_id)
    (job_dir / "stderr.log").write_text("ERR: something happened\n")
    (job_dir / "stdout.log").write_text("OK\n")

    res_err = runner.invoke(app, ["logs", job_id, "--stream", "stderr", "--json"])
    assert res_err.exit_code == 0, res_err.stdout
    err_lines = json.loads(res_err.stdout)["data"]["lines"]
    assert any("something happened" in ln for ln in err_lines)
    assert json.loads(res_err.stdout)["data"]["stream"] == "stderr"

    res_out = runner.invoke(app, ["logs", job_id, "--json"])
    out_lines = json.loads(res_out.stdout)["data"]["lines"]
    assert any("OK" in ln for ln in out_lines)
    assert json.loads(res_out.stdout)["data"]["stream"] == "stdout"


def test_logs_stream_rejects_unknown_value(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    job_id = _submit_noop(runner)
    res = runner.invoke(
        app, ["logs", job_id, "--stream", "trace", "--json"],
    )
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["error"]["code"] == "E_INVALID_PARAMS"


def test_results_on_unfinished_job_errors(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    job_id = _submit_noop(runner)
    res = runner.invoke(app, ["results", job_id, "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_NOT_FOUND"
    assert payload["error"]["details"]["state"] == "queued"


def test_cancel_transitions_to_cancelled(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    job_id = _submit_noop(runner)
    cancel_res = runner.invoke(app, ["cancel", job_id, "--json"])
    assert cancel_res.exit_code == 0, cancel_res.stdout
    cdata = json.loads(cancel_res.stdout)["data"]
    assert cdata["state"] == "cancelled"

    status_res = runner.invoke(app, ["status", job_id, "--json"])
    assert status_res.exit_code == 0
    sdata = json.loads(status_res.stdout)["data"]["job"]
    assert sdata["state"] == "cancelled"


def test_status_unknown_job_returns_not_found(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    res = runner.invoke(app, ["status", "job_" + "0" * 32, "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["error"]["code"] == "E_NOT_FOUND"


def test_wait_timeout_zero_returns_immediately(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    job_id = _submit_noop(runner)
    res = runner.invoke(app, ["wait", job_id, "--timeout-ms", "0", "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    assert payload["data"]["timed_out"] is True
    assert payload["data"]["all_terminal"] is False

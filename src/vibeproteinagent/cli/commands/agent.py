from __future__ import annotations

from typing import Any

import typer

from vibeproteinagent.agents import AgentConfig, AgentController, AgentSessionStore
from vibeproteinagent.agents.execution import ExecutionPolicy
from vibeproteinagent.agents.execution.controller import load_plan_or_workflow
from vibeproteinagent.cli.output import fail_from_exception, print_envelope_ok
from vibeproteinagent.core.types import load_params_file, parse_kv_pairs
from vibeproteinagent.workspace import Workspace

agent_app = typer.Typer(no_args_is_help=True, help="Optional LLM agent orchestration.")


def _ensure_workspace(name: str | None) -> Workspace:
    ws = Workspace.resolve(name)
    if not ws.db_path.exists():
        ws = Workspace.create(ws.name)
    return ws


def _merge_params(
    *,
    params_file: str | None,
    param: list[str] | None,
    target: str | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if params_file:
        merged.update(load_params_file(params_file))
    merged.update(parse_kv_pairs(param))
    if target:
        merged.setdefault("target_pdb", target)
        merged.setdefault("pdb_path", target)
    return merged


@agent_app.command("run")
def agent_run(
    request: list[str] = typer.Argument(  # noqa: B008
        ..., help="Natural language request or direct run text."
    ),
    goal: str | None = typer.Option(None, "--goal", help="Additional scientific goal text."),
    tool: str | None = typer.Option(None, "--tool", help="Force direct single-tool mode."),
    workflow: str | None = typer.Option(None, "--workflow", help="Force explicit workflow mode."),
    target: str | None = typer.Option(None, "--target", help="Target PDB/path convenience param."),
    param: list[str] = typer.Option(None, "--param", help="KEY=VALUE agent/workflow param."),  # noqa: B008
    params_file: str | None = typer.Option(None, "--params-file", help="JSON/YAML params file."),
    workspace_name: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    model: str | None = typer.Option(None, "--model", help="Agent model id."),
    provider: str | None = typer.Option(None, "--provider", help="Agent provider."),
    agent_profile: str | None = typer.Option(
        None,
        "--agent-profile",
        help="Agent profile name for provider backends such as Claude Code or Codex.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="Provider API key for this invocation only.",
        hide_input=True,
        show_default=False,
    ),
    force_new: bool = typer.Option(False, "--force-new", help="Force fresh direct job IDs."),
    no_execute: bool = typer.Option(
        False,
        "--no-execute",
        help="Submit validated workflows/jobs but do not drive workflow execution.",
    ),
    plan_only: bool = typer.Option(
        False,
        "--plan-only",
        help="Classify/plan/validate only; do not submit jobs or workflows.",
    ),
    no_warm: bool = typer.Option(False, "--no-warm", help="Skip pre-warming tools."),
    max_rounds: int = typer.Option(8, "--max-rounds", help="Maximum agent optimization rounds."),
    max_entries: int = typer.Option(
        3,
        "--max-entries",
        help=(
            "Maximum failed LLM planning/decision entries before the agent run fails "
            "fast."
        ),
    ),
    max_wait_s: int = typer.Option(3600, "--max-wait-s", help="Workflow execution wait budget."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Classify, optionally plan, execute through VPA, and persist an agent run."""
    try:
        text = " ".join(request)
        if goal:
            text = f"{text}\nGoal: {goal}"
        ws = _ensure_workspace(workspace_name)
        params = _merge_params(params_file=params_file, param=param, target=target)
        config = AgentConfig.from_env(
            provider=provider,
            model=model,
            api_key=api_key,
            agent_profile=agent_profile,
            max_rounds=max_rounds,
            max_entries=max_entries,
        )
        controller = AgentController(ws, config=config)
        data = controller.run_sync(
            text,
            params=params,
            tool=tool,
            workflow=workflow,
            force_new=force_new,
            policy=ExecutionPolicy(
                submit=not plan_only,
                execute=not no_execute and not plan_only,
                warm=not no_warm and not plan_only,
                max_wait_s=max_wait_s,
            ),
        )
        print_envelope_ok(data, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@agent_app.command("ask")
def agent_ask(
    request: list[str] = typer.Argument(  # noqa: B008
        ..., help="Request to classify or plan."
    ),
    param: list[str] = typer.Option(None, "--param", help="KEY=VALUE param."),  # noqa: B008
    workspace_name: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    model: str | None = typer.Option(None, "--model", help="Agent model id."),
    provider: str | None = typer.Option(None, "--provider", help="Agent provider."),
    agent_profile: str | None = typer.Option(
        None,
        "--agent-profile",
        help="Agent profile name for provider backends such as Claude Code or Codex.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="Provider API key for this invocation only.",
        hide_input=True,
        show_default=False,
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Alias for `agent run --no-execute --no-warm`."""
    try:
        ws = _ensure_workspace(workspace_name)
        controller = AgentController(
            ws,
            config=AgentConfig.from_env(
                provider=provider,
                model=model,
                api_key=api_key,
                agent_profile=agent_profile,
            ),
        )
        data = controller.run_sync(
            " ".join(request),
            params=parse_kv_pairs(param),
            policy=ExecutionPolicy(submit=False, execute=False, warm=False, max_wait_s=0),
        )
        print_envelope_ok(data, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@agent_app.command("validate-plan")
def agent_validate_plan(
    path: str = typer.Argument(..., help="Agent plan JSON or workflow YAML."),
    workspace_name: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Validate an LLM-generated plan locally without executing it."""
    try:
        ws = _ensure_workspace(workspace_name)
        controller = AgentController(ws)
        plan = load_plan_or_workflow(path)
        valid = controller.validate_plan(plan)
        print_envelope_ok({"valid": True, "plan": valid}, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@agent_app.command("submit-plan")
def agent_submit_plan(
    path: str = typer.Argument(..., help="Agent plan JSON or workflow YAML."),
    param: list[str] = typer.Option(None, "--param", help="KEY=VALUE workflow param."),  # noqa: B008
    params_file: str | None = typer.Option(None, "--params-file"),
    workspace_name: str | None = typer.Option(None, "--workspace"),
    max_wait_s: int = typer.Option(3600, "--max-wait-s"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Submit a pre-authored, locally validated agent workflow plan."""
    try:
        ws = _ensure_workspace(workspace_name)
        controller = AgentController(ws)
        plan = load_plan_or_workflow(path)
        params = _merge_params(params_file=params_file, param=param, target=None)
        data = controller.submit_plan(plan, params, max_wait_s=max_wait_s)
        print_envelope_ok(data, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@agent_app.command("runs")
def agent_runs(
    workspace_name: str | None = typer.Option(None, "--workspace"),
    limit: int = typer.Option(50, "--limit"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List persisted agent runs in the workspace."""
    try:
        ws = _ensure_workspace(workspace_name)
        rows = [
            r.model_dump()
            for r in AgentSessionStore(ws).list_runs(limit=limit)
        ]
        print_envelope_ok({"runs": rows}, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@agent_app.command("show")
def agent_show(
    run_id: str = typer.Argument(..., help="Agent run id."),
    workspace_name: str | None = typer.Option(None, "--workspace"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show a persisted agent run record."""
    try:
        ws = _ensure_workspace(workspace_name)
        details = AgentSessionStore(ws).get_run_details(run_id)
        print_envelope_ok(details, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)

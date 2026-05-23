from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents import AgentConfig, AgentController, AgentSessionStore, ModeRouter
from vibeproteinagent.agents.evaluation import DecisionEngine
from vibeproteinagent.agents.execution import ExecutionPolicy
from vibeproteinagent.agents.planning import WorkflowPlanner
from vibeproteinagent.agents.planning.workflow_planner import _fallback_plan_from_prompt
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse, provider_from_config
from vibeproteinagent.agents.provider.cli_agent import CliAgentProvider
from vibeproteinagent.agents.tools.schemas import compact_tool_outline, compact_tool_schema
from vibeproteinagent.cli.main import app
from vibeproteinagent.core.errors import InvalidParamsError, TimeoutError_
from vibeproteinagent.mcp.tools import SessionCtx, list_workflows
from vibeproteinagent.workflows import WorkflowStore
from vibeproteinagent.workspace import ArtifactStore, JobState, JobStore, Workspace


def test_decision_engine_does_not_terminal_on_representative_sequence() -> None:
    decision = DecisionEngine(None).local_decision(
        {
            "state": "completed",
            "nodes": [
                {
                    "node_id": "sequence_summary",
                    "output": {
                        "representative_sequences": [
                            {"sequence": "ACDEFGHIK", "length": 9}
                        ]
                    },
                }
            ],
            "metrics_used": ["metrics.family_summary.sequence_count"],
        },
        success_criteria={"final_response_includes_one_output_protein_sequence": True},
    )

    assert decision["decision"] == "completed_review_metrics"
    assert decision["next_round"]["recommended"] is False


def test_decision_engine_does_not_terminal_on_ranked_design() -> None:
    decision = DecisionEngine(None).local_decision(
        {
            "state": "completed",
            "top_designs": [{"rank": 1, "design": {"sequence": "ACDEFGHIK"}}],
        },
        success_criteria={"return_top_ranked_sequence": True},
    )

    assert decision["decision"] == "completed_review_metrics"
    assert decision["next_round"]["recommended"] is False


def _run_pending_jobs(store: JobStore, registry: ToolRegistry, stop: threading.Event) -> None:
    import typing

    import torch

    from vibeproteinagent.adapters.context import null_context

    device = torch.device("cpu")
    weights_dir = Path("/tmp/vpa_test_weights")
    weights_dir.mkdir(exist_ok=True)
    while not stop.is_set():
        recs = store.list(state=JobState.QUEUED, limit=8)
        if not recs:
            time.sleep(0.02)
            continue
        for rec in recs:
            claimed = store.claim_next_queued(worker_id=0, gpu_ids=[], job_id=rec.job_id)
            if claimed is None:
                continue
            _, cls = registry.get(rec.tool_name, "*")
            adapter = cls(config={}, weights_dir=weights_dir, device=device)
            hints = typing.get_type_hints(adapter.run)
            input_model = hints.get("params")
            params = input_model(**rec.params) if input_model is not None else rec.params
            out = adapter.run(params, null_context(rec.job_id))
            payload = out.model_dump() if hasattr(out, "model_dump") else dict(out)
            store.transition(rec.job_id, JobState.COMPLETED, output=payload)


@pytest.fixture
def inline_supervisor(tmp_workspace: Workspace):
    store = JobStore(tmp_workspace)
    registry = ToolRegistry()
    registry.discover()
    stop = threading.Event()
    thread = threading.Thread(
        target=_run_pending_jobs,
        args=(store, registry, stop),
        daemon=True,
    )
    thread.start()
    try:
        yield object()
    finally:
        stop.set()
        thread.join(timeout=2)


def test_mode_router_preserves_direct_tool_without_llm(tmp_workspace: Workspace) -> None:
    registry = ToolRegistry()
    registry.discover()
    route = ModeRouter(registry).classify("run noop payload=hello")

    assert route.mode == "direct_tool"
    assert route.requires_llm_planning is False
    assert route.requested_tools == ["noop"]
    assert route.params == {"payload": "hello"}


def test_mode_router_splits_direct_tool_set_params(tmp_workspace: Workspace) -> None:
    registry = ToolRegistry()
    registry.discover()
    route = ModeRouter(registry).classify(
        "run noop then protein_qc noop.payload=hello protein_qc.input_dir=outputs/qc"
    )

    assert route.mode == "direct_tool_set"
    assert route.execution == {"type": "sequential"}
    by_tool = {job["tool"]: job["params"] for job in route.direct_jobs}
    assert by_tool["noop"] == {"payload": "hello"}
    assert by_tool["protein_qc"] == {"input_dir": "outputs/qc"}


def test_mode_router_defaults_direct_tool_set_to_parallel(tmp_workspace: Workspace) -> None:
    registry = ToolRegistry()
    registry.discover()
    route = ModeRouter(registry).classify("run noop and protein_qc")

    assert route.mode == "direct_tool_set"
    assert route.execution == {"type": "parallel"}


def test_mode_router_keeps_design_chains_in_agent_planning(tmp_workspace: Workspace) -> None:
    registry = ToolRegistry()
    registry.discover()
    route = ModeRouter(registry).classify(
        "Use rfdiffusion, proteinmpnn, protenix_v2, structure_analysis, "
        "protein_qc, and rank_designs to design and return an output sequence."
    )

    assert route.mode == "agent_plan"
    assert route.requires_llm_planning is True


def test_mode_router_result_word_inside_scientific_text_is_not_direct(
    tmp_workspace: Workspace,
) -> None:
    registry = ToolRegistry()
    registry.discover()
    route = ModeRouter(registry).classify(
        "Design an enzyme for damaged NAD(P)H that is a result of hydration."
    )

    assert route.mode == "agent_plan"


def test_mode_router_fresh_analysis_request_uses_planning(tmp_workspace: Workspace) -> None:
    registry = ToolRegistry()
    registry.discover()
    route = ModeRouter(registry).classify("analyze this sequence for liabilities")

    assert route.mode == "agent_plan"
    assert route.requires_llm_planning is True


def test_template_recovery_handles_semantic_function_requests(
    tmp_workspace: Workspace,
) -> None:
    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))

    plan = _fallback_plan_from_prompt(
        {
            "request": "Design a protein with NADPHX epimerase activity and Mg2+ binding.",
            "workflow_templates": workflows,
            "tool_selection": {},
        },
        "",
    )

    assert plan is not None
    assert plan["workflow_plan"]["name"] == "semantic_constraint_to_family_sequences"
    assert plan["workflow_ref"] == "semantic_constraint_to_family_sequences"
    assert plan["params"]["query"].startswith("Design a protein")


def test_template_recovery_handles_target_sequence_binder_requests(
    tmp_workspace: Workspace,
) -> None:
    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))
    target_sequence = (
        "MKAVEVANNILFSVMMFNGAQTVIQKISWLRTAIAFLKGYMETTGATKKELEQVEK"
        "LKERVDEIATAVNWDVYAQYARGDFNLLSDDEYKEIQKALLVLEDIKEQIIVEM"
        "LRVGLAQGQMGTLKISDYLDSLIKERK"
    )
    request = (
        "Design a protein binder that targets a specific site.\n"
        f"Sequence: {target_sequence}\n"
        "Target binding site: Y73, Y76, Y128, F36, L132, V72, Y40, A35, Q25, G39"
    )

    plan = _fallback_plan_from_prompt(
        {"request": request, "workflow_templates": workflows, "tool_selection": {}},
        "",
    )

    assert plan is not None
    assert plan["workflow_plan"]["name"] == "target_sequence_binder_design_round"
    params = plan["params"]
    assert params["target_sequence"] == target_sequence
    assert params["target_hotspots"] == [73, 76, 128, 36, 132, 72, 40, 35, 25, 39]


def test_template_recovery_defaults_de_novo_design_length(
    tmp_workspace: Workspace,
) -> None:
    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))
    request = (
        "Use rfdiffusion, proteinmpnn, protenix_v2, structure_analysis, "
        "protein_qc, and rank_designs to design a de novo soluble protein candidate."
    )

    plan = _fallback_plan_from_prompt(
        {"request": request, "workflow_templates": workflows, "tool_selection": {}},
        "",
    )

    assert plan is not None
    assert plan["workflow_plan"]["name"] == "rfdiffusion_backbone_design_round"
    assert plan["params"]["length"] == 100


def test_template_recovery_extracts_requested_de_novo_length(
    tmp_workspace: Workspace,
) -> None:
    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))

    plan = _fallback_plan_from_prompt(
        {
            "request": "Design a de novo 85 aa soluble protein with RFdiffusion.",
            "workflow_templates": workflows,
            "tool_selection": {},
        },
        "",
    )

    assert plan is not None
    assert plan["params"]["length"] == 85


def test_template_recovery_canonicalizes_runtime_param_values(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner.descriptor_availability",
        lambda *args, **kwargs: {"available": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )
    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))
    request = "Design a protein with NADPHX epimerase activity and Mg2+ binding."
    raw_plan = {
        "mode": "agent_plan",
        "goal": request,
        "assumptions": [],
        "missing_inputs": [],
        "required_tools": ["literature_search", "go_sequences", "sequence_analysis"],
        "workflow_plan": {
            "name": "semantic_constraint_to_family_sequences",
            "version": "1.0.0",
            "params": {"query": request, "max_sequences": 200},
            "nodes": [],
        },
        "success_criteria": {},
        "warm_tools": [],
    }

    planner = WorkflowPlanner(registry)
    plan = planner._canonicalize_template_plan(
        {"request": request, "workflow_templates": workflows, "tool_selection": {}},
        raw_plan,
    )
    validated = planner.validate_plan(plan)

    params = validated["workflow_plan"]["params"]
    assert validated["params"]["query"] == request
    assert validated["params"]["max_sequences"] == 200
    assert params["query"]["type"] == "str"
    assert validated["workflow_plan"]["nodes"]


def test_template_recovery_overrides_bad_provider_selection(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner.descriptor_availability",
        lambda *args, **kwargs: {"available": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )

    class FakeProvider:
        async def chat(self, messages, **kwargs):
            prompt = json.loads(messages[-1].content)
            if prompt["output_contract"]["mode"] == "tool_selection":
                return ProviderResponse(
                    provider="fake",
                    model="fake",
                    content=json.dumps(
                        {
                            "mode": "tool_selection",
                            "candidate_tools": ["mpnn_to_protenix_input"],
                            "reason": "bad internal glue selection",
                            "missing_inputs": [],
                        }
                    ),
                )
            return ProviderResponse(
                provider="fake",
                model="fake",
                content=json.dumps({"mode": "agent_plan", "goal": "incomplete"}),
            )

    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))
    request = "Design a protein with NADPHX epimerase activity and Mg2+ binding."

    plan = asyncio.run(
        WorkflowPlanner(registry, FakeProvider()).draft_plan(
            request,
            available_tools=[
                {"name": "literature_search", "input_schema": {}},
                {"name": "go_sequences", "input_schema": {}},
                {"name": "sequence_analysis", "input_schema": {}},
            ],
            workflows=workflows,
        )
    )

    assert plan["workflow_plan"]["name"] == "semantic_constraint_to_family_sequences"
    assert plan["params"]["query"] == request


def test_workflow_ref_contract_accepts_runtime_params(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner.descriptor_availability",
        lambda *args, **kwargs: {"available": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )

    class FakeProvider:
        async def chat(self, messages, **kwargs):
            prompt = json.loads(messages[-1].content)
            if prompt["output_contract"]["mode"] == "tool_selection":
                return ProviderResponse(
                    provider="fake",
                    model="fake",
                    content=json.dumps(
                        {
                            "mode": "tool_selection",
                            "candidate_tools": [
                                "rfdiffusion",
                                "proteinmpnn",
                                "protenix_v2",
                            ],
                            "reason": "de novo backbone workflow",
                            "missing_inputs": [],
                        }
                    ),
                )
            return ProviderResponse(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "mode": "agent_plan",
                        "goal": "Design an 85 aa protein.",
                        "assumptions": [],
                        "missing_inputs": [],
                        "required_tools": ["rfdiffusion", "proteinmpnn", "protenix_v2"],
                        "workflow_ref": "rfdiffusion_backbone_design_round",
                        "params": {"length": 85},
                        "success_criteria": {"workflow_completes": True},
                        "warm_tools": ["rfdiffusion", "proteinmpnn", "protenix_v2"],
                        "estimated_llm_calls": 1,
                    }
                ),
            )

    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))

    plan = asyncio.run(
        WorkflowPlanner(registry, FakeProvider()).draft_plan(
            "Design an 85 aa protein with RFdiffusion.",
            available_tools=[
                {"name": "rfdiffusion", "input_schema": {}},
                {"name": "proteinmpnn", "input_schema": {}},
                {"name": "protenix_v2", "input_schema": {}},
                {"name": "structure_analysis", "input_schema": {}},
                {"name": "protein_qc", "input_schema": {}},
                {"name": "rank_designs", "input_schema": {}},
            ],
            workflows=workflows,
        )
    )

    assert plan["workflow_ref"] == "rfdiffusion_backbone_design_round"
    assert plan["params"] == {"length": 85}
    assert plan["workflow_plan"]["params"]["length"]["type"] == "int"
    WorkflowPlanner(registry).validate_invocation(plan, {})


def test_workflow_ref_contract_rejects_wrong_runtime_param_type(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner.descriptor_availability",
        lambda *args, **kwargs: {"available": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )
    registry = ToolRegistry()
    registry.discover()
    workflows = list_workflows(SessionCtx(tmp_workspace, registry))
    planner = WorkflowPlanner(registry)

    plan = planner._canonicalize_template_plan(
        {
            "request": "Design a de novo protein.",
            "workflow_templates": workflows,
            "tool_selection": {},
        },
        {
            "mode": "agent_plan",
            "workflow_ref": "rfdiffusion_backbone_design_round",
            "params": {"length": "eighty-five"},
        },
    )

    with pytest.raises(InvalidParamsError, match="runtime param has the wrong type"):
        planner.validate_invocation(plan, {})


def test_agent_direct_tool_set_accepts_explicit_jobs(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)
    data = controller.run_sync(
        "run noop and protein_qc",
        params={
            "jobs": [
                {"tool": "noop", "params": {"payload": "a"}},
                {"tool": "noop", "params": {"payload": "b"}},
            ]
        },
    )

    assert data["mode"] == "direct_tool_set"
    assert len(data["job_ids"]) == 2
    assert data["result"]["jobs"][0]["tool"] == "noop"


def test_agent_direct_tool_set_reports_submission_failures(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)
    data = controller.run_sync(
        "run noop and protein_qc",
        params={
            "jobs": [
                {"tool": "noop", "params": {"payload": "a"}},
                {"tool": "noop", "params": {"x": "missing-payload"}},
            ]
        },
    )

    assert data["mode"] == "direct_tool_set"
    assert data["state"] == "partial_failed"
    assert data["result"]["jobs"][1]["code"] == "E_INVALID_PARAMS"


def test_agent_plan_only_direct_tool_validates_params(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)

    with pytest.raises(Exception) as exc_info:
        controller.run_sync(
            "run noop",
            params={"x": "missing-payload"},
            policy=ExecutionPolicy(submit=False, execute=False, warm=False),
        )

    assert getattr(exc_info.value, "code", None) == "E_INVALID_PARAMS"


def test_agent_sequential_tool_set_without_daemon_does_not_submit_downstream(
    tmp_workspace: Workspace,
) -> None:
    controller = AgentController(tmp_workspace)

    data = controller.run_sync(
        "run noop then protein_qc",
        params={
            "jobs": [
                {"tool": "noop", "params": {"payload": "a"}},
                {"tool": "protein_qc", "params": {"input_dir": "outputs/qc"}},
            ]
        },
    )

    assert data["mode"] == "direct_tool_set"
    assert data["classification"]["execution"] == {"type": "sequential"}
    assert len(data["job_ids"]) == 1


def test_agent_direct_tool_submits_and_persists_run(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)
    data = controller.run_sync("run noop payload=hello")

    assert data["mode"] == "direct_tool"
    assert data["job_ids"]
    assert data["state"] == "queued"

    runs = AgentSessionStore(tmp_workspace).list_runs()
    assert runs[0].run_id == data["run_id"]
    assert runs[0].mode == "direct_tool"
    assert runs[0].job_ids == data["job_ids"]
    assert runs[0].final_report is not None


def test_agent_direct_warm_does_not_submit_job(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)
    data = controller.run_sync("warm noop")

    assert data["mode"] == "direct_tool"
    assert data["classification"]["action"] == "warm"
    assert data["job_ids"] == []
    assert data["result"]["code"] == "E_DAEMON_UNAVAILABLE"


def test_agent_status_request_does_not_submit_new_job(tmp_workspace: Workspace) -> None:
    created = AgentController(tmp_workspace).run_sync("run noop payload=hello")
    job_id = created["job_ids"][0]
    before = len(JobStore(tmp_workspace).list())

    status = AgentController(tmp_workspace).run_sync(f"status noop {job_id}")

    assert status["mode"] == "direct_tool"
    assert status["classification"]["action"] == "get_status"
    assert status["job_ids"] == [job_id]
    assert len(JobStore(tmp_workspace).list()) == before


def test_agent_config_accepts_invocation_api_key_without_env() -> None:
    config = AgentConfig.from_env(api_key="sk-test")

    assert config.api_key == "sk-test"


def test_agent_config_accepts_openrouter_api_key_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("VPA_OPENROUTER_CREDENTIAL", raising=False)

    config = AgentConfig.from_env(provider="openrouter")

    assert config.api_key == "sk-test"


def test_agent_config_defaults_to_local_cli_provider(monkeypatch) -> None:
    monkeypatch.delenv("VPA_AGENT_PROVIDER", raising=False)
    monkeypatch.delenv("VPA_AGENT_MODEL", raising=False)
    monkeypatch.delenv("VPA_AGENT_MAX_ROUNDS", raising=False)
    monkeypatch.delenv("VPA_AGENT_MAX_ENTRIES", raising=False)
    config = AgentConfig.from_env()

    assert config.provider == "codex"
    assert config.max_rounds == 8
    assert config.max_entries == 3
    assert isinstance(provider_from_config(config), CliAgentProvider)


def test_agent_config_reads_max_entries_env(monkeypatch) -> None:
    monkeypatch.setenv("VPA_AGENT_MAX_ENTRIES", "4")
    monkeypatch.setenv("VPA_AGENT_MAX_ROUNDS", "6")

    config = AgentConfig.from_env()

    assert config.max_entries == 4
    assert config.max_rounds == 6


def test_agent_config_parses_provider_model_refs() -> None:
    claude = AgentConfig.from_env(model="claude-code/sonnet", agent_profile="plan")
    codex = AgentConfig.from_env(model="codex/gpt-5.2")
    openrouter = AgentConfig.from_env(model="openrouter/openai/gpt-5.1")
    explicit_openrouter = AgentConfig.from_env(
        provider="openrouter",
        model="openrouter/anthropic/claude-sonnet-4.5",
    )

    assert claude.provider == "claude-code"
    assert claude.model == "sonnet"
    assert claude.agent_profile == "plan"
    assert codex.provider == "codex"
    assert codex.model == "gpt-5.2"
    assert openrouter.provider == "openrouter"
    assert openrouter.model == "openai/gpt-5.1"
    assert explicit_openrouter.provider == "openrouter"
    assert explicit_openrouter.model == "anthropic/claude-sonnet-4.5"
    assert isinstance(provider_from_config(claude), CliAgentProvider)
    assert isinstance(provider_from_config(codex), CliAgentProvider)


def test_cli_agent_provider_reads_json_from_subprocess(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import sys\n"
        "_ = sys.stdin.read()\n"
        "print('{\"mode\":\"agent_plan\",\"workflow_plan\":{\"name\":\"x\",\"nodes\":[]}}')\n"
    )
    monkeypatch.setenv("VPA_CODEX_COMMAND", "python")
    monkeypatch.setenv("VPA_CODEX_ARGS", str(script))
    provider = provider_from_config(AgentConfig.from_env(provider="codex", model="fake"))

    import asyncio

    response = asyncio.run(
        provider.chat(
            [ChatMessage(role="user", content="return json")],
            response_format={"type": "json_object"},
        )
    )

    assert json.loads(response.content)["workflow_plan"]["name"] == "x"


def test_cli_agent_provider_passes_model_argument(
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "fake_agent.py"
    argv_path = tmp_path / "argv.json"
    script.write_text(
        "import json, os, sys\n"
        "open(os.environ['ARGV_PATH'], 'w').write(json.dumps(sys.argv))\n"
        "print('{\"mode\":\"agent_plan\",\"workflow_plan\":{\"name\":\"x\",\"nodes\":[]}}')\n"
    )
    monkeypatch.setenv("ARGV_PATH", str(argv_path))
    monkeypatch.setenv("VPA_CODEX_COMMAND", "python")
    monkeypatch.setenv("VPA_CODEX_ARGS", str(script))
    provider = provider_from_config(AgentConfig.from_env(provider="codex", model="fake-model"))

    import asyncio

    asyncio.run(
        provider.chat(
            [ChatMessage(role="user", content="return json")],
            response_format={"type": "json_object"},
        )
    )

    assert json.loads(argv_path.read_text())[-2:] == ["--model", "fake-model"]


def test_direct_mode_does_not_require_valid_provider(tmp_workspace: Workspace) -> None:
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="not-a-provider", model="unused"),
    )

    data = controller.run_sync("run noop payload=hello")

    assert data["mode"] == "direct_tool"
    assert data["job_ids"]


def test_agent_ask_plans_without_submitting_workflow(
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "fake_planner.py"
    script.write_text(
        "import sys\n"
        "_ = sys.stdin.read()\n"
        "print('''{\"mode\":\"agent_plan\",\"goal\":\"x\",\"assumptions\":[],"
        "\"missing_inputs\":[],\"required_tools\":[\"protein_qc\"],"
        "\"workflow_plan\":{\"name\":\"ask_qc\",\"version\":\"0.1.0\","
        "\"params\":{\"sequence\":{\"type\":\"str\",\"required\":true}},"
        "\"nodes\":[{\"id\":\"n\",\"tool\":\"protein_qc\","
        "\"inputs\":{\"sequence\":\"${params.sequence}\"}}]},"
        "\"success_criteria\":{},\"warm_tools\":[\"protein_qc\"],"
        "\"estimated_llm_calls\":1}''')\n"
    )
    monkeypatch.setenv("VPA_CODEX_COMMAND", "python")
    monkeypatch.setenv("VPA_CODEX_ARGS", str(script))
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake"),
    )

    data = controller.run_sync(
        "check this sequence",
        params={"sequence": "ACDEFGHIK"},
        policy=ExecutionPolicy(submit=False, execute=False, warm=False),
    )

    assert data["mode"] == "agent_plan"
    assert data["state"] == "planned"
    assert data["workflow_ids"] == []
    assert WorkflowStore(tmp_workspace).list() == []


def test_explicit_workflow_plan_only_validates_descriptor(
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
name: bad
version: 0.1.0
nodes:
  - id: n
    tool: not_a_tool
    inputs: {}
"""
    )
    controller = AgentController(tmp_workspace)

    with pytest.raises(Exception) as exc_info:
        controller.run_sync(
            "run workflow",
            workflow=str(bad),
            policy=ExecutionPolicy(submit=False, execute=False, warm=False),
        )

    assert getattr(exc_info.value, "code", None) == "E_UNKNOWN_TOOL"


def test_explicit_workflow_no_execute_submits_without_running(
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    wf_path = tmp_path / "noop_wf.yaml"
    wf_path.write_text(
        """
name: noop_wf
version: 0.1.0
params:
  payload:
    type: str
    required: true
nodes:
  - id: n
    tool: noop
    inputs:
      payload: ${params.payload}
"""
    )
    controller = AgentController(tmp_workspace)

    data = controller.run_sync(
        "run workflow",
        workflow=str(wf_path),
        params={"payload": "hello"},
        policy=ExecutionPolicy(submit=True, execute=False, warm=False),
    )

    assert data["state"] == "submitted_not_executed"
    assert data["workflow_ids"]
    assert data["job_ids"] == []
    nodes = WorkflowStore(tmp_workspace).list_nodes(data["workflow_id"])
    assert [node.state for node in nodes] == ["pending"]


def test_agent_plan_blocks_without_daemon_before_submit(
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "fake_planner.py"
    script.write_text(
        "import sys\n"
        "_ = sys.stdin.read()\n"
        "print('''{\"mode\":\"agent_plan\",\"goal\":\"x\",\"assumptions\":[],"
        "\"missing_inputs\":[],\"required_tools\":[\"protein_qc\"],"
        "\"workflow_plan\":{\"name\":\"blocked_qc\",\"version\":\"0.1.0\","
        "\"params\":{\"sequence\":{\"type\":\"str\",\"required\":true}},"
        "\"nodes\":[{\"id\":\"n\",\"tool\":\"protein_qc\","
        "\"inputs\":{\"sequence\":\"${params.sequence}\"}}]},"
        "\"success_criteria\":{},\"warm_tools\":[],"
        "\"estimated_llm_calls\":1}''')\n"
    )
    monkeypatch.setenv("VPA_CODEX_COMMAND", "python")
    monkeypatch.setenv("VPA_CODEX_ARGS", str(script))
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake"),
    )

    data = controller.run_sync("check this sequence", params={"sequence": "ACDEFGHIK"})

    assert data["state"] == "daemon_unavailable"
    assert data["workflow_ids"] == []
    assert data["decision"]["decision"] == "workflow_not_completed"
    assert WorkflowStore(tmp_workspace).list() == []
    run = AgentSessionStore(tmp_workspace).get_run(data["run_id"])
    assert run.state == "failed"


def test_agent_plan_timeout_preserves_workflow_context(
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch,
) -> None:
    script = tmp_path / "fake_planner.py"
    script.write_text(
        "import sys\n"
        "_ = sys.stdin.read()\n"
        "print('''{\"mode\":\"agent_plan\",\"goal\":\"x\",\"assumptions\":[],"
        "\"missing_inputs\":[],\"required_tools\":[\"protein_qc\"],"
        "\"workflow_plan\":{\"name\":\"timeout_qc\",\"version\":\"0.1.0\","
        "\"params\":{\"sequence\":{\"type\":\"str\",\"required\":true}},"
        "\"nodes\":[{\"id\":\"n\",\"tool\":\"protein_qc\","
        "\"inputs\":{\"sequence\":\"${params.sequence}\"}}]},"
        "\"success_criteria\":{},\"warm_tools\":[],"
        "\"estimated_llm_calls\":1}''')\n"
    )
    monkeypatch.setenv("VPA_CODEX_COMMAND", "python")
    monkeypatch.setenv("VPA_CODEX_ARGS", str(script))
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.SupervisorClient",
        lambda: type("FakeSupervisor", (), {"is_running": lambda self: True})(),
    )

    def _timeout(self, workflow_id, *, max_wait_s=3600, poll_s=0.25):
        self.store.update_state(workflow_id, "failed", finished=True)
        raise TimeoutError_("workflow timed out", details={"workflow_id": workflow_id})

    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.WorkflowExecutor.run_sync",
        _timeout,
    )
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake"),
    )

    data = controller.run_sync("check this sequence", params={"sequence": "ACDEFGHIK"})

    assert data["state"] == "timeout"
    assert data["workflow_ids"]
    assert data["summary"]["latest"]["error"]["code"] == "E_TIMEOUT"
    run = AgentSessionStore(tmp_workspace).get_run(data["run_id"])
    assert run.state == "failed"


def test_agent_submit_plan_timeout_preserves_workflow_context(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.SupervisorClient",
        lambda: type("FakeSupervisor", (), {"is_running": lambda self: True})(),
    )

    def _timeout(self, workflow_id, *, max_wait_s=3600, poll_s=0.25):
        self.store.update_state(workflow_id, "failed", finished=True)
        raise TimeoutError_("workflow timed out", details={"workflow_id": workflow_id})

    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.WorkflowExecutor.run_sync",
        _timeout,
    )
    plan = {
        "mode": "agent_plan",
        "goal": "x",
        "assumptions": [],
        "missing_inputs": [],
        "required_tools": ["protein_qc"],
        "workflow_plan": {
            "name": "submit_timeout_qc",
            "version": "0.1.0",
            "params": {"sequence": {"type": "str", "required": True}},
            "nodes": [
                {
                    "id": "n",
                    "tool": "protein_qc",
                    "inputs": {"sequence": "${params.sequence}"},
                }
            ],
        },
        "success_criteria": {},
        "warm_tools": [],
        "estimated_llm_calls": 0,
    }
    controller = AgentController(tmp_workspace)

    data = controller.submit_plan(plan, {"sequence": "ACDEFGHIK"}, max_wait_s=0)

    assert data["state"] == "timeout"
    assert data["workflow_ids"]
    assert data["summary"]["error"]["code"] == "E_TIMEOUT"
    assert data["result"]["stored_state"] == "failed"
    run = AgentSessionStore(tmp_workspace).get_run(data["run_id"])
    assert run.state == "failed"


def test_agent_plan_controller_executes_and_logs_provider_messages(
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch,
    inline_supervisor,
) -> None:
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import sys\n"
        "prompt = sys.stdin.read()\n"
        "if 'result critic' in prompt.lower():\n"
        "    print('{\"decision\":\"done\",\"next_round\":{\"recommended\":false,"
        "\"changes\":[],\"reason\":\"ok\"},\"metrics_used\":[],\"notes\":[]}')\n"
        "else:\n"
        "    print('''{\"mode\":\"agent_plan\",\"goal\":\"x\",\"assumptions\":[],"
        "\"missing_inputs\":[],\"required_tools\":[\"protein_qc\"],"
        "\"workflow_plan\":{\"name\":\"exec_qc\",\"version\":\"0.1.0\","
        "\"params\":{\"sequence\":{\"type\":\"str\",\"required\":true}},"
        "\"nodes\":[{\"id\":\"n\",\"tool\":\"protein_qc\","
        "\"inputs\":{\"sequence\":\"${params.sequence}\"}}]},"
        "\"success_criteria\":{},\"warm_tools\":[],"
        "\"estimated_llm_calls\":2}''')\n"
    )
    monkeypatch.setenv("VPA_CODEX_COMMAND", "python")
    monkeypatch.setenv("VPA_CODEX_ARGS", str(script))
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.SupervisorClient",
        lambda: type("FakeSupervisor", (), {"is_running": lambda self: True})(),
    )
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake"),
    )

    data = controller.run_sync("check this sequence", params={"sequence": "ACDEFGHIK"})

    assert data["mode"] == "agent_plan"
    assert data["state"] == "completed"
    assert data["workflow_ids"]
    run = AgentSessionStore(tmp_workspace).get_run(data["run_id"])
    assert run.provider_snapshot is not None
    assert run.final_report["decision"]["decision"] == "done"


def test_agent_plan_retries_bad_llm_entries_within_budget(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    class FlakyProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            if self.calls <= 2:
                return ProviderResponse(provider="fake", model="fake", content="not json")
            if self.calls == 3:
                return ProviderResponse(
                    provider="fake",
                    model="fake",
                    content=json.dumps(
                        {
                            "mode": "tool_selection",
                            "candidate_tools": ["protein_qc"],
                            "reason": "sequence quality check",
                            "missing_inputs": [],
                        }
                    ),
                )
            return ProviderResponse(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "mode": "agent_plan",
                        "goal": "qc",
                        "assumptions": [],
                        "missing_inputs": [],
                        "required_tools": ["protein_qc"],
                        "workflow_plan": {
                            "name": "qc",
                            "version": "0.1.0",
                            "params": {"sequence": {"type": "str", "required": True}},
                            "nodes": [
                                {
                                    "id": "qc",
                                    "tool": "protein_qc",
                                    "inputs": {"sequence": "${params.sequence}"},
                                }
                            ],
                        },
                        "success_criteria": {},
                        "warm_tools": [],
                        "estimated_llm_calls": 2,
                    }
                ),
            )

    provider = FlakyProvider()
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda config: provider,
    )
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake", max_entries=3),
    )

    data = controller.run_sync(
        "check this sequence",
        params={"sequence": "ACDEFGHIK"},
        policy=ExecutionPolicy(submit=False, execute=False, warm=False),
    )

    assert data["state"] == "planned"
    assert provider.calls == 4


def test_agent_plan_fails_after_max_entries(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    class BadProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, **kwargs):
            del messages, kwargs
            self.calls += 1
            return ProviderResponse(provider="fake", model="fake", content="not json")

    provider = BadProvider()
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda config: provider,
    )
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake", max_entries=2),
    )

    with pytest.raises(Exception) as exc_info:
        controller.run_sync(
            "check this sequence",
            params={"sequence": "ACDEFGHIK"},
            policy=ExecutionPolicy(submit=False, execute=False, warm=False),
        )

    assert getattr(exc_info.value, "code", None) == "E_INVALID_PARAMS"
    assert exc_info.value.details["max_entries"] == 2
    assert exc_info.value.details["failure_count"] == 2
    assert provider.calls == 2


def test_job_and_workflow_summaries_include_artifact_refs(
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    created = AgentController(tmp_workspace).run_sync("run noop payload=artifact")
    job_id = created["job_ids"][0]
    artifact_file = tmp_path / "result.txt"
    artifact_file.write_text("small artifact")
    ArtifactStore(tmp_workspace).register(job_id, "report", artifact_file, kind="text")

    summary = AgentController(tmp_workspace).run_sync(f"summarize {job_id}")

    artifacts = summary["summary"]["jobs"][0]["artifacts"]
    assert artifacts["report"]["artifact_id"].startswith("art_")

    run = AgentSessionStore(tmp_workspace).get_run(summary["run_id"])
    assert run.artifacts["jobs"][0]["summary"]["report"]["artifact_id"].startswith("art_")


def test_workflow_summary_extracts_ranked_candidates() -> None:
    from types import SimpleNamespace

    from vibeproteinagent.agents.summarization.result_summarizer import _extract_top_designs

    nodes = [
        SimpleNamespace(
            node_id="rank",
            output={
                "ranked_candidates": [
                    {"candidate_id": "a", "score": 1.0},
                    {"candidate_id": "b", "score": 0.5},
                ]
            },
        )
    ]

    designs = _extract_top_designs(nodes)

    assert designs[0]["node_id"] == "rank"
    assert designs[0]["design"]["candidate_id"] == "a"


def test_workflow_summary_exposes_representative_sequences_as_candidates() -> None:
    from types import SimpleNamespace

    from vibeproteinagent.agents.summarization.result_summarizer import _extract_top_designs

    nodes = [
        SimpleNamespace(
            node_id="sequence_summary",
            output={
                "representative_sequences": [
                    {
                        "candidate_id": "representative:1:abc",
                        "sequence": "ACDEFGHIK",
                    }
                ]
            },
        )
    ]

    designs = _extract_top_designs(nodes)

    assert designs[0]["node_id"] == "sequence_summary"
    assert designs[0]["design"]["candidate_id"] == "representative:1:abc"


def test_compact_tool_schema_keeps_only_planner_facing_fields() -> None:
    compact = compact_tool_schema(
        {
            "name": "rfdiffusion",
            "description": "Backbone generation.",
            "available": False,
            "availability_status": "unavailable",
            "availability_reason": "missing weights",
            "availability_details": {"weights": {"ok": False}},
            "support_status": "verified",
            "gpu_required": True,
            "vram_min_gib": 24,
            "sharding": {
                "strategy": "design_seed_shards",
                "shardable_fields": ["num_designs", "seed"],
                "recommended_shard_size": 1,
                "default_max_parallel": 4,
            },
            "input_schema": {
                "type": "object",
                "required": ["mode"],
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["unconditional", "binder_backbone_design"],
                        "default": "unconditional",
                        "description": "RFdiffusion design mode.",
                    }
                },
            },
        }
    )

    assert compact == {
        "name": "rfdiffusion",
        "description": "Backbone generation.",
        "sharding": {
            "strategy": "design_seed_shards",
            "shardable_fields": ["num_designs", "seed"],
            "recommended_shard_size": 1,
            "default_max_parallel": 4,
        },
        "parameters": {
            "required": ["mode"],
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["unconditional", "binder_backbone_design"],
                    "default": "unconditional",
                    "description": "RFdiffusion design mode.",
                }
            },
        },
    }


def test_compact_tool_outline_excludes_parameters_and_runtime_metadata() -> None:
    assert compact_tool_outline(
        {
            "name": "rfdiffusion",
            "description": "Backbone generation.",
            "gpu_required": True,
            "input_schema": {"properties": {"mode": {"type": "string"}}},
        }
    ) == {
        "name": "rfdiffusion",
        "description": "Backbone generation.",
    }


def test_workflow_planner_two_stage_prompt_resolves_selected_parameters(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    import asyncio

    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner.descriptor_availability",
        lambda descriptor: {"available": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )

    class FakeProvider:
        provider_name = "fake"

        def __init__(self) -> None:
            self.prompts: list[dict] = []

        async def chat(self, messages, **kwargs):
            prompt = json.loads(messages[-1].content)
            self.prompts.append(prompt)
            if prompt["output_contract"]["mode"] == "tool_selection":
                return ProviderResponse(
                    provider="fake",
                    model="fake",
                    content=json.dumps(
                        {
                            "mode": "tool_selection",
                            "candidate_tools": ["protein_qc"],
                            "reason": "sequence quality check",
                            "missing_inputs": [],
                        }
                    ),
                )
            return ProviderResponse(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "mode": "agent_plan",
                        "goal": "qc",
                        "assumptions": [],
                        "missing_inputs": [],
                        "required_tools": ["protein_qc"],
                        "workflow_plan": {
                            "name": "qc",
                            "version": "0.1.0",
                            "params": {"sequence": {"type": "str", "required": True}},
                            "nodes": [
                                {
                                    "id": "qc",
                                    "tool": "protein_qc",
                                    "inputs": {"sequence": "${params.sequence}"},
                                }
                            ],
                        },
                        "success_criteria": {},
                        "warm_tools": [],
                        "estimated_llm_calls": 2,
                    }
                ),
            )

    provider = FakeProvider()
    registry = ToolRegistry()
    registry.discover()

    plan = asyncio.run(
        WorkflowPlanner(registry, provider).draft_plan(
            "check sequence quality",
            available_tools=[
                {
                    "name": "protein_qc",
                    "description": "Protein sequence and structure QC.",
                    "input_schema": "schemas/protein_qc.input.json",
                    "sharding": {
                        "strategy": "candidate_shards",
                        "shardable_fields": ["sequence"],
                        "recommended_shard_size": 1,
                        "default_max_parallel": 4,
                    },
                }
            ],
            workflows=[],
        )
    )

    assert plan["workflow_plan"]["nodes"][0]["tool"] == "protein_qc"
    first_prompt, second_prompt = provider.prompts
    assert first_prompt["available_tools"] == [
        {
            "name": "protein_qc",
            "description": "Protein sequence and structure QC.",
            "sharding": {
                "strategy": "candidate_shards",
                "shardable_fields": ["sequence"],
                "recommended_shard_size": 1,
                "default_max_parallel": 4,
            },
        }
    ]
    selected = second_prompt["selected_tools"][0]
    assert selected["name"] == "protein_qc"
    assert selected["sharding"]["strategy"] == "candidate_shards"
    assert "sequence" in selected["parameters"]["properties"]
    assert selected["parameters"]["properties"]["sequence"]["pattern"] == "^[A-Za-z]*$"


def test_agent_show_includes_persisted_run_details(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    Workspace.create("default")
    created = runner.invoke(app, ["agent", "run", "run", "noop", "payload=hi", "--json"])
    assert created.exit_code == 0, created.stdout
    run_id = json.loads(created.stdout)["data"]["run_id"]

    shown = runner.invoke(app, ["agent", "show", run_id, "--json"])

    assert shown.exit_code == 0, shown.stdout
    data = json.loads(shown.stdout)["data"]
    assert data["run"]["run_id"] == run_id
    assert data["messages"]
    assert data["tool_calls"]


async def _fake_next_round_plan(tmp_workspace: Workspace) -> dict:
    class FakeProvider:
        async def chat(self, *args, **kwargs):
            return ProviderResponse(
                provider="fake",
                model="fake",
                content=json.dumps(
                    {
                        "mode": "agent_plan",
                        "goal": "next_round",
                        "assumptions": ["lowered batch size after failed filters"],
                        "missing_inputs": [],
                        "required_tools": ["protein_qc"],
                        "workflow_plan": {
                            "name": "next_qc",
                            "version": "0.1.0",
                            "params": {"sequence": {"type": "str", "required": True}},
                            "nodes": [
                                {
                                    "id": "n",
                                    "tool": "protein_qc",
                                    "inputs": {"sequence": "${params.sequence}"},
                                }
                            ],
                        },
                        "success_criteria": {},
                        "warm_tools": ["protein_qc"],
                        "estimated_llm_calls": 1,
                    }
                ),
            )

    registry = ToolRegistry()
    registry.discover()
    return await WorkflowPlanner(registry, FakeProvider()).draft_next_round_plan(
        previous_plan={"success_criteria": {}},
        summary={"workflow_id": "wf_x", "state": "completed"},
        decision={
            "next_round": {
                "recommended": True,
                "changes": [{"param": "num_designs", "value": 4}],
            }
        },
        available_tools=[{"name": "protein_qc", "input_schema": {}}],
        workflows=[],
    )


def test_next_round_planner_drafts_validated_plan(tmp_workspace: Workspace) -> None:
    import asyncio

    plan = asyncio.run(_fake_next_round_plan(tmp_workspace))

    assert plan["workflow_plan"]["name"] == "next_qc"
    assert plan["workflow_plan"]["nodes"][0]["tool"] == "protein_qc"


def test_agent_validate_plan_rejects_unknown_tool(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)
    plan = {
        "mode": "agent_plan",
        "workflow_plan": {
            "name": "bad",
            "version": "0.1.0",
            "nodes": [{"id": "x", "tool": "not_a_tool", "inputs": {}}],
        },
    }

    try:
        controller.validate_plan(plan)
    except Exception as exc:
        assert getattr(exc, "code", None) == "E_UNKNOWN_TOOL"
    else:  # pragma: no cover
        raise AssertionError("expected validation failure")


def test_agent_validate_plan_rejects_internal_tool(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace)
    plan = {
        "mode": "agent_plan",
        "required_tools": ["noop"],
        "workflow_plan": {
            "name": "bad_internal",
            "version": "0.1.0",
            "nodes": [{"id": "x", "tool": "noop", "inputs": {"payload": "x"}}],
        },
    }

    with pytest.raises(Exception) as exc_info:
        controller.validate_plan(plan)

    assert getattr(exc_info.value, "code", None) == "E_INVALID_PARAMS"
    assert exc_info.value.details["tool"] == "noop"
    assert "internal" in exc_info.value.message


def test_agent_validate_plan_rejects_unavailable_probe_tool(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner.descriptor_availability",
        lambda descriptor: {
            "available": False,
            "availability_status": "unavailable",
            "availability_reason": "weights missing",
        },
    )
    controller = AgentController(tmp_workspace)
    plan = {
        "mode": "agent_plan",
        "workflow_plan": {
            "name": "bad_rfdiffusion",
            "version": "0.1.0",
            "nodes": [
                {
                    "id": "backbone",
                    "tool": "rfdiffusion",
                    "inputs": {"mode": "unconditional", "length_min": 80},
                }
            ],
        },
    }

    with pytest.raises(Exception) as exc_info:
        controller.validate_plan(plan)

    assert getattr(exc_info.value, "code", None) == "E_INVALID_PARAMS"
    assert exc_info.value.details["tool"] == "rfdiffusion"
    assert exc_info.value.details["availability"]["available"] is False


def test_cli_agent_run_direct_tool_json(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    Workspace.create("default")

    result = runner.invoke(app, ["agent", "run", "run", "noop", "payload=hi", "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert data["mode"] == "direct_tool"
    assert data["job_ids"]
    assert data["classification"]["requires_llm_planning"] is False


def test_cli_agent_validate_plan_accepts_workflow_yaml(
    runner: CliRunner,
    isolated_vpa_home: Path,
    tmp_path: Path,
) -> None:
    Workspace.create("default")
    plan = tmp_path / "wf.yaml"
    plan.write_text(
        """
name: agent_qc
version: 0.1.0
params:
  sequence:
    type: str
    required: true
nodes:
  - id: n
    tool: protein_qc
    inputs:
      sequence: ${params.sequence}
"""
    )

    result = runner.invoke(app, ["agent", "validate-plan", str(plan), "--json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["data"]["valid"] is True

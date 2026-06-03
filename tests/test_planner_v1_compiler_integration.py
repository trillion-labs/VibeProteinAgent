from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution.controller import AgentController
from vibeproteinagent.agents.planning.compiler import (
    compile_planner_ir,
    normalize_pxdesign_full_structure_ir,
)
from vibeproteinagent.agents.planning.ir import validate_planner_ir_v1
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.workspace import Workspace

FIXTURE = Path(__file__).parent / "fixtures" / "planner_v1" / "go_rank_request.json"


class _PlannerProvider:
    provider_name = "fake"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        payload = self.payloads[min(len(self.messages) - 1, len(self.payloads) - 1)]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _fixture_ir() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())["planner_ir"]


def _tool_catalog() -> list[dict[str, str]]:
    return [
        {"name": "go_sequences", "description": "Retrieve GO sequences."},
        {"name": "sequence_analysis", "description": "Analyze protein sequences."},
        {"name": "rank_designs", "description": "Rank candidates."},
    ]


def _pxdesign_tool_catalog() -> list[dict[str, str]]:
    return [
        {"name": "pxdesign", "description": "Design protein-target binders."},
        {"name": "protenix_v2", "description": "Predict target-binder complexes."},
        {"name": "protein_qc", "description": "Evaluate protein structures."},
        {"name": "rank_designs", "description": "Rank candidates."},
    ]


def _selection(*tools: str) -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {"name": tool, "reason": "Selected tools needed for the requested workflow."}
            for tool in tools
        ],
        "missing_inputs": [],
    }


def _pxdesign_ir(*, include_rank: bool = False) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {
            "id": "binders",
            "tool": "pxdesign",
            "intent": "Design protein binders against the supplied target.",
            "inputs": {"target_pdb": "/tmp/target.pdb", "top_k_designs": 4},
            "upstream": [],
        }
    ]
    if include_rank:
        nodes.append(
            {
                "id": "rank_binders",
                "tool": "rank_designs",
                "intent": "Rank binder candidates.",
                "inputs": {"top_k": 4},
                "upstream": [
                    {"node": "binders", "relationship": "candidate_source"}
                ],
            }
        )
    return {
        "version": "planner_ir_v1",
        "goal": "Design full-structure protein binder candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": nodes,
        "finalization": {"success_criteria": {"ranked_binders": True}},
    }


def _pxdesign_structure_analysis_ir() -> dict[str, Any]:
    ir = _pxdesign_ir()
    ir["nodes"].append(
        {
            "id": "analyze_binders",
            "tool": "structure_analysis",
            "intent": "Analyze predicted binder complex structures.",
            "inputs": {},
            "upstream": [{"node": "binders", "relationship": "candidate_source"}],
        }
    )
    return ir


def _two_pxdesign_roots_with_shared_node(
    *,
    shared_tool: str,
    shared_node_id: str,
    terminal_tool: str = "rank_designs",
    terminal_node_id: str = "rank_all",
) -> dict[str, Any]:
    shared_upstream = [
        {"node": "binders_a", "relationship": "candidate_source"},
        {"node": "binders_b", "relationship": "candidate_source"},
    ]
    terminal_upstream_node = shared_node_id
    nodes: list[dict[str, Any]] = [
        {
            "id": "binders_a",
            "tool": "pxdesign",
            "intent": "Design binders for target A.",
            "inputs": {"target_pdb": "/tmp/target_a.pdb"},
            "upstream": [],
        },
        {
            "id": "binders_b",
            "tool": "pxdesign",
            "intent": "Design binders for target B.",
            "inputs": {"target_pdb": "/tmp/target_b.pdb"},
            "upstream": [],
        },
    ]
    if shared_tool != "rank_designs":
        nodes.append(
            {
                "id": shared_node_id,
                "tool": shared_tool,
                "intent": f"Shared {shared_tool} node authored by the planner.",
                "inputs": {},
                "upstream": shared_upstream,
            }
        )
    else:
        terminal_upstream_node = None
    nodes.append(
        {
            "id": terminal_node_id,
            "tool": terminal_tool,
            "intent": f"Run {terminal_tool} over shared PXDesign candidates.",
            "inputs": {"top_k": 2} if terminal_tool == "rank_designs" else {},
            "upstream": (
                shared_upstream
                if terminal_upstream_node is None
                else [{"node": terminal_upstream_node, "relationship": "candidate_source"}]
            ),
        }
    )
    return {
        "version": "planner_ir_v1",
        "goal": "Design and process two target-conditioned binder sets.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": nodes,
        "finalization": {"success_criteria": {"processed": True}},
    }


def _assert_pxdesign_shared_normalization_is_valid(ir: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_pxdesign_full_structure_ir(ir)

    validate_planner_ir_v1(normalized)
    assert normalize_pxdesign_full_structure_ir(normalized) == normalized
    node_ids = [node["id"] for node in normalized["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    assert "binders_a_protenix" in node_ids
    assert "binders_a_qc" in node_ids
    assert "binders_b_protenix" in node_ids
    assert "binders_b_qc" in node_ids
    return normalized


def test_plan_v1_compiles_single_producer_consumer() -> None:
    provider = _PlannerProvider([
        _selection("go_sequences", "rank_designs"),
        _fixture_ir(),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates and rank them.",
            available_tools=_tool_catalog(),
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == ["go_candidates", "rank_candidates"]
    assert nodes[1]["inputs"]["candidates"] == "${outputs.go_candidates.candidates}"


def test_plan_v1_expands_pxdesign_only_to_full_structure_ranked_workflow() -> None:
    provider = _PlannerProvider([
        _selection("pxdesign"),
        _pxdesign_ir(),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design protein binders against target_pdb=/tmp/target.pdb.",
            available_tools=_pxdesign_tool_catalog(),
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == [
        "binders",
        "binders_protenix",
        "binders_qc",
        "binders_rank",
    ]
    assert [node["tool"] for node in nodes] == [
        "pxdesign",
        "protenix_v2",
        "protein_qc",
        "rank_designs",
    ]
    assert nodes[1]["inputs"]["candidates"] == "${outputs.binders.candidates}"
    assert nodes[2]["inputs"]["candidates"] == "${outputs.binders_protenix.candidates}"
    assert nodes[3]["inputs"]["candidates"] == "${outputs.binders_qc.candidates}"
    assert nodes[3]["inputs"]["top_k"] == 4
    assert plan["required_tools"] == [
        "pxdesign",
        "protenix_v2",
        "protein_qc",
        "rank_designs",
    ]


def test_plan_v1_rewires_existing_pxdesign_rank_without_duplicate_rank() -> None:
    provider = _PlannerProvider([
        _selection("pxdesign", "rank_designs"),
        _pxdesign_ir(include_rank=True),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design and rank protein binders against target_pdb=/tmp/target.pdb.",
            available_tools=_pxdesign_tool_catalog(),
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == [
        "binders",
        "binders_protenix",
        "binders_qc",
        "rank_binders",
    ]
    assert [node["tool"] for node in nodes].count("rank_designs") == 1
    assert nodes[-1]["inputs"]["candidates"] == "${outputs.binders_qc.candidates}"
    tail_warnings = [
        warning
        for warning in plan.get("planner_audit_warnings", [])
        if warning.get("event_type") == "pxdesign_full_structure_tail_inserted"
    ]
    assert tail_warnings
    assert tail_warnings[0]["pxdesign_nodes"] == ["binders"]


def test_plan_v1_validates_pxdesign_effective_full_structure_before_compile() -> None:
    provider = _PlannerProvider([
        _selection("pxdesign", "structure_analysis"),
        _pxdesign_structure_analysis_ir(),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design target-conditioned binders and analyze their structures.",
            available_tools=[
                *_pxdesign_tool_catalog(),
                {"name": "structure_analysis", "description": "Analyze structures."},
            ],
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == [
        "binders",
        "binders_protenix",
        "binders_qc",
        "binders_rank",
        "analyze_binders",
    ]
    assert nodes[-1]["inputs"]["candidates"] == "${outputs.binders_qc.candidates}"


def test_pxdesign_shared_rank_normalization_is_duplicate_safe() -> None:
    ir = _two_pxdesign_roots_with_shared_node(
        shared_tool="rank_designs",
        shared_node_id="rank_all",
    )
    normalized = _assert_pxdesign_shared_normalization_is_valid(ir)

    assert [node["id"] for node in normalized["nodes"]] == [
        "binders_a",
        "binders_a_protenix",
        "binders_a_qc",
        "binders_b",
        "binders_b_protenix",
        "binders_b_qc",
        "rank_all",
    ]
    rank = normalized["nodes"][-1]
    assert rank["upstream"] == [
        {"node": "binders_a_qc", "relationship": "candidate_source"},
        {"node": "binders_b_qc", "relationship": "candidate_source"},
    ]

    plan = compile_planner_ir(ir)
    node_ids = [node["id"] for node in plan["workflow_plan"]["nodes"]]
    assert len(node_ids) == len(set(node_ids))
    assert "merge_rank_all_candidates" in node_ids
    assert plan["workflow_plan"]["nodes"][-1]["id"] == "rank_all"


def test_pxdesign_shared_protenix_normalization_is_duplicate_safe() -> None:
    ir = _two_pxdesign_roots_with_shared_node(
        shared_tool="protenix_v2",
        shared_node_id="pred_all",
    )
    normalized = _assert_pxdesign_shared_normalization_is_valid(ir)

    node_ids = [node["id"] for node in normalized["nodes"]]
    assert "pred_all" not in node_ids
    assert normalized["nodes"][-1]["id"] == "rank_all"
    assert normalized["nodes"][-1]["upstream"] == [
        {"node": "binders_a_qc", "relationship": "candidate_source"},
        {"node": "binders_b_qc", "relationship": "candidate_source"},
    ]

    plan = compile_planner_ir(ir)
    compiled_ids = [node["id"] for node in plan["workflow_plan"]["nodes"]]
    assert len(compiled_ids) == len(set(compiled_ids))
    assert "pred_all" not in compiled_ids
    assert "merge_rank_all_candidates" in compiled_ids
    assert plan["workflow_plan"]["nodes"][-1]["id"] == "rank_all"


def test_pxdesign_shared_qc_normalization_is_duplicate_safe() -> None:
    ir = _two_pxdesign_roots_with_shared_node(
        shared_tool="protein_qc",
        shared_node_id="qc_all",
    )
    normalized = _assert_pxdesign_shared_normalization_is_valid(ir)

    node_ids = [node["id"] for node in normalized["nodes"]]
    assert "qc_all" not in node_ids
    assert normalized["nodes"][-1]["id"] == "rank_all"
    assert normalized["nodes"][-1]["upstream"] == [
        {"node": "binders_a_qc", "relationship": "candidate_source"},
        {"node": "binders_b_qc", "relationship": "candidate_source"},
    ]

    plan = compile_planner_ir(ir)
    compiled_ids = [node["id"] for node in plan["workflow_plan"]["nodes"]]
    assert len(compiled_ids) == len(set(compiled_ids))
    assert "qc_all" not in compiled_ids
    assert "merge_rank_all_candidates" in compiled_ids
    assert plan["workflow_plan"]["nodes"][-1]["id"] == "rank_all"


def test_pxdesign_shared_intermediate_analysis_rewires_to_local_qc_streams() -> None:
    ir = _two_pxdesign_roots_with_shared_node(
        shared_tool="protenix_v2",
        shared_node_id="pred_all",
        terminal_tool="structure_analysis",
        terminal_node_id="analyze_all",
    )
    normalized = _assert_pxdesign_shared_normalization_is_valid(ir)

    analyze = normalized["nodes"][-1]
    assert analyze["id"] == "analyze_all"
    assert analyze["upstream"] == [
        {"node": "binders_a_qc", "relationship": "candidate_source"},
        {"node": "binders_b_qc", "relationship": "candidate_source"},
    ]

    plan = compile_planner_ir(ir)
    compiled_ids = [node["id"] for node in plan["workflow_plan"]["nodes"]]
    assert "pred_all" not in compiled_ids
    assert "merge_analyze_all_candidates" in compiled_ids
    assert plan["workflow_plan"]["nodes"][-1]["id"] == "analyze_all"


def test_plan_v1_validates_shared_pxdesign_intermediate_before_compile() -> None:
    provider = _PlannerProvider([
        _selection("pxdesign", "rank_designs"),
        _two_pxdesign_roots_with_shared_node(
            shared_tool="protenix_v2",
            shared_node_id="pred_all",
        ),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design two protein-target binder sets and rank the resulting structures.",
            available_tools=_pxdesign_tool_catalog(),
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    node_ids = [node["id"] for node in nodes]
    assert "pred_all" not in node_ids
    assert "merge_rank_all_candidates" in node_ids
    assert nodes[-1]["id"] == "rank_all"
    tail_warnings = [
        warning
        for warning in plan.get("planner_audit_warnings", [])
        if warning.get("event_type") == "pxdesign_full_structure_tail_inserted"
    ]
    assert tail_warnings
    assert tail_warnings[0]["pxdesign_nodes"] == ["binders_a", "binders_b"]


def test_plan_v1_compiles_multi_upstream_merge() -> None:
    ir = _fixture_ir()
    second = copy.deepcopy(ir["nodes"][0])
    second["id"] = "go_more_candidates"
    second["inputs"] = {"go_id": "GO:0008150", "max_sequences": 2}
    ir["nodes"].insert(1, second)
    ir["nodes"][2]["upstream"].append(
        {"node": "go_more_candidates", "relationship": "candidate_source"}
    )
    provider = _PlannerProvider([
        _selection("go_sequences", "rank_designs"),
        ir,
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1("Find and rank GO candidates.", available_tools=_tool_catalog())
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == [
        "go_candidates",
        "go_more_candidates",
        "merge_rank_candidates_candidates",
        "rank_candidates",
    ]
    assert nodes[2]["tool"] == "merge_candidates"
    assert nodes[2]["inputs"]["candidate_sets"] == [
        "${outputs.go_candidates.candidates}",
        "${outputs.go_more_candidates.candidates}",
    ]
    assert nodes[3]["inputs"]["candidates"] == (
        "${outputs.merge_rank_candidates_candidates.candidates}"
    )


def test_plan_v1_reuses_controller_fixup_transport(tmp_workspace: Workspace) -> None:
    ir = _fixture_ir()
    ir["nodes"][1]["tool"] = "go_sequences"
    ir["nodes"][1]["inputs"] = {}
    provider = _PlannerProvider(
        [
            _selection("go_sequences", "rank_designs"),
            ir,
            {
                "add_selected_tools": [],
                "drop_selected_tools": ["rank_designs"],
                "remove_node_ids": [],
                "reason": (
                    "Let compiler fixup resolve the downstream consumer swap "
                    "without treating the unused selected tool as a planner IR boundary error."
                ),
            },
            {"choice_id": "swap_tool", "tool": "rank_designs"},
        ]
    )
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    planner = WorkflowPlanner(ToolRegistry(), provider)

    async def compile_with_fixup(planner_ir: dict[str, Any]) -> dict[str, Any]:
        return await controller._compile_with_fixup(planner_ir, provider=provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates and rank them.",
            available_tools=_tool_catalog(),
            compile_with_fixup=compile_with_fixup,
        )
    )

    assert len(provider.messages) == 4
    assert plan["workflow_plan"]["nodes"][1]["tool"] == "rank_designs"
    assert plan["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == (
        "${outputs.go_candidates.candidates}"
    )

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from vibeproteinagent.adapters import ToolRegistry, validate_params
from vibeproteinagent.adapters.availability import descriptor_availability
from vibeproteinagent.adapters.descriptor import resolve_input_schema_dict
from vibeproteinagent.adapters.visibility import is_agent_visible_tool
from vibeproteinagent.agents.provider import ChatMessage
from vibeproteinagent.agents.provider.types import ChatProvider
from vibeproteinagent.agents.tools.schemas import (
    compact_tool_outline,
    compact_tool_schema,
    tool_names_from_plan,
)
from vibeproteinagent.core.errors import InvalidParamsError, VPAError
from vibeproteinagent.workflows import Workflow, serialize_workflow
from vibeproteinagent.workflows.planner import params_dict_from_workflow
from vibeproteinagent.workflows.resolver import iter_refs, resolve

from .prompts import (
    NEXT_ROUND_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)


def _install_readiness(tool_name: str) -> dict[str, Any]:
    try:
        from vibeproteinagent.install.installer import install_readiness

        return install_readiness((tool_name,), runtime_probe=False)
    except Exception as exc:
        return {
            "install_ready": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _descriptor_availability(descriptor: Any, adapter_cls: type[Any]) -> dict[str, Any]:
    try:
        return descriptor_availability(descriptor, adapter_cls)
    except TypeError:
        return descriptor_availability(descriptor)


def _rfdiffusion_mode_readiness(params: dict[str, Any]) -> dict[str, Any]:
    try:
        from vibeproteinagent.adapters.rfdiffusion import mode_readiness

        return dict(mode_readiness(params))
    except VPAError:
        raise
    except Exception as exc:
        raise InvalidParamsError(
            "could not validate RFdiffusion mode readiness",
            details={"error": str(exc), "exception_type": type(exc).__name__},
        ) from exc


def _phase_idempotency_key(idempotency_key: str | None, phase: str) -> str | None:
    if idempotency_key is None:
        return None
    return f"{idempotency_key}:{phase}"


def _candidate_tool_names(selection: dict[str, Any]) -> list[str]:
    names = selection.get("candidate_tools")
    if not isinstance(names, list):
        names = tool_names_from_plan(selection)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _load_json_object_response(
    content: str,
    *,
    label: str,
    error_message: str,
) -> dict[str, Any]:
    try:
        obj: Any = json.loads(content)
        for _ in range(2):
            if not isinstance(obj, str):
                break
            obj = json.loads(obj)
    except json.JSONDecodeError as exc:
        raise InvalidParamsError(
            error_message,
            details={"error": str(exc), "content": content[:1000]},
        ) from exc
    if not isinstance(obj, dict):
        raise InvalidParamsError(
            f"{label} must return a JSON object",
            details={"got_type": type(obj).__name__},
        )
    return obj


def _fallback_tool_selection(
    prompt: dict[str, Any],
    content: str,
) -> dict[str, Any] | None:
    available = prompt.get("available_tools")
    if not isinstance(available, list):
        return None
    allowed = {
        str(item.get("name"))
        for item in available
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    template = _pick_workflow_template(prompt)
    names: list[str] = []
    if template is not None:
        for name in template.get("tool_sequence") or []:
            if isinstance(name, str) and name in allowed and name not in names:
                names.append(name)
    haystack = f"{prompt.get('request') or ''}\n{content}".lower()
    for name in sorted(allowed):
        if name.lower() in haystack and name not in names:
            names.append(name)
    if not names:
        return None
    return {
        "mode": "tool_selection",
        "candidate_tools": names,
        "reason": (
            "Recovered from a non-JSON provider response using matching local "
            "workflow metadata."
        ),
        "missing_inputs": [],
    }


def _template_tool_selection(
    prompt: dict[str, Any],
    available_tools: list[dict[str, Any]],
    *,
    reason: str,
) -> dict[str, Any] | None:
    allowed = {
        str(item.get("name"))
        for item in available_tools
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    template = _pick_workflow_template(prompt)
    if template is None:
        return None
    names: list[str] = []
    for name in template.get("tool_sequence") or []:
        if isinstance(name, str) and name in allowed and name not in names:
            names.append(name)
    if not names:
        return None
    return {
        "mode": "tool_selection",
        "candidate_tools": names,
        "reason": reason,
        "missing_inputs": [],
    }


def _fallback_plan_from_prompt(
    prompt: dict[str, Any],
    content: str,
) -> dict[str, Any] | None:
    del content
    template = _pick_workflow_template(prompt)
    if template is None:
        return None
    runtime_params = _template_runtime_params_for_request(
        template,
        str(prompt.get("request") or ""),
    )
    return _plan_from_template(
        prompt,
        template,
        runtime_params=runtime_params,
        assumptions=[
            "Provider response was not valid JSON; selected the closest "
            "validated local workflow template."
        ],
        fallback="template_recovery",
    )


def _plan_from_template(
    prompt: dict[str, Any],
    template: dict[str, Any],
    *,
    runtime_params: dict[str, Any] | None = None,
    assumptions: list[str] | None = None,
    fallback: str | None = None,
) -> dict[str, Any] | None:
    nodes = []
    for spec in template.get("node_specs") or []:
        if not isinstance(spec, dict):
            continue
        nodes.append(
            {
                "id": spec.get("id"),
                "tool": spec.get("tool"),
                "version": spec.get("version") or "*",
                "foreach": spec.get("foreach"),
                "max_parallel": spec.get("max_parallel"),
                "when": spec.get("when"),
                "inputs": spec.get("inputs") or {},
            }
        )
    if not nodes:
        return None
    tool_sequence = [
        str(name)
        for name in template.get("tool_sequence") or []
        if isinstance(name, str)
    ]
    plan = {
        "mode": "agent_plan",
        "source_kind": "workflow_template",
        "workflow_ref": template.get("name"),
        "goal": str(prompt.get("request") or template.get("description") or template.get("name")),
        "assumptions": list(assumptions or []),
        "missing_inputs": [],
        "required_tools": list(dict.fromkeys(tool_sequence)),
        "params": dict(runtime_params or {}),
        "workflow_plan": {
            "name": template.get("name"),
            "version": template.get("version") or "1.0.0",
            "description": template.get("description"),
            "params": dict(template.get("params") or {}),
            "nodes": nodes,
        },
        "success_criteria": {
            "workflow_completes": True,
            "report_and_visualization_artifacts": True,
        },
        "warm_tools": list(dict.fromkeys(tool_sequence)),
        "estimated_llm_calls": 1,
    }
    if fallback:
        plan["fallback"] = fallback
    return plan


def _pick_workflow_template(prompt: dict[str, Any]) -> dict[str, Any] | None:
    workflows = prompt.get("workflow_templates")
    if not isinstance(workflows, list):
        return None
    by_name = {
        item.get("name"): item
        for item in workflows
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    request = str(prompt.get("request") or "").lower()
    selection = prompt.get("tool_selection") or {}
    selected = set(_candidate_tool_names(selection)) if isinstance(selection, dict) else set()

    if _is_target_sequence_binder_request(request):
        candidate = by_name.get("target_sequence_binder_design_round")
        if candidate is not None:
            return candidate

    if _is_semantic_function_request(request):
        candidate = by_name.get("semantic_constraint_to_family_sequences")
        if candidate is not None:
            return candidate

    if _is_de_novo_backbone_request(request, selected):
        candidate = by_name.get("rfdiffusion_backbone_design_round")
        if candidate is not None and "pdb_path" not in request and "target_pdb" not in request:
            return candidate
    if {"proteinmpnn", "protenix_v2"} & selected or "refold" in request or "backbone" in request:
        candidate = by_name.get("mpnn_then_refold")
        if candidate is not None:
            return candidate
    return None


def _is_de_novo_backbone_request(request: str, selected: set[str]) -> bool:
    return (
        "rfdiffusion" in request
        or "rf diffusion" in request
        or "de novo" in request
        or "backbone design" in request
        or "backbone generation" in request
        or "inverse-fold" in request
        or "inverse fold" in request
        or {"rfdiffusion", "proteinmpnn", "protenix_v2"} & selected
    )


def _is_semantic_function_request(request: str) -> bool:
    semantic_markers = (
        "activity",
        "specificity",
        "catalyzes",
        "epimerization",
        "enzyme",
        "metabolic",
        "go:",
        "gene ontology",
        "interpro",
        "family",
        "annotation",
        "literature",
    )
    structure_markers = (
        "target protein:",
        "target binding site:",
        "target residues",
        "binder",
        "pdb_path",
        "target_pdb",
        "backbone",
        "de novo",
    )
    return any(marker in request for marker in semantic_markers) and not any(
        marker in request for marker in structure_markers
    )


def _is_target_sequence_binder_request(request: str) -> bool:
    return (
        "binder" in request
        and (
            "target protein" in request
            or "target sequence" in request
            or "sequence:" in request
        )
        and (
            "target binding site" in request
            or "target residues" in request
            or "hotspot" in request
            or "interface toward residues" in request
            or re.search(r"\bresidues?\s+[a-z]\d+", request) is not None
        )
    )


def _template_runtime_params_for_request(template: dict[str, Any], request: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    name = str(template.get("name") or "")
    template_params = template.get("params") if isinstance(template.get("params"), dict) else {}
    if name == "semantic_constraint_to_family_sequences":
        params["query"] = request
    elif name == "rfdiffusion_backbone_design_round":
        params["length"] = _extract_design_length(request) or 100
    elif name == "target_sequence_binder_design_round":
        sequence = _extract_target_sequence(request)
        hotspots = _extract_hotspot_positions(request)
        if sequence:
            params["target_sequence"] = sequence
        if hotspots:
            params["target_hotspots"] = hotspots
    if "submit_final" in template_params and _request_implies_final_submission(request):
        params["submit_final"] = True
    return params


def _request_implies_final_submission(request: str) -> bool:
    text = request.lower()
    return any(
        marker in text
        for marker in (
            "submit the final",
            "explicitly submit",
            "final candidate",
            "final top",
            "return reason:",
            "answer:",
        )
    )


def _extract_design_length(request: str) -> int | None:
    match = re.search(
        r"\b(?:length|len|residues?|aa|amino acids?)\D{0,12}(\d{2,4})\b",
        request,
        flags=re.I,
    )
    if match:
        value = int(match.group(1))
        if 20 <= value <= 2000:
            return value
    match = re.search(r"\b(\d{2,4})\s*(?:aa|residues?|amino acids?)\b", request, flags=re.I)
    if match:
        value = int(match.group(1))
        if 20 <= value <= 2000:
            return value
    return None


def _extract_target_sequence(request: str) -> str | None:
    match = re.search(r"(?im)^\s*[-*]?\s*sequence\s*:\s*([A-Z]{20,})\s*$", request)
    if match:
        return match.group(1).strip()
    blocks = re.findall(r"\b([ACDEFGHIKLMNPQRSTVWY]{40,})\b", request.upper())
    return max(blocks, key=len) if blocks else None


def _extract_hotspot_positions(request: str) -> list[int]:
    positions = [
        int(match.group(2))
        for match in re.finditer(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,5})\b", request.upper())
    ]
    out: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        if pos in seen:
            continue
        seen.add(pos)
        out.append(pos)
    return out


def _workflow_template_by_name(
    prompt: dict[str, Any],
    name: str | None,
) -> dict[str, Any] | None:
    if not name:
        return None
    workflows = prompt.get("workflow_templates")
    for item in workflows or []:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    return None


def _workflow_params_look_like_schema(params: Any) -> bool:
    if not isinstance(params, dict):
        return False
    return all(isinstance(value, dict) and "type" in value for value in params.values())


def _runtime_params_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    runtime: dict[str, Any] = {}
    params = plan.get("params")
    if isinstance(params, dict):
        runtime.update(params)
    workflow_plan = plan.get("workflow_plan")
    if isinstance(workflow_plan, dict):
        workflow_params = workflow_plan.get("params")
        if isinstance(workflow_params, dict) and not _workflow_params_look_like_schema(
            workflow_params
        ):
            runtime.update(workflow_params)
    return runtime


def _merge_runtime_params_for_template(
    template: dict[str, Any],
    request: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        **_template_runtime_params_for_request(template, request),
        **_runtime_params_from_plan(plan),
    }


def _invocation_params_from_plan(
    plan: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    plan_params = plan.get("params")
    if not isinstance(plan_params, dict):
        return dict(params)
    # Explicit CLI / params-file values win over LLM-suggested defaults.
    return {**plan_params, **params}


def _validate_runtime_param_types(
    workflow: Workflow,
    runtime_params: dict[str, Any],
) -> None:
    for name, value in runtime_params.items():
        spec = workflow.params.get(name)
        if spec is None or value is None:
            continue
        expected = spec.type
        ok = (
            (expected in {"str", "path"} and isinstance(value, str))
            or (expected == "int" and isinstance(value, int) and not isinstance(value, bool))
            or (
                expected == "float"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
            or (expected == "bool" and isinstance(value, bool))
            or (expected == "list" and isinstance(value, list))
            or (expected == "dict" and isinstance(value, dict))
        )
        if not ok:
            raise InvalidParamsError(
                "agent workflow runtime param has the wrong type",
                details={
                    "workflow": workflow.name,
                    "param": name,
                    "expected_type": expected,
                    "actual_type": type(value).__name__,
                    "value": value,
                },
            )


class WorkflowPlanner:
    """LLM-backed workflow plan proposal plus deterministic local validation."""

    def __init__(self, registry: ToolRegistry, provider: ChatProvider | None = None) -> None:
        self.registry = registry
        self.provider = provider
        self.registry.discover()

    def template_plan(
        self,
        request: str,
        *,
        workflows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = {
            "request": request,
            "tool_selection": {},
            "workflow_templates": workflows,
        }
        plan = _fallback_plan_from_prompt(prompt, "")
        if plan is None:
            raise InvalidParamsError(
                "no local workflow template matches the agent request",
                details={"request": request},
            )
        plan = self._canonicalize_template_plan(prompt, plan)
        return self.validate_plan(plan)

    async def draft_plan(
        self,
        request: str,
        *,
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]],
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        selection = await self._select_candidate_tools(
            {
                "request": request,
                "available_tools": [compact_tool_outline(t) for t in available_tools],
                "workflow_templates": workflows,
                "output_contract": {
                    "mode": "tool_selection",
                    "candidate_tools": ["tool_name"],
                    "reason": "string",
                    "missing_inputs": ["string"],
                },
            },
            label="planner tool selection",
            model=model,
            idempotency_key=_phase_idempotency_key(idempotency_key, "tool-selection"),
            system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
        )
        selection_prompt = {
            "request": request,
            "workflow_templates": workflows,
            "tool_selection": selection,
        }
        template_selection = _template_tool_selection(
            selection_prompt,
            available_tools,
            reason=(
                "Matched a validated local workflow template for the user's "
                "scientific request."
            ),
        )
        if template_selection is not None:
            selection = template_selection
        selected_tools = self._selected_tool_schemas(selection, available_tools)
        prompt = {
            "request": request,
            "tool_selection": selection,
            "selected_tools": selected_tools,
            "workflow_templates": workflows,
            "output_contract": {
                "mode": "agent_plan",
                "goal": "string",
                "assumptions": ["string"],
                "missing_inputs": ["string"],
                "required_tools": ["tool_name"],
                "workflow_ref": "one workflow_templates.name to run",
                "params": {"param_name": "runtime value"},
                "success_criteria": {},
                "warm_tools": ["tool_name"],
                "estimated_llm_calls": 2,
            },
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=_phase_idempotency_key(idempotency_key, "workflow-plan"),
        )
        try:
            plan = _load_json_object_response(
                response.content,
                label="planner",
                error_message="planner returned non-JSON content",
            )
        except InvalidParamsError:
            plan = _fallback_plan_from_prompt(prompt, response.content)
            if plan is None:
                raise
        plan = self._canonicalize_template_plan(prompt, plan)
        try:
            return self.validate_plan(plan)
        except InvalidParamsError:
            fallback = _fallback_plan_from_prompt(prompt, "")
            if fallback is None:
                raise
            return self.validate_plan(fallback)

    async def draft_next_round_plan(
        self,
        *,
        previous_plan: dict[str, Any],
        summary: dict[str, Any],
        decision: dict[str, Any],
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]],
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            raise InvalidParamsError(
                "next-round planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        selection = await self._select_candidate_tools(
            {
                "previous_plan": previous_plan,
                "compact_summary": summary,
                "decision": decision,
                "available_tools": [compact_tool_outline(t) for t in available_tools],
                "workflow_templates": workflows,
                "output_contract": {
                    "mode": "tool_selection",
                    "candidate_tools": ["tool_name"],
                    "reason": "string",
                    "missing_inputs": [],
                },
            },
            label="next-round tool selection",
            model=model,
            idempotency_key=_phase_idempotency_key(idempotency_key, "tool-selection"),
            system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
        )
        selected_tools = self._selected_tool_schemas(selection, available_tools)
        prompt = {
            "previous_plan": previous_plan,
            "compact_summary": summary,
            "decision": decision,
            "tool_selection": selection,
            "selected_tools": selected_tools,
            "workflow_templates": workflows,
            "output_contract": {
                "mode": "agent_plan",
                "goal": "string",
                "assumptions": ["include parameter changes and reasons"],
                "missing_inputs": [],
                "required_tools": ["tool_name"],
                "workflow_ref": previous_plan.get("workflow_ref") or "workflow_templates.name",
                "params": {"param_name": "runtime value"},
                "success_criteria": previous_plan.get("success_criteria", {}),
                "warm_tools": ["tool_name"],
                "estimated_llm_calls": 1,
            },
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=NEXT_ROUND_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=_phase_idempotency_key(idempotency_key, "workflow-plan"),
        )
        plan = _load_json_object_response(
            response.content,
            label="next-round planner",
            error_message="next-round planner returned non-JSON content",
        )
        plan = self._canonicalize_template_plan(prompt, plan)
        return self.validate_plan(plan)

    async def _select_candidate_tools(
        self,
        prompt: dict[str, Any],
        *,
        label: str,
        model: str | None,
        idempotency_key: str | None,
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        if self.provider is None:  # pragma: no cover - guarded by callers
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=idempotency_key,
        )
        try:
            return _load_json_object_response(
                response.content,
                label=label,
                error_message=f"{label} returned non-JSON content",
            )
        except InvalidParamsError:
            selection = _fallback_tool_selection(prompt, response.content)
            if selection is None:
                raise
            return selection

    def _selected_tool_schemas(
        self,
        selection: dict[str, Any],
        available_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_name = {
            str(tool.get("name")): tool
            for tool in available_tools
            if isinstance(tool.get("name"), str)
        }
        names = _candidate_tool_names(selection)
        if not names:
            raise InvalidParamsError(
                "planner did not select any candidate tools",
                details={"selection": selection},
            )
        unknown = [name for name in names if name not in by_name]
        if unknown:
            raise InvalidParamsError(
                "planner selected unknown tools",
                details={"tools": unknown, "available_tools": sorted(by_name)},
            )
        return [
            compact_tool_schema(self._tool_with_resolved_input_schema(by_name[name]))
            for name in names
        ]

    def _tool_with_resolved_input_schema(self, tool: dict[str, Any]) -> dict[str, Any]:
        schema = tool.get("input_schema")
        if isinstance(schema, dict):
            return tool
        name = tool.get("name")
        if not isinstance(name, str):
            return {**tool, "input_schema": {}}
        descriptor, _adapter_cls = self.registry.get(name, "*")
        return {**tool, "input_schema": resolve_input_schema_dict(descriptor)}

    def _canonicalize_template_plan(
        self,
        prompt: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        workflow_plan = plan.get("workflow_plan")
        workflow_name = (
            workflow_plan.get("name")
            if isinstance(workflow_plan, dict)
            and isinstance(workflow_plan.get("name"), str)
            else None
        )
        workflow_ref = plan.get("workflow_ref") or plan.get("workflow")
        workflow_ref = workflow_ref if isinstance(workflow_ref, str) else None
        preferred_template = _pick_workflow_template(prompt)
        preferred_name = (
            preferred_template.get("name")
            if isinstance(preferred_template, dict)
            and isinstance(preferred_template.get("name"), str)
            else None
        )
        request = str(plan.get("goal") or prompt.get("request") or "")

        ref_template = _workflow_template_by_name(prompt, workflow_ref)
        if ref_template is not None:
            assumptions = [
                *[
                    str(item)
                    for item in plan.get("assumptions", [])
                    if isinstance(item, str)
                ],
                "Provider selected a workflow template by reference; expanded it "
                "to the validated local workflow descriptor.",
            ]
            canonical_prompt = {**prompt, "request": request}
            canonical = _plan_from_template(
                canonical_prompt,
                ref_template,
                runtime_params=_merge_runtime_params_for_template(
                    ref_template,
                    request,
                    plan,
                ),
                assumptions=list(dict.fromkeys(assumptions)),
                fallback=plan.get("fallback") if isinstance(plan.get("fallback"), str) else None,
            )
            if canonical is not None:
                canonical["missing_inputs"] = [
                    item for item in plan.get("missing_inputs", []) if isinstance(item, str)
                ]
                if isinstance(plan.get("success_criteria"), dict):
                    canonical["success_criteria"] = plan["success_criteria"]
                return canonical

        if preferred_name and workflow_name != preferred_name:
            canonical = _fallback_plan_from_prompt(prompt, "")
            if canonical is not None:
                assumptions = [
                    *[
                        str(item)
                        for item in plan.get("assumptions", [])
                        if isinstance(item, str)
                    ],
                    (
                        "Provider selected a workflow template that does not match the "
                        "available request inputs; canonicalized to the validated local template."
                    ),
                ]
                canonical["params"].update(_runtime_params_from_plan(plan))
                canonical["assumptions"] = list(dict.fromkeys(assumptions))
                return canonical

        template = _workflow_template_by_name(prompt, workflow_name)
        if (
            template is not None
            and isinstance(workflow_plan, dict)
            and not _workflow_params_look_like_schema(
                workflow_plan.get("params")
            )
        ):
            canonical_prompt = {**prompt, "request": request}
            canonical = _plan_from_template(
                canonical_prompt,
                template,
                runtime_params=_merge_runtime_params_for_template(
                    template,
                    request,
                    plan,
                ),
                fallback=plan.get("fallback") if isinstance(plan.get("fallback"), str) else None,
            )
            if canonical is not None:
                assumptions = [
                    *[
                        str(item)
                        for item in plan.get("assumptions", [])
                        if isinstance(item, str)
                    ],
                    (
                        "Provider selected a local workflow template but returned "
                        "runtime parameter values in workflow_plan.params; "
                        "canonicalized to the validated template schema."
                    ),
                ]
                canonical["assumptions"] = list(dict.fromkeys(assumptions))
                if isinstance(plan.get("success_criteria"), dict):
                    canonical["success_criteria"] = plan["success_criteria"]
                return canonical

        internal_tools = []
        for tool_name in tool_names_from_plan(plan):
            descriptor, _adapter_cls = self.registry.get(tool_name, "*")
            if not is_agent_visible_tool(descriptor):
                internal_tools.append(tool_name)
        if not internal_tools:
            return plan

        template = _workflow_template_by_name(prompt, workflow_name)
        if template is None:
            template = preferred_template
        if template is None:
            return plan

        canonical_prompt = {**prompt, "request": plan.get("goal") or prompt.get("request")}
        canonical = _plan_from_template(
            canonical_prompt,
            template,
            runtime_params=_merge_runtime_params_for_template(
                template,
                str(canonical_prompt.get("request") or ""),
                plan,
            ),
        )
        if canonical is None:
            return plan
        assumptions = [
            *[
                str(item)
                for item in plan.get("assumptions", [])
                if isinstance(item, str)
            ],
            "Provider selected a matching local workflow template; canonicalized "
            "glue-tool steps from the validated template.",
        ]
        canonical["assumptions"] = list(dict.fromkeys(assumptions))
        return canonical

    def validate_plan(
        self,
        plan: dict[str, Any],
        *,
        allow_internal_tools: bool = False,
    ) -> dict[str, Any]:
        allow_internal_tools = allow_internal_tools or plan.get("source_kind") in {
            "workflow_yaml",
            "workflow_template",
        }
        if not isinstance(plan, dict):
            raise InvalidParamsError(
                "agent workflow plan must be an object",
                details={"got_type": type(plan).__name__},
            )
        if plan.get("mode") not in (None, "agent_plan"):
            raise InvalidParamsError(
                "planner returned an unsupported mode for workflow planning",
                details={"mode": plan.get("mode")},
            )
        workflow_plan = plan.get("workflow_plan")
        if not isinstance(workflow_plan, dict):
            raise InvalidParamsError(
                "agent plan missing workflow_plan object",
                details={"field": "workflow_plan"},
            )

        for tool_name in tool_names_from_plan(plan):
            descriptor, adapter_cls = self.registry.get(tool_name, "*")
            if not allow_internal_tools and not is_agent_visible_tool(descriptor):
                raise InvalidParamsError(
                    "agent plan requested an internal tool",
                    details={
                        "tool": tool_name,
                        "reason": (
                            "internal/glue tools are not part of the LLM agent "
                            "first-class tool surface"
                        ),
                    },
                )
            availability = _descriptor_availability(descriptor, adapter_cls)
            install_status = _install_readiness(tool_name)
            if (
                availability.get("available") is not True
                or install_status.get("install_ready") is not True
            ):
                raise InvalidParamsError(
                    "agent plan requested an unavailable tool",
                    details={
                        "tool": tool_name,
                        "availability": availability,
                        "install_readiness": install_status,
                    },
                )

        workflow = self.compile_workflow(plan)
        workflow.validate_dag()
        for node in workflow.nodes:
            if node.tool != "rfdiffusion":
                continue
            has_runtime_ref = any(
                ref.startswith(("params.", "outputs.", "item"))
                for ref in iter_refs(node.inputs)
            )
            if has_runtime_ref:
                continue
            mode_status = _rfdiffusion_mode_readiness(node.inputs)
            if mode_status.get("ready") is not True:
                raise InvalidParamsError(
                    "agent plan requested an unavailable RFdiffusion mode",
                    details={"tool": node.tool, "mode_readiness": mode_status},
                )
        plan["workflow_plan"] = workflow.model_dump(mode="json", exclude_none=True)
        return plan

    def validate_invocation(
        self,
        plan: dict[str, Any],
        params: dict[str, Any],
        *,
        allow_internal_tools: bool = False,
    ) -> None:
        workflow = self.compile_workflow(
            self.validate_plan(dict(plan), allow_internal_tools=allow_internal_tools)
        )
        runtime_params = self.invocation_params(plan, params)
        _validate_runtime_param_types(workflow, runtime_params)
        merged = params_dict_from_workflow(workflow, runtime_params)
        scope = {"params": merged, "outputs": {}}
        for node in workflow.nodes:
            refs = iter_refs(node.inputs)
            if any(ref.startswith(("outputs.", "jobs.", "item")) for ref in refs):
                continue
            try:
                resolved = resolve(node.inputs, scope)
            except VPAError:
                raise
            if not isinstance(resolved, dict):
                raise InvalidParamsError(
                    "agent workflow node inputs resolved to a non-object",
                    details={"node_id": node.id, "type": type(resolved).__name__},
                )
            descriptor, _ = self.registry.get(node.tool, node.version)
            validate_params(descriptor, resolved)

    def invocation_params(self, plan: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return _invocation_params_from_plan(plan, params)

    def compile_workflow(self, plan: dict[str, Any]) -> Workflow:
        workflow_plan = plan.get("workflow_plan")
        if not isinstance(workflow_plan, dict):
            raise InvalidParamsError(
                "agent plan missing workflow_plan object",
                details={"field": "workflow_plan"},
            )
        try:
            return Workflow.model_validate(workflow_plan)
        except ValidationError as exc:
            raise InvalidParamsError(
                "agent workflow plan failed local validation",
                details={"errors": exc.errors()},
            ) from exc

    def compile_workflow_yaml(self, plan: dict[str, Any]) -> str:
        return serialize_workflow(self.compile_workflow(plan))

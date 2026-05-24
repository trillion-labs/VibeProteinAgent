from __future__ import annotations

import json
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
    names: list[str] = []
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
            "Recovered tool names from a non-JSON provider response using only "
            "the public tool inventory."
        ),
        "missing_inputs": [],
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


def _reject_post_round_action_tools(plan: dict[str, Any]) -> None:
    if "submit_candidate" not in tool_names_from_plan(plan):
        return
    raise InvalidParamsError(
        "submit_candidate is a post-round decision action, not a planner workflow node",
        details={"tool": "submit_candidate"},
    )


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

    async def draft_plan(
        self,
        request: str,
        *,
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del workflows
        if self.provider is None:
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        selection = await self._select_candidate_tools(
            {
                "request": request,
                "available_tools": [compact_tool_outline(t) for t in available_tools],
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
        selected_tools = self._selected_tool_schemas(selection, available_tools)
        prompt = {
            "request": request,
            "tool_selection": selection,
            "selected_tools": selected_tools,
            "output_contract": {
                "mode": "agent_plan",
                "goal": "string",
                "assumptions": ["string"],
                "missing_inputs": ["string"],
                "required_tools": ["tool_name"],
                "params": {"param_name": "runtime value"},
                "workflow_plan": {
                    "name": "string",
                    "version": "string",
                    "params": {"param_name": {"type": "str|int|float|bool|path|list|dict"}},
                    "nodes": [
                        {
                            "id": "node_id",
                            "tool": "tool_name",
                            "inputs": {
                                "param": "runtime value or ${params.x}/${outputs.node.field}"
                            },
                        }
                    ],
                },
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
        plan = _load_json_object_response(
            response.content,
            label="planner",
            error_message="planner returned non-JSON content",
        )
        return self.validate_round_plan(plan)

    async def draft_next_round_plan(
        self,
        *,
        previous_plan: dict[str, Any],
        summary: dict[str, Any],
        decision: dict[str, Any],
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del workflows
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
            "output_contract": {
                "mode": "agent_plan",
                "goal": "string",
                "assumptions": ["include parameter changes and reasons"],
                "missing_inputs": [],
                "required_tools": ["tool_name"],
                "params": {"param_name": "runtime value"},
                "workflow_plan": {
                    "name": "string",
                    "version": "string",
                    "nodes": [{"id": "node_id", "tool": "tool_name", "inputs": {}}],
                },
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
        return self.validate_round_plan(plan)

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

    def validate_plan(
        self,
        plan: dict[str, Any],
        *,
        allow_internal_tools: bool = False,
    ) -> dict[str, Any]:
        allow_internal_tools = allow_internal_tools or plan.get("source_kind") == "workflow_yaml"
        if not isinstance(plan, dict):
            raise InvalidParamsError(
                "agent workflow plan must be an object",
                details={"got_type": type(plan).__name__},
            )
        if "workflow_ref" in plan or "workflow" in plan:
            raise InvalidParamsError(
                "LLM agent plans must include an explicit workflow_plan, not workflow_ref",
                details={"fields": [key for key in ("workflow_ref", "workflow") if key in plan]},
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

    def validate_round_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        valid = self.validate_plan(plan)
        _reject_post_round_action_tools(valid)
        return valid

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

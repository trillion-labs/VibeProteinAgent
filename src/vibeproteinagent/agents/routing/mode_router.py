from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.session.types import ModeClassification
from vibeproteinagent.core.types import parse_kv_pairs
from vibeproteinagent.workflows.loader import load_workflow

_KV_RE = re.compile(r"(?<!\S)([A-Za-z_][A-Za-z0-9_.-]*)=(\S+)")
_JOB_RE = re.compile(r"\bjob_[0-9a-f]{32}\b")
_WF_RE = re.compile(r"\bwf_[0-9a-f]{32}\b")


class ModeRouter:
    """Local-first request classifier.

    The router deliberately handles direct execution without an LLM whenever the
    user named concrete tools, workflows, jobs, params, or artifact/status verbs.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.registry.discover()

    def classify(
        self,
        request: str,
        *,
        params: dict[str, Any] | None = None,
        tool: str | None = None,
        workflow: str | None = None,
        force_new: bool = False,
    ) -> ModeClassification:
        merged_params = {**_params_from_text(request), **(params or {})}
        text = request.strip()
        lower = text.lower()

        if workflow:
            return ModeClassification(
                mode="explicit_workflow",
                reason="Workflow was provided explicitly.",
                workflow_ref=workflow,
                params=merged_params,
                force_new=force_new,
                action="run_workflow",
            )

        if tool:
            direct_jobs = _direct_jobs([tool], merged_params)
            return ModeClassification(
                mode="direct_tool",
                reason="Tool was provided explicitly.",
                requested_tools=[tool],
                params=direct_jobs[0]["params"] if direct_jobs else merged_params,
                force_new=force_new,
                action=_direct_action(lower),
                direct_jobs=direct_jobs,
                execution=_direct_execution(lower),
            )

        workflow_ref = _workflow_ref_from_text(text)
        job_refs = _JOB_RE.findall(text)
        workflow_match = _WF_RE.search(text)
        if job_refs:
            action = _direct_action(lower)
            if action in {"get_status", "get_results"}:
                return ModeClassification(
                    mode="direct_tool",
                    reason="Request references existing job results or status.",
                    requested_tools=self._mentioned_tools(text),
                    workflow_ref=workflow_match.group(0) if workflow_match else None,
                    job_refs=job_refs,
                    params=merged_params,
                    action=action,
                )
            return ModeClassification(
                mode="analysis_only",
                reason="Request references existing jobs for analysis.",
                requested_tools=self._mentioned_tools(text),
                workflow_ref=workflow_match.group(0) if workflow_match else None,
                job_refs=job_refs,
                params=merged_params,
                action="summarize",
            )
        if workflow_ref is not None:
            return ModeClassification(
                mode="explicit_workflow",
                reason="Request references a workflow descriptor or workflow id.",
                workflow_ref=workflow_ref,
                params=merged_params,
                force_new=force_new,
                action="run_workflow",
            )

        tool_names = self._mentioned_tools(text)
        if len(tool_names) == 1 and _looks_direct_execution_request(lower):
            direct_jobs = _direct_jobs(tool_names, merged_params)
            return ModeClassification(
                mode="direct_tool",
                reason="Request names one registered tool and asks to execute or inspect it.",
                requested_tools=tool_names,
                params=direct_jobs[0]["params"] if direct_jobs else merged_params,
                force_new=force_new or "force_new=true" in lower or "--force-new" in lower,
                action=_direct_action(lower),
                direct_jobs=direct_jobs,
                execution=_direct_execution(lower),
            )
        if len(tool_names) > 1 and _looks_direct_execution_request(lower):
            direct_jobs = _direct_jobs(tool_names, merged_params)
            return ModeClassification(
                mode="direct_tool_set",
                reason=(
                    "Request names multiple registered tools with a clear "
                    "direct execution intent."
                ),
                requested_tools=tool_names,
                params=merged_params,
                force_new=force_new,
                action=_direct_action(lower),
                direct_jobs=direct_jobs,
                execution=_direct_execution(lower),
            )

        workflow_match = _WF_RE.search(text)
        if workflow_match:
            return ModeClassification(
                mode="analysis_only",
                reason=(
                    "Request references existing jobs/workflows and asks for "
                    "analysis or status."
                ),
                requested_tools=tool_names,
                workflow_ref=workflow_match.group(0) if workflow_match else None,
                job_refs=[],
                params=merged_params,
                action="summarize",
            )

        if (
            (job_refs or workflow_match)
            and any(
                word in lower
                for word in ("summarize", "rank", "compare", "interpret", "analyze")
            )
        ):
            return ModeClassification(
                mode="analysis_only",
                reason="Request asks for interpretation without a concrete new execution.",
                requested_tools=tool_names,
                params=merged_params,
                action="summarize",
            )

        missing = []
        target_keys = ("target", "target_pdb", "pdb_path")
        if "target" in lower and not any(k in merged_params for k in target_keys):
            missing.append("target_pdb")
        return ModeClassification(
            mode="agent_plan",
            reason="Request is a scientific goal or asks for planning without a concrete workflow.",
            requires_llm_planning=True,
            requested_tools=tool_names,
            missing_inputs=missing,
            params=merged_params,
            force_new=force_new,
            action="plan",
        )

    def _mentioned_tools(self, text: str) -> list[str]:
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.lower()))
        names = []
        for descriptor in self.registry.list():
            if descriptor.name.lower() in tokens or descriptor.name.lower() in text.lower():
                names.append(descriptor.name)
        return sorted(set(names))


def _params_from_text(text: str) -> dict[str, Any]:
    return parse_kv_pairs([f"{m.group(1)}={m.group(2)}" for m in _KV_RE.finditer(text)])


def _workflow_ref_from_text(text: str) -> str | None:
    match = _WF_RE.search(text)
    if match:
        return match.group(0)
    for token in re.findall(r"[^\s]+\.ya?ml", text):
        cleaned = token.strip("'\".,:;()[]")
        if Path(cleaned).is_file():
            return cast(str, cleaned)
        try:
            load_workflow(cleaned)
            return cast(str, cleaned)
        except Exception:
            return cast(str, cleaned)
    return None


def _looks_execution_request(lower: str) -> bool:
    return _has_any_word(
        lower,
        (
            "run",
            "submit",
            "execute",
            "warm",
            "retry",
            "fetch",
            "result",
            "results",
            "status",
            "inspect",
        ),
    )


def _looks_direct_execution_request(lower: str) -> bool:
    """Return true only for explicit tool/job operations.

    Scientific requests often contain words like "result" ("as a result of
    hydration") or tool names ("use rfdiffusion then ProteinMPNN") while still
    requiring a workflow plan. Keep direct mode for operator-style commands and
    let design/generation objectives go through the planner.
    """
    if _has_scientific_design_intent(lower):
        return False
    return _looks_execution_request(lower)


def _has_scientific_design_intent(lower: str) -> bool:
    return _has_any_word(
        lower,
        (
            "design",
            "generate",
            "generation",
            "binder",
            "binding",
            "backbone",
            "inverse-fold",
            "inverse",
            "refold",
            "rank",
            "specificity",
            "activity",
            "objective",
            "constraint",
            "constraints",
        ),
    )


def _has_any_word(lower: str, words: tuple[str, ...]) -> bool:
    tokens = set(re.findall(r"[a-z0-9_+-]+", lower))
    return any(word in tokens for word in words)


def _direct_action(lower: str) -> str:
    if _has_any_word(lower, ("result", "results", "fetch")):
        return "get_results"
    if _has_any_word(lower, ("status", "state")):
        return "get_status"
    if _has_any_word(lower, ("warm", "preload")):
        return "warm"
    if _has_any_word(lower, ("inspect", "info", "describe")):
        return "tool_info"
    return "submit"


def _direct_jobs(tools: list[str], params: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = params.get("jobs")
    if isinstance(explicit, list):
        jobs = []
        for item in explicit:
            if not isinstance(item, dict):
                continue
            tool = item.get("tool")
            if not isinstance(tool, str):
                continue
            job_params = item.get("params") if isinstance(item.get("params"), dict) else {}
            jobs.append(
                {
                    "tool": tool,
                    "params": job_params,
                    "version": item.get("version", "*"),
                    "call_id": item.get("call_id"),
                    "priority": item.get("priority"),
                    "force_new": item.get("force_new"),
                    "max_retries": item.get("max_retries"),
                }
            )
        if jobs:
            return jobs

    tool_set = set(tools)
    shared = {
        key: value
        for key, value in params.items()
        if "." not in key and key != "jobs"
    }
    jobs = []
    for tool in tools:
        prefix = f"{tool}."
        job_params = dict(shared)
        job_params.update(
            {
                key.removeprefix(prefix): value
                for key, value in params.items()
                if key.startswith(prefix)
            }
        )
        # Also support category-like aliases only if they exactly match the
        # requested tool set; unrelated dotted keys are left out deliberately.
        for key, value in params.items():
            head, sep, tail = key.partition(".")
            if sep and head in tool_set and head == tool:
                job_params[tail] = value
        jobs.append({"tool": tool, "params": job_params, "version": "*"})
    return jobs


def _direct_execution(lower: str) -> dict[str, Any]:
    if any(word in lower for word in (" then ", " after ", " sequential")):
        return {"type": "sequential"}
    if any(word in lower for word in ("parallel", "concurrent", "independent")):
        return {"type": "parallel"}
    return {"type": "parallel"}

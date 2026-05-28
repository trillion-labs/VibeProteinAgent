from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from vibeproteinagent.adapters import ToolRegistry, validate_params
from vibeproteinagent.agents.config import (
    AgentConfig,
    resolve_decision_normalization_mode,
)
from vibeproteinagent.agents.evaluation import DecisionEngine, DecisionSchemaViolation
from vibeproteinagent.agents.planning import WorkflowPlanner
from vibeproteinagent.agents.planning.compiler import (
    PLANNER_COMPILED_SOURCE_KIND,
    apply_fixup_choice,
    compile_planner_ir,
)
from vibeproteinagent.agents.planning.fixup import (
    StructuredFixup,
    malformed_fixup_response,
)
from vibeproteinagent.agents.provider import (
    ChatMessage,
    ChatProvider,
    SessionLoggingProvider,
    provider_from_config,
)
from vibeproteinagent.agents.routing import ModeRouter
from vibeproteinagent.agents.session import AgentSessionStore, ModeClassification
from vibeproteinagent.agents.summarization import ResultSummarizer
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.agents.tools.schemas import tool_names_from_plan
from vibeproteinagent.core.errors import ErrorCode, InvalidParamsError, TimeoutError_, VPAError
from vibeproteinagent.workers.supervisor_client import SupervisorClient
from vibeproteinagent.workflows import WorkflowExecutor, load_workflow
from vibeproteinagent.workspace import Workspace

from .policies import ExecutionPolicy
from .memory_bank import MemoryBank
from .memory_summarizer import MemorySummarizer
from .round_state import PriorRoundContext
from .run_result import AgentRunResult

T = TypeVar("T")
audit_log = logging.getLogger("vibeproteinagent.audit")


@dataclass(frozen=True)
class WorkflowExecutionOutcome:
    workflow_id: str | None
    state: str
    executed: bool
    summary: dict[str, Any]


class AgentController:
    """Optional orchestration layer over VPA's deterministic execution substrate."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        session_store: AgentSessionStore | None = None,
    ) -> None:
        self.workspace = workspace
        self.registry = registry or ToolRegistry()
        self.registry.discover()
        self.config = config or AgentConfig.from_env()
        self.session_store = session_store or AgentSessionStore(workspace)
        self.router = ModeRouter(self.registry)
        self._round_state: PriorRoundContext | None = None

    def run_sync(
        self,
        request: str,
        *,
        params: dict[str, Any] | None = None,
        tool: str | None = None,
        workflow: str | None = None,
        force_new: bool = False,
        policy: ExecutionPolicy | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.run(
                request,
                params=params,
                tool=tool,
                workflow=workflow,
                force_new=force_new,
                policy=policy,
            )
        )

    async def run(
        self,
        request: str,
        *,
        params: dict[str, Any] | None = None,
        tool: str | None = None,
        workflow: str | None = None,
        force_new: bool = False,
        policy: ExecutionPolicy | None = None,
    ) -> dict[str, Any]:
        policy = policy or ExecutionPolicy()
        run = self.session_store.create_run(
            request,
            provider=self.config.provider,
            model=self.config.model,
        )
        self.session_store.add_message(run.run_id, "user", request)
        broker = AgentToolBroker(self.workspace, self.registry, self.session_store, run.run_id)

        classification = self.router.classify(
            request,
            params=params,
            tool=tool,
            workflow=workflow,
            force_new=force_new,
        )
        self.session_store.update_run(run.run_id, state="classified", classification=classification)

        self._round_state = None
        try:
            if classification.mode == "direct_tool":
                report = self._run_direct_tool(run.run_id, broker, classification, policy)
            elif classification.mode == "direct_tool_set":
                report = self._run_direct_tool_set(run.run_id, broker, classification, policy)
            elif classification.mode == "explicit_workflow":
                report = self._run_explicit_workflow(run.run_id, broker, classification, policy)
            elif classification.mode == "analysis_only":
                report = self._run_analysis_only(run.run_id, classification)
            else:
                report = await self._run_agent_plan(
                    run.run_id,
                    broker,
                    request,
                    classification,
                    policy,
                )
            self.session_store.update_run(
                run.run_id,
                state=_agent_run_state(report),
                workflow_ids=report.get("workflow_ids", []),
                job_ids=report.get("job_ids", []),
                artifacts=report.get("artifacts", {}),
                final_report=report,
            )
            return {"run_id": run.run_id, **report}
        except Exception:
            self.session_store.update_run(run.run_id, state="failed")
            raise
        finally:
            self._round_state = None

    def _run_direct_tool(
        self,
        run_id: str,
        broker: AgentToolBroker,
        classification: ModeClassification,
        policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        tool = classification.requested_tools[0] if classification.requested_tools else None
        if not tool:
            if classification.action in {"get_status", "get_results"} and classification.job_refs:
                return self._run_job_ref_action(run_id, broker, classification)
            raise InvalidParamsError(
                "direct_tool mode requires a tool or job reference",
                details={},
            )
        if classification.action in {"get_status", "get_results"}:
            return self._run_job_ref_action(run_id, broker, classification)
        if classification.action == "warm":
            result = broker.warm_tool(tool)
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="completed" if result.get("ok") else "failed",
                job_ids=[],
                workflow_ids=[],
                result=result,
            )
        if classification.action == "tool_info":
            result = broker.tool_info(tool)
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="completed" if result.get("ok", True) is not False else "failed",
                job_ids=[],
                workflow_ids=[],
                result=result,
            )
        job_spec = (
            classification.direct_jobs[0]
            if classification.direct_jobs
            else {"tool": tool, "params": classification.params, "version": "*"}
        )
        if not policy.submit:
            _validate_direct_job_specs(self.registry, [job_spec])
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="planned",
                job_ids=[],
                workflow_ids=[],
                result={"job": _planned_job(job_spec, policy)},
            )
        result = broker.submit_job(
            str(job_spec.get("tool") or tool),
            _job_params(job_spec),
            version=str(job_spec.get("version") or "*"),
            call_id=_optional_str(job_spec.get("call_id")),
            priority=str(job_spec.get("priority") or policy.priority),
            force_new=_optional_bool(job_spec.get("force_new"), classification.force_new),
            max_retries=_optional_int(job_spec.get("max_retries"), policy.max_retries),
        )
        if result.get("ok") is False and result.get("code"):
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="failed",
                job_ids=[],
                workflow_ids=[],
                result=result,
            )
        job_ids = [result["job_id"]]
        summary = ResultSummarizer(self.workspace).summarize_job(result["job_id"])
        self.session_store.add_summary(run_id, summary, job_id=result["job_id"])
        return _final_report(
            mode=classification.mode,
            classification=classification,
            state=result.get("state", "submitted"),
            job_ids=job_ids,
            workflow_ids=[],
            result=result,
            summary=summary,
        )

    def _run_direct_tool_set(
        self,
        run_id: str,
        broker: AgentToolBroker,
        classification: ModeClassification,
        policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        jobs = []
        specs = classification.direct_jobs or [
            {"tool": tool, "params": classification.params, "version": "*"}
            for tool in classification.requested_tools
        ]
        if not policy.submit and classification.action == "submit":
            _validate_direct_job_specs(self.registry, specs)
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="planned",
                job_ids=[],
                workflow_ids=[],
                result={"jobs": [_planned_job(spec, policy) for spec in specs]},
            )
        sequential = classification.execution.get("type") == "sequential"
        supervisor_running = SupervisorClient().is_running() if sequential else False
        for job_spec in specs:
            tool = str(job_spec.get("tool") or "")
            if classification.action == "warm":
                result = broker.warm_tool(tool, version=str(job_spec.get("version") or "*"))
            elif classification.action == "tool_info":
                result = broker.tool_info(tool, version=str(job_spec.get("version") or "*"))
            else:
                result = broker.submit_job(
                    tool,
                    _job_params(job_spec),
                    version=str(job_spec.get("version") or "*"),
                    call_id=_optional_str(job_spec.get("call_id")),
                    priority=str(job_spec.get("priority") or policy.priority),
                    force_new=_optional_bool(job_spec.get("force_new"), classification.force_new),
                    max_retries=_optional_int(job_spec.get("max_retries"), policy.max_retries),
                )
            jobs.append(result)
            if (
                classification.action == "submit"
                and sequential
                and isinstance(result, dict)
                and result.get("job_id")
            ):
                if policy.execute and supervisor_running:
                    waited = broker.wait_job(
                        str(result["job_id"]),
                        timeout_ms=policy.wait_job_timeout_ms,
                    )
                    jobs[-1] = waited
                    if (
                        waited.get("timed_out")
                        or waited.get("state") not in {"completed"}
                        or waited.get("ok") is False
                    ):
                        break
                elif policy.execute:
                    break
        job_ids = [j["job_id"] for j in jobs if isinstance(j, dict) and j.get("job_id")]
        summaries = [ResultSummarizer(self.workspace).summarize_job(jid) for jid in job_ids]
        for jid, summary in zip(job_ids, summaries, strict=False):
            self.session_store.add_summary(run_id, summary, job_id=jid)
        state = _aggregate_job_state(jobs)
        return _final_report(
            mode=classification.mode,
            classification=classification,
            state=state,
            job_ids=job_ids,
            workflow_ids=[],
            result={"jobs": jobs},
            summary={"jobs": summaries},
        )

    def _run_job_ref_action(
        self,
        run_id: str,
        broker: AgentToolBroker,
        classification: ModeClassification,
    ) -> dict[str, Any]:
        results = []
        summaries = []
        for job_id in classification.job_refs:
            if classification.action == "get_results":
                result = broker.get_results(job_id)
            else:
                result = broker.get_status(job_id)
            results.append(result)
            try:
                summary = ResultSummarizer(self.workspace).summarize_job(job_id)
            except Exception:
                summary = None
            if summary is not None:
                summaries.append(summary)
                self.session_store.add_summary(run_id, summary, job_id=job_id)
        return _final_report(
            mode=classification.mode,
            classification=classification,
            state=_aggregate_job_state(results),
            job_ids=classification.job_refs,
            workflow_ids=[],
            result={"jobs": results},
            summary={"jobs": summaries},
        )

    def _run_explicit_workflow(
        self,
        run_id: str,
        broker: AgentToolBroker,
        classification: ModeClassification,
        policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        ref = classification.workflow_ref
        if not ref:
            raise InvalidParamsError("explicit_workflow mode requires workflow_ref", details={})
        if ref.startswith("wf_"):
            status = broker.get_workflow_status(ref)
            summary = ResultSummarizer(self.workspace).summarize_workflow(ref)
            self.session_store.add_summary(run_id, summary, workflow_id=ref)
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state=status.get("state", "unknown"),
                job_ids=summary.get("job_ids", []),
                workflow_ids=[ref],
                result=status,
                summary=summary,
            )
        if not policy.submit:
            workflow_obj = _load_workflow_ref(ref)
            WorkflowPlanner(self.registry).validate_invocation(
                {
                    "mode": "agent_plan",
                    "workflow_plan": workflow_obj.model_dump(mode="json", exclude_none=True),
                },
                classification.params,
                allow_internal_tools=True,
            )
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="planned",
                job_ids=[],
                workflow_ids=[],
                result={"workflow_ref": ref, "params": classification.params},
            )
        if not policy.execute:
            workflow_obj = _load_workflow_ref(ref)
            WorkflowPlanner(self.registry).validate_invocation(
                {
                    "mode": "agent_plan",
                    "workflow_plan": workflow_obj.model_dump(mode="json", exclude_none=True),
                },
                classification.params,
                allow_internal_tools=True,
            )
            executor = WorkflowExecutor(self.workspace, self.registry, None)
            workflow_id = executor.submit(workflow_obj, classification.params)
            summary = ResultSummarizer(self.workspace).summarize_workflow(workflow_id)
            summary["state"] = "submitted_not_executed"
            self.session_store.add_summary(run_id, summary, workflow_id=workflow_id)
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="submitted_not_executed",
                job_ids=summary.get("job_ids", []),
                workflow_ids=[workflow_id],
                result={
                    "workflow_id": workflow_id,
                    "state": "submitted_not_executed",
                    "workflow_ref": ref,
                    "params": classification.params,
                },
                summary=summary,
            )
        result = broker.run_workflow(ref, classification.params, max_wait_s=policy.max_wait_s)
        workflow_id = result.get("workflow_id")
        workflow_summary: dict[str, Any] | None = None
        if workflow_id:
            workflow_summary = ResultSummarizer(self.workspace).summarize_workflow(workflow_id)
            self.session_store.add_summary(run_id, workflow_summary, workflow_id=workflow_id)
        return _final_report(
            mode=classification.mode,
            classification=classification,
            state=_workflow_result_state(result),
            job_ids=(workflow_summary or {}).get("job_ids", []),
            workflow_ids=[workflow_id] if workflow_id else [],
            result=result,
            summary=workflow_summary,
        )

    def _run_analysis_only(
        self,
        run_id: str,
        classification: ModeClassification,
    ) -> dict[str, Any]:
        summarizer = ResultSummarizer(self.workspace)
        job_summaries: list[dict[str, Any]] = []
        workflow_summaries: list[dict[str, Any]] = []
        job_ids: list[str] = []
        for job_id in classification.job_refs:
            summary = summarizer.summarize_job(job_id)
            job_summaries.append(summary)
            job_ids.append(job_id)
            self.session_store.add_summary(run_id, summary, job_id=job_id)
        if classification.workflow_ref and classification.workflow_ref.startswith("wf_"):
            summary = summarizer.summarize_workflow(classification.workflow_ref)
            workflow_summaries.append(summary)
            self.session_store.add_summary(
                run_id, summary, workflow_id=classification.workflow_ref
            )
        return _final_report(
            mode=classification.mode,
            classification=classification,
            state="completed",
            job_ids=job_ids,
            workflow_ids=[classification.workflow_ref] if classification.workflow_ref else [],
            result={"jobs": job_summaries, "workflows": workflow_summaries},
            summary={"jobs": job_summaries, "workflows": workflow_summaries},
        )

    async def _run_agent_plan(
        self,
        run_id: str,
        broker: AgentToolBroker,
        request: str,
        classification: ModeClassification,
        policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        self._round_state = None
        try:
            return await self._run_agent_plan_impl(
                run_id,
                broker,
                request,
                classification,
                policy,
            )
        finally:
            self._round_state = None

    async def _run_agent_plan_impl(
        self,
        run_id: str,
        broker: AgentToolBroker,
        request: str,
        classification: ModeClassification,
        policy: ExecutionPolicy,
    ) -> dict[str, Any]:
        provider = SessionLoggingProvider(
            provider_from_config(self.config),
            self.session_store,
            run_id,
        )
        available_tools = broker.list_tools()
        planner = WorkflowPlanner(self.registry, provider)
        entry_failures: list[dict[str, Any]] = []

        async def draft_initial_plan(
            attempt: int,
            retry_context: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del retry_context
            async def compile_with_fixup(planner_ir: dict[str, Any]) -> dict[str, Any]:
                return await self._compile_with_fixup(
                    planner_ir,
                    provider=provider,
                    idempotency_key_prefix=(
                        f"{run_id}:initial-plan-fixup:{attempt}"
                    ),
                    prior_round_context=None,
                )

            candidate = await planner._plan_v1(
                request,
                available_tools=available_tools,
                known_inputs=classification.params,
                model=self.config.model_for_role("planner"),
                idempotency_key=f"{run_id}:initial-plan:{attempt}",
                compile_with_fixup=compile_with_fixup,
            )
            planner.validate_invocation(
                candidate,
                classification.params,
                allow_internal_tools=_plan_allows_internal_tools(candidate),
            )
            return candidate

        plan = await self._run_llm_entry(
            phase="initial_plan",
            failures=entry_failures,
            factory=draft_initial_plan,
        )
        planner.validate_invocation(
            plan,
            classification.params,
            allow_internal_tools=_plan_allows_internal_tools(plan),
        )
        self.session_store.add_plan(run_id, plan, status="validated")
        if not policy.submit:
            return _final_report(
                mode=classification.mode,
                classification=classification,
                state="planned",
                job_ids=[],
                workflow_ids=[],
                result={"plan": plan},
                plan=plan,
            )

        workflow_ids: list[str] = []
        summaries: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        current_plan = plan
        final_state = "submitted"
        final_result: AgentRunResult | None = None
        self._round_state = None
        memory_bank = MemoryBank()
        prompt_digest: dict[str, Any] | None = None
        survivor_pool: list[dict[str, Any]] = []

        for round_index in range(max(1, self.config.max_rounds)):
            prior_round_context = self._round_state
            outcome = await self._execute_plan_thread(
                run_id=run_id,
                broker=broker,
                current_plan=current_plan,
                params=classification.params,
                policy=policy,
                prior_round_context=prior_round_context,
            )
            final_state = outcome.state
            if outcome.workflow_id:
                workflow_ids.append(outcome.workflow_id)
            summaries.append(outcome.summary)
            self.session_store.update_run(
                run_id,
                state="running" if outcome.executed else final_state,
                workflow_ids=workflow_ids,
                job_ids=[jid for item in summaries for jid in item.get("job_ids", [])],
                final_report=_partial_report(classification, workflow_ids, summaries, final_state),
            )
            if not outcome.executed or outcome.state != "completed":
                decision = DecisionEngine(None).local_decision(
                    outcome.summary,
                    success_criteria=current_plan.get("success_criteria") or {},
                )
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                break
            current_round_candidates = self._collect_current_round_candidates(outcome.summary)
            survivor_pool = _merge_candidate_pool(survivor_pool, current_round_candidates)
            max_round_reached = round_index >= self.config.max_rounds - 1
            round_record = memory_bank.record_round(
                round_index=round_index,
                workflow_summary=outcome.summary,
                current_round_candidates=current_round_candidates,
                survivor_pool=survivor_pool,
                max_round_reached=max_round_reached,
            )
            prior_prompt_digest = prompt_digest
            prompt_digest = MemorySummarizer().summarize(
                previous_digest=prior_prompt_digest,
                latest_round_record=round_record,
                memory_bank=memory_bank,
            )
            outcome.summary["current_round_candidate_summary"] = round_record[
                "current_round_candidate_summary"
            ]
            outcome.summary["survivor_pool_summary"] = round_record["survivor_pool_summary"]
            outcome.summary["empty_candidate_events"] = round_record["empty_candidate_events"]
            outcome.summary["skipped_nodes"] = round_record["skipped_nodes"]
            outcome.summary["memory_digest"] = prompt_digest
            if (
                max_round_reached
                and not survivor_pool
                and not outcome.summary.get("final_submission")
                and (
                    _has_empty_candidate_outcome(round_record)
                    or _summary_declares_no_available_candidates(outcome.summary)
                )
            ):
                decision = {
                    "decision": "no_available_candidate",
                    "reasoning": "Max rounds reached with no current candidates and no survivor pool.",
                    "technical_failure": False,
                    "next_round": {
                        "recommended": False,
                        "changes": [],
                        "reason": "No viable candidates are available.",
                    },
                    "resolved_branch": {
                        "action": "stop",
                        "reason": "no_available_candidate",
                        "failure": False,
                    },
                    "metrics_used": [],
                    "notes": [],
                }
                memory_bank.record_branch(round_index, decision["resolved_branch"])
                memory_bank.record_decision(round_index, decision)
                latest_round_record = memory_bank.latest_round_record()
                if latest_round_record is not None:
                    prompt_digest = MemorySummarizer().summarize(
                        previous_digest=prior_prompt_digest,
                        latest_round_record=latest_round_record,
                        memory_bank=memory_bank,
                    )
                    outcome.summary["memory_digest"] = prompt_digest
                outcome.summary["branch_decision"] = decision["resolved_branch"]
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                final_state = "completed"
                final_result = AgentRunResult(
                    kind="no_available_candidate",
                    reasoning="No available candidate was produced before max rounds.",
                )
                break
            if survivor_pool and not current_round_candidates:
                outcome.summary["survivor_pool"] = survivor_pool[:20]
            final_submission = outcome.summary.get("final_submission")
            if isinstance(final_submission, dict):
                decision = {
                    "decision": "submitted_final_candidate",
                    "next_round": {
                        "recommended": False,
                        "changes": [],
                        "reason": "workflow completed with an explicit submit_candidate result",
                    },
                    "metrics_used": outcome.summary.get("metrics_used") or [],
                    "notes": [],
                    "submission": final_submission,
                }
                memory_bank.record_decision(round_index, decision)
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                break
            round_empty = not current_round_candidates
            if (
                _has_empty_candidate_outcome(round_record)
                and round_empty
                and not max_round_reached
            ):
                decision = _deterministic_empty_next_round_decision(
                    survivor_pool=survivor_pool,
                    current_round_candidates=current_round_candidates,
                )
            else:
                try:
                    decision = await self._run_llm_entry(
                        phase=f"decision_round_{round_index}",
                        failures=entry_failures,
                        factory=lambda attempt,
                        retry_context=None,
                        round_index=round_index,
                        outcome=outcome,
                        current_plan=current_plan: DecisionEngine(provider).decide(
                            outcome.summary,
                            success_criteria=current_plan.get("success_criteria")
                            or {},
                            model=self.config.model_for_role("analyst"),
                            idempotency_key=f"{run_id}:decision:{round_index}:{attempt}",
                            retry_error=retry_context,
                        ),
                    )
                except DecisionSchemaViolation:
                    raise
                except VPAError as exc:
                    decision = DecisionEngine(None).local_decision(
                        outcome.summary,
                        success_criteria=current_plan.get("success_criteria") or {},
                    )
                    notes = decision.setdefault("notes", [])
                    if isinstance(notes, list):
                        notes.append(
                            {
                                "level": "warning",
                                "message": (
                                    "LLM decision step failed after workflow completion; "
                                    "used deterministic local decision instead."
                                ),
                                "code": exc.code,
                                "details": _json_safe_decision_details(exc.details),
                            }
                        )
            memory_bank.record_decision(round_index, decision)
            if decision.get("decision") == "answer":
                final_result = AgentRunResult.from_answer_decision(decision)
                outcome.summary["answer"] = final_result.to_dict()
                final_state = "completed"
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                break
            if (
                decision.get("decision") == "next_round_with_surviving_candidate"
                and round_index >= self.config.max_rounds - 1
            ):
                ref = _first_submit_candidate_ref(outcome.summary)
                if ref is not None:
                    decision = {
                        **decision,
                        "decision": "submit_candidate",
                        "submit_candidate": {
                            "candidates": [ref],
                        },
                        "reasoning": (
                            str(decision.get("reasoning") or "").strip()
                            or "No local rounds remain; submit the best surviving candidate."
                        ),
                    }
                    decision.pop("next_round", None)
            branch = _resolve_decision_branch(
                decision,
                workflow_state=outcome.state,
                round_index=round_index,
                max_rounds=self.config.max_rounds,
            )
            decision["resolved_branch"] = branch
            memory_bank.record_branch(round_index, branch)
            latest_round_record = memory_bank.latest_round_record()
            if latest_round_record is not None:
                prompt_digest = MemorySummarizer().summarize(
                    previous_digest=prior_prompt_digest,
                    latest_round_record=latest_round_record,
                    memory_bank=memory_bank,
                )
                outcome.summary["memory_digest"] = prompt_digest
            outcome.summary["branch_decision"] = branch
            if branch["action"] == "submit_candidate":
                try:
                    submission, submit_job_id = self._execute_submit_candidate_decision(
                        run_id,
                        broker,
                        decision,
                        round_index=round_index,
                        policy=policy,
                    )
                except VPAError as exc:
                    notes = decision.setdefault("notes", [])
                    if isinstance(notes, list):
                        notes.append(
                            {
                                "level": "error",
                                "message": "submit_candidate decision failed during execution",
                                "code": exc.code,
                                "details": _json_safe_decision_details(exc.details),
                            }
                        )
                    final_state = "failed"
                    self.session_store.add_decision(run_id, decision, round_index=round_index)
                    decisions.append(decision)
                    break
                outcome.summary["final_submission"] = submission
                job_ids = outcome.summary.setdefault("job_ids", [])
                if isinstance(job_ids, list) and submit_job_id not in job_ids:
                    job_ids.append(submit_job_id)
                decision["submission"] = submission
                decision["submit_job_id"] = submit_job_id
                self.session_store.add_summary(
                    run_id,
                    {
                        "state": "completed",
                        "job_id": submit_job_id,
                        "tool": "submit_candidate",
                        "final_submission": submission,
                        "metrics_used": decision.get("metrics_used") or [],
                    },
                    job_id=submit_job_id,
                )
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                self.session_store.update_run(
                    run_id,
                    state=final_state,
                    workflow_ids=workflow_ids,
                    job_ids=[jid for item in summaries for jid in item.get("job_ids", [])],
                    final_report=_partial_report(
                        classification, workflow_ids, summaries, final_state
                    ),
                )
                break
            if branch["action"] == "stop":
                if (
                    not branch.get("failure")
                    and decision.get("decision")
                    in {
                        "next_round_with_no_acceptable_candidate",
                        "no_acceptable_candidate",
                        "next_round_with_surviving_candidate",
                    }
                ):
                    try:
                        submission, submit_job_id = self._execute_no_acceptable_candidate_submission(
                            run_id,
                            broker,
                            decision,
                            round_index=round_index,
                            policy=policy,
                        )
                    except VPAError as exc:
                        notes = decision.setdefault("notes", [])
                        if isinstance(notes, list):
                            notes.append(
                                {
                                    "level": "error",
                                    "message": (
                                        "no_acceptable_candidate terminal submission "
                                        "failed during execution"
                                    ),
                                    "code": exc.code,
                                    "details": _json_safe_decision_details(exc.details),
                                }
                            )
                        final_state = "failed"
                        self.session_store.add_decision(run_id, decision, round_index=round_index)
                        decisions.append(decision)
                        break
                    outcome.summary["final_submission"] = submission
                    job_ids = outcome.summary.setdefault("job_ids", [])
                    if isinstance(job_ids, list) and submit_job_id not in job_ids:
                        job_ids.append(submit_job_id)
                    source_decision = str(decision.get("decision") or "")
                    decision["source_decision"] = source_decision
                    decision["decision"] = "no_acceptable_candidate"
                    decision["submission"] = submission
                    decision["submit_job_id"] = submit_job_id
                    self.session_store.add_summary(
                        run_id,
                        {
                            "state": "completed",
                            "job_id": submit_job_id,
                            "tool": "submit_candidate",
                            "final_submission": submission,
                            "metrics_used": decision.get("metrics_used") or [],
                        },
                        job_id=submit_job_id,
                    )
                    final_state = "completed"
                    final_result = AgentRunResult(
                        kind="no_acceptable",
                        reasoning=_negative_submission_reason(decision),
                    )
                elif branch.get("failure"):
                    final_state = "failed"
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                break
            next_round = decision.get("next_round") or {}
            self.session_store.add_decision(run_id, decision, round_index=round_index)
            decisions.append(decision)
            self._round_state = PriorRoundContext(
                round_number=round_index + 2,
                surviving_candidates=survivor_pool,
                prior_decision_reason=_next_round_reason(decision),
                summary=self._prior_round_summary(outcome.summary),
                prompt_digest=prompt_digest,
            )
            try:
                async def compile_with_fixup(planner_ir: dict[str, Any]) -> dict[str, Any]:
                    return await self._compile_with_fixup(
                        planner_ir,
                        provider=provider,
                        idempotency_key_prefix=(
                            f"{run_id}:next-round-fixup:{round_index}"
                        ),
                        prior_round_context=self._round_state,
                    )

                current_plan = await self._run_llm_entry(
                    phase=f"next_round_plan_{round_index}",
                    failures=entry_failures,
                    factory=lambda attempt,
                    retry_context=None,
                    prior_round_context=self._round_state: planner._plan_v1(
                        request,
                        available_tools=available_tools,
                        known_inputs=classification.params,
                        model=self.config.model_for_role("planner"),
                        idempotency_key=(
                            f"{run_id}:next-round:{round_index}:{attempt}"
                        ),
                        compile_with_fixup=compile_with_fixup,
                        prior_round_context=prior_round_context,
                    ),
                )
                planner.validate_invocation(
                    current_plan,
                    classification.params,
                    allow_internal_tools=_plan_allows_internal_tools(current_plan),
                )
            except VPAError as exc:
                reason = (
                    "A completed workflow did not produce an explicit final "
                    "submission, and the next-round planner could not produce "
                    "a valid executable plan."
                )
                decision["decision"] = "next_round_with_no_acceptable_candidate"
                decision["reasoning"] = reason
                decision.pop("submit_candidate", None)
                decision.pop("next_round", None)
                decision["resolved_branch"] = {
                    "action": "stop",
                    "reason": reason,
                    "failure": True,
                }
                decision["notes"] = (
                    "LLM next-round planning failed after workflow completion; "
                    "recorded deterministic stop without submit_candidate."
                )
                self.session_store.add_decision(run_id, decision, round_index=round_index)
                decisions.append(decision)
                final_state = "failed"
                break
            self.session_store.add_plan(run_id, current_plan, status="validated_next_round")

        summary = summaries[-1]
        final_decision = decisions[-1] if decisions else None
        return _final_report(
            mode=classification.mode,
            classification=classification,
            state=final_state,
            job_ids=[jid for item in summaries for jid in item.get("job_ids", [])],
            workflow_ids=workflow_ids,
            result=(
                final_result.to_dict()
                if final_result is not None
                else {"workflow_ids": workflow_ids, "state": final_state}
            ),
            summary={"rounds": summaries, "latest": summary},
            decision=final_decision,
            decisions=decisions,
            plan=plan,
        )

    def _collect_surviving_candidates(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        return self._collect_current_round_candidates(summary)

    def _collect_current_round_candidates(self, summary: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for node in summary.get("nodes") or []:
            if not isinstance(node, dict):
                continue
            output = node.get("output")
            if not isinstance(output, dict):
                continue
            for candidate in output.get("candidates") or []:
                if not _is_candidate_v1(candidate):
                    continue
                key = str(candidate.get("candidate_id"))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(dict(candidate))
        return candidates

    def _prior_round_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "workflow_id": summary.get("workflow_id"),
            "state": summary.get("state"),
            "job_ids": list(summary.get("job_ids") or [])[:20],
            "metrics_used": list(summary.get("metrics_used") or [])[:20],
            "top_designs": list(summary.get("top_designs") or [])[:5],
        }

    def _build_executor_scope(
        self,
        *,
        prior_round: PriorRoundContext | None = None,
    ) -> dict[str, Any]:
        if prior_round is None:
            return {"outputs": {}}
        return {
            "outputs": {
                "prior_round": {
                    "round_number": prior_round.round_number,
                    "candidates": prior_round.surviving_candidates,
                    "reason": prior_round.prior_decision_reason,
                    "summary": prior_round.summary or {},
                }
            }
        }

    async def _execute_plan_thread(
        self,
        *,
        run_id: str,
        broker: AgentToolBroker,
        current_plan: dict[str, Any],
        params: dict[str, Any],
        policy: ExecutionPolicy,
        prior_round_context: PriorRoundContext | None,
    ) -> WorkflowExecutionOutcome:
        args = (run_id, broker, current_plan, params, policy)
        if prior_round_context is None or not _accepts_prior_round_context(
            self._execute_validated_plan
        ):
            return await asyncio.to_thread(self._execute_validated_plan, *args)
        return await asyncio.to_thread(
            self._execute_validated_plan,
            *args,
            prior_round_context,
        )

    def _execute_submit_candidate_decision(
        self,
        run_id: str,
        broker: AgentToolBroker,
        decision: dict[str, Any],
        *,
        round_index: int,
        policy: ExecutionPolicy,
    ) -> tuple[dict[str, Any], str]:
        params = _submit_candidate_params_from_decision(_decision_v1_payload(decision))
        result = broker.submit_job(
            "submit_candidate",
            params,
            call_id=_safe_call_id(f"{run_id}-submit-candidate-{round_index}"),
            priority=policy.priority,
            max_retries=policy.max_retries,
        )
        if result.get("ok") is False:
            raise InvalidParamsError(
                "submit_candidate decision could not be submitted",
                details={"submit_result": result},
            )
        job_id = str(result.get("job_id") or "")
        if not job_id:
            raise InvalidParamsError(
                "submit_candidate submission did not return a job_id",
                details={"submit_result": result},
            )
        final = broker.wait_job(job_id, timeout_ms=policy.wait_job_timeout_ms)
        if final.get("state") != "completed":
            raise InvalidParamsError(
                "submit_candidate decision did not complete",
                details={"job_id": job_id, "job": final},
            )
        output = final.get("output")
        if not isinstance(output, dict) or output.get("terminal") is not True:
            raise InvalidParamsError(
                "submit_candidate completed without a terminal output",
                details={"job_id": job_id, "output": output},
            )
        return output, job_id

    def _execute_no_acceptable_candidate_submission(
        self,
        run_id: str,
        broker: AgentToolBroker,
        decision: dict[str, Any],
        *,
        round_index: int,
        policy: ExecutionPolicy,
    ) -> tuple[dict[str, Any], str]:
        params = {
            "outcome": "no_acceptable_candidate",
            "candidates": [],
            "reasoning": _negative_submission_reason(decision),
        }
        result = broker.submit_job(
            "submit_candidate",
            params,
            call_id=_safe_call_id(f"{run_id}-no-acceptable-candidate-{round_index}"),
            priority=policy.priority,
            max_retries=policy.max_retries,
        )
        if result.get("ok") is False:
            raise InvalidParamsError(
                "no_acceptable_candidate terminal submission could not be submitted",
                details={"submit_result": result},
            )
        job_id = str(result.get("job_id") or "")
        if not job_id:
            raise InvalidParamsError(
                "no_acceptable_candidate submission did not return a job_id",
                details={"submit_result": result},
            )
        final = broker.wait_job(job_id, timeout_ms=policy.wait_job_timeout_ms)
        if final.get("state") != "completed":
            raise InvalidParamsError(
                "no_acceptable_candidate terminal submission did not complete",
                details={"job_id": job_id, "job": final},
            )
        output = final.get("output")
        if (
            not isinstance(output, dict)
            or output.get("terminal") is not True
            or output.get("outcome") != "no_acceptable_candidate"
        ):
            raise InvalidParamsError(
                "no_acceptable_candidate submission completed without terminal output",
                details={"job_id": job_id, "output": output},
            )
        return output, job_id

    async def _run_llm_entry(
        self,
        *,
        phase: str,
        failures: list[dict[str, Any]],
        factory: Callable[[int, dict[str, Any] | None], Awaitable[T]],
    ) -> T:
        max_entries = max(1, self.config.max_entries)
        phase_failures: list[dict[str, Any]] = []
        remaining = max_entries - len(failures)
        if remaining <= 0:
            raise InvalidParamsError(
                "agent LLM entry budget exhausted",
                details={
                    "phase": phase,
                    "max_entries": max_entries,
                    "failure_count": len(failures),
                    "failures": failures,
                },
            )
        retry_context: dict[str, Any] | None = None
        for attempt in range(1, remaining + 1):
            try:
                return await factory(attempt, retry_context)
            except DecisionSchemaViolation as exc:
                entry = {
                    "phase": phase,
                    "attempt": attempt,
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
                phase_failures.append(entry)
                failures.append(entry)
                retry_context = {
                    "previous_output": exc.payload,
                    "structured_error": exc.structured_error,
                }
                if len(failures) >= max_entries:
                    raise exc
            except VPAError as exc:
                entry = {
                    "phase": phase,
                    "attempt": attempt,
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
                phase_failures.append(entry)
                failures.append(entry)
                if _is_terminal_planner_entry(entry):
                    raise InvalidParamsError(
                        "agent planning stopped on terminal missing executable inputs",
                        details={
                            "phase": phase,
                            "max_entries": max_entries,
                            "failure_count": len(failures),
                            "phase_failures": phase_failures,
                            "failures": failures,
                        },
                    ) from exc
                if len(failures) >= max_entries:
                    raise InvalidParamsError(
                        "agent LLM entry budget exhausted",
                        details={
                            "phase": phase,
                            "max_entries": max_entries,
                            "failure_count": len(failures),
                            "phase_failures": phase_failures,
                            "failures": failures,
                        },
                    ) from exc
        raise InvalidParamsError(
            "agent LLM entry budget exhausted",
            details={"phase": phase, "max_entries": max_entries, "failures": phase_failures},
        )

    async def _compile_with_fixup(
        self,
        llm_ir: dict[str, Any],
        *,
        provider: ChatProvider,
        idempotency_key_prefix: str = "planner-ir-fixup",
        max_retries: int | None = None,
        prior_round_context: PriorRoundContext | None = None,
    ) -> dict[str, Any]:
        retries = max(0, self.config.max_entries - 1) if max_retries is None else max_retries
        current_ir = llm_ir
        pending_prompt_fixup: dict[str, Any] | None = None
        pending_apply_fixup: dict[str, Any] | None = None
        for attempt in range(retries + 1):
            if pending_prompt_fixup is None:
                try:
                    return compile_planner_ir(
                        current_ir,
                        prior_round_context=prior_round_context,
                    )
                except StructuredFixup as fx:
                    prompt_fixup = fx.payload
                    apply_fixup = fx.payload
            else:
                prompt_fixup = pending_prompt_fixup
                apply_fixup = pending_apply_fixup or pending_prompt_fixup
                pending_prompt_fixup = None
                pending_apply_fixup = None

            if attempt == retries:
                raise StructuredFixup(prompt_fixup)

            choice = await self._ask_planner_for_fixup_choice(
                provider=provider,
                planner_ir=current_ir,
                fixup_payload=prompt_fixup,
                idempotency_key=f"{idempotency_key_prefix}:{attempt + 1}",
            )
            try:
                current_ir = apply_fixup_choice(current_ir, apply_fixup, choice)
            except ValueError as exc:
                pending_prompt_fixup = malformed_fixup_response(
                    original_payload=apply_fixup,
                    received_choice=choice,
                    parse_error=str(exc),
                ).payload
                pending_apply_fixup = apply_fixup
                continue
        raise AssertionError("unreachable planner IR compiler retry state")

    async def _ask_planner_for_fixup_choice(
        self,
        *,
        provider: ChatProvider,
        planner_ir: dict[str, Any],
        fixup_payload: dict[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        prompt = {
            "planner_ir": planner_ir,
            "structured_fixup": fixup_payload,
            "response_shape": {"choice_id": "remove_node | remove_upstream | swap_tool"},
            "instruction": (
                "Choose exactly one listed option. Return only a JSON object "
                "containing the enum choice fields."
            ),
        }
        response = await provider.chat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You resolve VPA planner compiler fixups using only the "
                        "provided enum options."
                    ),
                ),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=self.config.model_for_role("planner"),
            temperature=0,
            response_format={"type": "json_object"},
            idempotency_key=idempotency_key,
        )
        return _load_json_object(
            response.content,
            label="planner fixup choice",
            error_message="planner fixup choice was not valid JSON",
        )

    def _execute_validated_plan(
        self,
        run_id: str,
        broker: AgentToolBroker,
        plan: dict[str, Any],
        params: dict[str, Any],
        policy: ExecutionPolicy,
        prior_round_context: PriorRoundContext | None = None,
    ) -> WorkflowExecutionOutcome:
        planner = WorkflowPlanner(self.registry)
        planner.validate_invocation(
            plan,
            params,
            allow_internal_tools=_plan_allows_internal_tools(plan),
        )
        runtime_params = planner.invocation_params(plan, params)
        warm_results = []
        if policy.warm:
            for tool in tool_names_from_plan(plan):
                result = broker.warm_tool(tool, poll_timeout_s=0)
                warm_results.append(result)
                if policy.warm_required and result.get("ok") is False:
                    return WorkflowExecutionOutcome(
                        workflow_id=None,
                        state="warm_failed",
                        executed=False,
                        summary={
                            "state": "warm_failed",
                            "job_ids": [],
                            "warm_results": warm_results,
                            "metrics_used": [],
                            "notes": ["required warm step failed before workflow submission"],
                        },
                    )

        workflow = planner.compile_workflow(plan)
        supervisor = SupervisorClient()
        sup = supervisor if supervisor.is_running() else None
        if policy.execute and sup is None:
            return WorkflowExecutionOutcome(
                workflow_id=None,
                state="daemon_unavailable",
                executed=False,
                summary={
                    "state": "daemon_unavailable",
                    "job_ids": [],
                    "warm_results": warm_results,
                    "metrics_used": [],
                    "notes": ["supervisor is not running; workflow was not submitted"],
                },
            )
        executor_scope = self._build_executor_scope(prior_round=prior_round_context)
        executor = WorkflowExecutor(
            self.workspace,
            self.registry,
            sup,
            virtual_outputs=executor_scope["outputs"],
        )
        workflow_id = executor.submit(workflow, runtime_params)
        self.session_store.add_plan(run_id, plan, status="submitted", workflow_id=workflow_id)

        if policy.execute and sup is not None:
            try:
                final = executor.run_sync(workflow_id, max_wait_s=policy.max_wait_s)
            except TimeoutError_ as exc:
                final = executor.store.get(workflow_id)
                summary = ResultSummarizer(self.workspace).summarize_workflow(workflow_id)
                summary["warm_results"] = warm_results
                summary["state"] = "timeout"
                summary["error"] = {
                    "code": ErrorCode.TIMEOUT,
                    "message": exc.message,
                    "details": exc.details,
                }
                self.session_store.add_summary(run_id, summary, workflow_id=workflow_id)
                return WorkflowExecutionOutcome(workflow_id, "timeout", True, summary)
            executed = True
        else:
            final = executor.store.get(workflow_id)
            executed = False

        summary = ResultSummarizer(self.workspace).summarize_workflow(workflow_id)
        summary["warm_results"] = warm_results
        self.session_store.add_summary(run_id, summary, workflow_id=workflow_id)
        state = final.state if executed else "submitted_not_executed"
        summary["state"] = state
        return WorkflowExecutionOutcome(workflow_id, state, executed, summary)

    def validate_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        return WorkflowPlanner(self.registry).validate_plan(
            plan,
            allow_internal_tools=_plan_allows_internal_tools(plan),
        )

    def submit_plan(
        self,
        plan: dict[str, Any],
        params: dict[str, Any] | None = None,
        *,
        max_wait_s: int = 3600,
    ) -> dict[str, Any]:
        run = self.session_store.create_run("submit validated agent plan")
        planner = WorkflowPlanner(self.registry)
        allow_internal = _plan_allows_internal_tools(plan)
        valid = planner.validate_plan(plan, allow_internal_tools=allow_internal)
        planner.validate_invocation(
            valid,
            params or {},
            allow_internal_tools=allow_internal,
        )
        runtime_params = planner.invocation_params(valid, params or {})
        workflow = planner.compile_workflow(valid)
        supervisor = SupervisorClient()
        sup = supervisor if supervisor.is_running() else None
        executor = WorkflowExecutor(self.workspace, self.registry, sup)
        workflow_id = executor.submit(workflow, runtime_params)
        state = "submitted"
        timeout_error: TimeoutError_ | None = None
        if sup is not None:
            try:
                final = executor.run_sync(workflow_id, max_wait_s=max_wait_s)
                state = final.state
            except TimeoutError_ as exc:
                timeout_error = exc
                final = executor.store.get(workflow_id)
                state = "timeout"
        else:
            final = executor.store.get(workflow_id)
            state = "daemon_unavailable"
        summary = ResultSummarizer(self.workspace).summarize_workflow(workflow_id)
        if timeout_error is not None:
            summary["error"] = {
                "code": ErrorCode.TIMEOUT,
                "message": timeout_error.message,
                "details": timeout_error.details,
            }
        summary["state"] = state
        report = _final_report(
            mode="agent_plan",
            classification=ModeClassification(
                mode="agent_plan",
                reason="Submitted prevalidated plan",
                requires_llm_planning=False,
            ),
            state=state,
            job_ids=summary.get("job_ids", []),
            workflow_ids=[workflow_id],
            result={
                "workflow_id": workflow_id,
                "state": state,
                "stored_state": final.state,
                "error": summary.get("error"),
            },
            summary=summary,
            plan=valid,
        )
        self.session_store.add_plan(run.run_id, valid, status="submitted", workflow_id=workflow_id)
        self.session_store.add_summary(run.run_id, summary, workflow_id=workflow_id)
        self.session_store.update_run(
            run.run_id,
            state=_agent_run_state(report),
            workflow_ids=[workflow_id],
            job_ids=summary.get("job_ids", []),
            artifacts=_report_artifacts(summary),
            final_report=report,
        )
        return {"run_id": run.run_id, **report}


def _final_report(
    *,
    mode: str,
    classification: ModeClassification,
    state: str,
    job_ids: list[str],
    workflow_ids: list[str],
    result: Any,
    summary: Any | None = None,
    decision: dict[str, Any] | None = None,
    decisions: list[dict[str, Any]] | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics_used = (
        (summary or {}).get("metrics_used", [])
        if isinstance(summary, dict)
        else []
    )
    return {
        "mode": mode,
        "classification": classification.model_dump(),
        "state": state,
        "workflow_ids": workflow_ids,
        "workflow_id": workflow_ids[0] if workflow_ids else None,
        "job_ids": job_ids,
        "artifacts": _report_artifacts(summary),
        "metrics_used": metrics_used,
        "summary": summary,
        "result": result,
        "decision": decision,
        "decisions": decisions or ([] if decision is None else [decision]),
        "next_round": (decision or {}).get("next_round") if decision else None,
        "plan": plan,
    }


def current_decision_normalization_mode() -> str:
    return resolve_decision_normalization_mode()


def _safe_call_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    if not safe:
        safe = "call"
    return safe[:128]


def _resolve_decision_branch(
    decision: dict[str, Any],
    *,
    workflow_state: str,
    round_index: int,
    max_rounds: int,
) -> dict[str, Any]:
    if workflow_state != "completed":
        return {
            "action": "stop",
            "reason": f"workflow_state={workflow_state!r}",
            "failure": True,
        }
    label = decision.get("decision")
    remaining_rounds = max(0, max_rounds - round_index - 1)
    if label == "submit_candidate":
        submit = decision.get("submit_candidate")
        candidates = submit.get("candidates") if isinstance(submit, dict) else None
        if isinstance(candidates, list) and candidates:
            return {
                "action": "submit_candidate",
                "reason": "submit intent has at least one candidate reference",
                "failure": False,
            }
        return {
            "action": "stop",
            "reason": "submit intent did not include a valid candidate reference",
            "failure": True,
        }
    if label in {
        "next_round_with_surviving_candidate",
        "next_round_with_no_acceptable_candidate",
        "next_round",
        "no_acceptable_candidate",
    }:
        if remaining_rounds > 0:
            return {
                "action": "next_round",
                "reason": f"{remaining_rounds} local round(s) remain",
                "failure": False,
            }
        return {
            "action": "stop",
            "reason": "no remaining local rounds",
            "failure": False,
        }
    return {
        "action": "stop",
        "reason": f"unrecognized decision intent {label!r}",
        "failure": True,
    }


def _first_submit_candidate_ref(summary: dict[str, Any]) -> dict[str, Any] | None:
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        job_id = node.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            continue
        output = node.get("output")
        if not isinstance(output, dict):
            continue
        candidates = output.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("candidate_id")
            return {
                "job_id": job_id,
                "candidate_id": str(candidate_id) if candidate_id else None,
                "index": index,
                "justification": (
                    "Selected automatically because the decision requested another "
                    "round with surviving candidates but no local rounds remain."
                ),
                "scores_to_highlight": [],
            }
    return None


def _load_json_object(
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


_DECISION_V1_FIELDS = {
    "decision",
    "reasoning",
    "submit_candidate",
    "next_round",
    "metrics_used",
    "notes",
}


def _decision_v1_payload(decision: dict[str, Any]) -> dict[str, Any]:
    """Return only the public decision_v1 fields from a controller decision record."""

    payload: dict[str, Any] = {}
    for key in _DECISION_V1_FIELDS:
        if key not in decision:
            continue
        if key == "notes" and not isinstance(decision.get(key), str):
            continue
        payload[key] = decision[key]
    return payload


def _json_safe_decision_details(details: Any) -> Any:
    return _json_safe_value(details, seen=set(), depth=0)


def _json_safe_value(value: Any, *, seen: set[int], depth: int) -> Any:
    if depth > 8:
        return "<truncated>"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    value_id = id(value)
    if value_id in seen:
        return "<circular_reference>"
    if isinstance(value, dict):
        seen.add(value_id)
        out: dict[str, Any] = {}
        for key, item in list(value.items())[:50]:
            if key == "payload":
                out[str(key)] = _json_safe_payload_summary(item)
            else:
                out[str(key)] = _json_safe_value(item, seen=seen, depth=depth + 1)
        seen.remove(value_id)
        return out
    if isinstance(value, list | tuple):
        seen.add(value_id)
        out = [_json_safe_value(item, seen=seen, depth=depth + 1) for item in value[:50]]
        seen.remove(value_id)
        return out
    return repr(value)


def _json_safe_payload_summary(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return _json_safe_value(payload, seen=set(), depth=0)
    submit = payload.get("submit_candidate")
    refs = submit.get("candidates") if isinstance(submit, dict) else None
    return {
        "decision": payload.get("decision"),
        "keys": sorted(str(key) for key in payload.keys())[:30],
        "candidate_ref_count": len(refs) if isinstance(refs, list) else 0,
    }


def _submit_candidate_params_from_decision(decision: dict[str, Any]) -> dict[str, Any]:
    mode = current_decision_normalization_mode()
    if mode in {"strict", "off"}:
        from vibeproteinagent.agents.evaluation.decision_validator import validate_decision_v1

        validate_decision_v1(decision)
        return _strict_submit_candidate_params(decision)
    try:
        from vibeproteinagent.agents.evaluation.decision_validator import validate_decision_v1

        validate_decision_v1(decision)
    except DecisionSchemaViolation as exc:
        first = exc.violations[0] if exc.violations else None
        audit_log.warning(
            "audit.decision_schema_violation_in_compat_mode",
            extra={
                "pattern_key": first.issue if first else "unknown",
                "structured_error": exc.structured_error,
            },
        )
    return _legacy_lift_submit_candidate_params(decision)


def _strict_submit_candidate_params(decision: dict[str, Any]) -> dict[str, Any]:
    label = decision.get("decision")
    if label == "submit_candidate":
        params = decision.get("submit_candidate")
        if not isinstance(params, dict):
            raise InvalidParamsError(
                "submit_candidate decision missing submit_candidate payload",
                details={"decision": decision},
            )
        out = dict(params)
        out["outcome"] = "accepted"
        out["reasoning"] = str(decision.get("reasoning") or "").strip()
        return out
    raise InvalidParamsError(
        "decision is not a submit_candidate action",
        details={"decision": decision},
    )


def _legacy_lift_submit_candidate_params(decision: dict[str, Any]) -> dict[str, Any]:
    label = decision.get("decision")
    if label == "submit_candidate":
        params = decision.get("submit_candidate")
        if not isinstance(params, dict):
            raise InvalidParamsError(
                "submit_candidate decision missing submit_candidate payload",
                details={"decision": decision},
            )
        out = dict(params)
        out.setdefault("outcome", "accepted")
        out.setdefault("candidates", [])
        candidates = out.get("candidates")
        if isinstance(candidates, list):
            normalized_candidates: list[Any] = []
            lifted_reasoning: list[str] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    normalized_candidates.append(candidate)
                    continue
                item = dict(candidate)
                reasoning = item.pop("reasoning", None)
                if isinstance(reasoning, str) and reasoning.strip():
                    lifted_reasoning.append(reasoning.strip())
                normalized_candidates.append(_normalize_submit_candidate_ref(item))
            out["candidates"] = normalized_candidates
            if not isinstance(out.get("reasoning"), str) or not out["reasoning"].strip():
                if lifted_reasoning:
                    audit_log.warning(
                        "decision_reasoning_normalization.compat_lift_used",
                        extra={
                            "pattern_key": "decision.reasoning.lifted_from_candidate",
                        },
                    )
                    out["reasoning"] = " ".join(lifted_reasoning)
        if not isinstance(out.get("reasoning"), str) or not out["reasoning"].strip():
            next_round = decision.get("next_round")
            if isinstance(next_round, dict) and isinstance(next_round.get("reason"), str):
                out["reasoning"] = next_round["reason"].strip()
        return out
    raise InvalidParamsError(
        "decision is not a submit_candidate action",
        details={"decision": decision},
    )


_SUBMIT_CANDIDATE_REF_KEYS = {
    "job_id",
    "candidate_id",
    "index",
    "justification",
    "scores_to_highlight",
}


def _normalize_submit_candidate_ref(candidate: dict[str, Any]) -> dict[str, Any]:
    """Keep only the submit_candidate ref contract from LLM-authored rows."""

    return {
        key: value
        for key, value in candidate.items()
        if key in _SUBMIT_CANDIDATE_REF_KEYS and value is not None
    }


def _negative_submission_reason(decision: dict[str, Any]) -> str:
    reasoning = decision.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    no_candidate = decision.get("no_acceptable_candidate")
    if isinstance(no_candidate, dict):
        reason = no_candidate.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    next_round = decision.get("next_round")
    if isinstance(next_round, dict):
        reason = next_round.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    notes = decision.get("notes")
    if isinstance(notes, list):
        text = "; ".join(str(item) for item in notes if str(item).strip())
        if text:
            return text
    return "No acceptable candidate met the stated success criteria."


def _is_candidate_v1(candidate: Any) -> bool:
    return (
        isinstance(candidate, dict)
        and candidate.get("schema_version") == "candidate_v1"
        and isinstance(candidate.get("candidate_id"), str)
        and isinstance(candidate.get("source"), dict)
        and isinstance(candidate["source"].get("tool"), str)
        and isinstance(candidate["source"].get("local_id"), str)
    )


def _next_round_reason(decision: dict[str, Any]) -> str:
    next_round = decision.get("next_round")
    if isinstance(next_round, dict):
        reason = next_round.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
    reasoning = decision.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return "Decision requested another local workflow round."


def _merge_candidate_pool(
    survivor_pool: list[dict[str, Any]],
    current_round_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in [*survivor_pool, *current_round_candidates]:
        if not _is_candidate_v1(candidate):
            continue
        key = str(candidate.get("candidate_id"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(dict(candidate))
    return merged


def _has_empty_candidate_outcome(round_record: dict[str, Any]) -> bool:
    if round_record.get("empty_candidate_events"):
        return True
    for node in round_record.get("skipped_nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("skip_kind") in {
            "skipped_no_candidates",
            "skipped_missing_candidate_capability",
        }:
            return True
    return False


def _summary_declares_no_available_candidates(summary: dict[str, Any]) -> bool:
    no_candidate_statuses = {
        "no_hits",
        "no_candidates",
        "no_available_candidate",
        "filtered_out",
    }
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        output = node.get("output")
        if not isinstance(output, dict):
            continue
        status = output.get("status")
        if status not in no_candidate_statuses:
            continue
        for key in ("candidates", "records", "sequences", "representative_sequences"):
            value = output.get(key)
            if isinstance(value, list) and value:
                return False
        return True
    return False


def _deterministic_empty_next_round_decision(
    *,
    survivor_pool: list[dict[str, Any]],
    current_round_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    del current_round_candidates
    if survivor_pool:
        return {
            "decision": "next_round_with_surviving_candidate",
            "reasoning": (
                "Current round produced no usable candidates; continue deterministically "
                "with the preserved survivor pool."
            ),
            "next_round": {
                "recommended": True,
                "changes": [],
                "reason": "Empty current candidate stream with survivor pool preserved.",
            },
            "metrics_used": [],
            "notes": [],
        }
    return {
        "decision": "next_round_with_no_acceptable_candidate",
        "reasoning": (
            "Current round produced no usable candidates and the survivor pool is empty; "
            "start the next round with explicit empty-pool state."
        ),
        "next_round": {
            "recommended": True,
            "changes": [],
            "reason": "Empty current candidate stream and empty survivor pool.",
        },
        "metrics_used": [],
        "notes": [],
    }


def _accepts_prior_round_context(callable_obj: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return True
    return (
        "prior_round_context" in parameters
        or any(param.kind is inspect.Parameter.VAR_POSITIONAL for param in parameters.values())
        or any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    )


def _agent_run_state(report: dict[str, Any]) -> str:
    state = str(report.get("state") or "unknown")
    if state in {"completed", "planned"}:
        return state
    if state in {
        "failed",
        "partial_failed",
        "warm_failed",
        "daemon_unavailable",
        "timeout",
    }:
        return "failed"
    return state


def _workflow_result_state(result: dict[str, Any]) -> str:
    if result.get("ok") is False:
        code = result.get("code")
        if code == ErrorCode.DAEMON_UNAVAILABLE:
            return "daemon_unavailable"
        if code == ErrorCode.TIMEOUT:
            return "timeout"
        return "failed"
    return str(result.get("state") or "submitted")


def _load_workflow_ref(ref: str):
    try:
        from vibeproteinagent.cli.commands.workflow import _resolve_descriptor

        return load_workflow(_resolve_descriptor(ref))
    except Exception:
        return load_workflow(ref)


def load_plan_or_workflow(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() in {".yaml", ".yml"}:
        wf = load_workflow(p)
        return {
            "mode": "agent_plan",
            "source_kind": "workflow_yaml",
            "workflow_plan": wf.model_dump(mode="json", exclude_none=True),
            "required_tools": sorted({node.tool for node in wf.nodes}),
            "warm_tools": sorted({node.tool for node in wf.nodes}),
            "success_criteria": {},
        }
    import json

    obj = json.loads(p.read_text())
    if not isinstance(obj, dict):
        raise InvalidParamsError(
            "plan file must contain a JSON/YAML object",
            details={"path": str(p)},
        )
    return obj


def _partial_report(
    classification: ModeClassification,
    workflow_ids: list[str],
    summaries: list[dict[str, Any]],
    state: str,
) -> dict[str, Any]:
    return _final_report(
        mode=classification.mode,
        classification=classification,
        state=state,
        job_ids=[jid for item in summaries for jid in item.get("job_ids", [])],
        workflow_ids=workflow_ids,
        result={"workflow_ids": workflow_ids, "state": state},
        summary={"rounds": summaries, "latest": summaries[-1] if summaries else None},
    )


def _job_params(job_spec: dict[str, Any]) -> dict[str, Any]:
    params = job_spec.get("params")
    if isinstance(params, dict):
        return params
    return {}


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _plan_allows_internal_tools(plan: dict[str, Any]) -> bool:
    return plan.get("source_kind") in {
        PLANNER_COMPILED_SOURCE_KIND,
        "workflow_yaml",
        "workflow_template",
    }


def _optional_bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _optional_int(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _planned_job(job_spec: dict[str, Any], policy: ExecutionPolicy) -> dict[str, Any]:
    return {
        "tool": job_spec.get("tool"),
        "version": job_spec.get("version") or "*",
        "params": _job_params(job_spec),
        "validation_required": True,
        "force_new": _optional_bool(job_spec.get("force_new"), False),
        "priority": str(job_spec.get("priority") or policy.priority),
        "max_retries": _optional_int(job_spec.get("max_retries"), policy.max_retries),
    }


def _is_terminal_planner_entry(entry: dict[str, Any]) -> bool:
    details = entry.get("details")
    if not isinstance(details, dict):
        return False
    structured = details.get("structured_error")
    if not isinstance(structured, dict):
        return False
    violations = structured.get("violations")
    if not isinstance(violations, list):
        return False
    terminal_issues = {
        "required_missing_input_blocks_compile",
        "unsatisfied_root_required_input",
    }
    return any(
        isinstance(item, dict) and item.get("issue") in terminal_issues
        for item in violations
    )


def _aggregate_job_state(results: list[Any]) -> str:
    if not results:
        return "completed"
    error_count = sum(
        1 for item in results
        if isinstance(item, dict) and item.get("ok") is False and item.get("code")
    )
    if error_count == len(results):
        return "failed"
    if error_count:
        return "partial_failed"
    states: set[str] = {
        str(item.get("state"))
        for item in results
        if isinstance(item, dict) and isinstance(item.get("state"), str)
    }
    if not states:
        return "completed"
    if states <= {"completed"}:
        return "completed"
    if states & {"running", "queued", "submitted"}:
        return "submitted"
    return sorted(states)[0]


def _validate_direct_job_specs(registry: ToolRegistry, specs: list[dict[str, Any]]) -> None:
    registry.discover()
    for spec in specs:
        tool = str(spec.get("tool") or "")
        descriptor, _ = registry.get(tool, str(spec.get("version") or "*"))
        validate_params(descriptor, _job_params(spec))


def _report_artifacts(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    artifacts: dict[str, Any] = {}
    direct = summary.get("artifacts")
    if isinstance(direct, dict) and direct:
        artifacts["summary"] = direct
    for key in ("latest",):
        nested = _report_artifacts(summary.get(key))
        if nested:
            artifacts[key] = nested
    for key in ("jobs", "workflows", "rounds"):
        items = summary.get(key)
        if isinstance(items, list):
            collected = [
                nested for item in items
                if (nested := _report_artifacts(item))
            ]
            if collected:
                artifacts[key] = collected
    return artifacts

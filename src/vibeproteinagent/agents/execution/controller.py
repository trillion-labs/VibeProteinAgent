from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from vibeproteinagent.adapters import ToolRegistry, validate_params
from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.evaluation import DecisionEngine
from vibeproteinagent.agents.planning import WorkflowPlanner
from vibeproteinagent.agents.provider import SessionLoggingProvider, provider_from_config
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

T = TypeVar("T")


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
        provider = SessionLoggingProvider(
            provider_from_config(self.config),
            self.session_store,
            run_id,
        )
        available_tools = broker.list_tools()
        workflows = broker.list_workflows()
        planner = WorkflowPlanner(self.registry, provider)
        entry_failures: list[dict[str, Any]] = []
        plan = await self._run_llm_entry(
            phase="initial_plan",
            failures=entry_failures,
            factory=lambda attempt: planner.draft_plan(
                request,
                available_tools=available_tools,
                workflows=workflows,
                model=self.config.model_for_role("planner"),
                idempotency_key=f"{run_id}:initial-plan:{attempt}",
            ),
        )
        try:
            planner.validate_invocation(plan, classification.params)
        except VPAError:
            plan = planner.template_plan(request, workflows=workflows)
            planner.validate_invocation(plan, classification.params)
            assumptions = plan.setdefault("assumptions", [])
            if isinstance(assumptions, list):
                assumptions.append(
                    "Provider plan did not match the available invocation inputs; used a validated local workflow template."
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

        for round_index in range(max(1, self.config.max_rounds)):
            outcome = await asyncio.to_thread(
                self._execute_validated_plan,
                run_id,
                broker,
                current_plan,
                classification.params,
                policy,
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
            try:
                decision = await self._run_llm_entry(
                    phase=f"decision_round_{round_index}",
                    failures=entry_failures,
                    factory=lambda attempt, round_index=round_index: DecisionEngine(provider).decide(
                        outcome.summary,
                        success_criteria=current_plan.get("success_criteria") or {},
                        model=self.config.model_for_role("analyst"),
                        idempotency_key=f"{run_id}:decision:{round_index}:{attempt}",
                    ),
                )
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
                            "details": exc.details,
                        }
                    )
            self.session_store.add_decision(run_id, decision, round_index=round_index)
            decisions.append(decision)

            next_round = decision.get("next_round") or {}
            if not next_round.get("recommended") or round_index + 1 >= self.config.max_rounds:
                break
            next_plan = next_round.get("plan") or next_round.get("workflow_plan")
            try:
                if isinstance(next_plan, dict):
                    if "workflow_plan" not in next_plan:
                        next_plan = {
                            "mode": "agent_plan",
                            "workflow_plan": next_plan,
                            "success_criteria": current_plan.get("success_criteria") or {},
                        }
                    current_plan = planner.validate_plan(next_plan)
                    planner.validate_invocation(current_plan, classification.params)
                else:
                    current_plan = await self._run_llm_entry(
                        phase=f"next_round_plan_{round_index}",
                        failures=entry_failures,
                        factory=lambda attempt, round_index=round_index: planner.draft_next_round_plan(
                            previous_plan=current_plan,
                            summary=outcome.summary,
                            decision=decision,
                            available_tools=available_tools,
                            workflows=workflows,
                            model=self.config.model_for_role("planner"),
                            idempotency_key=f"{run_id}:next-round:{round_index}:{attempt}",
                        ),
                    )
                    planner.validate_invocation(current_plan, classification.params)
            except VPAError as exc:
                decision["next_round"] = {
                    "recommended": False,
                    "changes": [],
                    "reason": "next-round planner failed after a completed workflow",
                }
                notes = decision.setdefault("notes", [])
                if isinstance(notes, list):
                    notes.append(
                        {
                            "level": "warning",
                            "message": (
                                "LLM next-round planning failed after workflow completion; "
                                "stopped after the completed round."
                            ),
                            "code": exc.code,
                            "details": exc.details,
                        }
                    )
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
            result={"workflow_ids": workflow_ids, "state": final_state},
            summary={"rounds": summaries, "latest": summary},
            decision=final_decision,
            decisions=decisions,
            plan=plan,
        )

    async def _run_llm_entry(
        self,
        *,
        phase: str,
        failures: list[dict[str, Any]],
        factory: Callable[[int], Awaitable[T]],
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
        for attempt in range(1, remaining + 1):
            try:
                return await factory(attempt)
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

    def _execute_validated_plan(
        self,
        run_id: str,
        broker: AgentToolBroker,
        plan: dict[str, Any],
        params: dict[str, Any],
        policy: ExecutionPolicy,
    ) -> WorkflowExecutionOutcome:
        planner = WorkflowPlanner(self.registry)
        planner.validate_invocation(plan, params)
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
        executor = WorkflowExecutor(self.workspace, self.registry, sup)
        workflow_id = executor.submit(workflow, params)
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
        workflow = planner.compile_workflow(valid)
        supervisor = SupervisorClient()
        sup = supervisor if supervisor.is_running() else None
        executor = WorkflowExecutor(self.workspace, self.registry, sup)
        workflow_id = executor.submit(workflow, params or {})
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
    return plan.get("source_kind") in {"workflow_yaml", "workflow_template"}


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

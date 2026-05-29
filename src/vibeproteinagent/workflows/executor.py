from __future__ import annotations

import contextlib
import time
from typing import Any

from vibeproteinagent.adapters import ToolRegistry, validate_params
from vibeproteinagent.adapters.availability import descriptor_availability
from vibeproteinagent.adapters.descriptor import (
    drop_optional_none_params,
    resolve_input_schema_dict,
)
from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_CONSUMERS,
    CANONICAL_TERM_CONSUMERS,
    annotate_candidate_capabilities,
    candidate_capabilities_from_record,
    candidate_contract_for_inputs,
)
from vibeproteinagent.core.errors import (
    DependencyError,
    ErrorCode,
    InvalidParamsError,
    TimeoutError_,
    VPAError,
)
from vibeproteinagent.workspace import (
    JobState,
    JobStore,
    Workspace,
)
from vibeproteinagent.workspace.store import params_hash
from vibeproteinagent.workspace.types import JobRecord, WorkflowNodeRecord, WorkflowRecord

from .cache import compute_cache_key, env_fingerprint
from .descriptor import Workflow
from .failure import FailurePolicy
from .loader import serialize_workflow
from .parameter_resolution import audit_adapter_defaults, normalize_tool_parameters
from .planner import params_dict_from_workflow as _params_dict_from_workflow
from .planner import resolve_iterable as _resolve_iterable
from .planner import static_upstream as _static_upstream
from .resolver import eval_when, iter_refs, resolve
from .state import FOREACH_CHILD_RE as _FOREACH_CHILD_RE
from .state import SUCCESS_NODE_STATES as _SUCCESS_NODE_STATES
from .state import TERMINAL_NODE_STATES as _TERMINAL_NODE_STATES
from .state import NodeRuntime as _NodeRuntime
from .state import WorkflowStateMachine
from .store import WorkflowStore


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


def _annotate_output_candidate_capabilities(output: Any) -> Any:
    if not isinstance(output, dict):
        return output
    candidates = output.get("candidates")
    if not isinstance(candidates, list):
        return output
    annotated = []
    changed = False
    for candidate in candidates:
        if isinstance(candidate, dict):
            annotated_candidate = annotate_candidate_capabilities(candidate)
            annotated.append(annotated_candidate)
            changed = changed or annotated_candidate != candidate
        else:
            annotated.append(candidate)
    if not changed:
        return output
    return {**output, "candidates": annotated}


def _candidate_list(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    candidates: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        candidates.append(item)
    return candidates


def _output_candidates(output: Any) -> list[dict[str, Any]] | None:
    if not isinstance(output, dict):
        return None
    return _candidate_list(output.get("candidates"))


def _candidate_stream_ref_nodes(value: Any) -> set[str]:
    refs: set[str] = set()
    for ref in iter_refs(value):
        parts = ref.split(".")
        if len(parts) >= 3 and parts[0] == "outputs" and parts[-1] == "candidates":
            refs.add(parts[1])
    return refs


def _first_candidate_input_source(rt: _NodeRuntime) -> str | None:
    refs = sorted(_candidate_stream_ref_nodes(rt.desc.inputs))
    return refs[0] if refs else None


def _node_consumes_candidate_stream(rt: _NodeRuntime) -> bool:
    slot = CANONICAL_CONSUMERS.get(rt.desc.tool)
    if not slot:
        return False
    inputs = rt.desc.inputs if isinstance(rt.desc.inputs, dict) else {}
    return slot in inputs


def _candidate_runtime_gate(
    rt: _NodeRuntime,
    resolved: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    slot = CANONICAL_CONSUMERS.get(rt.desc.tool)
    if not slot or slot not in resolved:
        return None, None
    candidates = _candidate_list(resolved.get(slot))
    if candidates is None:
        return None, None
    if not candidates:
        return {
            "status": "skipped_no_candidates",
            "reason": "required candidate stream is empty",
            "input_field": slot,
            "candidate_count": 0,
            "technical_failure": False,
        }, None
    contract = candidate_contract_for_inputs(rt.desc.tool, resolved)
    if len(candidates) < contract.min_candidate_items:
        return {
            "status": "skipped_missing_candidate_capability",
            "reason": "candidate stream has fewer items than this tool requires",
            "input_field": slot,
            "candidate_count": len(candidates),
            "compatible_candidate_count": 0,
            "skipped_candidate_count": len(candidates),
            "missing_capabilities": [
                f"at_least_{contract.min_candidate_items}_candidate_items"
            ],
            "filtered_candidates": _bounded_filtered_candidates(
                candidates,
                missing_capabilities=[
                    f"at_least_{contract.min_candidate_items}_candidate_items"
                ],
            ),
            "technical_failure": False,
        }, None
    missing_by_candidate: list[dict[str, Any]] = []
    compatible_candidates: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        capabilities = candidate_capabilities_from_record(candidate)
        missing = sorted(contract.consumes - capabilities)
        if missing:
            missing_by_candidate.append(
                {
                    "index": index,
                    "candidate_id": candidate.get("candidate_id"),
                    "source": _candidate_source_summary(candidate),
                    "missing_capabilities": missing,
                    "capabilities": sorted(capabilities),
                }
            )
        else:
            compatible_candidates.append(candidate)
    if missing_by_candidate:
        if len(compatible_candidates) >= contract.min_candidate_items:
            resolved[slot] = compatible_candidates
            return None, {
                "status": "partitioned",
                "reason": (
                    "candidate stream contained incompatible candidates; "
                    "executing tool with compatible candidate subset"
                ),
                "input_field": slot,
                "candidate_count": len(candidates),
                "compatible_candidate_count": len(compatible_candidates),
                "skipped_candidate_count": len(missing_by_candidate),
                "required_capabilities": sorted(contract.consumes),
                "filtered_candidates": missing_by_candidate,
                "technical_failure": False,
            }
        missing_all = sorted(
            {
                capability
                for item in missing_by_candidate
                for capability in item["missing_capabilities"]
            }
        )
        return {
            "status": "skipped_missing_candidate_capability",
            "reason": "candidate stream lacks capabilities required by this tool",
            "input_field": slot,
            "candidate_count": len(candidates),
            "compatible_candidate_count": len(compatible_candidates),
            "skipped_candidate_count": len(missing_by_candidate),
            "required_capabilities": sorted(contract.consumes),
            "missing_capabilities": missing_all,
            "missing_by_candidate": missing_by_candidate,
            "filtered_candidates": missing_by_candidate,
            "technical_failure": False,
        }, None
    return None, None


def _bounded_filtered_candidates(
    candidates: list[dict[str, Any]],
    *,
    missing_capabilities: list[str],
    limit: int = 20,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates[:limit]):
        capabilities = candidate_capabilities_from_record(candidate)
        out.append(
            {
                "index": index,
                "candidate_id": candidate.get("candidate_id"),
                "source": _candidate_source_summary(candidate),
                "missing_capabilities": list(missing_capabilities),
                "capabilities": sorted(capabilities),
            }
        )
    return out


def _candidate_source_summary(candidate: dict[str, Any]) -> Any:
    source = candidate.get("source")
    if isinstance(source, dict):
        return {
            key: source.get(key)
            for key in ("tool", "node_id", "local_id", "native_id", "workflow_id")
            if source.get(key) not in (None, "")
        }
    if isinstance(source, str) and source:
        return source
    return None


def _skip_output(
    rt: _NodeRuntime,
    gate: dict[str, Any],
    *,
    upstream_empty_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status = str(gate.get("status") or "skipped_no_candidates")
    output = {
        "status": status,
        "skipped": True,
        "skip_reason": gate.get("reason") or status,
        "skip_kind": status,
        "consumer_node": rt.runtime_id,
        "consumer_tool": rt.desc.tool,
        "input_field": gate.get("input_field"),
        "candidates": [],
        "technical_failure": False,
        "runtime_gate_event": dict(gate),
    }
    if upstream_empty_event is not None:
        output["upstream_empty_candidate_event"] = upstream_empty_event
    if status == "skipped_missing_candidate_capability":
        output["candidate_capability_filter_event"] = {
            "status": "all_incompatible",
            "reason": output["skip_reason"],
            "input_field": gate.get("input_field"),
            "candidate_count": gate.get("candidate_count", 0),
            "compatible_candidate_count": gate.get("compatible_candidate_count", 0),
            "skipped_candidate_count": gate.get(
                "skipped_candidate_count",
                len(gate.get("filtered_candidates") or []),
            ),
            "required_capabilities": list(gate.get("required_capabilities") or []),
            "missing_capabilities": list(gate.get("missing_capabilities") or []),
            "filtered_candidates": list(gate.get("filtered_candidates") or []),
            "technical_failure": False,
        }
    return output


def _empty_candidate_event(
    rt: _NodeRuntime,
    *,
    upstream_node: str | None = None,
    reason: str = "node produced an empty candidate stream",
) -> dict[str, Any]:
    return {
        "event_type": "empty_candidate_event",
        "node_id": rt.runtime_id,
        "tool": rt.desc.tool,
        "upstream_node": upstream_node,
        "reason": reason,
        "technical_failure": False,
    }


def _unresolved_term_source_error(
    rt: _NodeRuntime,
    resolved: dict[str, Any],
    runtime: dict[str, _NodeRuntime],
) -> InvalidParamsError | None:
    input_field = CANONICAL_TERM_CONSUMERS.get(rt.desc.tool)
    if not input_field or input_field not in resolved:
        return None
    resolved_value = resolved.get(input_field)
    if isinstance(resolved_value, str) and resolved_value.strip():
        return None

    producer_node: str | None = None
    raw_value = rt.desc.inputs.get(input_field) if isinstance(rt.desc.inputs, dict) else None
    for ref in iter_refs(raw_value):
        parts = ref.split(".")
        if len(parts) >= 2 and parts[0] == "outputs":
            producer_node = parts[1]
            break
    if producer_node is None:
        return None

    producer_rt = runtime.get(producer_node)
    producer_output = producer_rt.output if producer_rt is not None else None
    resolver_status = None
    matches: Any = None
    if isinstance(producer_output, dict):
        resolver_status = producer_output.get("status")
        matches = producer_output.get("matches")

    return InvalidParamsError(
        "term_source did not resolve to an executable identifier",
        details={
            "error_kind": "unresolved_term_source",
            "producer_node": producer_node,
            "consumer_node": rt.runtime_id,
            "consumer_tool": rt.desc.tool,
            "input_field": input_field,
            "resolved_value": resolved_value,
            "resolver_status": resolver_status,
            "matches": matches if isinstance(matches, list) else [],
        },
    )


def _empty_term_source_output(
    rt: _NodeRuntime,
    error: InvalidParamsError,
) -> dict[str, Any]:
    return {
        "status": "skipped_unresolved_term_source",
        "skipped": True,
        "skip_reason": error.message,
        "error_details": error.details,
        "sequences": [],
        "candidates": [],
        "kept_sequences": 0,
        "num_sequences_returned": 0,
        "summary": {
            "consumer_tool": rt.desc.tool,
            "reason": "term_source resolved to an empty identifier",
        },
    }


class WorkflowExecutor:
    """Client-side DAG driver that submits each node as a normal JobStore job."""

    def __init__(
        self,
        workspace: Workspace,
        registry: ToolRegistry,
        supervisor: Any | None = None,
        *,
        virtual_outputs: dict[str, Any] | None = None,
    ) -> None:
        self.workspace = workspace
        self.registry = registry
        self.supervisor = supervisor  # SupervisorClient or None (skip-execution mode)
        self.store = WorkflowStore(workspace)
        self.jobs = JobStore(workspace)
        self.virtual_outputs = dict(virtual_outputs or {})

    # ------------------------------------------------------------ submit

    def submit(
        self,
        workflow: Workflow,
        params: dict[str, Any],
        *,
        resume_workflow_id: str | None = None,
    ) -> str:
        """Register a workflow + every node row in DB; return workflow_id.

        If ``resume_workflow_id`` is given, skip insertion and reuse the existing
        workflow + node rows (foreach expansions, however, are deterministic so
        the node set is identical).
        """
        if resume_workflow_id is not None:
            wf_rec = self.store.get(resume_workflow_id)
            return wf_rec.workflow_id

        # Reject missing-upstream / cycle workflows at submit time so the
        # row never lands in the DB (Codex round-3 RC2). load_workflow
        # already does this for YAML descriptors; the Python-builder /
        # MCP submit path comes through here.
        workflow.validate_dag()
        merged = _params_dict_from_workflow(workflow, params)
        wf_id = self.store.create(
            name=workflow.name,
            version=workflow.version,
            descriptor_yaml=serialize_workflow(workflow),
            params=merged,
        )
        # Insert one row per (declared) node. foreach expansion happens at runtime
        # because the iterable may depend on upstream outputs.
        for node in workflow.nodes:
            # Pre-resolve what we can with just params (upstream outputs are unknown
            # at submit time → cache_key recomputed when the node actually runs).
            self.store.add_node(
                workflow_id=wf_id,
                node_id=node.id,
                tool_name=node.tool,
                tool_version=node.version,
                upstream=_static_upstream(node),
                params_hash="",  # filled on first run
                input_hash="",
                cache_key="",
                state="pending",
            )
        return wf_id

    # ------------------------------------------------------------ run

    def run_sync(
        self,
        workflow_id: str,
        *,
        max_wait_s: int = 3600,
        poll_s: float = 0.25,
    ) -> WorkflowRecord:
        """Drive the workflow to completion or failure. Blocks the caller."""
        wf_rec = self.store.get(workflow_id)
        workflow = _parse_descriptor(wf_rec.descriptor)
        params = wf_rec.params

        # Mark running (idempotent on re-runs).
        if wf_rec.state == "submitted":
            self.store.update_state(workflow_id, "running")

        # Hydrate runtime state from existing rows (for resume).
        runtime: dict[str, _NodeRuntime] = {}
        existing_nodes = {n.node_id: n for n in self.store.list_nodes(workflow_id)}
        descriptor_by_id = {node.id: node for node in workflow.nodes}
        for desc in workflow.nodes:
            rt = _NodeRuntime(
                desc=desc,
                runtime_id=desc.id,
                upstream=_static_upstream(desc),
            )
            if desc.id in existing_nodes:
                row = existing_nodes[desc.id]
                self._hydrate_runtime_from_row(rt, row)
            runtime[desc.id] = rt

        for node_id, row in existing_nodes.items():
            if node_id in runtime:
                continue
            match = _FOREACH_CHILD_RE.fullmatch(node_id)
            if match is None:
                continue
            parent_id = match.group("parent")
            parent_desc = descriptor_by_id.get(parent_id)
            if parent_desc is None or not parent_desc.foreach:
                continue
            child_desc = parent_desc.model_copy(update={"foreach": None, "id": node_id})
            rt = _NodeRuntime(
                desc=child_desc,
                runtime_id=node_id,
                upstream=row.upstream,
                foreach_parent=parent_id,
                foreach_index=int(match.group("index")),
            )
            self._hydrate_runtime_from_row(rt, row)
            if rt.state in {"pending", "ready"}:
                try:
                    parent_rt = runtime[parent_id]
                    if not self._upstream_ok(parent_rt, runtime):
                        raise InvalidParamsError(
                            "cannot resume foreach child before parent upstream is ready",
                            details={"node_id": node_id, "parent": parent_id},
                        )
                    scope = self._build_scope(params, runtime, current=parent_rt)
                    items = self._resolve_iterable(parent_desc.foreach, scope)
                    if rt.foreach_index is None or rt.foreach_index >= len(items):
                        raise InvalidParamsError(
                            "foreach child index is outside resolved iterable",
                            details={
                                "node_id": node_id,
                                "parent": parent_id,
                                "index": rt.foreach_index,
                                "items": len(items),
                            },
                        )
                    rt.item = items[rt.foreach_index]
                    rt.state = "pending"
                    rt.job_id = None
                except VPAError as exc:
                    self._fail_node_from_error(workflow_id, rt, exc)
            runtime[node_id] = rt

        env = env_fingerprint()
        deadline = time.time() + float(max_wait_s)
        failed = False
        capability_filter_events: dict[str, dict[str, Any]] = {}
        parameter_resolution_events: dict[str, list[dict[str, Any]]] = {}

        while time.time() < deadline:
            progressed = False

            # 0) Foreach placeholders complete only after all runtime children do.
            # The placeholder then exposes an ordered list of child outputs under
            # outputs.<base_id>, allowing downstream aggregate nodes to depend on
            # the fanout as a normal upstream.
            for rt in list(runtime.values()):
                if not rt.desc.foreach:
                    continue
                if rt.state in {"failed", "skipped"}:
                    continue
                if self._complete_foreach_base_if_ready(workflow_id, rt, runtime):
                    progressed = True

            # 1) Pick up completion / failure of in-flight jobs.
            for rt in list(runtime.values()):
                if rt.state == "running" and rt.job_id:
                    job = self._safe_get_job(rt.job_id)
                    if job is None:
                        continue
                    if job.state is JobState.COMPLETED:
                        rt.state = "completed"
                        rt.output = _annotate_output_candidate_capabilities(job.output)
                        filter_event = capability_filter_events.pop(rt.runtime_id, None)
                        param_events = parameter_resolution_events.pop(rt.runtime_id, None)
                        if filter_event is not None and isinstance(rt.output, dict):
                            rt.output = {
                                **rt.output,
                                "candidate_capability_filter_event": filter_event,
                            }
                        if param_events and isinstance(rt.output, dict):
                            rt.output = {
                                **rt.output,
                                "parameter_resolution_events": param_events,
                            }
                        if _output_candidates(rt.output) == []:
                            event = _empty_candidate_event(rt)
                            rt.output = {
                                **rt.output,
                                "empty_candidate_event": event,
                                "technical_failure": False,
                            }
                        self.store.update_node_state(
                            workflow_id, rt.runtime_id, "completed",
                            job_id=rt.job_id,
                            output=rt.output,
                        )
                        if _output_candidates(rt.output) == []:
                            self._cascade_empty_candidate_skip(
                                workflow_id,
                                rt,
                                runtime,
                                upstream_empty_event=rt.output.get("empty_candidate_event")
                                if isinstance(rt.output, dict)
                                else None,
                            )
                        progressed = True
                    elif job.state in (JobState.FAILED, JobState.CANCELLED):
                        self._fail_node(
                            workflow_id,
                            rt,
                            code=job.error_code or ErrorCode.INTERNAL,
                            message=job.error_message
                            or f"job {job.job_id} ended in state {job.state.value}",
                            details=job.error_details,
                            job_id=rt.job_id,
                        )
                        if self._node_failed(workflow_id, rt, runtime):
                            failed = True
                        progressed = True

            if failed:
                self.store.update_state(workflow_id, "failed", finished=True)
                return self.store.get(workflow_id)

            # 2) Try to advance any pending node whose upstream is ready.
            for rt in list(runtime.values()):
                if rt.state != "pending":
                    continue
                if not self._upstream_ok(rt, runtime):
                    continue

                scope = self._build_scope(params, runtime, current=rt)
                # Evaluate `when` (skip if false).
                if rt.desc.when and not eval_when(rt.desc.when, scope):
                    rt.state = "skipped"
                    self.store.update_node_state(
                        workflow_id, rt.runtime_id, "skipped",
                    )
                    progressed = True
                    continue

                # foreach base nodes never run themselves — they're a placeholder
                # whose `inputs` reference `${item}` (only valid for the expanded
                # children). Expand first, before trying to resolve inputs.
                if rt.desc.foreach:
                    try:
                        items = self._resolve_iterable(rt.desc.foreach, scope)
                    except VPAError as exc:
                        self._fail_node_from_error(workflow_id, rt, exc)
                        if self._node_failed(workflow_id, rt, runtime):
                            failed = True
                        progressed = True
                        continue
                    self._expand_foreach(workflow_id, rt, items, runtime, env)
                    if items:
                        rt.state = "running"
                        self.store.update_node_state(
                            workflow_id, rt.runtime_id, "running",
                        )
                    else:
                        rt.state = "completed"
                        rt.output = []
                        self.store.update_node_state(
                            workflow_id, rt.runtime_id, "completed",
                            output=rt.output,
                        )
                    progressed = True
                    continue

                if not self._foreach_parallel_slot_available(rt, runtime):
                    continue

                # Resolve inputs.
                try:
                    resolved = resolve(rt.desc.inputs, scope)
                except VPAError as exc:
                    self._fail_node_from_error(workflow_id, rt, exc)
                    if self._node_failed(workflow_id, rt, runtime):
                        failed = True
                    progressed = True
                    continue

                if not isinstance(resolved, dict):
                    self._fail_node(
                        workflow_id,
                        rt,
                        code=ErrorCode.INVALID_PARAMS,
                        message=(
                            f"workflow node {rt.runtime_id!r} resolved inputs "
                            f"to {type(resolved).__name__}; expected object"
                        ),
                        details={
                            "node_id": rt.runtime_id,
                            "type": type(resolved).__name__,
                        },
                    )
                    if self._node_failed(workflow_id, rt, runtime):
                        failed = True
                    progressed = True
                    continue

                if error := _unresolved_term_source_error(rt, resolved, runtime):
                    rt.state = "completed"
                    rt.output = _empty_term_source_output(rt, error)
                    rt.output["empty_candidate_event"] = _empty_candidate_event(
                        rt,
                        reason="term resolver produced no executable candidate query",
                    )
                    rt.output["technical_failure"] = False
                    self.store.update_node_state(
                        workflow_id,
                        rt.runtime_id,
                        "completed",
                        output=rt.output,
                    )
                    self._cascade_empty_candidate_skip(
                        workflow_id,
                        rt,
                        runtime,
                        upstream_empty_event=rt.output["empty_candidate_event"],
                    )
                    progressed = True
                    continue

                runtime_gate, capability_filter_event = _candidate_runtime_gate(rt, resolved)
                if runtime_gate is not None:
                    event = _empty_candidate_event(
                        rt,
                        upstream_node=_first_candidate_input_source(rt),
                        reason=str(
                            runtime_gate.get("reason")
                            or "candidate runtime gate skipped node"
                        ),
                    )
                    rt.state = "skipped"
                    rt.output = _skip_output(rt, runtime_gate, upstream_empty_event=event)
                    rt.output["empty_candidate_event"] = event
                    self.store.update_node_state(
                        workflow_id,
                        rt.runtime_id,
                        "skipped",
                        output=rt.output,
                    )
                    self._cascade_empty_candidate_skip(
                        workflow_id,
                        rt,
                        runtime,
                        upstream_empty_event=event,
                    )
                    progressed = True
                    continue
                if capability_filter_event is not None:
                    capability_filter_events[rt.runtime_id] = capability_filter_event

                # Resolve the tool descriptor up-front so the cache key uses
                # the concrete adapter version (not the PEP-440 spec).
                try:
                    descriptor, adapter_cls = self.registry.get(rt.desc.tool, rt.desc.version)
                except VPAError as exc:
                    self._fail_node_from_error(workflow_id, rt, exc)
                    if self._node_failed(workflow_id, rt, runtime):
                        failed = True
                    progressed = True
                    continue
                availability = descriptor_availability(descriptor, adapter_cls)
                try:
                    from vibeproteinagent.install.installer import install_readiness

                    install_status = install_readiness(
                        (str(descriptor.name),),
                        runtime_probe=False,
                    )
                except Exception as exc:
                    install_status = {
                        "install_ready": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                if (
                    availability.get("available") is not True
                    or install_status.get("install_ready") is not True
                ):
                    self._fail_node_from_error(
                        workflow_id,
                        rt,
                        DependencyError(
                            f"tool '{descriptor.name}' is not ready to run",
                            details={
                                "tool": descriptor.name,
                                "version": descriptor.version,
                                "availability": availability,
                                "install_readiness": install_status,
                            },
                        ),
                    )
                    if self._node_failed(workflow_id, rt, runtime):
                        failed = True
                    progressed = True
                    continue

                # Validate resolved inputs against the tool's input_schema.
                # Same JSON-Schema validator the CLI uses, so workflow nodes can't
                # bypass param validation (Architecture v1 §6 + §8).
                try:
                    resolved, resolved_provenance, param_events = normalize_tool_parameters(
                        tool=descriptor.name,
                        inputs=resolved,
                        provenance=rt.desc.input_provenance,
                        original_inputs=rt.desc.inputs,
                    )
                    resolved = drop_optional_none_params(descriptor, resolved)
                    default_events = audit_adapter_defaults(
                        descriptor=descriptor,
                        resolved_inputs=resolved,
                        sources=resolved_provenance,
                    )
                    param_events.extend(default_events)
                    if param_events:
                        parameter_resolution_events[rt.runtime_id] = param_events
                    validate_params(descriptor, resolved)
                    if descriptor.name == "rfdiffusion":
                        mode_status = _rfdiffusion_mode_readiness(resolved)
                        if mode_status.get("ready") is not True:
                            raise DependencyError(
                                "RFdiffusion mode checkpoint is not ready",
                                details={
                                    "tool": descriptor.name,
                                    "version": descriptor.version,
                                    "mode_readiness": mode_status,
                                },
                            )
                except VPAError as exc:
                    self._fail_node_from_error(workflow_id, rt, exc)
                    if self._node_failed(workflow_id, rt, runtime):
                        failed = True
                    progressed = True
                    continue

                input_schema = resolve_input_schema_dict(descriptor)
                cache_key = compute_cache_key(
                    rt.desc, resolved, env,
                    tool_version_resolved=descriptor.version,
                    input_schema=input_schema,
                )
                rt.cache_key = cache_key
                if rt.desc.cache:
                    hit = self.store.find_cached_node(cache_key)
                    if hit and hit.workflow_id != workflow_id:
                        cached_output = None
                        if hit.job_id:
                            try:
                                cached_job = self.jobs.get(hit.job_id)
                                cached_output = _annotate_output_candidate_capabilities(
                                    cached_job.output
                                )
                            except VPAError:
                                cached_output = None
                        rt.state = "cached"
                        rt.output = cached_output
                        param_events = parameter_resolution_events.pop(rt.runtime_id, None)
                        if (
                            capability_filter_event is not None
                            and isinstance(rt.output, dict)
                        ):
                            rt.output["candidate_capability_filter_event"] = (
                                capability_filter_event
                            )
                        if param_events and isinstance(rt.output, dict):
                            rt.output["parameter_resolution_events"] = param_events
                        if _output_candidates(rt.output) == []:
                            event = _empty_candidate_event(rt)
                            base_output = rt.output if isinstance(rt.output, dict) else {}
                            rt.output = {
                                **base_output,
                                "empty_candidate_event": event,
                                "technical_failure": False,
                            }
                        rt.job_id = hit.job_id
                        self.store.update_node_state(
                            workflow_id, rt.runtime_id, "cached",
                            job_id=hit.job_id, cache_key=cache_key,
                            params_hash=params_hash(resolved),
                            input_hash=cache_key,
                            output=rt.output,
                        )
                        if _output_candidates(rt.output) == []:
                            self._cascade_empty_candidate_skip(
                                workflow_id,
                                rt,
                                runtime,
                                upstream_empty_event=rt.output.get("empty_candidate_event")
                                if isinstance(rt.output, dict)
                                else None,
                            )
                        progressed = True
                        continue

                # Submit a real job.
                if self.supervisor is None:
                    rt.state = "pending"
                    continue

                ph = params_hash(resolved)
                self.store.update_node_state(
                    workflow_id, rt.runtime_id, "ready",
                    cache_key=cache_key, input_hash=cache_key, params_hash=ph,
                )

                try:
                    job = self.jobs.submit(
                        descriptor.name,
                        descriptor.version,
                        resolved,
                        priority=rt.desc.priority,
                        workspace_name=self.workspace.name,
                        workflow_id=workflow_id,
                        workflow_node_id=rt.runtime_id,
                        max_retries=int(rt.desc.max_retries),
                        input_schema=input_schema,
                    )
                    if job.state is JobState.SUBMITTED:
                        job = self.jobs.transition(job.job_id, JobState.QUEUED)
                except VPAError as exc:
                    self._fail_node_from_error(workflow_id, rt, exc)
                    if self._node_failed(workflow_id, rt, runtime):
                        failed = True
                    progressed = True
                    continue

                rt.state = "running"
                rt.job_id = job.job_id
                self.store.update_node_state(
                    workflow_id, rt.runtime_id, "running",
                    job_id=job.job_id, cache_key=cache_key,
                )
                progressed = True

            if failed:
                self.store.update_state(workflow_id, "failed", finished=True)
                return self.store.get(workflow_id)

            # 3) Done?
            if all(rt.state in _TERMINAL_NODE_STATES for rt in runtime.values()):
                any_failed = any(rt.state == "failed" for rt in runtime.values())
                final = "failed" if any_failed else "completed"
                self.store.update_state(workflow_id, final, finished=True)
                return self.store.get(workflow_id)

            if not progressed:
                time.sleep(poll_s)

        # Timed out.
        for rt in runtime.values():
            if rt.state not in _TERMINAL_NODE_STATES:
                self._fail_node(
                    workflow_id,
                    rt,
                    code=ErrorCode.TIMEOUT,
                    message=f"workflow {workflow_id} timed out waiting for node {rt.runtime_id}",
                    details={
                        "workflow_id": workflow_id,
                        "node_id": rt.runtime_id,
                        "max_wait_s": max_wait_s,
                    },
                    job_id=rt.job_id,
                )
                if rt.job_id:
                    with contextlib.suppress(VPAError):
                        self.jobs.request_cancel(rt.job_id)
        self.store.update_state(workflow_id, "failed", finished=True)
        raise TimeoutError_(
            f"workflow {workflow_id} did not finish within {max_wait_s}s",
            details={"workflow_id": workflow_id, "max_wait_s": max_wait_s},
        )

    def _hydrate_runtime_from_row(
        self,
        rt: _NodeRuntime,
        row: WorkflowNodeRecord,
    ) -> None:
        rt.state = row.state
        rt.job_id = row.job_id
        rt.cache_key = row.cache_key or None
        if row.state == "failed":
            rt.state = "failed"
            return
        if row.output is not None:
            rt.output = row.output
        if row.state in _SUCCESS_NODE_STATES and row.job_id:
            try:
                job = self.jobs.get(row.job_id)
                rt.output = job.output
            except VPAError:
                rt.state = "pending"
                rt.job_id = None
        if row.state == "ready":
            rt.state = "pending"
            rt.job_id = None

    # ------------------------------------------------------------ failure handling

    def _fail_node_from_error(
        self,
        workflow_id: str,
        rt: _NodeRuntime,
        exc: VPAError,
    ) -> None:
        self._fail_node(
            workflow_id,
            rt,
            code=exc.code,
            message=exc.message,
            details=exc.details or None,
        )

    def _fail_node(
        self,
        workflow_id: str,
        rt: _NodeRuntime,
        *,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> None:
        rt.state = "failed"
        if job_id is not None:
            rt.job_id = job_id
        self.store.update_node_state(
            workflow_id,
            rt.runtime_id,
            "failed",
            job_id=rt.job_id,
            error_code=code,
            error_message=message,
            error_details=details,
        )

    def _node_failed(
        self,
        workflow_id: str,
        failed_rt: _NodeRuntime,
        runtime: dict[str, _NodeRuntime],
    ) -> bool:
        """Apply the failed node's ``on_failure`` policy to its descendants.

        Returns ``True`` when the workflow should halt immediately (the ``stop``
        policy). For ``continue`` / ``skip`` the workflow keeps running on its
        independent branches and we return ``False``.

        Codex round-2 RC7: ``continue`` and ``skip`` both cascade to the full
        transitive descendant set. Phase 7's ``continue`` only marked direct
        dependents skipped, leaving second-order descendants (a node Y that
        depends on a node Z that depends on the failed X) ``pending`` forever.
        Both policies now share one cascade — they only differ from ``stop`` in
        that independent branches keep running.
        """
        policy = failed_rt.desc.on_failure
        if policy == "stop":
            # Halt: every non-terminal sibling is removed from the workflow and
            # running jobs are asked to cancel so status cannot remain "running"
            # after the workflow has failed.
            for rt in runtime.values():
                if rt.runtime_id == failed_rt.runtime_id:
                    continue
                if rt.state in _TERMINAL_NODE_STATES:
                    continue
                if rt.job_id:
                    with contextlib.suppress(VPAError):
                        self.jobs.request_cancel(rt.job_id)
                rt.state = "skipped"
                self.store.update_node_state(
                    workflow_id, rt.runtime_id, "skipped", job_id=rt.job_id,
                )
            return True

        if policy in ("continue", "skip"):
            # Transitive descendant set: every node that (directly or
            # indirectly) depends on the failed node. Computed once via
            # iterative forward traversal over ``upstream`` references.
            self._cascade_skip(workflow_id, failed_rt, runtime)
            return False

        # Unknown policy — defensive: treat as stop.
        return True

    def _cascade_skip(
        self,
        workflow_id: str,
        failed_rt: _NodeRuntime,
        runtime: dict[str, _NodeRuntime],
    ) -> None:
        """Mark every transitive descendant of ``failed_rt`` as skipped (RC7)."""
        for rid in FailurePolicy.descendant_runtime_ids(failed_rt, runtime):
            rt = runtime[rid]
            if rt.state == "pending":
                rt.state = "skipped"
                self.store.update_node_state(
                    workflow_id, rt.runtime_id, "skipped",
                )

    def _cascade_empty_candidate_skip(
        self,
        workflow_id: str,
        empty_rt: _NodeRuntime,
        runtime: dict[str, _NodeRuntime],
        *,
        upstream_empty_event: dict[str, Any] | None,
    ) -> None:
        """Skip only candidate consumers that require an empty candidate stream."""
        queue = [empty_rt.runtime_id]
        seen: set[str] = set()
        skipped_nodes: list[dict[str, Any]] = []
        while queue:
            source_id = queue.pop(0)
            if source_id in seen:
                continue
            seen.add(source_id)
            for rt in runtime.values():
                if rt.runtime_id == source_id or rt.state in _TERMINAL_NODE_STATES:
                    continue
                candidate_ref_nodes = _candidate_stream_ref_nodes(rt.desc.inputs)
                consumes_source_candidate_stream = source_id in candidate_ref_nodes
                if not consumes_source_candidate_stream:
                    continue
                if consumes_source_candidate_stream and not _node_consumes_candidate_stream(rt):
                    continue
                gate = {
                    "status": "skipped_no_candidates",
                    "reason": "upstream candidate stream is empty",
                    "input_field": CANONICAL_CONSUMERS.get(rt.desc.tool),
                    "upstream_node": source_id,
                    "candidate_count": 0,
                }
                rt.state = "skipped"
                rt.output = _skip_output(
                    rt,
                    gate,
                    upstream_empty_event=upstream_empty_event,
                )
                self.store.update_node_state(
                    workflow_id,
                    rt.runtime_id,
                    "skipped",
                    output=rt.output,
                )
                skipped_nodes.append({"node_id": rt.runtime_id, "tool": rt.desc.tool})
                queue.append(rt.runtime_id)
        if skipped_nodes and isinstance(empty_rt.output, dict):
            empty_rt.output = {
                **empty_rt.output,
                "skipped_no_candidates": skipped_nodes,
            }
            self.store.update_node_state(
                workflow_id,
                empty_rt.runtime_id,
                empty_rt.state,
                output=empty_rt.output,
            )

    def _complete_foreach_base_if_ready(
        self,
        workflow_id: str,
        base: _NodeRuntime,
        runtime: dict[str, _NodeRuntime],
    ) -> bool:
        children = [
            rt for rt in runtime.values() if rt.foreach_parent == base.desc.id
        ]
        if not children:
            return False
        if any(child.state == "failed" for child in children):
            self._fail_node(
                workflow_id,
                base,
                code=ErrorCode.INTERNAL,
                message=f"foreach node {base.runtime_id!r} has failed children",
                details={
                    "node_id": base.runtime_id,
                    "failed_children": [
                        child.runtime_id for child in children if child.state == "failed"
                    ],
                },
            )
            return True
        if not all(child.state in _SUCCESS_NODE_STATES for child in children):
            return False
        ordered = sorted(children, key=lambda child: child.foreach_index or 0)
        base.output = [child.output for child in ordered]
        if base.state != "completed":
            base.state = "completed"
            self.store.update_node_state(
                workflow_id, base.runtime_id, "completed", output=base.output
            )
            return True
        return False

    # ------------------------------------------------------------ helpers

    def _safe_get_job(self, job_id: str) -> JobRecord | None:
        try:
            return self.jobs.get(job_id)
        except VPAError:
            return None

    @staticmethod
    def _upstream_ok(rt: _NodeRuntime, runtime: dict[str, _NodeRuntime]) -> bool:
        return WorkflowStateMachine.upstream_ok(rt, runtime)

    def _build_scope(
        self,
        params: dict[str, Any],
        runtime: dict[str, _NodeRuntime],
        *,
        current: _NodeRuntime | None = None,
    ) -> dict[str, Any]:
        scope = WorkflowStateMachine.build_scope(params, runtime, current=current)
        if self.virtual_outputs:
            scope["outputs"] = {**self.virtual_outputs, **scope.get("outputs", {})}
        return scope

    def _resolve_iterable(self, expr: str, scope: dict[str, Any]) -> list[Any]:
        return _resolve_iterable(expr, scope)

    def _expand_foreach(
        self,
        workflow_id: str,
        base: _NodeRuntime,
        items: list[Any],
        runtime: dict[str, _NodeRuntime],
        env: dict[str, Any],
    ) -> None:
        """Create one runtime child per item. Each runs as a normal node next tick."""
        for i, item in enumerate(items):
            cid = f"{base.desc.id}[{i}]"
            if cid in runtime:
                continue
            child_desc = base.desc.model_copy(update={"foreach": None, "id": cid})
            rt = _NodeRuntime(
                desc=child_desc,
                runtime_id=cid,
                upstream=list(base.upstream),
                item=item,
                foreach_parent=base.desc.id,
                foreach_index=i,
            )
            runtime[cid] = rt
            self.store.add_node(
                workflow_id=workflow_id,
                node_id=cid,
                tool_name=child_desc.tool,
                tool_version=child_desc.version,
                upstream=list(base.upstream),
                params_hash="",
                input_hash="",
                cache_key="",
                state="pending",
            )

    @staticmethod
    def _foreach_parallel_slot_available(
        rt: _NodeRuntime,
        runtime: dict[str, _NodeRuntime],
    ) -> bool:
        if rt.foreach_parent is None or rt.desc.max_parallel is None:
            return True
        active = 0
        for sibling in runtime.values():
            if sibling.runtime_id == rt.runtime_id:
                continue
            if sibling.foreach_parent != rt.foreach_parent:
                continue
            if sibling.state in {"ready", "running"}:
                active += 1
        return active < rt.desc.max_parallel


# ---------------------------------------------------------------- helpers


def _parse_descriptor(blob: str) -> Workflow:
    """Re-parse the YAML stored alongside the workflow row."""
    import yaml

    raw = yaml.safe_load(blob)
    return Workflow.model_validate(raw)


__all__ = ["WorkflowExecutor", "WorkflowNodeRecord", "WorkflowRecord"]

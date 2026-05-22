from __future__ import annotations

import asyncio
import copy
import time
from typing import TYPE_CHECKING, Any

from pydantic import create_model

from vibeproteinagent.adapters import ToolDescriptor
from vibeproteinagent.adapters.visibility import is_agent_visible_tool
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import TERMINAL_STATES

from .tools import SessionCtx, get_results, submit_job, wait_job
from .util import resolve_input_schema, validate_against, vpa_error_to_dict

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def _augment_schema_with_runtime_controls(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` advertising optional VPA runtime controls.

    The descriptor's schema typically sets ``additionalProperties: false`` and
    would reject the orchestration-level ``force_new`` keyword. We deep-copy
    and add it as a declared optional boolean so MCP clients see the knob
    and validation still passes when callers send it.
    """
    augmented = copy.deepcopy(schema) if isinstance(schema, dict) else {}
    props = augmented.setdefault("properties", {})
    props["force_new"] = {
        "type": "boolean",
        "default": False,
        "description": (
            "Force a fresh job even if an identical submission already exists "
            "(parity with CLI --force-new). Defaults to False."
        ),
    }
    props["max_retries"] = {
        "type": "integer",
        "minimum": 0,
        "default": 0,
        "description": "Maximum automatic retries on retryable runtime errors.",
    }
    return augmented


def _wait_timeout_s_for(descriptor: ToolDescriptor) -> float:
    """Sync-wait budget for an ergonomic per-tool MCP tool.

    Heavy tools (pin_warm or startup_cost=high) get 600s; everything else 300s.
    Agents who want non-blocking semantics use submit_job + wait_job by hand.
    """
    pin_warm = bool(getattr(descriptor, "pin_warm", False))
    cost = getattr(descriptor, "startup_cost", None)
    cost_v = getattr(cost, "value", cost)
    if pin_warm or cost_v == "high":
        return 600.0
    return 300.0


def _make_passthrough_tool(
    *,
    name: str,
    description: str,
    parameters_schema: dict[str, Any],
    runner,  # async (dict) -> Any
):
    """Construct a FastMCP Tool whose advertised schema is the descriptor's.

    FastMCP's @app.tool() derives the schema from the function signature, which would
    flatten the descriptor schema to `{params: object}`. We instead build a Tool with:
    - `parameters` = descriptor's `input_schema` (what tools/list will advertise).
    - `fn` = an async callable that receives the raw arguments dict.
    - `run` = bypassed: a PassthroughTool subclass calls `fn(arguments)` directly,
      skipping the arg_model validation that would otherwise reject unknown fields.
    """
    from mcp.server.fastmcp.tools.base import Tool
    from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata

    class PassthroughTool(Tool):
        async def run(  # type: ignore[override]
            self,
            arguments: dict[str, Any],
            context: Any = None,
            convert_result: bool = False,
        ) -> Any:
            result = await self.fn(arguments or {})
            if convert_result:
                result = self.fn_metadata.convert_result(result)
            return result

    arg_model = create_model("PassthroughArgs", __base__=ArgModelBase)
    fn_meta = FuncMetadata(arg_model=arg_model)
    return PassthroughTool(
        fn=runner,
        name=name,
        description=description,
        parameters=parameters_schema,
        fn_metadata=fn_meta,
        is_async=True,
    )


def register_dynamic_tools(app: FastMCP, ctx: SessionCtx) -> list[str]:
    """Register one MCP tool per discovered adapter.

    Names: `run_<descriptor.name>`. Each tool:
      1. Submits a job with the agent's params (validated by submit_job against the schema).
      2. Synchronously waits up to a tool-specific timeout for the job to terminate.
      3. Returns `{job_id, state, output, ...}` on success, or `{ok: false, ...}` on failure.
    """
    ctx.registry.discover()
    descriptors = ctx.registry.list()
    registered: list[str] = []
    for descriptor in descriptors:
        if not is_agent_visible_tool(descriptor):
            continue
        # Resolve here so per-call dispatch is fast; if it fails, skip the tool.
        try:
            schema = resolve_input_schema(descriptor)
        except Exception:  # pragma: no cover — schema missing, skip
            continue
        tool_name = f"run_{descriptor.name}"
        wait_s = _wait_timeout_s_for(descriptor)
        desc = descriptor.description or f"Run {descriptor.name} (version {descriptor.version})."
        full_desc = (
            f"{desc}\n\n"
            f"Submits a `{descriptor.name}` job and waits up to {wait_s:.0f}s for it "
            "to terminate. Returns the completed job record (with output) or a domain-error "
            "dict on failure."
        )
        runner = _make_runner(ctx, descriptor, wait_s)
        tool = _make_passthrough_tool(
            name=tool_name,
            description=full_desc,
            parameters_schema=_augment_schema_with_runtime_controls(schema),
            runner=runner,
        )
        # Inject directly into the tool manager (bypasses add_tool's schema derivation).
        app._tool_manager._tools[tool_name] = tool
        registered.append(tool_name)
    return registered


def _make_runner(ctx: SessionCtx, descriptor: ToolDescriptor, wait_s: float):
    async def _run(arguments: dict[str, Any]) -> dict[str, Any]:
        # Runtime controls are orchestration-level knobs; strip them before
        # validating params against the descriptor's input schema.
        args = dict(arguments or {})
        schema = _augment_schema_with_runtime_controls(resolve_input_schema(descriptor))
        try:
            validate_against(schema, args)
        except InvalidParamsError as exc:
            return vpa_error_to_dict(exc)
        force_new_raw = args.pop("force_new", False)
        if not isinstance(force_new_raw, bool):
            return vpa_error_to_dict(
                InvalidParamsError(
                    "force_new must be a boolean",
                    details={"field": "force_new", "got_type": type(force_new_raw).__name__},
                )
            )
        force_new = force_new_raw
        max_retries_raw = args.pop("max_retries", 0)
        if (
            not isinstance(max_retries_raw, int)
            or isinstance(max_retries_raw, bool)
            or max_retries_raw < 0
        ):
            return vpa_error_to_dict(
                InvalidParamsError(
                    "max_retries must be a non-negative integer",
                    details={
                        "field": "max_retries",
                        "got_type": type(max_retries_raw).__name__,
                    },
                )
            )
        submission = submit_job(
            ctx,
            tool=descriptor.name,
            params=args,
            version=f"=={descriptor.version}",
            force_new=force_new,
            max_retries=max_retries_raw,
        )
        if not submission.get("job_id"):
            return submission  # error envelope
        job_id = submission["job_id"]
        idempotent_hit = bool(submission.get("idempotent_hit", False))

        # Cheap poll loop with the existing wait_job helper, then load results.
        deadline = time.time() + wait_s
        store = ctx.store()
        while True:
            rec = store.get(job_id)
            if rec.state in TERMINAL_STATES:
                break
            if time.time() >= deadline:
                return {
                    "ok": False,
                    "code": "E_TIMEOUT",
                    "job_id": job_id,
                    "state": rec.state.value,
                    "idempotent_hit": idempotent_hit,
                    "message": f"timed out waiting {wait_s:.0f}s for job {job_id}",
                }
            await asyncio.sleep(0.5)

        final = wait_job(ctx, job_id, timeout_ms=0)
        if final.get("state") == "completed":
            results = get_results(ctx, job_id)
            return {
                "ok": True,
                "job_id": job_id,
                "state": "completed",
                "idempotent_hit": idempotent_hit,
                "output": results.get("output"),
            }
        return {
            "ok": False,
            "job_id": job_id,
            "state": final.get("state"),
            "idempotent_hit": idempotent_hit,
            "code": final.get("error_code"),
            "message": final.get("error_message"),
            "details": final.get("error_details"),
        }

    return _run


__all__ = ["register_dynamic_tools"]

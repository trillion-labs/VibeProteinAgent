from __future__ import annotations

from typing import Any, cast

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.adapters.visibility import is_agent_visible_tool
from vibeproteinagent.agents.session.store import AgentSessionStore
from vibeproteinagent.mcp import tools as mcp_tools
from vibeproteinagent.mcp.tools import SessionCtx
from vibeproteinagent.workspace import Workspace


class AgentToolBroker:
    """Agent-facing facade over VPA's existing MCP/Python execution substrate."""

    def __init__(
        self,
        workspace: Workspace,
        registry: ToolRegistry,
        session_store: AgentSessionStore | None = None,
        run_id: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.registry = registry
        self.session_store = session_store
        self.run_id = run_id
        self.ctx = SessionCtx(workspace=workspace, registry=registry)

    def list_tools(self, category: str | None = None) -> list[dict[str, Any]]:
        raw = mcp_tools.list_tools(self.ctx, category, include_internal=False)
        visible_names = {
            descriptor.name
            for descriptor in self.registry.list(category)
            if is_agent_visible_tool(descriptor)
        }
        tools = [tool for tool in raw if tool.get("name") in visible_names]
        return cast(list[dict[str, Any]], self._log(
            "list_tools",
            {"category": category},
            tools,
        ))

    def tool_info(self, name: str, version: str = "*") -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "tool_info",
            {"name": name, "version": version},
            mcp_tools.tool_info(self.ctx, name, version),
        ))

    def submit_job(
        self,
        tool: str,
        params: dict[str, Any],
        *,
        version: str = "*",
        call_id: str | None = None,
        priority: str = "normal",
        workflow_id: str | None = None,
        force_new: bool = False,
        max_retries: int = 0,
    ) -> dict[str, Any]:
        result = mcp_tools.submit_job(
            self.ctx,
            tool,
            params,
            version,
            call_id,
            priority,
            workflow_id,
            force_new=force_new,
            max_retries=max_retries,
        )
        return cast(dict[str, Any], self._log("submit_job", locals_without_self(locals()), result))

    def wait_job(self, job_id: str, timeout_ms: int = 30_000) -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "wait_job", {"job_id": job_id, "timeout_ms": timeout_ms},
            mcp_tools.wait_job(self.ctx, job_id, timeout_ms),
        ))

    def get_status(self, job_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "get_status", {"job_id": job_id}, mcp_tools.get_status(self.ctx, job_id)
        ))

    def get_results(self, job_id: str, fields: list[str] | None = None) -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "get_results", {"job_id": job_id, "fields": fields},
            mcp_tools.get_results(self.ctx, job_id, fields),
        ))

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], self._log(
            "list_artifacts", {"job_id": job_id}, mcp_tools.list_artifacts(self.ctx, job_id)
        ))

    def warm_tool(
        self,
        tool: str,
        version: str = "*",
        poll_timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "warm_tool",
            {"tool": tool, "version": version, "poll_timeout_s": poll_timeout_s},
            mcp_tools.warm_tool(self.ctx, tool, version, poll_timeout_s=poll_timeout_s),
        ))

    def list_workflows(self) -> list[dict[str, Any]]:
        return cast(
            list[dict[str, Any]],
            self._log("list_workflows", {}, mcp_tools.list_workflows(self.ctx)),
        )

    def run_workflow(
        self,
        name: str,
        params: dict[str, Any],
        *,
        max_wait_s: int = 3600,
    ) -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "run_workflow",
            {"name": name, "params": params, "max_wait_s": max_wait_s},
            mcp_tools.run_workflow(self.ctx, name, params, max_wait_s=max_wait_s),
        ))

    def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._log(
            "get_workflow_status", {"workflow_id": workflow_id},
            mcp_tools.get_workflow_status(self.ctx, workflow_id),
        ))

    def daemon_status(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            self._log("daemon_status", {}, mcp_tools.daemon_status(self.ctx)),
        )

    def _log(self, tool_name: str, args: dict[str, Any], result: Any) -> Any:
        if self.session_store is not None and self.run_id is not None:
            ok = not (
                isinstance(result, dict)
                and result.get("ok") is False
                and result.get("code")
            )
            self.session_store.add_tool_call(self.run_id, tool_name, args, result, ok=ok)
        return result


def locals_without_self(values: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in values.items() if k not in {"self", "result"}}

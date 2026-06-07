from __future__ import annotations

import typer

from vibeproteinagent.adapters import ToolRegistry, serialize_descriptor
from vibeproteinagent.adapters.availability import attach_availability
from vibeproteinagent.adapters.descriptor import (
    resolve_input_schema_dict,
    resolve_output_schema_dict,
)
from vibeproteinagent.adapters.schema_affordance import conditional_required_summary
from vibeproteinagent.adapters.visibility import is_public_tool
from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_ok,
    protect_json_stdout,
)
from vibeproteinagent.cli.render import render_tool_info, render_tool_list

tools_app = typer.Typer(help="List registered tools.")
tool_app = typer.Typer(no_args_is_help=True, help="Single-tool introspection.")


@tools_app.callback(invoke_without_command=True)
def tools(
    ctx: typer.Context,
    category: str | None = typer.Option(None, "--category", "-c", help="Filter by category."),
    include_internal: bool = typer.Option(
        False,
        "--include-internal",
        help="Include developer-only runtime/glue adapters in the listing.",
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Re-scan filesystem + entry points."),
    schema_only: bool = typer.Option(
        False,
        "--schema-only",
        help="Return resolved schemas without live availability probes.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List registered tools."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        with protect_json_stdout(json_):
            registry = ToolRegistry()
            registry.discover(refresh=refresh)
            descriptors = [
                d
                for d in registry.list(category=category)
                if include_internal or is_public_tool(d)
            ]
            tools_payload = [_tool_payload(d, schema_only=schema_only) for d in descriptors]
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {
            "category": category,
            "include_internal": include_internal,
            "schema_only": schema_only,
            "tools": tools_payload,
        },
        json_mode=json_,
        render=render_tool_list,
    )


@tool_app.command("info")
def tool_info(
    name: str = typer.Argument(..., help="Tool name."),
    version: str | None = typer.Option(
        None, "--version", help="PEP-440 version specifier (e.g. '>=1.0.0,<2')."
    ),
    schema_only: bool = typer.Option(
        False,
        "--schema-only",
        help="Return resolved schemas without live availability probes.",
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show a tool descriptor."""
    try:
        with protect_json_stdout(json_):
            registry = ToolRegistry()
            registry.discover()
            descriptor, _ = registry.get(name, version or "*")
            tool_payload = _tool_payload(descriptor, schema_only=schema_only)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"tool": tool_payload},
        json_mode=json_,
        render=render_tool_info,
    )


def _tool_payload(descriptor, *, schema_only: bool) -> dict:
    payload = serialize_descriptor(descriptor)
    if schema_only:
        input_schema = resolve_input_schema_dict(descriptor)
        payload["input_schema_resolved"] = input_schema
        payload["conditional_required"] = conditional_required_summary(input_schema)
        payload["output_schema_resolved"] = resolve_output_schema_dict(descriptor)
        return payload
    return attach_availability(payload, descriptor)


__all__ = ["tool_app", "tools_app"]

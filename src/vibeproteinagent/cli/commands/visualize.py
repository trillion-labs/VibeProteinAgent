from __future__ import annotations

import json
import webbrowser
from pathlib import Path
from typing import Any

import typer

from vibeproteinagent.cli.output import fail_from_exception, print_envelope_ok
from vibeproteinagent.visualization import (
    build_protein_viewer,
    resolve_job_structures,
    structure_entry_from_path,
)
from vibeproteinagent.workspace import Workspace


def visualize(
    structure: str | None = typer.Argument(
        None,
        help="PDB/mmCIF file to visualize. Omit when using --job-id.",
    ),
    job_id: str | None = typer.Option(
        None,
        "--job-id",
        help="Build a viewer from PDB/CIF artifacts produced by a job.",
    ),
    field: str | None = typer.Option(
        None,
        "--field",
        help="For --job-id, select one artifact/output field.",
    ),
    summary_json: str | None = typer.Option(
        None,
        "--summary-json",
        help="Optional JSON summary to show in the side panel.",
    ),
    out: str | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output HTML path. Defaults to ./<structure>_viewer.html or ./<job_id>_viewer.html.",
    ),
    title: str | None = typer.Option(None, "--title", help="Viewer title."),
    viewer: str = typer.Option(
        "online",
        "--viewer",
        help="Viewer implementation: online | offline.",
    ),
    open_: bool = typer.Option(False, "--open", help="Open the generated HTML in a browser."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a standalone browser-based protein structure viewer."""
    try:
        if bool(structure) == bool(job_id):
            from vibeproteinagent.core.errors import InvalidParamsError

            raise InvalidParamsError(
                "provide exactly one of STRUCTURE or --job-id",
                details={"structure": structure, "job_id": job_id},
            )
        summary: dict[str, Any] = {}
        if summary_json:
            summary = _load_summary(summary_json)

        if job_id:
            ws = Workspace.resolve(workspace)
            structures, job_summary = resolve_job_structures(ws, job_id, field=field)
            if summary:
                job_summary = {**job_summary, "provided_summary": summary}
            summary = job_summary
            default_name = f"{job_id}_viewer.html"
            resolved_title = title or f"VPA job {job_id}"
            source = {"kind": "job", "job_id": job_id, "workspace": ws.name}
        else:
            assert structure is not None
            entry = structure_entry_from_path(structure)
            structures = [entry]
            default_name = f"{entry.path.stem}_viewer.html"
            resolved_title = title or entry.path.name
            source = {"kind": "path", "path": str(entry.path)}

        out_path = Path(out) if out else Path.cwd() / default_name
        html_path = build_protein_viewer(
            structures,
            out_path,
            title=resolved_title,
            summary=summary,
            viewer_mode=viewer,
        )
        if open_:
            webbrowser.open(html_path.as_uri())

        print_envelope_ok(
            {
                "viewer_html": str(html_path),
                "structure_count": len(structures),
                "structures": [
                    {
                        "label": item.label,
                        "path": str(item.path),
                        "format": item.format,
                        "metadata": item.metadata,
                        "source": item.source,
                    }
                    for item in structures
                ],
                "source": source,
                "viewer_mode": viewer,
                "opened": open_,
            },
            json_mode=json_,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


def _load_summary(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    data = json.loads(p.read_text())
    return data if isinstance(data, dict) else {"summary": data}


__all__ = ["visualize"]

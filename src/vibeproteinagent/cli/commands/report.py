from __future__ import annotations

import webbrowser
from pathlib import Path

import typer

from vibeproteinagent.cli.output import fail_from_exception, print_envelope_ok
from vibeproteinagent.reporting import collect_report_model, write_report
from vibeproteinagent.workspace import Workspace


def report(
    id: str = typer.Argument(..., help="Job id (job_*) or workflow id (wf_*) to report."),
    out: str | None = typer.Option(None, "--out", "-o", help="Output HTML path or bundle dir."),
    top_k: int = typer.Option(5, "--top-k", min=1, max=100, help="Top candidates to show."),
    viewer: str = typer.Option(
        "offline",
        "--viewer",
        help="Viewer implementation: offline | online.",
    ),
    format_: str = typer.Option(
        "html",
        "--format",
        help="Report format: html | bundle.",
    ),
    share: bool = typer.Option(False, "--share", help="Prefer self-contained inline assets."),
    open_: bool = typer.Option(False, "--open", help="Open the generated report in a browser."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a human-readable HTML report for a job or workflow."""
    try:
        ws = Workspace.resolve(workspace)
        model = collect_report_model(ws, id)
        default_name = f"{id}_report_bundle" if format_ == "bundle" else f"{id}_report.html"
        default_out = ws.root / "reports" / default_name
        html_path = write_report(
            model,
            Path(out) if out else default_out,
            viewer_mode=viewer,
            report_format=format_,
            share=share,
            top_k=top_k,
        )
        if open_:
            webbrowser.open(html_path.as_uri())
        data = {
            "report_html": str(html_path),
            "format": format_,
            "source": {"kind": model.summary.kind, "id": model.summary.id},
            "candidate_count": len(model.candidates),
            "top_k": top_k,
            "warnings": [
                {"level": w.level, "message": w.message, "source": w.source}
                for w in model.warnings
            ],
            "opened": open_,
        }
        if not json_:
            data["lines"] = [
                f"Report written: {html_path}",
                f"Source: {model.summary.kind} {model.summary.id} ({model.summary.state})",
                f"Candidates: {len(model.candidates)}",
            ]
        print_envelope_ok(data, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


__all__ = ["report"]

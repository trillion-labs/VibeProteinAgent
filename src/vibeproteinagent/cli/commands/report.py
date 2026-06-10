from __future__ import annotations

import webbrowser
from pathlib import Path

import typer

from vibeproteinagent.cli.output import fail_from_exception, print_envelope_ok
from vibeproteinagent.reporting import (
    collect_report_model,
    resolve_report_output_path,
    write_report,
)
from vibeproteinagent.reporting.enrich import ensure_sequence_analysis_enrichment
from vibeproteinagent.workspace import TrajectoryStore, Workspace


def report(
    id: str = typer.Argument(
        ...,
        help="Trajectory id (traj_*), workflow id (wf_*), or job id (job_*) to report.",
    ),
    out: str | None = typer.Option(None, "--out", "-o", help="Output HTML path or bundle dir."),
    top_k: int = typer.Option(
        20,
        "--top-k",
        min=1,
        max=100,
        help=(
            "Maximum displayed candidates in candidate-heavy report sections. "
            "Final submitted candidates are always included."
        ),
    ),
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
    replace_report: bool = typer.Option(
        False,
        "--replace-report",
        help="Replace an existing trajectory final report record.",
    ),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Generate a human-readable report for a trajectory, workflow, or job id."""
    try:
        ws = Workspace.resolve(workspace)
        model = collect_report_model(ws, id)
        enrichment_job_ids = ensure_sequence_analysis_enrichment(ws, model.summary.id, model)
        if enrichment_job_ids:
            model = collect_report_model(ws, id)
        default_name = f"{id}_report_bundle" if format_ == "bundle" else f"{id}_report.html"
        default_out = ws.root / default_name
        target_out = Path(out) if out else default_out
        canonical_report_path = resolve_report_output_path(target_out, report_format=format_)
        trajectory_store = TrajectoryStore(ws) if id.startswith("traj_") else None
        if trajectory_store is not None:
            trajectory_store.preflight_report(
                id,
                str(canonical_report_path),
                format=format_,
                replace=replace_report,
            )
        html_path = write_report(
            model,
            target_out,
            viewer_mode=viewer,
            report_format=format_,
            share=share,
            top_k=top_k,
        )
        trajectory_status = None
        if trajectory_store is not None:
            trajectory_status = trajectory_store.record_report(
                id,
                str(html_path),
                source_id=id,
                format=format_,
                replace=replace_report,
            ).model_dump()
        if open_:
            webbrowser.open(html_path.as_uri())
        data = {
            "report_html": str(html_path),
            "format": format_,
            "source": {"kind": model.summary.kind, "id": model.summary.id},
            "candidate_count": len(model.candidates),
            "enrichment_job_ids": enrichment_job_ids,
            "enrichment_enabled": True,
            "top_k": top_k,
            "warnings": [
                {"level": w.level, "message": w.message, "source": w.source}
                for w in model.warnings
            ],
            "opened": open_,
            "replace_report": replace_report,
        }
        if trajectory_status is not None:
            data["trajectory"] = trajectory_status
        if not json_:
            data["lines"] = [
                f"Report written: {html_path}",
                f"Source: {model.summary.kind} {model.summary.id} ({model.summary.state})",
                f"Candidates: {len(model.candidates)}",
                (
                    f"Sequence analysis enrichment: {', '.join(enrichment_job_ids)}"
                    if enrichment_job_ids
                    else (
                        "Sequence analysis enrichment: already present or no sequences"
                    )
                ),
            ]
        print_envelope_ok(data, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


__all__ = ["report"]

# HTML/CSS/JS templates are intentionally kept readable as literal snippets.
# ruff: noqa: E501

from __future__ import annotations

import base64
import functools
import html
import json
import math
import mimetypes
import re
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vibeproteinagent._version import __version__
from vibeproteinagent.core.errors import InvalidParamsError

from ._brand import vpa_logo
from .metrics import (
    DUAL_SCALE_METRICS as _DUAL_SCALE_METRICS,
)
from .metrics import (
    METRIC_REGISTRY as _METRIC_REGISTRY,
)
from .metrics import (
    metric_explainer as _metric_explainer,
)
from .metrics import (
    pretty_metric_name as _pretty_metric_name,
)
from .models import (
    AgentDecision,
    AgentMessage,
    AgentStepSummary,
    AgentToolCall,
    AgentTranscript,
    MSABlock,
    PlotSpec,
    ReportArtifact,
    ReportCandidate,
    ReportModel,
    WorkflowNodeSummary,
)
from .plots import PLDDT_HOVER_JS, build_plots


def write_report(
    model: ReportModel,
    out_path: str | Path,
    *,
    viewer_mode: str = "offline",
    report_format: str = "html",
    share: bool = False,
    top_k: int = 5,
) -> Path:
    if viewer_mode not in {"offline", "online"}:
        raise InvalidParamsError(
            f"unknown viewer mode {viewer_mode!r}",
            details={"viewer_mode": viewer_mode, "allowed": ["offline", "online"]},
        )
    if report_format not in {"html", "bundle"}:
        raise InvalidParamsError(
            f"unknown report format {report_format!r}",
            details={"format": report_format, "allowed": ["html", "bundle"]},
        )

    if report_format == "bundle":
        root = Path(out_path).expanduser().resolve()
        if root.suffix:
            root = root.with_suffix("")
        root.mkdir(parents=True, exist_ok=True)
        (root / "assets").mkdir(exist_ok=True)
        (root / "structures").mkdir(exist_ok=True)
        (root / "plots").mkdir(exist_ok=True)
        model = _bundle_model(model, root)
        html_path = root / "report.html"
    else:
        html_path = Path(out_path).expanduser().resolve()
        html_path.parent.mkdir(parents=True, exist_ok=True)

    html_text = render_report_html(
        model,
        viewer_mode=viewer_mode,
        report_format=report_format,
        share=share,
        top_k=top_k,
    )
    html_path.write_text(html_text, encoding="utf-8")
    return html_path


def _bundle_model(model: ReportModel, root: Path) -> ReportModel:
    path_map: dict[str, str] = {}
    structures_dir = root / "structures"
    for idx, artifact in enumerate(model.artifacts, start=1):
        src = Path(artifact.path)
        if not src.is_file() or src.suffix.lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
            continue
        if artifact.path in path_map:
            continue
        token = (artifact.sha256 or artifact.artifact_id or str(idx))[:12]
        target = structures_dir / f"{_safe_filename(artifact.field)}_{token}{src.suffix}"
        shutil.copyfile(src, target)
        path_map[artifact.path] = str(target)
        path_map[str(src.expanduser())] = str(target)

    if not path_map:
        return model

    def rewrite_artifact(artifact: ReportArtifact) -> ReportArtifact:
        return replace(artifact, path=path_map.get(artifact.path, artifact.path))

    def rewrite_path(path: str | None) -> str | None:
        if not path:
            return None
        return path_map.get(path, path_map.get(str(Path(path).expanduser()), path))

    artifacts = [rewrite_artifact(artifact) for artifact in model.artifacts]
    nodes = [
        replace(node, artifacts=[rewrite_artifact(artifact) for artifact in node.artifacts])
        for node in model.graph.nodes
    ]
    candidates = [
        replace(
            candidate,
            structure_path=rewrite_path(candidate.structure_path),
            artifacts=[rewrite_artifact(artifact) for artifact in candidate.artifacts],
        )
        for candidate in model.candidates
    ]
    raw_summary = _rewrite_paths(model.raw_summary, path_map)
    return replace(
        model,
        graph=replace(model.graph, nodes=nodes),
        candidates=candidates,
        artifacts=artifacts,
        raw_summary=raw_summary if isinstance(raw_summary, dict) else model.raw_summary,
    )


def _rewrite_paths(value: Any, path_map: dict[str, str]) -> Any:
    if isinstance(value, str):
        return path_map.get(value, path_map.get(str(Path(value).expanduser()), value))
    if isinstance(value, list):
        return [_rewrite_paths(item, path_map) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, path_map) for key, item in value.items()}
    return value


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe.strip("_") or "structure"


_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


# ----------------------------------------------------------------- download helpers
# Reports may be opened (a) as a local file, (b) over HTTP from a dev server,
# or (c) sent to a coworker as a single self-contained HTML attachment. The
# safest way to make download buttons work in all three is to inline each
# artifact's bytes as a ``data:`` URI rather than linking to ``file://`` —
# modern browsers refuse ``file://`` links from HTTP-served pages, which made
# the buttons fire a "site can't be reached" error. We cap per-file size to
# keep the report itself from ballooning; anything beyond that falls back to
# a disabled chip with the path tooltip.

_DOWNLOAD_MAX_BYTES = 8 * 1024 * 1024  # 8 MiB per artifact

_MIME_OVERRIDES = {
    ".pdb": "chemical/x-pdb",
    ".ent": "chemical/x-pdb",
    ".cif": "chemical/x-cif",
    ".mmcif": "chemical/x-cif",
    ".fasta": "text/plain; charset=utf-8",
    ".fa": "text/plain; charset=utf-8",
    ".faa": "text/plain; charset=utf-8",
    ".a3m": "text/plain; charset=utf-8",
    ".sto": "text/plain; charset=utf-8",
    ".aln": "text/plain; charset=utf-8",
    ".tsv": "text/tab-separated-values; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".npz": "application/octet-stream",
    ".npy": "application/octet-stream",
}


def _mime_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def _data_uri(path: Path) -> tuple[str, int] | None:
    """Return ``(data_uri, size_bytes)`` or ``None`` when not embeddable."""
    if not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _DOWNLOAD_MAX_BYTES:
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{_mime_for(path)};base64,{b64}", size


def _download_link(
    path_str: str | Path,
    label: str,
    *,
    kind: str | None = None,
    css: str = "artifact-link",
) -> str:
    """Render a self-contained download chip for one artifact.

    The link's ``href`` is a base64 ``data:`` URI containing the file bytes,
    so the browser triggers a real save without ever needing to fetch the
    backing file — works identically whether the report is opened locally
    or served from an HTTP host. Big files (>8 MiB) degrade to a disabled
    chip with the original path in the tooltip.
    """
    p = Path(path_str)
    filename = p.name or label
    if p.is_file():
        size = p.stat().st_size
    else:
        size = None
    size_hint = _size(size)
    kind_label = kind or (p.suffix.lstrip(".") or "file")
    encoded = _data_uri(p) if p.is_file() else None
    if encoded is None:
        suffix = "missing" if not p.is_file() else "too large"
        return (
            f'<span class="{_h(css)} dl-disabled" title="{_attr(str(p))}">'
            f'<span class="artifact-name">{_h(label)}</span>'
            f'<span class="artifact-meta">{_h(kind_label)} · {_h(suffix)}</span></span>'
        )
    href, _bytes = encoded
    return (
        f'<a class="{_h(css)}" href="{href}" download="{_attr(filename)}" '
        f'title="{_attr(filename)}">'
        f'<span class="artifact-name">{_h(label)}</span>'
        f'<span class="artifact-meta">{_h(kind_label)} · {_h(size_hint)}</span></a>'
    )


def _load_3dmol_min_js() -> str | None:
    """Return the inlined 3Dmol-min.js asset for offline embedding.

    Vendored under ``reporting/assets/3Dmol-min.js`` and force-included into
    the wheel by ``pyproject.toml``. When the file is missing the report
    still renders — the JS guard ``typeof $3Dmol === 'undefined'`` shows a
    "viewer unavailable" notice in each viewer slot instead of breaking.
    """
    candidate = _ASSETS_DIR / "3Dmol-min.js"
    if not candidate.is_file():
        return None
    try:
        return candidate.read_text(encoding="utf-8")
    except OSError:
        return None


# ----------------------------------------------------------------- main render


def render_report_html(
    model: ReportModel,
    *,
    viewer_mode: str = "offline",
    report_format: str = "html",
    share: bool = False,
    top_k: int = 5,
) -> str:
    """Build a self-contained product-style HTML report.

    Layout: fixed left sidebar (logo + scroll-spy section nav), top header
    strip (workspace · run id · status · download), main content as a stack
    of cards. Each card is a widget with a clear title + explainer text.
    All narrative copy is template-rendered (no LLM calls) so re-running on
    the same workspace produces byte-equivalent output (modulo the footer
    timestamp).
    """
    del share, report_format  # share/bundle behaviour handled upstream in write_report
    plots = build_plots(model)
    title = f"VPA report · {model.summary.id}"
    final_top = model.final_candidates[:top_k]
    explored_top = model.candidates[:top_k]
    top = final_top or explored_top

    headline = _executive_headline(model)
    dashboard_html = _executive_dashboard(model)
    verdict_html = _exec_verdict_callout(model)

    show_workflow = len(model.graph.nodes) >= 2
    show_plots = bool(plots)
    show_msa = bool(model.msa_blocks)
    show_candidates = bool(model.final_candidates or model.candidates)
    show_conversation = model.agent_transcript is not None

    nav_items: list[tuple[str, str, str]] = [("summary", "Summary", "◎")]
    if show_conversation:
        nav_items.append(("conversation", "Conversation", "❝"))
    if show_candidates:
        nav_items.append(("candidates", "Candidates", "★"))
    if show_workflow:
        nav_items.append(("workflow", "Workflow", "⌥"))
    if show_plots:
        nav_items.append(("plots", "Plots", "▦"))
    if show_msa:
        nav_items.append(("msa", "MSA", "≡"))
    nav_items.append(("details", "Per-tool results", "⊞"))
    nav_items.append(("reproducibility", "Reproducibility", "⟲"))

    nav_html = "".join(
        f'<a href="#{anchor}" class="side-nav-item">'
        f'<span class="side-nav-glyph">{_h(glyph)}</span>'
        f'<span class="side-nav-label">{_h(label)}</span></a>'
        for anchor, label, glyph in nav_items
    )

    viewer_payloads = _collect_viewer_payloads(top)
    viewer_data_script = (
        f'<script id="vpa-structures" type="application/json">{_script_json(viewer_payloads)}</script>'
        if viewer_payloads
        else ""
    )
    # Inline 3Dmol.js so the report stays a single self-contained file with
    # no CDN fetches. The script tag matches the spec's "load $3Dmol before
    # app.js" ordering; if vendoring is missing we skip the script and let
    # the offline-viewer JS render an "unavailable" notice.
    threedmol_src = _load_3dmol_min_js()
    threedmol_script = (
        f'<script>/* 3Dmol-min.js (vendored) */{threedmol_src}</script>'
        if threedmol_src
        else ""
    )

    generated_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    # ----- Summary section -----
    pills = [
        ("state-" + _state_class(model.summary.state), model.summary.state),
        ("mono", _short(model.summary.id, 18)),
        ("", _duration(model.summary.runtime_ms)),
        ("", f"{viewer_mode} viewer"),
    ]
    pills_html = "".join(
        f'<span class="pill {_h(cls)}" title="{_attr(model.summary.id if cls == "mono" else label)}">{_h(label)}</span>'
        for cls, label in pills
    )
    summary_card = f"""<section id="summary" class="card card-summary">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Executive Summary</p>
      <h2 class="card-title">{_h(headline)}</h2>
    </div>
    <div class="card-header-pills">{pills_html}</div>
  </header>
  {dashboard_html}
  {verdict_html}
  {_hero_sequence_block(model)}
  {_warning_list(model)}
</section>"""

    # ----- Conversation section (LLM-agent runs only) -----
    conversation_section = (
        _agent_transcript_section(model.agent_transcript)
        if show_conversation
        else ""
    )

    # ----- Candidates section -----
    final_block = ""
    if model.final_candidates:
        final_label = "Auto-selected" if (
            model.final_submission or {}
        ).get("auto_selected") else "Submitted"
        final_block = f"""
  <div class="candidate-group candidate-group-final">
    <div class="candidate-group-head">
      <p class="card-eyebrow">Final picks</p>
      <span class="card-meta">{_h(final_label)}</span>
    </div>
    {_candidate_cards(final_top, total_candidates=len(model.final_candidates))}
  </div>"""
    explored_block = ""
    if model.candidates:
        explored_block = f"""
  <div class="candidate-group candidate-group-explored">
    <div class="candidate-group-head">
      <p class="card-eyebrow">Explored candidates</p>
      <span class="card-meta">{len(model.candidates)} total</span>
    </div>
    {_candidate_cards(explored_top, total_candidates=len(model.candidates))}
    {_all_designs_table(model.candidates)}
  </div>"""
    candidates_section = (
        f"""<section id="candidates" class="card card-candidates">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Designs</p>
      <h2 class="card-title">Top Candidates</h2>
    </div>
    <span class="card-meta">{len(model.final_candidates)} final · {len(model.candidates)} explored</span>
  </header>
  {final_block}
  {explored_block}
</section>"""
        if show_candidates else ""
    )

    workflow_section = (
        f"""<section id="workflow" class="card">
  <header class="card-header"><div><p class="card-eyebrow">Pipeline</p><h2 class="card-title">Workflow graph</h2></div></header>
  <p class="widget-explainer">Colour shows each node's completion state; arrows trace data dependencies between jobs.</p>
  {_workflow_svg(model.graph.nodes)}
  {_workflow_composition(model)}
</section>"""
        if show_workflow else ""
    )

    plots_section = (
        f"""<section id="plots" class="card">
  <header class="card-header"><div><p class="card-eyebrow">Analytics</p><h2 class="card-title">Plots</h2></div></header>
  {_plots_html(plots)}
</section>"""
        if show_plots else ""
    )

    msa_section = (
        f"""<section id="msa" class="card">
  <header class="card-header"><div><p class="card-eyebrow">Homologs</p><h2 class="card-title">MSA / Sequence Context</h2></div></header>
  {_msa_html(model.msa_blocks)}
</section>"""
        if show_msa else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_h(title)}</title>
  <style>{_css()}</style>
</head>
<body>
  <aside class="sidebar">
    <div class="sidebar-brand">
      {vpa_logo("brand-mark")}
      <span class="brand-text">VibeProteinAgent</span>
    </div>
    <div class="sidebar-meta">
      <span class="sidebar-meta-label">Workspace</span>
      <code class="sidebar-meta-value" title="{_attr(model.summary.workspace)}">{_h(model.summary.workspace)}</code>
    </div>
    <nav class="side-nav">{nav_html}</nav>
    <div class="sidebar-foot">
      <span>v{_h(__version__)}</span>
      <span>{_h(generated_at)}</span>
    </div>
  </aside>
  <header class="top-bar">
    <div class="top-bar-left">
      <span class="top-bar-kind">{_h(model.summary.kind.title())}</span>
      <h1 class="top-bar-title">{_h(model.summary.name or model.summary.id)}</h1>
      <span class="status-chip state-{_state_class(model.summary.state)}">{_h(model.summary.state)}</span>
    </div>
    <div class="top-bar-right">
      <code class="top-bar-id" title="{_attr(model.summary.id)}">{_h(_short(model.summary.id, 28))}</code>
    </div>
  </header>
  <main class="content">
    {summary_card}
    {conversation_section}
    {candidates_section}
    {workflow_section}
    {plots_section}
    {msa_section}
    <section id="details" class="card">
      <header class="card-header">
        <div><p class="card-eyebrow">Tool results</p><h2 class="card-title">Per-tool widgets</h2></div>
      </header>
      <p class="widget-explainer">Each tool used in this workflow gets its own structured panel below — metrics, tables, downloadable artifacts. The raw JSON output is collapsed at the bottom of each panel for reference.</p>
      {_node_details(model.graph.nodes)}
    </section>
    <section id="reproducibility" class="card">
      <header class="card-header">
        <div><p class="card-eyebrow">Reproducibility</p><h2 class="card-title">Replay command</h2></div>
      </header>
      {_repro_html(model)}
    </section>
  </main>
  {threedmol_script}
  {viewer_data_script}
  <script id="report-data" type="application/json">{_script_json(_public_model(model))}</script>
  <script>{_js()}</script>
</body>
</html>
"""


# ----------------------------------------------------------------- hero / summary


def _executive_dashboard(model: ReportModel) -> str:
    """CELLKEY-style 3-panel dashboard for the Executive Summary card.

    Replaces the previous flat KPI strip with a structured dashboard:

    * **Run Vitals** (left) — KPI tile grid with status, runtime, candidate
      count, tool count, artifact count, warning count. Each tile gets an
      icon, value, and a colour accent driven by the underlying state /
      quality.
    * **Confidence at a Glance** (middle) — donut gauges for the top
      candidate's headline confidence metrics (pLDDT / pTM / ipTM / ΔG /
      DockQ / score). Up to three, picked deterministically by priority.
    * **Pipeline Health** (right) — single donut breaking down the
      workflow nodes by state (completed / running / failed / etc.) with
      a small legend underneath. Hidden for single-tool jobs since the
      breakdown is trivially 1-node.

    The whole dashboard adapts to runs that produced no scalar metrics
    (sequence-only adapters, info-tools) — empty panels just stay quiet.
    """
    parts = [
        _exec_vitals_panel(model),
        _exec_confidence_panel(model),
        _exec_pipeline_panel(model),
    ]
    panels = "".join(p for p in parts if p)
    return f'<div class="exec-dashboard">{panels}</div>'


def _exec_vitals_panel(model: ReportModel) -> str:
    """Left panel — six KPI tiles covering the run's headline numbers."""
    tiles: list[str] = []
    state = (model.summary.state or "unknown").lower()
    state_class = _state_class(model.summary.state)
    state_glyph = {
        "completed": "✓",
        "running": "●",
        "queued": "◔",
        "pending": "◔",
        "failed": "✕",
        "cancelled": "○",
        "skipped": "—",
    }.get(state, "●")
    tiles.append(_kpi_tile(state_glyph, state.title(), "Status", accent=state_class))
    tiles.append(_kpi_tile("⏱", _duration(model.summary.runtime_ms), "Runtime"))

    n_candidates = len(model.candidates)
    cand_accent = "good" if n_candidates >= 1 else "neutral"
    tiles.append(_kpi_tile(
        "", str(n_candidates), "Candidates", accent=cand_accent,
        raw_icon=vpa_logo("kpi-logo"),
    ))

    n_nodes = len(model.graph.nodes)
    tiles.append(_kpi_tile("⚙", str(n_nodes), "Tools" if n_nodes == 1 else "Pipeline nodes"))

    tiles.append(_kpi_tile("⊞", str(len(model.artifacts)), "Artifacts"))

    n_warn = len(model.warnings)
    n_err = sum(1 for w in model.warnings if w.level == "error")
    if n_err:
        warn_accent = "poor"
    elif n_warn:
        warn_accent = "ok"
    else:
        warn_accent = "good"
    warn_label = "Errors" if n_err else "Warnings" if n_warn else "Clean"
    warn_value = str(n_err or n_warn or 0)
    warn_glyph = "✕" if n_err else "⚠" if n_warn else "✓"
    tiles.append(_kpi_tile(warn_glyph, warn_value, warn_label, accent=warn_accent))

    return (
        '<section class="exec-panel exec-panel-vitals">'
        '<header class="exec-panel-head">'
        '<h3 class="exec-panel-title">Run vitals</h3>'
        '<span class="exec-panel-sub">Headline numbers from this run</span>'
        '</header>'
        f'<div class="kpi-grid">{"".join(tiles)}</div>'
        '</section>'
    )


def _kpi_tile(
    icon: str, value: str, label: str, *, accent: str = "neutral",
    raw_icon: str | None = None,
) -> str:
    """Single dashboard KPI tile (icon + value + label, with state accent).

    ``raw_icon`` injects pre-built markup (e.g. the brand logo) in place of
    the escaped glyph icon.
    """
    icon_html = (
        raw_icon if raw_icon is not None
        else f'<span class="kpi-icon" aria-hidden="true">{_h(icon)}</span>'
    )
    return (
        f'<div class="kpi-tile kpi-{_h(accent)}">'
        f'{icon_html}'
        f'<div class="kpi-body">'
        f'<div class="kpi-value">{_h(value)}</div>'
        f'<div class="kpi-label">{_h(label)}</div>'
        f'</div></div>'
    )


# Priority list of confidence metrics surfaced in the executive donut row,
# with their gauge ranges and one-line explainers. The first three with
# data available end up as donuts; the rest stay hidden so the panel never
# overflows.
_EXEC_CONFIDENCE_SPECS: tuple[tuple[str, str, str, float, float], ...] = (
    ("plddt", "pLDDT", "Per-residue confidence. >=70 confident, >=90 very high.", 0.0, 100.0),
    ("iptm", "ipTM", "Interface pTM (0-1). >0.6 generally trustworthy.", 0.0, 1.0),
    ("ptm", "pTM", "Predicted TM-score (0-1). >0.5 = likely-correct fold.", 0.0, 1.0),
    ("dockq", "DockQ", "Docking pose quality (0-1).", 0.0, 1.0),
    ("delta_g", "ΔG", "Binding free energy (kcal/mol). More negative = tighter.", -16.0, 0.0),
    ("ranking_confidence", "Conf.", "Composite confidence head.", 0.0, 1.0),
    ("score", "Score", "Pipeline-specific composite score.", 0.0, 1.0),
)


def _exec_confidence_panel(model: ReportModel) -> str:
    """Middle panel — up to three confidence donuts for the top candidate."""
    top = model.candidates[0] if model.candidates else None
    if top is None:
        return ""
    donuts: list[str] = []
    for key, label, tip, lo, hi in _EXEC_CONFIDENCE_SPECS:
        if len(donuts) >= 3:
            break
        if key == "score":
            value = top.score
        else:
            value = top.scores.get(key)
        num = _num(value)
        if num is None:
            continue
        # Auto-scale dual-range metrics (e.g. pLDDT as 0-1 vs 0-100).
        gauge_hi = hi
        if key == "plddt" and num <= 1.5:
            gauge_hi = 1.0
        quality = _quality_class(key, num)
        donut_html = _gauge_svg(
            value=num, lo=lo, hi=gauge_hi, quality=quality,
            size=96, stroke=8,
            center_text=_fmt_donut(num),
            sub_text=_donut_label(label),
        )
        donuts.append(
            f'<div class="exec-donut">'
            f'<div class="exec-donut-viz" title="{_attr(tip)}">{donut_html}</div>'
            f'<div class="exec-donut-meta">'
            f'<div class="exec-donut-label">{_h(label)}</div>'
            f'<div class="exec-donut-tip">{_h(tip)}</div>'
            f'</div></div>'
        )
    if not donuts:
        # No confidence metrics — show the top-candidate identity and rank
        # as a soft fallback so the middle column isn't a void.
        rank = top.rank or 1
        donuts.append(
            f'<div class="exec-donut exec-donut-empty">'
            f'<div class="exec-donut-meta">'
            f'<div class="exec-donut-label">Top candidate</div>'
            f'<div class="exec-donut-id"><code>{_h(top.candidate_id)}</code></div>'
            f'<div class="exec-donut-tip">Rank #{rank} of {len(model.candidates)}</div>'
            f'</div></div>'
        )
    return (
        '<section class="exec-panel exec-panel-confidence">'
        '<header class="exec-panel-head">'
        '<h3 class="exec-panel-title">Confidence at a glance</h3>'
        '<span class="exec-panel-sub">Top candidate · headline scores</span>'
        '</header>'
        f'<div class="exec-donut-row">{"".join(donuts)}</div>'
        '</section>'
    )


def _exec_pipeline_panel(model: ReportModel) -> str:
    """Right panel — node-state breakdown donut + stages list.

    Hidden for single-tool jobs. For multi-node runs we show:

    * a centred stacked-arc donut (130px) with the "% done" centre
    * a vertical stages list (one row per node) so the panel fills its
      vertical space instead of leaving a void below a tiny legend
    * a compact legend strip showing the state-count breakdown

    The stages list shows the tool name + a coloured state pill so the
    user can scan the pipeline composition without leaving the summary.
    """
    nodes = model.graph.nodes
    if len(nodes) < 2:
        return ""
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node.state or "unknown"] = counts.get(node.state or "unknown", 0) + 1
    total = sum(counts.values()) or 1

    state_colours = {
        # Drawn entirely from the user's brand palette — no red/green/blue.
        # Sky=good, violet=running (lively brand accent), orange=waiting,
        # pink=critical/cancelled, lavender-grey=skipped.
        "completed": "#4cc4e4",   # sky (matches --good)
        "running": "#a78bfa",     # bright violet (active — distinct from sky)
        "queued": "#f39e3a",      # orange (warm waiting)
        "pending": "#f39e3a",
        "failed": "#cc0980",      # pink (matches --poor)
        "cancelled": "#cc0980",   # pink (warning state)
        "skipped": "#a09bbf",     # lavender-grey (deprioritised)
    }
    state_glyphs = {
        "completed": "✓",
        "running": "●",
        "queued": "◔",
        "pending": "◔",
        "failed": "✕",
        "cancelled": "○",
        "skipped": "—",
    }

    # Donut with stacked-arc state breakdown. Sized small enough that the
    # stages list to the right has room to show full tool names on a
    # single line (no wrap, no ellipsis).
    size = 108
    stroke = 10
    cx = cy = size / 2
    r = size / 2 - stroke
    circ = 2 * math.pi * r
    arcs: list[str] = []
    cumulative = 0.0
    state_order = ["completed", "running", "queued", "pending", "failed", "cancelled", "skipped"]
    for state in [*state_order, *sorted(s for s in counts if s not in state_order)]:
        n = counts.get(state)
        if not n:
            continue
        share = n / total
        arc_len = share * circ
        offset = -cumulative
        colour = state_colours.get(state, "#a09bbf")
        arcs.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" '
            f'stroke="{colour}" stroke-width="{stroke}" '
            f'stroke-dasharray="{arc_len:.2f} {circ - arc_len:.2f}" '
            f'stroke-dashoffset="{offset:.2f}" '
            f'transform="rotate(-90 {cx} {cy})"/>'
        )
        cumulative += arc_len
    n_done = counts.get("completed", 0)
    pct_done = round((n_done / total) * 100)
    # ``dominant-baseline="central"`` centres each glyph on its y. Stacking
    # the two lines at +/-10px from the donut centre keeps the combined
    # block visually balanced around the geometric centre — without this,
    # the default alphabetic baseline drifts the percentage upward.
    centre = (
        f'<text x="{cx}" y="{cy - 10:.0f}" text-anchor="middle" '
        f'dominant-baseline="central" class="exec-pipe-val">{pct_done}%</text>'
        f'<text x="{cx}" y="{cy + 10:.0f}" text-anchor="middle" '
        f'dominant-baseline="central" class="exec-pipe-sub">{n_done}/{total} done</text>'
    )
    donut_svg = (
        f'<svg class="exec-pipe-donut" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" aria-hidden="true">'
        f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" '
        f'stroke="#ede9fe" stroke-width="{stroke}"/>'
        + "".join(arcs)
        + centre
        + "</svg>"
    )

    # Stages list — one row per node. The coloured glyph already encodes
    # state (✓ green = completed, ✕ red = failed, ●/◔ for active/pending),
    # so we drop the textual state badge entirely and give the tool name
    # the full remaining width — that's the user-facing identifier that
    # should never be ellipsis-clipped. Every node is listed; the list
    # scrolls within the panel rather than truncating with a "+N more" row.
    stages_rows: list[str] = []
    for node in nodes:
        state = (node.state or "unknown").lower()
        colour = state_colours.get(state, "#a09bbf")
        glyph = state_glyphs.get(state, "•")
        tool = node.tool or node.node_id or "tool"
        stages_rows.append(
            f'<li class="exec-stage-row" title="{_attr(state)}">'
            f'<span class="exec-stage-glyph" style="color:{colour}">{_h(glyph)}</span>'
            f'<span class="exec-stage-name">{_h(tool)}</span>'
            f'</li>'
        )

    # Compact legend strip below the donut + stages.
    legend_chips: list[str] = []
    for state in [*state_order, *sorted(s for s in counts if s not in state_order)]:
        n = counts.get(state)
        if not n:
            continue
        colour = state_colours.get(state, "#a09bbf")
        legend_chips.append(
            f'<span class="exec-pipe-legend-chip">'
            f'<span class="exec-pipe-legend-swatch" style="background:{colour}"></span>'
            f'<span class="exec-pipe-legend-label">{_h(state)}</span>'
            f'<span class="exec-pipe-legend-count">{n}</span>'
            f'</span>'
        )

    return (
        '<section class="exec-panel exec-panel-pipeline">'
        '<header class="exec-panel-head">'
        '<h3 class="exec-panel-title">Pipeline health</h3>'
        f'<span class="exec-panel-sub">{total} node{"s" if total != 1 else ""} · '
        f'{n_done} completed</span>'
        '</header>'
        '<div class="exec-pipe-body">'
        f'<div class="exec-pipe-donut-wrap">{donut_svg}</div>'
        '<ul class="exec-stage-list">'
        f'{"".join(stages_rows)}'
        '</ul>'
        '</div>'
        f'<div class="exec-pipe-legend">{"".join(legend_chips)}</div>'
        '</section>'
    )


def _exec_verdict_callout(model: ReportModel) -> str:
    """Single-line plain-English verdict beneath the dashboard.

    Distils ``_executive_secondary`` into a more visually prominent callout
    so the reader has a clear "should I trust this?" cue right below the
    numbers. Icon + colour reflect the verdict severity.
    """
    top = model.candidates[0] if model.candidates else None
    if top is None:
        return (
            '<div class="exec-verdict exec-verdict-neutral">'
            '<span class="exec-verdict-icon" aria-hidden="true">•</span>'
            '<div class="exec-verdict-body">'
            '<div class="exec-verdict-head">No candidate designs produced</div>'
            '<div class="exec-verdict-detail">Review the per-tool widgets and artifact list to inspect inputs and intermediate outputs.</div>'
            '</div></div>'
        )
    cautions: list[str] = []
    plddt = _as_unit(top.scores.get("plddt"))
    if plddt is not None and plddt < 0.7:
        cautions.append(f"mean pLDDT {plddt:.2f} suggests low structural confidence")
    ptm = _as_unit(top.scores.get("ptm"))
    if ptm is not None and ptm < 0.5:
        cautions.append(f"pTM {ptm:.2f} below 0.5 — overall fold may be unreliable")
    # ipTM is the "interface trustworthiness" call — only meaningful when
    # the structure actually HAS an interface. Skip the caution on a
    # single-chain output so we don't issue an "interface unlikely to be
    # trustworthy" warning for a monomer that has no interface to begin
    # with.
    if not _is_monomer_candidate(top):
        iptm = _as_unit(top.scores.get("iptm"))
        if iptm is not None and iptm < 0.5:
            cautions.append(
                f"ipTM {iptm:.2f} below 0.5 — interface unlikely to be trustworthy"
            )
    if top.qc_flags:
        cautions.append(f"{len(top.qc_flags)} QC flag(s) on the top candidate")
    n_err = sum(1 for w in model.warnings if w.level == "error")
    if n_err:
        cautions.insert(0, f"{n_err} error(s) recorded during the run")

    if not cautions:
        return (
            '<div class="exec-verdict exec-verdict-good">'
            '<span class="exec-verdict-icon" aria-hidden="true">✓</span>'
            '<div class="exec-verdict-body">'
            '<div class="exec-verdict-head">No confidence or QC red flags on the top candidate.</div>'
            '<div class="exec-verdict-detail">Scores fall in the trustworthy range — proceed to downstream validation.</div>'
            '</div></div>'
        )
    severity = "poor" if (n_err or len(cautions) >= 3) else "ok"
    icon = "✕" if severity == "poor" else "⚠"
    return (
        f'<div class="exec-verdict exec-verdict-{severity}">'
        f'<span class="exec-verdict-icon" aria-hidden="true">{icon}</span>'
        '<div class="exec-verdict-body">'
        '<div class="exec-verdict-head">Review before downstream use</div>'
        f'<div class="exec-verdict-detail">{_h("; ".join(cautions))}.</div>'
        '</div></div>'
    )


def _hero_sequence_block(model: ReportModel) -> str:
    if not model.candidates:
        return ""
    seq = model.candidates[0].sequence
    if not seq:
        return ""
    seq_html = _colored_sequence(seq)
    return (
        f'<div class="hero-sequence">'
        f'<div class="hero-sequence-head">'
        f'<span class="eyebrow">Input sequence · {len(seq)} residues</span>'
        f'<button class="copy-btn" data-copy="{_attr(seq)}">Copy</button>'
        f'</div>'
        f'<div class="hero-sequence-body"><code>{seq_html}</code></div>'
        '</div>'
    )


def _executive_headline(model: ReportModel) -> str:
    """Tight one-line summary at the top of the card.

    Avoids re-stating the workflow name (already in the top bar) and keeps
    the headline focused on what actually happened — runtime + candidate
    count + the top headline scores + warning posture.
    """
    state = (model.summary.state or "unknown").lower()
    state_phrase = {
        "completed": "Completed",
        "running": "In progress",
        "queued": "Queued",
        "pending": "Pending",
        "failed": "Failed",
        "cancelled": "Cancelled",
        "skipped": "Skipped",
    }.get(state, state.title())
    runtime = _duration(model.summary.runtime_ms)
    pieces: list[str] = [
        f"{state_phrase} in {runtime}"
        if runtime != "runtime n/a"
        else state_phrase
    ]

    n_candidates = len(model.candidates)
    if n_candidates:
        pieces.append(
            f"{n_candidates} candidate" if n_candidates == 1 else f"{n_candidates} candidates"
        )

    top = model.candidates[0] if model.candidates else None
    if top is not None:
        bits: list[str] = []
        for key, label_short in (
            ("plddt", "pLDDT"),
            ("iptm", "ipTM"),
            ("ptm", "pTM"),
            ("delta_g", "ΔG"),
        ):
            value = top.scores.get(key)
            if value is not None and _num(value) is not None:
                bits.append(f"{label_short} {_fmt(value)}")
            if len(bits) == 2:
                break
        if not bits and top.score is not None:
            bits.append(f"score {_fmt(top.score)}")
        if bits:
            pieces.append("top " + ", ".join(bits))

    pieces.append(_state_clause(model))
    # Sentence-case join with " · " separators — reads like a dashboard
    # tagline rather than two stitched sentences.
    return " · ".join(p.rstrip(".") for p in pieces if p)


def _state_clause(model: ReportModel) -> str:
    state = model.summary.state
    if model.warnings:
        worst = max((w.level for w in model.warnings), default="warning", key=lambda level: {"error": 2, "warning": 1}.get(level, 0))
        return f"{len(model.warnings)} {worst}(s) need review"
    if state == "completed":
        return "no warnings"
    return f"state: {state}"


# ----------------------------------------------------------------- candidates


def _collect_viewer_payloads(candidates: list[ReportCandidate]) -> list[dict[str, Any]]:
    """Build inline structure payloads — once per top candidate, JSON-embedded.

    Each payload includes PDB/CIF text content plus pre-computed pLDDT
    per-residue (B-factor) for client-side colouring. No iframe, no escaped
    HTML attribute — the viewer JS reads from a single <script type="json">
    block.
    """
    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.structure_path:
            continue
        path = Path(candidate.structure_path)
        if not path.is_file():
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        if len(content) > 2_000_000:  # 2 MB hard cap to avoid blowing up the report
            content = content[:2_000_000]
        payloads.append(
            {
                "id": _structure_dom_id(candidate.candidate_id),
                "candidate_id": candidate.candidate_id,
                "format": path.suffix.lower().lstrip("."),
                "filename": path.name,
                "content": content,
            }
        )
    return payloads


def _structure_dom_id(candidate_id: str) -> str:
    return "viewer-" + "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in candidate_id)


# 3-letter → 1-letter map for structure-file → FASTA fallback. Mirrors the
# JS-side AA3_TO_1 used by the Pick-mode info card.
_AA3_TO_1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _extract_sequence_from_structure(path: Path) -> str:
    """Pull the 1-letter sequence from CA atoms of a PDB/CIF file.

    Used as a fallback for the sequence track when ``candidate.sequence``
    is empty (common for Boltz-2 / Protenix outputs that don't echo the
    input sequence in the ranked-candidates row). Returns an empty string
    on any read failure so callers can simply skip the sequence track.

    Handles two formats:

    * **PDB**: scans ``ATOM`` records, takes one ``CA`` per residue.
    * **CIF / mmCIF**: parses the ``_atom_site`` loop column layout and
      pulls one ``CA`` per ``label_seq_id``. Hetero-atom (HETATM) rows
      are skipped so cofactors and waters don't pollute the sequence.
    """
    if not path.is_file():
        return ""
    suffix = path.suffix.lower()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""

    if suffix in {".pdb", ".ent"}:
        seq: list[str] = []
        seen: set[tuple[str, int]] = set()
        for line in text.splitlines():
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            resn = line[17:20].strip().upper()
            try:
                resi = int(line[22:26].strip())
            except ValueError:
                continue
            chain = line[21:22].strip() or "?"
            key = (chain, resi)
            if key in seen:
                continue
            seen.add(key)
            seq.append(_AA3_TO_1.get(resn, "X"))
        return "".join(seq)

    if suffix in {".cif", ".mmcif"}:
        # Minimal CIF parser: find the ``_atom_site`` loop, read its
        # column order, then collect CA rows. We don't depend on any
        # external CIF library so the report stays self-contained.
        lines = text.splitlines()
        cols: list[str] = []
        in_header = False
        in_loop = False
        start = -1
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if stripped == "loop_":
                in_header = True
                in_loop = False
                cols = []
                continue
            if in_header and stripped.startswith("_atom_site."):
                cols.append(stripped.split(".", 1)[1].strip())
                continue
            if in_header and stripped.startswith("_") and not stripped.startswith("_atom_site."):
                # Different loop; reset.
                in_header = False
                cols = []
                continue
            if in_header and cols and not stripped.startswith("_"):
                # Data rows for _atom_site begin here.
                in_header = False
                in_loop = True
                start = idx
                break
        if not (in_loop and cols and start >= 0):
            return ""
        try:
            grp = cols.index("group_PDB") if "group_PDB" in cols else -1
            comp_idx = cols.index("label_comp_id") if "label_comp_id" in cols else cols.index("auth_comp_id")
            atom_idx = cols.index("label_atom_id") if "label_atom_id" in cols else cols.index("auth_atom_id")
            asym_idx = cols.index("label_asym_id") if "label_asym_id" in cols else cols.index("auth_asym_id")
            seq_idx = cols.index("label_seq_id") if "label_seq_id" in cols else cols.index("auth_seq_id")
        except ValueError:
            return ""
        seq2: list[str] = []
        seen2: set[tuple[str, str]] = set()
        for raw in lines[start:]:
            row = raw.strip()
            if not row or row.startswith("#") or row.startswith("loop_") or row.startswith("_"):
                break
            parts = row.split()
            if len(parts) < len(cols):
                continue
            if grp >= 0 and parts[grp] != "ATOM":
                continue
            if parts[atom_idx].strip('"').strip("'") != "CA":
                continue
            resn = parts[comp_idx].strip('"').strip("'").upper()
            chain = parts[asym_idx]
            resi = parts[seq_idx]
            key = (chain, resi)
            if key in seen2:
                continue
            seen2.add(key)
            seq2.append(_AA3_TO_1.get(resn, "X"))
        return "".join(seq2)

    return ""


def _extract_chains_from_structure(path: Path) -> list[tuple[str, str]]:
    """Parse ``path`` into ``[(chain_id, one_letter_sequence), ...]``.

    Used to identify binder vs target chains for binder-design tool outputs
    (BindCraft, BoltzGen, etc.) where the PDB carries a target+binder
    complex while the candidate's declared sequence is the binder only.

    Returns the chains in the order their CA atoms first appear (matches
    the natural author-ordering in the structure file). Empty list on any
    parse failure or unsupported format. Only ATOM/CA records are read;
    HETATM is skipped so cofactors don't pollute chain sequences.
    """
    if not path.is_file():
        return []
    suffix = path.suffix.lower()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []

    # Insertion-ordered dict so chain order matches file order.
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()  # (chain, residue-key) dedupe

    if suffix in {".pdb", ".ent"}:
        for line in text.splitlines():
            if not line.startswith("ATOM") or len(line) < 27:
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21:22].strip() or "?"
            try:
                resi = int(line[22:26])
            except ValueError:
                continue
            icode = line[26:27]
            key = (chain, f"{resi}{icode}")
            if key in seen:
                continue
            seen.add(key)
            resn = line[17:20].strip().upper()
            chains.setdefault(chain, []).append(_AA3_TO_1.get(resn, "X"))
        return [(cid, "".join(letters)) for cid, letters in chains.items()]

    if suffix in {".cif", ".mmcif"}:
        # Reuse the column-discovery pass from _extract_sequence_from_structure.
        lines = text.splitlines()
        cols: list[str] = []
        in_header = False
        in_loop = False
        start = -1
        for idx, raw in enumerate(lines):
            stripped = raw.strip()
            if stripped == "loop_":
                in_header = True
                in_loop = False
                cols = []
                continue
            if in_header and stripped.startswith("_atom_site."):
                cols.append(stripped.split(".", 1)[1].strip())
                continue
            if in_header and stripped.startswith("_") and not stripped.startswith("_atom_site."):
                in_header = False
                cols = []
                continue
            if in_header and cols and not stripped.startswith("_"):
                in_header = False
                in_loop = True
                start = idx
                break
        if not (in_loop and cols and start >= 0):
            return []
        try:
            grp = cols.index("group_PDB") if "group_PDB" in cols else -1
            comp_idx = cols.index("label_comp_id") if "label_comp_id" in cols else cols.index("auth_comp_id")
            atom_idx = cols.index("label_atom_id") if "label_atom_id" in cols else cols.index("auth_atom_id")
            asym_idx = cols.index("label_asym_id") if "label_asym_id" in cols else cols.index("auth_asym_id")
            seq_idx = cols.index("label_seq_id") if "label_seq_id" in cols else cols.index("auth_seq_id")
        except ValueError:
            return []
        for raw in lines[start:]:
            row = raw.strip()
            if not row or row.startswith("#") or row.startswith("loop_") or row.startswith("_"):
                break
            parts = row.split()
            if len(parts) < len(cols):
                continue
            if grp >= 0 and parts[grp] != "ATOM":
                continue
            if parts[atom_idx].strip('"').strip("'") != "CA":
                continue
            chain = parts[asym_idx]
            resi = parts[seq_idx]
            key = (chain, resi)
            if key in seen:
                continue
            seen.add(key)
            resn = parts[comp_idx].strip('"').strip("'").upper()
            chains.setdefault(chain, []).append(_AA3_TO_1.get(resn, "X"))
        return [(cid, "".join(letters)) for cid, letters in chains.items()]

    return []


@dataclass(frozen=True, slots=True)
class _ChainContext:
    """Per-candidate chain-composition hint.

    Populated when the candidate's structure is a binder-in-complex (i.e.
    multi-chain PDB/CIF where exactly one chain's sequence matches
    ``candidate.sequence``). Used to surface "binder chain B · 30 aa,
    target chain A · 109 aa" labelling so the obvious size mismatch
    between the displayed sequence and the rendered complex stops looking
    like a bug. Empty / default-constructed for single-chain structures
    and for candidates without a sequence to anchor against.
    """

    binder_chain_id: str | None = None
    # [(chain_id, residue_count), ...] in file order, excluding the binder.
    target_chains: tuple[tuple[str, int], ...] = ()

    @property
    def is_complex(self) -> bool:
        return bool(self.binder_chain_id and self.target_chains)

    def target_summary(self) -> str:
        """e.g. ``'target chain A (109 aa)'`` or ``'target chains A (109 aa), C (8 aa)'``."""
        if not self.target_chains:
            return ""
        parts = [f"chain {cid} ({n} aa)" for cid, n in self.target_chains]
        if len(parts) == 1:
            return f"target {parts[0]}"
        return "target chains " + ", ".join(parts)


def _chain_context_for(candidate: ReportCandidate) -> _ChainContext:
    """Detect binder-in-complex layout for ``candidate``.

    Returns an empty (non-complex) context unless the structure has 2+
    chains AND exactly one of them is an exact sequence match for
    ``candidate.sequence`` — the unambiguous "this chain is the
    designed binder, the rest are target context" signal.
    """
    if not candidate.sequence or not candidate.structure_path:
        return _ChainContext()
    chains = _extract_chains_from_structure(Path(candidate.structure_path))
    if len(chains) <= 1:
        return _ChainContext()
    matches = [cid for cid, cseq in chains if cseq == candidate.sequence]
    if len(matches) != 1:
        return _ChainContext()
    binder_id = matches[0]
    targets = tuple((cid, len(cseq)) for cid, cseq in chains if cid != binder_id)
    if not targets:
        return _ChainContext()
    return _ChainContext(binder_chain_id=binder_id, target_chains=targets)


# Metrics that only carry meaning on a complex (two+ chains forming an
# interface). Hidden from monomer candidate cards / cautions / hero
# fallback because reporting them on a single-chain structure is
# misleading at best — ipTM/ipSAE/DockQ are *defined* on the interface
# between chains, sap_score / pae_interaction / iplddt likewise, and
# binding affinity (dG, affinity*, *_probability) presupposes a binding
# partner. Conservative — leaves plddt / ptm / sequence-design metrics
# alone since those describe the chain itself.
_INTERFACE_ONLY_METRICS: frozenset[str] = frozenset({
    "iptm", "mean_iptm",
    "ipsae",
    "dockq",
    "iplddt",
    "pae_interaction",
    "sap_score",
    "interface_residues", "interface_contacts", "interface_size",
    "binding_probability", "binder_probability",
    "affinity", "affinity_value",
    "affinity_pred_value",
    "affinity_probability", "affinity_probability_binary",
    "confidence_score",  # Boltz-2 affinity composite head, complex-only
    "dg", "delta_g",
})


def _is_interface_only_metric(canonical: str) -> bool:
    """True for metrics that only make sense on a complex / interface.

    Matches both the explicit registry in :data:`_INTERFACE_ONLY_METRICS`
    and any key with the prefixes ``interface_*`` or ``binder_*`` so
    derived metrics (``interface_helix%``, ``binder_plddt``, ...) are
    also caught.
    """
    if canonical in _INTERFACE_ONLY_METRICS:
        return True
    return canonical.startswith("interface_") or canonical.startswith("binder_")


def _is_monomer_candidate(candidate: ReportCandidate) -> bool:
    """Return True iff the candidate's structure has exactly one protein chain.

    Used to gate interface-only metric rendering: a single-chain output
    can't have an interface, so ipTM / ipSAE / DockQ / binding-affinity
    rows are dropped from the candidate card and the executive verdict.
    Candidates without a structure file are treated as NOT monomer (we
    have no way to tell) — they fall through to the unfiltered path so
    we don't accidentally hide complex metrics for a sequence-only
    record that genuinely came from a binder-design tool.
    """
    if not candidate.structure_path:
        return False
    chains = _extract_chains_from_structure(Path(candidate.structure_path))
    return len(chains) == 1


def _structure_sequence_track(
    sequence: str,
    dom_id: str,
    chain_context: _ChainContext | None = None,
) -> str:
    """Inline residue-letter strip above the 3D viewer.

    Each letter is its own ``<span class="sst-cell" data-resi="N">`` so
    the picker-mode hover handler can flip a single cell into the
    ``is-hovered`` state and visually link the 3D residue to its FASTA
    position. Renders nothing when the sequence is unavailable.

    For long sequences (>= 100 aa) we interleave a small ruler tick
    every 50 residues so the reader can navigate a multi-hundred-aa
    chain without counting characters. The strip itself stays
    fixed-height with vertical scroll (``max-height`` + ``overflow-y``)
    so a 900-residue sequence doesn't push the 3D viewer off-screen.
    """
    if not sequence:
        return ""
    long_seq = len(sequence) >= 100
    ruler_every = 50
    cells: list[str] = []
    for idx, aa in enumerate(sequence, start=1):
        # Insert a position marker BEFORE residues at the start of each
        # 50-aa stride (and at position 1) so long sequences carry a
        # visible coordinate.
        if long_seq and (idx == 1 or idx % ruler_every == 1):
            cells.append(f'<span class="sst-tick" aria-hidden="true">{idx}</span>')
        cells.append(
            f'<span class="sst-cell" data-resi="{idx}" '
            f'data-aa="{_h(aa)}">{_h(aa)}</span>'
        )
    meta_extra = (
        f' · ruler every {ruler_every} residues' if long_seq else ''
    )
    # For binder-design complex outputs, prepend the binder-chain id and
    # follow with a target-context note so the sequence track stops looking
    # like it disagrees with the rendered structure size.
    if chain_context and chain_context.is_complex:
        label_html = (
            f'<span class="sst-label">Binder · chain {_h(chain_context.binder_chain_id)}</span>'
        )
        meta_html = (
            f'<span class="sst-meta">{len(sequence)} residues · '
            f'hover the structure to highlight · '
            f'complex also shows {_h(chain_context.target_summary())}{meta_extra}</span>'
        )
    else:
        label_html = '<span class="sst-label">Sequence</span>'
        meta_html = (
            f'<span class="sst-meta">{len(sequence)} residues · '
            f'hover the structure to highlight{meta_extra}</span>'
        )
    return (
        f'<div class="structure-sequence-track" data-target="{_attr(dom_id)}">'
        '<div class="sst-head">'
        f'{label_html}'
        f'{meta_html}'
        '</div>'
        f'<code class="sst-strip">{"".join(cells)}</code>'
        '</div>'
    )


def _candidate_cards(
    candidates: list[ReportCandidate],
    *,
    total_candidates: int | None = None,
) -> str:
    """Render top-candidate cards.

    ``total_candidates`` is the full candidate population size before any
    ``--top-k`` truncation. When the *full* run only produced one candidate
    we use the shorter "Candidate" eyebrow; when ``--top-k`` truncated a
    larger set down to 1, we still use "Top #N" so the contract that the
    chosen card is one of many remains visible.
    """
    if not candidates:
        return "<p class='empty'>No candidate cards available for this run.</p>"
    population = total_candidates if total_candidates is not None else len(candidates)
    single = population <= 1
    cards = []
    for candidate in candidates:
        eyebrow = "Candidate" if single else f"Top #{candidate.rank or '?'}"
        score_html = _hero_candidate_score(candidate)
        # Compute the binder/target chain context once so both the viewer
        # block and the right-side sequence block use the same labelling.
        chain_context = _chain_context_for(candidate)
        info_html = f"""<div class="candidate-info-col">
      {_score_bars(candidate)}
      {_flag_badges(candidate)}
      <p class="why"><strong>Why this one?</strong> {_h(_why_candidate(candidate))}</p>
      {_candidate_sequence_block(candidate, chain_context)}
      {_candidate_downloads(candidate)}
    </div>"""
        # Sequence-only tools (esm2_embedding, esm3_tiny, ligandmpnn_redesign,
        # ...) don't emit a structure artifact. Drop the empty left viewer
        # column for those so the card isn't half-empty; the info column
        # expands to full width via the .candidate-grid--solo modifier.
        # When structure_path *is* set we keep the two-column layout even if
        # the file is missing on disk — the viewer block surfaces that error
        # so the missing-file bug stays visible instead of getting hidden.
        if candidate.structure_path:
            viewer = _viewer_block(candidate, chain_context)
            grid_html = (
                f'<div class="candidate-grid">'
                f'<div class="candidate-viewer-col">{viewer}</div>'
                f'{info_html}'
                f'</div>'
            )
        else:
            grid_html = (
                f'<div class="candidate-grid candidate-grid--solo">'
                f'{info_html}'
                f'</div>'
            )
        cards.append(
            f"""<article class="candidate-card">
  <div class="candidate-head">
    <div><p class="eyebrow">{_h(eyebrow)}</p><h3>{_h(candidate.candidate_id)}</h3></div>
    {score_html}
  </div>
  {grid_html}
</article>"""
        )
    return "\n".join(cards)


def _hero_candidate_score(candidate: ReportCandidate) -> str:
    """Top-right corner of the card — show the *most relevant* score prominently."""
    if candidate.score is not None:
        return f'<div class="card-score"><strong>{_h(_fmt(candidate.score))}</strong><span>score</span></div>'
    # ipTM only makes sense for complexes — skip it on monomer candidates
    # so we don't surface an "interface" score for a single-chain output.
    monomer = _is_monomer_candidate(candidate)
    fallbacks = (("plddt", "pLDDT"), ("ptm", "pTM"))
    if not monomer:
        fallbacks = (*fallbacks, ("iptm", "ipTM"))
    for key, label in fallbacks:
        value = candidate.scores.get(key)
        if value is None:
            continue
        quality = _quality_class(key, value)
        return (
            f'<div class="card-score quality-{quality}">'
            f'<strong>{_h(_fmt(value))}</strong><span>{_h(label)}</span></div>'
        )
    return ""


def _viewer_block(
    candidate: ReportCandidate,
    chain_context: _ChainContext | None = None,
) -> str:
    """Inline 3Dmol.js viewer block per the per-candidate structure file.

    HTML scaffold follows the spec at docs/agent_layer.md (3D-viz section):
    header strip with sidechain toggle + fullscreen button, then a single
    empty <div> that $3Dmol.createViewer fills with its WebGL canvas, plus
    an absolutely-positioned status overlay for loading / error states.

    When ``chain_context`` flags a binder-design complex, the default
    colour mode flips from pLDDT to Chain so the binder and target are
    visually distinct on first paint, and the header hint calls out the
    complex composition.
    """
    if not candidate.structure_path:
        return "<div class='viewer-empty'>No structure artifact linked.</div>"
    path = Path(candidate.structure_path)
    if not path.is_file():
        return (
            f"<div class='viewer-empty'>Structure file not accessible: "
            f"<code>{_h(_short(str(path), 60))}</code></div>"
        )
    dom_id = _structure_dom_id(candidate.candidate_id)
    fmt = path.suffix.lower().lstrip(".")
    structure_label = path.name
    # Prefer the candidate's declared sequence; fall back to extracting one
    # from the structure file itself so the sequence track still renders
    # for adapters (Boltz-2 etc.) that don't echo the input sequence in
    # ranked-candidates rows.
    sequence_str = candidate.sequence or _extract_sequence_from_structure(path)
    is_complex = bool(chain_context and chain_context.is_complex)
    default_colour = "chain" if is_complex else "plddt"
    # Order options so the default one is the first (and only) ``selected``.
    plddt_option = '<option value="plddt">pLDDT (AlphaFold)</option>'
    spectrum_option = '<option value="spectrum">N to C spectrum</option>'
    chain_option = '<option value="chain">Chain</option>'
    if default_colour == "chain":
        chain_option = '<option value="chain" selected>Chain</option>'
    else:
        plddt_option = '<option value="plddt" selected>pLDDT (AlphaFold)</option>'
    if is_complex:
        # Explicit complex-composition hint that subsumes the generic
        # Pick-mode pointer. Reads more naturally than two separate hints.
        hint_text = (
            f"Binder chain {chain_context.binder_chain_id} in complex with "
            f"{chain_context.target_summary()}. Turn on Pick, then hover "
            f"residues for an instant residue card."
        )
    else:
        hint_text = "Turn on Pick, then hover residues for an instant residue card."
    return f"""<section class="structure-section" data-target="{_attr(dom_id)}" data-format="{_attr(fmt)}"
         data-candidate-id="{_attr(candidate.candidate_id)}"
         data-structure-name="{_attr(structure_label)}"
         data-sequence="{_attr(sequence_str)}"
         data-default-colour="{_attr(default_colour)}">
  <header class="structure-header">
    <span class="structure-title">3D structure</span>
    <span class="structure-hint">{_h(hint_text)}</span>
    <label class="structure-control"><span>Colour</span>
      <select class="structure-color" data-target="{_attr(dom_id)}">
        {plddt_option}
        {spectrum_option}
        {chain_option}
      </select>
    </label>
    <label class="structure-control">
      <input type="checkbox" class="structure-sidechains" data-target="{_attr(dom_id)}" />
      Sidechains
    </label>
    <button type="button" class="structure-control structure-picker-btn"
            data-target="{_attr(dom_id)}" aria-pressed="false"
            title="Pick mode: hover residues for an instant residue card (RCSB-style)">&#x2316; Pick</button>
    <button type="button" class="structure-control structure-fullscreen-btn"
            data-target="{_attr(dom_id)}" aria-label="Toggle fullscreen"
            title="Expand (press Esc to exit)">&#x2922; Expand</button>
  </header>
  {_structure_sequence_track(sequence_str, dom_id, chain_context)}
  <div class="structure-viewer" id="{_attr(dom_id)}" data-structure-id="{_attr(dom_id)}"></div>
  <div class="structure-legend" data-legend="{_attr(default_colour)}">
    <span class="legend-text">Very low</span>
    <span class="legend-gradient"></span>
    <span class="legend-text">Very high</span>
  </div>
  <aside class="structure-residue-info" data-target="{_attr(dom_id)}" hidden>
    <div class="sri-title">{_h(candidate.candidate_id)}</div>
    <div class="sri-detail">
      <span class="sri-source">{_h(structure_label)}</span>
      <span class="sri-sep">·</span>
      <span class="sri-chain sri-placeholder">Chain &mdash;</span>
      <span class="sri-sep">·</span>
      <span class="sri-resi sri-placeholder"><strong>&mdash;</strong></span>
      <span class="sri-sep">·</span>
      <span class="sri-seqpos sri-placeholder">seq &mdash;</span>
    </div>
  </aside>
</section>"""


def _candidate_sequence_block(
    candidate: ReportCandidate,
    chain_context: _ChainContext | None = None,
) -> str:
    if not candidate.sequence:
        return ""
    is_complex = bool(chain_context and chain_context.is_complex)
    # When the structure is a binder-design complex, surface the binder
    # chain id in the head and add a small caption pointing at the target
    # so the reader sees the sequence-vs-structure size gap is intentional.
    head_suffix = (
        f' · binder chain {_h(chain_context.binder_chain_id)}'
        if is_complex
        else ''
    )
    target_note = (
        f"<p class='note'>Structure also shows {_h(chain_context.target_summary())}.</p>"
        if is_complex
        else ""
    )
    truncation_note = (
        f"<p class='note'>… {len(candidate.sequence) - 240} more residues hidden</p>"
        if len(candidate.sequence) > 240
        else ""
    )
    return (
        '<div class="sequence-block">'
        f'<div class="sequence-head"><strong>Sequence</strong> · {len(candidate.sequence)} aa'
        f'{head_suffix}'
        f' <button class="copy-btn" data-copy="{_attr(candidate.sequence)}">Copy</button></div>'
        f"<code class='msa-seq'>{_colored_sequence(candidate.sequence[:240])}</code>"
        f"{truncation_note}"
        f"{target_note}"
        "</div>"
    )


def _candidate_downloads(candidate: ReportCandidate) -> str:
    if not candidate.artifacts and not candidate.structure_path:
        return ""
    links: list[str] = []
    seen: set[str] = set()
    for artifact in candidate.artifacts:
        if artifact.path in seen:
            continue
        seen.add(artifact.path)
        links.append(_download_link(artifact.path, artifact.field, kind=artifact.kind, css="dl-link"))
    if candidate.structure_path and candidate.structure_path not in seen:
        links.append(
            _download_link(
                candidate.structure_path,
                Path(candidate.structure_path).name,
                kind="structure",
                css="dl-link",
            )
        )
    if not links:
        return ""
    return f'<div class="downloads"><span class="downloads-label">Download:</span>{"".join(links)}</div>'


def _all_designs_table(candidates: list[ReportCandidate]) -> str:
    """Render the all-designs table — but only when there is meaningful info.

    A single-candidate run already shows everything in the hero card. Showing
    a 1-row table adds noise; suppress it.
    """
    if not candidates or len(candidates) == 1:
        return ""

    # Build column set — drop columns that are entirely empty / always zero
    # so the table doesn't carry information-free noise.
    has_score = any(c.score is not None for c in candidates)
    has_plddt = any(c.scores.get("plddt") is not None for c in candidates)
    has_iptm = any(c.scores.get("iptm") is not None for c in candidates)
    has_ptm = any(c.scores.get("ptm") is not None for c in candidates)
    has_liab = any(c.liabilities for c in candidates)
    has_flags = any(c.qc_flags for c in candidates)

    headers = ["Rank", "Candidate"]
    if has_score:
        headers.append("Score")
    if has_plddt:
        headers.append("pLDDT")
    if has_iptm:
        headers.append("ipTM")
    if has_ptm:
        headers.append("pTM")
    if has_liab:
        headers.append("Liabilities")
    if has_flags:
        headers.append("QC flags")
    headers.append("Structure")

    rows = []
    for candidate in candidates:
        cells: list[str] = [
            f"<td>{candidate.rank or ''}</td>",
            f"<td>{_h(candidate.candidate_id)}</td>",
        ]
        if has_score:
            cells.append(f"<td>{_h(_fmt(candidate.score))}</td>")
        if has_plddt:
            cells.append(_score_cell(candidate.scores.get("plddt"), key="plddt"))
        if has_iptm:
            cells.append(_score_cell(candidate.scores.get("iptm"), key="iptm"))
        if has_ptm:
            cells.append(_score_cell(candidate.scores.get("ptm"), key="ptm"))
        if has_liab:
            cells.append(f"<td>{len(candidate.liabilities) or ''}</td>")
        if has_flags:
            cells.append(f"<td>{_h(', '.join(candidate.qc_flags))}</td>")
        # Structure column: filename only, with title tooltip showing full path.
        struct = candidate.structure_path or ""
        struct_html = (
            f'<td title="{_attr(struct)}"><code>{_h(Path(struct).name)}</code></td>'
            if struct
            else "<td></td>"
        )
        cells.append(struct_html)
        rows.append("<tr>" + "".join(cells) + "</tr>")

    header_html = "".join(f"<th>{_h(h)}</th>" for h in headers)
    return (
        '<h3>All Designs</h3>'
        '<input id="candidate-filter" class="filter" placeholder="Filter candidates…">'
        f'<div class="table-wrap"><table id="candidate-table"><thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _score_cell(value: Any, *, key: str) -> str:
    if value is None:
        return "<td></td>"
    return f'<td class="quality-{_quality_class(key, value)}">{_h(_fmt(value))}</td>'


# ----------------------------------------------------------------- workflow


def _workflow_svg(nodes: list[WorkflowNodeSummary]) -> str:
    """Polished horizontal workflow DAG.

    Design influences (researched, mid-2024):
      - Restraint and lack of ornamentation — turbo.build / React-Flow
        "Turbo" example. Soft curved edges, no chunky borders, subtle drop
        shadow only.
      - Numbered step that becomes a status check on completion —
        Contra / Tailscale stepper patterns. A single circular indicator
        does double duty (step ordering + state).
      - Status conveyed by **icon + colour**, not by background fills
        across the whole card — accessibility benefit, less visual noise.

    Each node is a hairline white card with a single status circle on the
    left (numbered for pending, checkmark for completed, X for failed,
    spinner for running). Edges are subtle bezier curves with a fine
    chevron head. Cards have a real (very subtle) drop shadow rather than
    coloured fills.
    """
    if not nodes:
        return "<p class='empty'>No workflow graph available.</p>"

    NODE_W = 190
    NODE_H = 64
    H_GAP = 44
    PAD_X = 28
    PAD_Y = 32
    cols = len(nodes)
    width = PAD_X * 2 + cols * NODE_W + (cols - 1) * H_GAP
    height = PAD_Y * 2 + NODE_H

    positions = {
        node.node_id: (PAD_X + idx * (NODE_W + H_GAP), PAD_Y)
        for idx, node in enumerate(nodes)
    }

    parts: list[str] = []
    parts.append(
        '<defs>'
        # Real, very subtle dual-layer drop shadow (Linear / Vercel style).
        '<filter id="dag-shadow" x="-20%" y="-20%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.05"/>'
        '<feDropShadow dx="0" dy="3" stdDeviation="6" flood-opacity="0.04"/>'
        '</filter>'
        # Solid filled chevron — bold enough to read as a directional cue
        # without dominating the diagram. ``refX="10"`` lands the chevron
        # tip exactly at the line's endpoint, so the arrow visually touches
        # the destination card. Colour matches the line for a unified feel.
        '<marker id="dag-chev" viewBox="0 0 12 12" refX="10" refY="6" '
        'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
        '<path d="M1,1 L10,6 L1,11 Z" fill="#a78bfa" stroke="none"/>'
        '</marker>'
        '</defs>'
    )

    # ---------- edges (drawn behind cards) ----------
    seen_edges: set[tuple[str, str]] = set()
    for idx, node in enumerate(nodes):
        x, _ = positions[node.node_id]
        # Use declared upstreams when present, else fall back to sequential.
        upstreams = list(node.upstream) if node.upstream else (
            [nodes[idx - 1].node_id] if idx > 0 else []
        )
        for upstream in upstreams:
            if upstream not in positions or (upstream, node.node_id) in seen_edges:
                continue
            seen_edges.add((upstream, node.node_id))
            ux, uy = positions[upstream]
            x_dst, y_dst = positions[node.node_id]
            # Start the line 4px past the source card's right edge and end
            # it 4px short of the destination's left edge, letting the
            # filled chevron marker bridge the final 4px so the arrow
            # tip visually lands flush against the destination card.
            sx = ux + NODE_W + 4
            sy = uy + NODE_H / 2
            ex = x_dst - 4
            ey = y_dst + NODE_H / 2
            # Cubic Bezier with horizontal control points → smooth curve
            # (degenerates to a straight line when source and dst share a
            # row, which is the common case for horizontal pipelines).
            parts.append(
                f'<path d="M{sx} {sy} C {sx + 26} {sy}, {ex - 26} {ey}, {ex} {ey}" '
                f'class="dag-edge" marker-end="url(#dag-chev)"/>'
            )

    # ---------- nodes ----------
    for idx, node in enumerate(nodes):
        x, y = positions[node.node_id]
        state_class = _state_class(node.state)
        step_num = idx + 1

        parts.append('<g class="dag-card-group">')
        # Card body
        parts.append(
            f'<rect x="{x}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="12" '
            f'class="dag-card" filter="url(#dag-shadow)"/>'
        )

        # Status circle (left-aligned, vertically centered)
        cc_x = x + 22
        cc_y = y + NODE_H / 2
        parts.append(_dag_status_circle(state_class, cc_x, cc_y, step_num))

        # Title + tool to the right of the circle
        title_x = x + 46
        parts.append(
            f'<text x="{title_x}" y="{y + 26}" class="dag-title">'
            f'{_svg_text(node.node_id[:18])}</text>'
            f'<text x="{title_x}" y="{y + 44}" class="dag-tool">'
            f'{_svg_text(node.tool[:18])}</text>'
        )
        parts.append('</g>')

    return (
        f'<div class="dag-wrap">'
        f'<svg class="dag" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="xMinYMid meet" xmlns="http://www.w3.org/2000/svg">'
        + "".join(parts)
        + "</svg></div>"
    )


def _dag_status_circle(state_class: str, cx: float, cy: float, step_num: int) -> str:
    """Return an SVG status circle (single indicator for step + state).

    pending  → hollow circle with step number
    running  → filled blue with a small dot
    completed/cached → filled green with white check
    failed   → filled red with white X
    cancelled → filled grey with a horizontal bar
    """
    r = 13
    if state_class in {"completed", "cached"}:
        check = (
            f'<path d="M{cx - 5} {cy + 0.4} L{cx - 1.2} {cy + 4} L{cx + 5} {cy - 3}" '
            'fill="none" stroke="#ffffff" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" class="dag-circle state-{state_class}"/>'
            + check
        )
    if state_class == "failed":
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" class="dag-circle state-failed"/>'
            f'<path d="M{cx - 4} {cy - 4} L{cx + 4} {cy + 4}" stroke="#ffffff" '
            f'stroke-width="2" stroke-linecap="round"/>'
            f'<path d="M{cx + 4} {cy - 4} L{cx - 4} {cy + 4}" stroke="#ffffff" '
            f'stroke-width="2" stroke-linecap="round"/>'
        )
    if state_class == "running":
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" class="dag-circle state-running"/>'
            f'<circle cx="{cx}" cy="{cy}" r="3" fill="#ffffff"/>'
        )
    if state_class == "cancelled":
        return (
            f'<circle cx="{cx}" cy="{cy}" r="{r}" class="dag-circle state-cancelled"/>'
            f'<rect x="{cx - 4}" y="{cy - 1}" width="8" height="2" rx="1" fill="#ffffff"/>'
        )
    # pending / queued / submitted / unknown
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" class="dag-circle state-pending"/>'
        f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" class="dag-step-num">'
        f'{step_num}</text>'
    )


def _fmt_param_value(value: Any) -> str:
    """Compact one-line rendering of a single input-parameter value."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _fmt(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return ""
        # File-path-like strings collapse to their basename.
        if "/" in s and len(s) > 24:
            return Path(s).name
        return s if len(s) <= 52 else s[:48] + "…"
    if isinstance(value, list):
        if not value:
            return "[]"
        if all(isinstance(x, (str, int, float, bool)) for x in value):
            joined = ", ".join(str(x) for x in value[:8])
            if len(value) > 8:
                joined += f" +{len(value) - 8}"
            return joined if len(joined) <= 60 else f"{len(value)} items"
        return f"{len(value)} items"
    if isinstance(value, dict):
        return f"{len(value)} fields" if value else ""
    return _short(str(value), 52)


# Structure-prediction tools that accept MSA / template input parameters —
# the workflow panel surfaces *how the MSA was done* for these. New tools
# are also picked up automatically when their params carry one of these keys.
_WF_MSA_AWARE_TOOLS = frozenset({"protenix_v2", "boltz2_affinity"})
_WF_MSA_KEYS = frozenset({
    "use_msa", "msa_backend", "use_rna_msa", "use_template",
    "msa_paths", "msa_index_csv", "msa",
})
# Documented input-parameter defaults for the MSA-aware predictors. A run
# that recorded no resolved params still has a known *effective* config —
# these let the panel show it (tagged "default" so it reads as inferred).
_WF_MSA_DEFAULTS: dict[str, Any] = {
    "use_msa": False,
    "msa_backend": "auto",
    "use_rna_msa": False,
    "use_template": False,
}


def _wf_msa_info(node: WorkflowNodeSummary) -> list[tuple[str, str]]:
    """How a structure-prediction node handled its MSA / templates.

    Uses the recorded ``use_msa`` / ``msa_backend`` / ``use_template`` input
    parameters when present. When a run recorded no params the documented
    tool defaults are shown instead (tagged ``default``); boltz2_affinity's
    echoed ``msa_paths`` give direct evidence when available."""
    params = node.params if isinstance(node.params, dict) else {}
    out = node.output if isinstance(node.output, dict) else {}
    tool = (node.tool or "").lower()
    if tool not in _WF_MSA_AWARE_TOOLS and not any(
        k in params for k in _WF_MSA_DEFAULTS
    ):
        return []
    items: list[tuple[str, str]] = []

    # ---- MSA ----
    if "use_msa" in params:
        if params.get("use_msa"):
            backend = params.get("msa_backend") or _WF_MSA_DEFAULTS["msa_backend"]
            items.append(("MSA", f"enabled · {backend} backend"))
        else:
            items.append(("MSA", "off — single-sequence"))
    else:
        mp = out.get("msa_paths")
        if isinstance(mp, list) and mp:
            items.append(
                ("MSA", f"{len(mp)} alignment{'s' if len(mp) != 1 else ''} used")
            )
        elif isinstance(mp, list) or out.get("msa_index_csv"):
            # boltz2_affinity echoed its MSA inputs — empty means single-sequence.
            items.append((
                "MSA",
                "MSA index supplied" if out.get("msa_index_csv")
                else "off — single-sequence",
            ))
        else:
            items.append(("MSA", "off — single-sequence (default)"))

    # ---- templates ----
    if "use_template" in params:
        items.append(("Templates", "enabled" if params.get("use_template") else "off"))
    elif tool in _WF_MSA_AWARE_TOOLS:
        items.append(("Templates", "off (default)"))

    # ---- RNA MSA — only surfaced when actually enabled ----
    if params.get("use_rna_msa"):
        items.append(("RNA MSA", "enabled"))

    return items


def _wf_node_inputs(node: WorkflowNodeSummary) -> list[tuple[str, str]]:
    """Input parameters a node ran with, as ``(label, value)`` pairs.

    Prefers the resolved ``params`` recorded on the node (threaded through
    from the job record). Falls back to values the tool echoes in its own
    output for older report payloads that predate per-node params. For
    structure-prediction tools the MSA configuration is always surfaced
    first via :func:`_wf_msa_info`.
    """
    msa = _wf_msa_info(node)
    params = node.params if isinstance(node.params, dict) else {}
    items: list[tuple[str, str]] = []
    if params:
        for key, value in params.items():
            # MSA keys are rendered by _wf_msa_info — don't double them here.
            if value is None or str(key).startswith("_") or key in _WF_MSA_KEYS:
                continue
            shown = _fmt_param_value(value)
            if shown:
                items.append((str(key).replace("_", " "), shown))
            if len(items) >= 12:
                break
    if not items:
        items = _wf_node_inputs_from_output(node)
    return msa + items


def _wf_node_inputs_from_output(node: WorkflowNodeSummary) -> list[tuple[str, str]]:
    """Best-effort recovery of input parameters from a tool's *output*.

    Used only when a node carries no recorded ``params`` (legacy payloads):
    most tools echo their key inputs back in their output (``go_sequences``
    returns its ``go_id``, ``sequence_edit`` its ``mutations``, …).
    """
    out = node.output if isinstance(node.output, dict) else {}
    tool = (node.tool or "").lower()
    items: list[tuple[str, str]] = []

    def add(label: str, value: Any) -> None:
        if value is None:
            return
        s = str(value).strip()
        if s and s.lower() != "none":
            items.append((label, s))

    if tool == "sequence_edit":
        add("operation", out.get("operation"))
        muts = out.get("mutations")
        if isinstance(muts, list) and muts:
            shown = ", ".join(str(m) for m in muts[:8])
            if len(muts) > 8:
                shown += f" +{len(muts) - 8}"
            add("mutations", shown)
    elif tool == "structure_analysis":
        if out.get("structure_path"):
            add("structure", Path(str(out["structure_path"])).name)
        scope = out.get("annotation_scope")
        if isinstance(scope, dict) and isinstance(scope.get("requested_positions"), list):
            rp = scope["requested_positions"]
            if rp:
                add("selected positions", _format_position_list(rp))
        chains = out.get("chains")
        if isinstance(chains, list) and chains:
            add("chains", ", ".join(str(c) for c in chains))
    elif tool == "go_sequences":
        add("GO term", out.get("go_id"))
        summ = out.get("summary")
        if isinstance(summ, dict):
            add("max scan", summ.get("max_scan_accessions"))
    elif tool == "interpro_sequences":
        add("InterPro entry", out.get("interpro_id"))
        summ = out.get("summary")
        if isinstance(summ, dict):
            add("max scan", summ.get("max_scan_accessions"))
    elif tool == "literature_search":
        if out.get("query"):
            add("query", _short(str(out["query"]), 64))
        add("backend", out.get("backend"))
        df = out.get("date_filter")
        if isinstance(df, dict) and (df.get("published_after") or df.get("published_before")):
            add("dates", f"{df.get('published_after') or '...'} - {df.get('published_before') or '...'}")
    elif tool == "template_select":
        add("engine", out.get("engine"))
        add("query kind", out.get("query_kind"))
    elif tool == "colabdesign_utils":
        add("action", out.get("action"))
    elif tool == "sequence_analysis":
        m = out.get("metrics")
        if isinstance(m, dict):
            add("mode", m.get("analysis_mode"))
    elif tool == "tmalign":
        if out.get("length_a") is not None:
            add("structure A", f"{out['length_a']} aa")
        if out.get("length_b") is not None:
            add("structure B", f"{out['length_b']} aa")
    elif tool == "rank_designs":
        if out.get("candidate_count") is not None:
            add("candidates", out.get("candidate_count"))

    # Generic fallback: surface any unambiguous run-config keys the tool
    # echoes (model name, mode, seed, sample count, …) when no tool-specific
    # extractor matched.
    if not items:
        seen: set[str] = set()
        for key, label in (
            ("model_name", "model"), ("mode", "mode"), ("task_name", "task"),
            ("num_samples", "samples"), ("seed", "seed"), ("seed_used", "seed"),
            ("seeds_used", "seeds"),
        ):
            if key in out and out[key] is not None and label not in seen:
                shown = _fmt_param_value(out[key])
                if shown:
                    items.append((label, shown))
                    seen.add(label)
    return items


def _workflow_composition(model: ReportModel) -> str:
    """Explanatory panel for the Workflow section.

    The bare DAG shows *what* ran but not *how* it was driven. This panel
    lists every step in recorded execution order and, for each, the input
    parameters recovered from its output plus any upstream dependency — so
    the run's flow is legible without prior knowledge of the run.
    """
    nodes = model.graph.nodes
    if len(nodes) < 2:
        return ""
    summary = model.summary
    kind = (summary.kind or "").lower()
    name = summary.name or ""
    by_id = {n.node_id: n for n in nodes}
    edges = sum(len([u for u in (n.upstream or []) if u in by_id]) for n in nodes)

    if kind == "tool":
        mode_key, mode_label = "tool", "Single tool run"
        lead = "This run executed a single tool."
    elif kind == "agent" or name.lower().startswith("agent"):
        mode_key, mode_label = "agent", "LLM agent-composed"
        lead = (
            "An LLM agent selected and ordered these steps automatically — no "
            "explicit user plan is recorded with the run. The steps below are "
            "listed in execution order; each shows the input parameters "
            "recovered from its output."
        )
    else:
        mode_key, mode_label = "workflow", "Predefined workflow"
        lead = (
            "This run followed a predefined workflow. The steps below are "
            "listed in execution order; each shows its input parameters and "
            "any upstream dependency."
        )
    if edges > 0:
        lead += " A 'consumes' tag marks which earlier step's output a tool reads."
    else:
        lead += (
            " The tools ran independently — there are no data dependencies "
            "between them, characteristic of a coverage / catalog run that "
            "exercises each tool in isolation."
        )

    stats = [f'<span class="wf-comp-stat"><b>{len(nodes)}</b> steps</span>']
    if edges > 0:
        stats.append(
            f'<span class="wf-comp-stat"><b>{edges}</b> data '
            f'dependenc{"y" if edges == 1 else "ies"}</span>'
        )
    else:
        stats.append('<span class="wf-comp-stat"><b>independent</b> execution</span>')

    pad = len(str(len(nodes)))
    rows: list[str] = []
    for i, n in enumerate(nodes, 1):
        chips: list[str] = []
        ups = [by_id[u].tool for u in (n.upstream or []) if u in by_id]
        if ups:
            chips.append(
                '<span class="wf-param wf-param-dep">consumes '
                f'{_h(", ".join(ups))}</span>'
            )
        for label, value in _wf_node_inputs(n):
            chips.append(
                f'<span class="wf-param"><b>{_h(label)}</b>{_h(value)}</span>'
            )
        params_html = (
            f'<span class="wf-step-params">{"".join(chips)}</span>'
            if chips
            else '<span class="wf-step-params wf-step-noparam">'
            "no input parameters recorded</span>"
        )
        rows.append(
            '<div class="wf-step">'
            f'<span class="wf-step-num">{i:0{pad}d}</span>'
            f'<span class="wf-step-dot state-{_state_class(n.state)}"></span>'
            f'<span class="wf-step-tool">{_h(n.tool)}</span>'
            f"{params_html}</div>"
        )

    return (
        '<div class="wf-comp">'
        '<h3 class="wf-comp-title">How this pipeline was composed</h3>'
        '<div class="wf-comp-head">'
        f'<span class="wf-comp-mode wf-mode-{mode_key}">{_h(mode_label)}</span>'
        f'<p class="wf-comp-lead">{_h(lead)}</p></div>'
        f'<div class="wf-comp-stats">{"".join(stats)}</div>'
        f'<div class="wf-steps">{"".join(rows)}</div>'
        "</div>"
    )


def _plots_html(plots: list[PlotSpec]) -> str:
    if not plots:
        return "<p class='empty'>No numeric distributions were detected.</p>"
    return "".join(
        f"<div class='plot'><h3>{_h(plot.title)}</h3>{plot.svg}</div>"
        for plot in plots
    )


def _msa_html(blocks: list[MSABlock]) -> str:
    """Render a multi-sequence alignment block per attached MSA artifact.

    The visualisation mirrors what alignment tools (Jalview, MView, MMseqs2's
    web report) show: column-number ruler, residue cells coloured by scheme,
    per-row identity vs query, and a per-column conservation histogram.
    Identity % is computed on the full source alignment (not just the
    displayed top-K) so the header line stays accurate.
    """
    if not blocks:
        return ""
    html_blocks = []
    for idx, block in enumerate(blocks):
        seqs = list(block.sequences)
        column_count = block.column_count or (
            max((len(seq) for _, seq in seqs), default=0)
        )
        conservation = _conservation(seqs, column_count)
        ruler_cells = "".join(
            f'<span class="msa-cell msa-ruler-cell">{(col + 1) if (col % 10 == 0 or col == column_count - 1) else "&nbsp;"}</span>'
            for col in range(column_count)
        )
        rows_html: list[str] = []
        for row_idx, (label, seq) in enumerate(seqs):
            identity = block.identities[row_idx] if row_idx < len(block.identities) else 0.0
            is_query = row_idx == 0
            id_label = "query" if is_query else f"{identity * 100:.0f}%"
            cells = "".join(
                f'<span class="msa-cell msa-residue" data-aa="{_attr(ch)}" data-col="{col + 1}">{_h(ch)}</span>'
                for col, ch in enumerate(seq[:column_count].ljust(column_count, "-"))
            )
            row_class = "msa-row msa-row-query" if is_query else "msa-row"
            rows_html.append(
                f'<div class="{row_class}">'
                f'<span class="msa-label" title="{_attr(label)}">{_h(label[:28])}</span>'
                f'<span class="msa-identity">{_h(id_label)}</span>'
                f'<span class="msa-seq">{cells}</span>'
                "</div>"
            )
        cons_cells = "".join(
            f'<span class="msa-cell msa-cons-cell" style="--cons:{v:.2f}" title="col {col + 1}: {v * 100:.0f}% conserved"></span>'
            for col, v in enumerate(conservation[:column_count])
        )
        meta_bits = [
            f"{block.sequence_count} sequences",
            f"{column_count} match columns",
            f"{block.mean_identity * 100:.0f}% mean identity",
            f"showing top {len(seqs)}",
        ]
        if block.source_tool:
            meta_bits.append(f"from {block.source_tool}")
        block_id = f"msa-{idx}"
        html_blocks.append(
            f'<div class="msa-section" id="{_attr(block_id)}" data-scheme="clustal">'
            f'<h3>{_h(block.label)}</h3>'
            f'<p class="msa-stats">{_h(" · ".join(meta_bits))}</p>'
            f'<p class="note">{_h(_short(block.path, 110))}'
            f"{' · truncated' if block.truncated else ''}</p>"
            f'<div class="msa-controls">'
            f'<label>Scheme <select class="msa-scheme">'
            f'<option value="clustal" selected>Clustal</option>'
            f'<option value="hydrophobic">Hydrophobic</option>'
            f'<option value="charge">Charge</option>'
            f'<option value="taylor">Taylor</option>'
            f'<option value="none">None</option>'
            f'</select></label>'
            f'<label>Zoom <input type="range" class="msa-zoom" min="6" max="18" value="11"></label>'
            f'<span class="msa-cursor">Col - / {column_count}</span>'
            f'<span class="msa-hover">hover a residue</span>'
            f'</div>'
            f'<div class="msa-grid-wrap">'
            f'<div class="msa-grid" style="--msa-cell-w:11px">'
            f'<div class="msa-row msa-row-ruler"><span class="msa-label"></span>'
            f'<span class="msa-identity"></span><span class="msa-seq">{ruler_cells}</span></div>'
            + "".join(rows_html)
            + f'<div class="msa-row msa-row-cons"><span class="msa-label">conservation</span>'
            f'<span class="msa-identity"></span><span class="msa-seq">{cons_cells}</span></div>'
            f"</div></div></div>"
        )
    return "".join(html_blocks)


# Per-tool widgets that already render their own artifact downloads inside
# the widget body — for these we suppress the generic outer "Artifacts" list
# in _node_details to avoid duplicating the same file links twice. Other
# tools (proteinmpnn / sequence_edit / sequence_analysis / etc.) still get
# the generic list as their only artifact surface.
_WIDGETS_OWN_ARTIFACTS = frozenset({
    "structure_analysis",
    "boltz2_affinity",
    "protenix_v2",
})


# Tools whose job is purely to *transform* one tool's output into another
# tool's input (format / schema bridges). They don't carry a user-visible
# deliverable of their own — the output is a derivative file that the
# downstream tool consumes — so the per-tool widget would render an empty
# card with just the tool name + state. Skip those entirely. They still
# appear in the Workflow DAG so the run's structure stays auditable.
#
# Add a new tool here when its only output is "convert X to Y format" —
# i.e. nothing a human reviewer would want to inspect in the report.
_TOOLS_WITHOUT_DELIVERABLE = frozenset({
    "mpnn_to_protenix_input",
    "sequence_to_protenix_input",
    "noop",
})


def _node_details(nodes: list[WorkflowNodeSummary]) -> str:
    """Per-tool result widgets — each node gets a structured presentation
    chosen by tool name, with the raw JSON tucked into a collapsible
    "Raw output" details element at the bottom for reference only.

    Tools listed in :data:`_TOOLS_WITHOUT_DELIVERABLE` (format/input bridges)
    are omitted — their output is a derivative file the next tool consumes,
    not something a human reviewer needs to inspect. Errors on those nodes
    DO still surface (we don't want to swallow real failures), but a
    successful pass-through stays out of the report.
    """
    if not nodes:
        return "<p class='empty'>No node details available.</p>"
    chunks = []
    for node in nodes:
        tool_lc = (node.tool or "").lower()
        is_passthrough = tool_lc in _TOOLS_WITHOUT_DELIVERABLE
        node_failed = bool(node.error_code or node.error_message) or node.state in {"failed", "skipped"}
        # Skip passthrough nodes that succeeded — nothing to show. If they
        # failed, render them so the operator can see why the bridge step
        # blocked the rest of the workflow.
        if is_passthrough and not node_failed:
            continue
        widget = _per_tool_widget(node)
        output_obj = node.output if isinstance(node.output, dict) else {}
        raw_json = _h(json.dumps(_compact(output_obj), indent=2, sort_keys=True))
        error = (
            f"<p class='error'><strong>{_h(node.error_code or 'error')}:</strong> "
            f"{_h(node.error_message or '')}</p>"
            if node.error_code or node.error_message
            else ""
        )
        # Skip the generic "Artifacts" footer for tools whose own widget
        # already surfaces those files with friendlier labels — duplicating
        # them under both "Per-residue analyses" and "Artifacts" is just
        # visual noise.
        artifacts = "" if tool_lc in _WIDGETS_OWN_ARTIFACTS else _node_artifacts_list(node)
        chunks.append(
            f'<article class="node-block" data-tool="{_attr(node.tool)}">'
            f'<header class="node-block-head">'
            f'<div><span class="node-block-id">{_h(node.node_id)}</span>'
            f'<span class="node-block-tool">{_h(node.tool)}@{_h(node.version)}</span></div>'
            f'<span class="node-block-state state-{_state_class(node.state)}">{_h(node.state)}</span>'
            "</header>"
            f"{error}{widget}{artifacts}"
            f'<details class="node-block-raw"><summary>Raw output (JSON)</summary>'
            f"<pre>{raw_json}</pre></details>"
            "</article>"
        )
    if not chunks:
        return "<p class='empty'>No tool results to display for this run.</p>"
    return "".join(chunks)


def _node_artifacts_list(node: WorkflowNodeSummary) -> str:
    if not node.artifacts:
        return ""
    items = "".join(_download_link(a.path, a.field, kind=a.kind) for a in node.artifacts)
    return (
        '<div class="node-block-artifacts">'
        '<span class="artifact-section-label">Artifacts</span>'
        f'<div class="artifact-row">{items}</div>'
        '</div>'
    )


# ----------------------------------------------------------------- per-tool widgets


# Deterministic per-metric specs for the QC widget. Each entry:
#   (key, display_name, one-sentence_explanation, quality_fn)
# ``quality_fn(value) -> "good"|"ok"|"poor"|"neutral"`` keeps the coloring
# rule explicit (not LLM-generated, fully reproducible across runs).
_QC_METRIC_SPECS: tuple[
    tuple[str, str, str, Callable[[float], str]], ...
] = (
    (
        "plddt",
        "Mean pLDDT",
        "Predicted local-distance difference test — per-residue confidence averaged over the chain. ≥70 confident · ≥90 very high.",
        lambda v: "good" if v >= 70 else ("ok" if v >= 50 else "poor"),
    ),
    (
        "ptm",
        "pTM",
        "Predicted TM-score (0-1). >0.5 implies a likely-correct overall fold.",
        lambda v: "good" if v >= 0.7 else ("ok" if v >= 0.5 else "poor"),
    ),
    (
        "iptm",
        "ipTM",
        "Interface pTM (0-1). Used for complex / binder interfaces; >0.6 generally trustworthy.",
        lambda v: "good" if v >= 0.7 else ("ok" if v >= 0.5 else "poor"),
    ),
    (
        "instability_index",
        "Instability Index",
        "Guruprasad et al. (1990) heuristic from amino-acid pair frequencies. <40 = predicted stable in vitro.",
        lambda v: "good" if v < 40 else ("ok" if v < 50 else "poor"),
    ),
    (
        "gravy",
        "GRAVY",
        "Grand Average of Hydropathy (Kyte-Doolittle). Negative = hydrophilic / likely soluble; positive = hydrophobic / membrane-associated.",
        lambda v: "neutral",
    ),
    (
        "pi",
        "Isoelectric point",
        "pH at which the protein has net zero charge. Typical soluble proteins span 4-9. Buffer choice should sit at least 1 unit away.",
        lambda v: "neutral",
    ),
    (
        "polybasic_hits",
        "Polybasic motifs",
        "Count of K/R-rich runs that can drive non-specific binding to membranes or RNA. 0 is ideal.",
        lambda v: "good" if v == 0 else ("ok" if v <= 2 else "poor"),
    ),
    (
        "aromatic_fraction",
        "Aromatic fraction",
        "Fraction of F/Y/W residues. Useful for UV absorbance / aggregation propensity.",
        lambda v: "neutral",
    ),
)


# Expected display range for each QC metric — used to draw a mini progress
# donut on each card. The donut's filled arc is the value's position within
# this range, so e.g. pLDDT 48.5 fills slightly less than half a 0-100 ring.
# Ranges are deliberately tuned around the "useful" interval (not strict
# physical bounds) so the visual conveys "where in the operating range did
# this run land". Keys match ``_QC_METRIC_SPECS``.
_QC_GAUGE_RANGES: dict[str, tuple[float, float]] = {
    "plddt": (0.0, 100.0),
    "ptm": (0.0, 1.0),
    "iptm": (0.0, 1.0),
    "instability_index": (0.0, 80.0),
    "gravy": (-2.0, 2.0),
    "pi": (0.0, 14.0),
    "polybasic_hits": (0.0, 10.0),
    "aromatic_fraction": (0.0, 0.2),
}


# Deterministic per-flag explanations for the QC flag chips. Avoids the
# situation where a chip just says ``low_plddt`` with no context for the
# non-expert reader.
_QC_FLAG_EXPLAINERS: dict[str, str] = {
    "low_plddt": "Mean pLDDT is below the confident threshold — interpret the structure with caution.",
    "high_instability_index": "Instability index >40 suggests the sequence may not be stable in vitro.",
    "high_polybasic": "Polybasic stretches above the cut-off raise the chance of nonspecific binding.",
    "aggregation_prone": "Aggregation-prone score above threshold — solubility may be a concern.",
    "membrane_signal": "Predicted membrane / signal-peptide features that complicate cytoplasmic expression.",
    "extreme_pi": "Isoelectric point outside the soluble-protein band (≈4-9).",
}


# --------------------------------------------------------------- generic metric registry
#
# Metric metadata (registry, dual-scale set, per-tool explainers, helpers)
# lives in :mod:`.metrics` so render.py and plots.py share one source of
# truth. The renderer keeps short local aliases below so the rest of this
# file's call sites read the same as before.

# Keys that look like nested metric dicts inside an adapter's output. The
# generic widget recurses one level into these so a tool emitting
# ``confidence_scores: {plddt: 0.8, ptm: 0.4}`` still surfaces both as cards.
_NESTED_METRIC_KEYS = ("confidence_scores", "metrics", "scores")


# Footer-only keys: shown as kv rows under the card grid rather than cards
# themselves (they're context, not headline metrics).
_FOOTER_METRIC_KEYS = (
    "model_name", "model", "tool", "version", "device", "seed",
    "seed_used", "seeds_used",
    "num_residues", "length", "n_residues", "n_atoms",
    "runtime_ms", "runtime_s", "elapsed_ms", "elapsed_s",
)


def _resolve_metric_spec(canonical: str) -> tuple[str, str, float, float] | None:
    """Resolve a gauge spec ``(label, explainer, lo, hi)`` for a metric key.

    Tries the explicit registry first, then a small set of name-based
    families so suffix / variant keys (``helix_fraction``, ``tm_score_*``,
    ``ipsae_max``, ``delta_g_kcal_mol`` …) still get a properly-scaled donut
    instead of falling back to a range-less frame ring.
    """
    spec = _METRIC_REGISTRY.get(canonical)
    if spec is not None:
        return spec
    # Unit-suffixed variants map back onto the base metric.
    if canonical.endswith("_kcal_mol"):
        base = _METRIC_REGISTRY.get(canonical[: -len("_kcal_mol")])
        if base is not None:
            return base
    pretty = _pretty_metric_name(canonical)
    # Name-based families — metrics that are 0-1 scores by construction.
    if canonical.endswith("_fraction"):
        return (pretty, "Fraction of residues in this category (0-1 scale).", 0.0, 1.0)
    if canonical.startswith("tm_score") or canonical == "tmscore":
        return (pretty, "Template-modeling score (0-1). >0.5 implies a similar fold.", 0.0, 1.0)
    if canonical.startswith("ipsae"):
        return (pretty, "Interface predicted symmetry-aware score (0-1 scale).", 0.0, 1.0)
    if canonical == "lddt" or canonical.endswith("_lddt"):
        return (pretty, "Local Distance Difference Test — structural accuracy (0-1).", 0.0, 1.0)
    if canonical == "seq_identity" or canonical.endswith("_identity"):
        return (pretty, "Sequence identity — fraction of identical aligned residues (0-1).", 0.0, 1.0)
    return None


def _auto_score_card(key: str, value: Any) -> str | None:
    """Build a CELLKEY-style donut card for *any* numeric metric.

    Returns ``None`` when the value isn't numeric (so the caller can decide
    to render it as a footer row instead). Every numeric metric gets a
    circular card: registry / family metrics get a proportionally-filled
    gauge, and metrics with no defensible range get a neutral "frame" ring
    so the card grid stays visually uniform.
    """
    num = _num(value)
    if num is None:
        return None
    canonical = key.lower()
    quality = _quality_class(canonical, num)
    spec = _resolve_metric_spec(canonical)
    if spec is None:
        # No defensible range — render a range-less frame ring so the metric
        # still reads as a circular card alongside the gauged ones.
        return _score_card(
            _pretty_metric_name(canonical), _fmt(num),
            "Tool-emitted metric. No reference range registered — value shown as-is.",
            quality,
            gauge={"value": num, "frame": True},
        )
    label, explainer, lo, hi = spec
    # Autoscale gauges that may arrive on either 0-1 or 0-100.
    if canonical in _DUAL_SCALE_METRICS and num <= 1.5 and hi >= 50.0:
        hi = 1.0
    return _score_card(
        label, _fmt(num), explainer, quality,
        gauge={"value": num, "lo": lo, "hi": hi},
    )


def _per_tool_widget(node: WorkflowNodeSummary) -> str:
    """Dispatch to a per-tool renderer; tools without a specialised widget
    fall through to a compact metric summary, and tools whose specialised
    widget produces no content (e.g. an adapter that emits only a sequence
    string) fall through further to the output-shape inventory. The chain
    guarantees every node block has *something* readable, never a blank.
    """
    output = node.output if isinstance(node.output, dict) else {}
    tool = (node.tool or "").lower()
    specialised: Callable[[dict[str, Any]], str] | None = {
        "protein_qc": _qc_widget,
        "dssp": _dssp_widget,
        "ipsae": _metrics_only_widget,
        "mmseqs2_gpu": _metrics_only_widget,
        "tmalign": _metrics_only_widget,
        "vina": _vina_widget,
        "template_select": _template_select_widget,
        "pxmeter": _pxmeter_widget,
        "go_sequences": _sequence_retrieval_widget,
        "interpro_sequences": _sequence_retrieval_widget,
        "literature_search": _literature_search_widget,
        "structure_analysis": _structure_analysis_widget,
        "sequence_analysis": _sequence_analysis_widget,
        "sequence_edit": _sequence_edit_widget,
        "rank_designs": _rank_designs_widget,
        "boltz2_affinity": _boltz2_widget,
        "protenix_v2": _protenix_widget,
        "proteinmpnn": functools.partial(_mpnn_widget, tool="proteinmpnn"),
        "ligandmpnn": functools.partial(_mpnn_widget, tool="ligandmpnn"),
        "bindcraft": functools.partial(_design_widget, tool="bindcraft"),
        "pxdesign": functools.partial(_design_widget, tool="pxdesign"),
        "rfdiffusion": functools.partial(_design_widget, tool="rfdiffusion"),
        "rfdiffusion3": functools.partial(_design_widget, tool="rfdiffusion3"),
        "rfpeptides": functools.partial(_design_widget, tool="rfpeptides"),
        "boltzgen": functools.partial(_design_widget, tool="boltzgen"),
        "proteina_complexa": functools.partial(_design_widget, tool="proteina_complexa"),
        "esm2": functools.partial(_esm_widget, tool="esm2"),
        "esm3": functools.partial(_esm_widget, tool="esm3"),
    }.get(tool)
    if specialised is not None:
        html_str = specialised(output)
        if html_str:
            return html_str
        # Specialised widget had nothing to show — fall through to the
        # generic registry-driven widget, then to the output-shape panel.
    generic = _generic_metric_widget(output)
    if generic:
        return generic
    return _output_shape_widget(output)


def _qc_widget(output: dict[str, Any]) -> str:
    """Protein QC widget — explainer + per-metric quality cards + flag chips.

    All thresholds and explanations are *fixed* (see ``_QC_METRIC_SPECS``
    above), not LLM-generated, so the widget is byte-reproducible for the
    same input.
    """
    metrics = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}
    flags = output.get("flags") or []
    cards: list[str] = []
    for key, name, explanation, qual_fn in _QC_METRIC_SPECS:
        if key not in metrics:
            continue
        value = metrics.get(key)
        num = _num(value)
        if num is None:
            continue
        quality = qual_fn(num)
        if quality not in {"good", "ok", "poor", "neutral"}:
            quality = "neutral"
        gauge_range = _QC_GAUGE_RANGES.get(key)
        gauge = (
            {"value": num, "lo": gauge_range[0], "hi": gauge_range[1]}
            if gauge_range else None
        )
        cards.append(
            _score_card(name, _fmt(num), explanation, quality, gauge=gauge)
        )
    cards_html = (
        _card_grid(cards)
        if cards
        else '<p class="empty">No QC metrics emitted.</p>'
    )
    flags_html = ""
    if flags:
        chips: list[str] = []
        for flag in flags:
            key = str(flag).strip().lower()
            explainer = _QC_FLAG_EXPLAINERS.get(
                key, "QC flag — see the protein_qc adapter documentation."
            )
            chips.append(
                f'<div class="qc-flag" title="{_attr(explainer)}">'
                f'<strong>⚠️ {_h(flag)}</strong>'
                f'<span class="qc-flag-explain">{_h(explainer)}</span></div>'
            )
        flags_html = f'<div class="qc-flag-list">{"".join(chips)}</div>'
    explainer = (
        '<p class="widget-explainer">'
        "<strong>QC</strong> aggregates fast, deterministic biochemical checks computed from the sequence + predicted "
        "structure: mean pLDDT, instability index, GRAVY, pI, polybasic motifs, and pTM/ipTM when available. "
        "Each card below shows the value, the threshold-based quality (green / amber / red), and what it means. "
        "Flags below the cards summarise which thresholds were tripped."
        "</p>"
    )
    return (
        '<div class="widget widget-qc">'
        '<h4 class="widget-title">Quality Control</h4>'
        f"{explainer}{cards_html}{flags_html}</div>"
    )


def _read_fasta_file(path: Any, *, cap: int = 60) -> list[tuple[str, str]]:
    """Read a FASTA file -> list of ``(header, sequence)`` tuples (capped)."""
    if not isinstance(path, str):
        return []
    p = Path(path)
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq)))
                if len(records) >= cap:
                    return records
            header = line[1:].strip()
            seq = []
        elif header is not None:
            seq.append(line.strip())
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def _read_metadata_jsonl(path: Any) -> dict[str, dict[str, Any]]:
    """Read a per-line JSON metadata file -> ``{accession: record}``."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(path, str):
        return out
    p = Path(path)
    if not p.is_file():
        return out
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("accession"):
            out[str(obj["accession"])] = obj
    return out


def _cross_overlap_positions(sequences: list[str], k: int = 10) -> list[set[int]]:
    """Per sequence, the residue indices covered by an exact k-mer that also
    occurs in a *different* retrieved sequence.

    Only cross-sequence overlap is flagged — a stretch identical between two
    of the retrieved proteins. ``k`` is 10: long enough that an exact match
    between distinct proteins is very unlikely to arise by chance, so the
    shaded region instead marks a span plausibly under shared functional
    constraint (roughly the length of a short conserved motif / one
    secondary-structure element). Stretches that merely repeat *within* a
    single sequence are deliberately ignored — they carry no cross-sequence
    functional signal.
    """
    masks: list[set[int]] = [set() for _ in sequences]
    if len(sequences) < 2:
        return masks
    owners: dict[str, set[int]] = {}
    for si, s in enumerate(sequences):
        for i in range(len(s) - k + 1):
            owners.setdefault(s[i : i + k], set()).add(si)
    shared = {km for km, idxs in owners.items() if len(idxs) >= 2}
    if not shared:
        return masks
    for si, s in enumerate(sequences):
        for i in range(len(s) - k + 1):
            if s[i : i + k] in shared:
                masks[si].update(range(i, i + k))
    return masks


def _render_residues_html(seq: str, overlap: set[int]) -> str:
    """Render a residue string, wrapping cross-sequence overlap runs in a span."""
    parts: list[str] = []
    buf: list[str] = []
    run = False

    def _flush(highlight: bool) -> None:
        if not buf:
            return
        chunk = _h("".join(buf))
        parts.append(
            f'<span class="seqret-hl">{chunk}</span>' if highlight else chunk
        )
        buf.clear()

    for i, ch in enumerate(seq):
        hl = i in overlap
        if hl != run:
            _flush(run)
            run = hl
        buf.append(ch)
    _flush(run)
    return "".join(parts)


def _seq_ruler(n: int) -> str:
    """Monospace column ruler — a position number ending on every 10th
    column, padded with spaces so it lines up under the residue track."""
    chars = [" "] * n
    for pos in range(10, n + 1, 10):
        label = str(pos)
        start = pos - len(label)
        for j, ch in enumerate(label):
            if 0 <= start + j < n:
                chars[start + j] = ch
    return _h("".join(chars))


def _sequence_retrieval_widget(output: dict[str, Any]) -> str:
    """Widget for go_sequences / interpro_sequences — a professional view of
    the retrieved UniProt sequences.

    Shows the retrieval query + summary stats, then one card per kept
    sequence with its accession, length and the residue string; stretches
    shared across the retrieved set are lightly shaded. When the FASTA file
    is unavailable (e.g. the workspace was cleaned up) it degrades to the
    summary plus a short note.
    """
    if not output:
        return ""
    summary = output.get("summary") if isinstance(output.get("summary"), dict) else {}
    query_id = output.get("go_id") or output.get("interpro_id") or ""
    query_label = output.get("go_label")
    is_go = bool(output.get("go_id"))

    kept = _num(output.get("kept_sequences"))
    candidates = (
        _num(summary.get("candidate_accessions"))
        or _num(output.get("resolved_uniprot_accessions"))
        or _num(output.get("retrieved_accessions"))
    )
    runtime = _num(output.get("runtime_ms"))
    l_min, l_med, l_max = (
        _num(summary.get("length_min")),
        _num(summary.get("length_median")),
        _num(summary.get("length_max")),
    )

    stats: list[str] = []
    if kept is not None:
        stats.append(f'<span class="seqret-stat"><b>{int(kept)}</b> sequences kept</span>')
    if candidates is not None:
        stats.append(
            f'<span class="seqret-stat"><b>{int(candidates)}</b> candidates scanned</span>'
        )
    if l_min is not None and l_max is not None:
        med = f" · median {int(l_med)}" if l_med is not None else ""
        stats.append(
            f'<span class="seqret-stat">length <b>{int(l_min)}-{int(l_max)}</b>{med} aa</span>'
        )
    if runtime is not None:
        stats.append(f'<span class="seqret-stat"><b>{int(runtime)}</b> ms</span>')

    query_html = ""
    if query_id:
        label = (
            f'<span class="seqret-query-label">{_h(str(query_label))}</span>'
            if query_label else ""
        )
        query_html = (
            f'<div class="seqret-query"><span class="seqret-query-id">'
            f'{_h(str(query_id))}</span>{label}</div>'
        )
    summary_html = (
        '<div class="seqret-summary">'
        f'{query_html}<div class="seqret-stats">{"".join(stats)}</div>'
        '</div>'
    )

    records = _read_fasta_file(output.get("fasta_path"))
    metadata = _read_metadata_jsonl(output.get("metadata_jsonl"))
    list_html = ""
    note = ""
    if records:
        seqs = [s for _, s in records]
        masks = (
            _cross_overlap_positions(seqs)
            if sum(len(s) for s in seqs) <= 200_000
            else [set() for _ in seqs]
        )
        max_len = max(len(s) for s in seqs)
        # One MSA-style row per sequence: a pinned accession label plus a
        # fixed-pitch residue track; every row shares a single horizontal
        # scrollbar and a column ruler so long sequences never overflow.
        rows: list[str] = []
        for (header, seq), mask in zip(records, masks, strict=False):
            acc = header.split()[0] if header else "?"
            meta = metadata.get(acc, {})
            length = meta.get("length") or len(seq)
            dot = (
                '<span class="seqret-rev" title="reviewed entry"></span>'
                if meta.get("reviewed") is True else ""
            )
            label = (
                f'<span class="seqret-row-label">{dot}'
                f'<span class="seqret-row-acc">{_h(acc)}</span>'
                f'<span class="seqret-row-len">{int(length)} aa</span></span>'
            )
            rows.append(
                f'<div class="seqret-row">{label}'
                f'<code class="seqret-row-seq">'
                f"{_render_residues_html(seq, mask)}</code></div>"
            )
        rows.append(
            '<div class="seqret-row seqret-row-ruler">'
            '<span class="seqret-row-label"></span>'
            f'<code class="seqret-row-seq">{_seq_ruler(max_len)}</code></div>'
        )
        legend = (
            '<div class="seqret-legend"><span>'
            '<span class="seqret-legend-pip seqret-hl"></span>'
            'cross-sequence overlap — a stretch of ≥10 residues identical '
            'in another retrieved sequence</span></div>'
            if any(masks) else ""
        )
        list_html = (
            f'<div class="seqret-align-head">Retrieved sequences · {len(records)} shown</div>'
            f'<div class="seqret-align-scroll">{"".join(rows)}</div>{legend}'
        )
    elif kept == 0:
        dropped = summary.get("dropped") if isinstance(summary.get("dropped"), dict) else {}
        reason = (
            " — " + ", ".join(f"{v} {str(k).replace('_', ' ')}" for k, v in dropped.items())
            if dropped else ""
        )
        note = f'<p class="note">No sequences passed the retrieval filters{reason}.</p>'
    else:
        note = (
            '<p class="note">The retrieved FASTA is not embedded in this report. '
            "Re-run <code>vpa report</code> with the source workspace present to list "
            "the sequences here.</p>"
        )

    source = "GO term" if is_go else "InterPro family"
    explainer = (
        '<p class="widget-explainer">'
        f"Retrieves UniProt protein sequences linked to the queried {source}, applies "
        "review / temporal / taxonomy filters, and writes a FASTA set. Each kept sequence "
        "is listed below; yellow-shaded residues mark a cross-sequence overlap — a stretch "
        "of at least 10 residues that is identical in another retrieved sequence, flagging "
        "a region plausibly under shared functional constraint. Stretches that merely "
        "repeat within a single sequence are not flagged."
        "</p>"
    )
    return (
        '<div class="widget widget-seqret">'
        '<h4 class="widget-title">Retrieved sequences</h4>'
        f"{explainer}{summary_html}{list_html}{note}"
        "</div>"
    )


# DSSP 8-state secondary-structure codes -> 3-state buckets for the SS bar.
# H/G/I = helices, E/B = strands; turn / bend / polyproline / coil -> loop.
_DSSP_SS_3STATE: dict[str, str] = {
    "H": "helix", "G": "helix", "I": "helix",
    "E": "sheet", "B": "sheet",
}


def _dssp_ss_strip(chain_label: str, first_resi: int, ss_string: str) -> str:
    """One white strip-block: a colour-coded SS bar + residue-index ruler.

    Mirrors the structure_analysis strip styling so the two widgets read
    consistently; wrapped in a single horizontally-scrolling container.
    """
    n = len(ss_string)
    if n == 0:
        return ""
    last = first_resi + n - 1
    ss_cells: list[str] = []
    ruler_cells: list[str] = []
    for i, code in enumerate(ss_string):
        ri = first_resi + i
        bucket = _DSSP_SS_3STATE.get(code.upper(), "loop")
        tip = f"{ri} · {bucket} · DSSP '{code}'"
        ss_cells.append(
            f'<span class="strip-cell ss-{bucket}" title="{_attr(tip)}"></span>'
        )
        show = ri in (first_resi, last) or (ri % 10 == 0 and abs(ri - last) > 2)
        ruler_cells.append(
            f'<span class="strip-cell strip-ruler-cell">{ri if show else "&nbsp;"}</span>'
        )
    head = (
        f"Chain {_h(chain_label)} · residues {first_resi}-{last}"
        if chain_label else f"Residues {first_resi}-{last}"
    )
    return (
        '<div class="strip-block">'
        f'<div class="strip-head">{head}</div>'
        '<div class="strip-scroll">'
        '<div class="strip-row"><span class="strip-label">SS</span>'
        f'<div class="strip-cells">{"".join(ss_cells)}</div></div>'
        '<div class="strip-row strip-ruler"><span class="strip-label"></span>'
        f'<div class="strip-cells">{"".join(ruler_cells)}</div></div>'
        '</div>'
        '<div class="strip-legend">'
        '<span class="legend-pip ss-helix"></span>helix'
        '<span class="legend-pip ss-sheet"></span>sheet'
        '<span class="legend-pip ss-loop"></span>loop'
        '</div>'
        '</div>'
    )


def _format_position_list(positions: list[Any], *, cap: int = 40) -> str:
    """Render a sorted residue-position list, collapsing consecutive runs
    into ``start-end`` ranges (e.g. ``[5,6,7,9] -> "5-7, 9"``)."""
    nums = sorted(
        {int(p) for p in positions if isinstance(p, (int, float)) and not isinstance(p, bool)}
    )
    if not nums:
        return ""
    runs: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    runs.append(str(start) if start == prev else f"{start}-{prev}")
    if len(runs) > cap:
        return ", ".join(runs[:cap]) + f", ... (+{len(runs) - cap} more)"
    return ", ".join(runs)


def _selected_positions_note(scope: Any, warnings: Any = None) -> str:
    """Context callout for tools run with an explicit residue-position
    selection. When a tool annotates only a requested subset of positions,
    the report otherwise shows just a handful of residues with no hint of
    *why* — this note makes the input selection explicit and, when the tool
    warned that some requested positions were absent, names them."""
    if not isinstance(scope, dict):
        return ""
    if scope.get("annotation_scope") != "requested_positions":
        return ""
    requested = scope.get("requested_positions")
    if not isinstance(requested, list) or not requested:
        return ""
    pos_txt = _format_position_list(requested)
    if not pos_txt:
        return ""
    n_req = len(requested)
    annotated = scope.get("annotated_residue_count")
    summary_n = scope.get("summary_residue_count")

    # Requested positions absent from the parsed structure — mined from the
    # tool's own warning text so the reader sees exactly which ones dropped.
    missing: list[int] = []
    if isinstance(warnings, list):
        for w in warnings:
            ws = str(w)
            if "missing from parsed structure" in ws.lower():
                tail = ws.rsplit(":", 1)[-1]
                missing.extend(int(n) for n in re.findall(r"\d+", tail))
    missing = sorted(set(missing))
    missing_txt = _format_position_list(missing)

    detail = (
        f"The per-residue annotation below is restricted to the {n_req} "
        f"position{'s' if n_req != 1 else ''} requested as an input parameter."
    )
    if missing_txt:
        noun = "Position" if len(missing) == 1 else "Positions"
        verb = "is" if len(missing) == 1 else "are"
        detail += f" {noun} {missing_txt} {verb} absent from the parsed structure"
        if isinstance(annotated, int):
            detail += f", so {annotated} of {n_req} could be annotated"
        detail += "."
    elif isinstance(annotated, int) and annotated != n_req:
        detail += (
            f" Only {annotated} of {n_req} "
            f"{'was' if annotated == 1 else 'were'} present in the structure."
        )
    if isinstance(summary_n, int) and summary_n:
        detail += f" The summary cards above still cover all {summary_n} residues."

    missing_chip = (
        f'<span class="scope-note-missing">not in structure: {_h(missing_txt)}</span>'
        if missing_txt else ""
    )
    return (
        '<div class="scope-note">'
        '<span class="scope-note-label">Input-selected residues</span>'
        f'<span class="scope-note-pos">{_h(pos_txt)}</span>'
        f"{missing_chip}"
        f'<p class="scope-note-detail">{_h(detail)}</p>'
        "</div>"
    )


def _fmt_evalue(value: Any) -> str:
    """Compact E-value formatting — scientific for the tiny/huge magnitudes
    typical of homology search, 3 sig figs otherwise."""
    e = _num(value)
    if e is None:
        return "—"
    if e == 0:
        return "0"
    if e < 1e-3 or e >= 1e4:
        return f"{e:.2e}"
    return f"{e:.3g}"


def _template_card(rank: int, t: dict[str, Any]) -> str:
    """One ranked template panel — header, identity / coverage bars and the
    E-value / length chips."""
    tid = str(t.get("template_id") or t.get("description") or f"template_{rank}")
    desc = str(t.get("description") or "")
    badges: list[str] = []
    for key in ("source", "release_date"):
        val = t.get(key)
        if val:
            badges.append(f'<span class="tmpl-badge">{_h(str(val))}</span>')
    badge_html = (
        f'<span class="tmpl-badges">{"".join(badges)}</span>' if badges else ""
    )
    desc_html = (
        f'<span class="tmpl-desc">{_h(desc)}</span>'
        if desc and desc != tid else ""
    )
    head = (
        '<div class="tmpl-card-head">'
        f'<span class="tmpl-rank">#{rank}</span>'
        f'<span class="tmpl-id">{_h(tid)}</span>'
        f"{desc_html}{badge_html}</div>"
    )

    bars: list[str] = []
    for label, frac in (("Identity", _num(t.get("identity"))),
                        ("Coverage", _num(t.get("coverage")))):
        if frac is None:
            continue
        pct = max(0.0, min(100.0, frac * 100.0))
        quality = "good" if pct >= 70 else ("ok" if pct >= 40 else "poor")
        colour = _QUALITY_COLOR[quality]
        bars.append(
            '<div class="tmpl-bar-row">'
            f'<span class="tmpl-bar-label">{label}</span>'
            '<div class="tmpl-bar-track">'
            f'<i style="width:{pct:.1f}%;background:{colour}"></i></div>'
            f'<span class="tmpl-bar-val">{pct:.1f}%</span></div>'
        )
    chips: list[str] = []
    if t.get("evalue") is not None:
        chips.append(
            f'<span class="tmpl-chip">E-value <b>{_h(_fmt_evalue(t.get("evalue")))}</b></span>'
        )
    if t.get("length") is not None:
        chips.append(
            f'<span class="tmpl-chip">Length <b>{_h(str(t.get("length")))} aa</b></span>'
        )
    chip_html = f'<div class="tmpl-chips">{"".join(chips)}</div>' if chips else ""
    metrics = (
        f'<div class="tmpl-metrics">{"".join(bars)}{chip_html}</div>'
        if bars or chip_html else ""
    )
    return f'<div class="tmpl-card">{head}{metrics}</div>'


def _template_sequences_html(
    templates: list[dict[str, Any]], seqs: dict[str, str]
) -> str:
    """MSA-style comparison strip — every template sequence stacked in one
    horizontally-scrollable container with a shared position ruler."""
    present: list[tuple[dict[str, Any], str]] = []
    for t in templates:
        tid = str(t.get("template_id") or "")
        seq = seqs.get(tid) or seqs.get(str(t.get("description") or ""))
        if seq:
            present.append((t, seq))
    if not present:
        return ""
    max_len = max(len(s) for _, s in present)
    rows: list[str] = []
    for t, seq in present:
        tid = str(t.get("template_id") or t.get("description") or "?")
        ident = _num(t.get("identity"))
        badge = (
            f'<span class="tmpl-seq-id">{ident * 100:.0f}% id</span>'
            if ident is not None else ""
        )
        rows.append(
            '<div class="tmpl-seq-row">'
            '<span class="tmpl-seq-label">'
            f'<span class="tmpl-seq-name">{_h(tid)}</span>{badge}</span>'
            f'<code class="tmpl-seq-track">{_h(seq)}</code></div>'
        )
    rows.append(
        '<div class="tmpl-seq-row tmpl-seq-ruler">'
        '<span class="tmpl-seq-label"></span>'
        f'<code class="tmpl-seq-track">{_seq_ruler(max_len)}</code></div>'
    )
    return (
        '<h5 class="widget-subhead">Template sequences</h5>'
        f'<div class="tmpl-seq-scroll">{"".join(rows)}</div>'
    )


def _template_select_widget(output: dict[str, Any]) -> str:
    """template_select widget — ranked modelling templates with their
    alignment metrics (identity / coverage / E-value) plus a side-by-side
    comparison of the template sequences parsed from the FASTA artifact."""
    if not output:
        return ""
    templates = output.get("top_templates")
    if not isinstance(templates, list) or not templates:
        loaded = _load_json_artifact(output.get("templates_json"))
        if isinstance(loaded, list):
            templates = loaded
    templates = [t for t in (templates or []) if isinstance(t, dict)]

    explainer = (
        '<p class="widget-explainer">'
        "<strong>Template select</strong> searches for known structures / "
        "sequences homologous to the query and ranks them as modelling "
        "templates. Each card carries the alignment metrics — sequence "
        "identity, query coverage and E-value; the strip below compares the "
        "template sequences side by side."
        "</p>"
    )
    if not templates:
        return (
            '<div class="widget widget-tmpl">'
            '<h4 class="widget-title">Template selection</h4>'
            f'{explainer}<p class="empty">No templates were selected.</p></div>'
        )

    meta_bits: list[str] = []
    if output.get("engine"):
        meta_bits.append(f'{output["engine"]} engine')
    if output.get("query_kind"):
        meta_bits.append(f'{output["query_kind"]} query')
    meta_bits.append(
        f"{len(templates)} template{'s' if len(templates) != 1 else ''}"
    )
    rt = _num(output.get("runtime_ms"))
    if rt is not None:
        meta_bits.append(_duration(int(rt)))
    meta = f'<p class="note">{_h(" · ".join(meta_bits))}</p>'

    # Template sequences keyed by FASTA header (the template id).
    seqs: dict[str, str] = {}
    arts = output.get("artifacts") if isinstance(output.get("artifacts"), dict) else {}
    fasta_path = arts.get("template_sequences_fasta") if isinstance(arts, dict) else None
    for header, seq in _read_fasta_file(fasta_path):
        key = header.split()[0] if header else ""
        if key:
            seqs[key] = seq

    cards = "".join(_template_card(i + 1, t) for i, t in enumerate(templates))
    seq_html = _template_sequences_html(templates, seqs)
    note = ""
    if not seq_html and isinstance(fasta_path, str):
        note = (
            '<p class="note">Template sequences are not embedded in this '
            "report.</p>"
        )
    return (
        '<div class="widget widget-tmpl">'
        '<h4 class="widget-title">Template selection</h4>'
        f'{explainer}{meta}<div class="tmpl-list">{cards}</div>{seq_html}{note}</div>'
    )


def _dssp_widget(output: dict[str, Any]) -> str:
    """DSSP widget — helix / sheet / coil gauges plus a per-residue SS bar.

    The SS bar colours every residue by secondary-structure type with a
    residue ruler underneath. Multi-chain DSSP output
    (``secondary_structure_by_chain``) renders one white sub-block per
    chain; each strip scrolls left/right for long sequences.
    """
    if not output:
        return ""
    # Fraction gauges — kept; the bar is added alongside, not instead.
    cards: list[str] = []
    for key in ("helix_fraction", "sheet_fraction", "coil_fraction"):
        if key in output:
            card = _auto_score_card(key, output[key])
            if card:
                cards.append(card)
    cards_html = _card_grid(cards)

    # Context footer (residue count, runtime) — sits between the gauges and
    # the SS bar so the bar lands directly under the headline numbers.
    footer_rows: list[str] = []
    for key, value in output.items():
        if (
            key in _FOOTER_METRIC_KEYS
            and value is not None
            and not isinstance(value, (list, dict, bool))
        ):
            footer_rows.append(_kv_row(_pretty_metric_name(key.lower()), str(value)))
    footer_html = (
        f'<table class="kv-table"><tbody>{"".join(footer_rows)}</tbody></table>'
        if footer_rows else ""
    )

    # SS bar(s): one strip-block per chain when chain-resolved data exists,
    # otherwise a single strip from the flat secondary-structure string.
    strips: list[str] = []
    by_chain = output.get("secondary_structure_by_chain")
    if isinstance(by_chain, list) and by_chain:
        for entry in by_chain:
            if not isinstance(entry, dict):
                continue
            ss = str(entry.get("secondary_structure") or "")
            if ss:
                strips.append(_dssp_ss_strip(str(entry.get("chain_id") or "?"), 1, ss))
    else:
        ss_string = output.get("secondary_structure_string")
        if isinstance(ss_string, str) and ss_string:
            strips.append(_dssp_ss_strip("", 1, ss_string))

    if not cards_html and not strips:
        return ""

    explainer = (
        '<p class="widget-explainer">'
        "<strong>DSSP</strong> assigns a secondary-structure state to every residue from the "
        "backbone hydrogen-bond pattern. The cards summarise the helix / sheet / coil split; the "
        "bar below colours each residue by SS type along the chain."
        "</p>"
    )
    scope_note = _selected_positions_note(
        output.get("annotation_scope"), output.get("warnings")
    )
    return (
        '<div class="widget widget-dssp">'
        '<h4 class="widget-title">Secondary structure (DSSP)</h4>'
        f"{explainer}{scope_note}{cards_html}{footer_html}{''.join(strips)}"
        "</div>"
    )


def _structure_analysis_widget(output: dict[str, Any]) -> str:
    """Rich Structure-analysis widget.

    Pulls together: burial / SS summary cards (from output.summary or output
    itself), per-residue secondary-structure + RSA strips parsed from the
    DSSP / annotations TSV artifacts, liability markers placed on the
    sequence ruler, interface counts, and downloads at the bottom.
    """
    artifact_paths = output.get("artifacts") if isinstance(output.get("artifacts"), dict) else {}
    summary_path = artifact_paths.get("summary_json") if isinstance(artifact_paths, dict) else None
    summary_inline = output.get("summary") if isinstance(output.get("summary"), dict) else None
    summary_data = _load_json_artifact(summary_path) if isinstance(summary_path, str) else None
    summary: dict[str, Any] = {}
    if isinstance(summary_inline, dict):
        summary.update(summary_inline)
    if isinstance(summary_data, dict):
        # Inline output wins for keys both sources define; fall back to file.
        for k, v in summary_data.items():
            summary.setdefault(k, v)
        # Also surface top-level keys from output itself if missing.
    # ``output`` itself sometimes carries the canonical summary at the top
    # level (e.g. burial / interface dicts) — merge those keys in last so
    # they win over the file's older snapshot.
    for key in ("burial", "interface", "chains", "liabilities", "annotation_scope"):
        if key not in summary and key in output:
            summary[key] = output[key]

    burial = summary.get("burial") if isinstance(summary.get("burial"), dict) else {}
    interface = summary.get("interface") if isinstance(summary.get("interface"), dict) else {}
    chains = summary.get("chains") if isinstance(summary.get("chains"), list) else []
    liabilities = summary.get("liabilities") if isinstance(summary.get("liabilities"), list) else []
    scope = summary.get("annotation_scope") if isinstance(summary.get("annotation_scope"), dict) else {}
    annotated = scope.get("annotated_residue_count") if isinstance(scope, dict) else None

    # Per-residue parsed rows (chain, resi, aa, rsa, burial_class, ss, neighbours)
    annotations_rows = (
        _parse_residue_annotations_tsv(artifact_paths.get("residue_annotations_tsv"))
        if isinstance(artifact_paths, dict)
        else []
    )
    # If annotations TSV missing, fall back to DSSP-only rows so the SS strip
    # still renders.
    if not annotations_rows:
        dssp_rows = (
            _parse_dssp_tsv(artifact_paths.get("dssp_tsv"))
            if isinstance(artifact_paths, dict)
            else []
        )
        annotations_rows = [
            {"chain_id": r[0], "resseq": r[1], "aa": r[2], "secondary_structure": r[3]}
            for r in dssp_rows
        ]

    ss_counts: dict[str, int] = {}
    for r in annotations_rows:
        s = (r.get("secondary_structure") or "").lower()
        if s:
            ss_counts[s] = ss_counts.get(s, 0) + 1
    n_total = sum(ss_counts.values()) or (annotated or 1)

    # ---------- top metric cards ----------
    cards: list[str] = []
    helix_pct = (ss_counts.get("helix", 0) / n_total) * 100 if n_total else 0
    sheet_pct = (ss_counts.get("sheet", 0) / n_total) * 100 if n_total else 0
    loop_pct = (ss_counts.get("loop", 0) / n_total) * 100 if n_total else 0
    if n_total:
        cards.append(_score_card(
            "% helix", f"{helix_pct:.0f}%",
            "Fraction of residues classified as alpha-helix by DSSP / pydssp.",
            "neutral",
            gauge={"value": helix_pct, "lo": 0.0, "hi": 100.0},
        ))
        cards.append(_score_card(
            "% sheet", f"{sheet_pct:.0f}%",
            "Fraction of residues classified as beta-sheet / strand.",
            "neutral",
            gauge={"value": sheet_pct, "lo": 0.0, "hi": 100.0},
        ))
        cards.append(_score_card(
            "% loop", f"{loop_pct:.0f}%",
            "Fraction of residues classified as loop / coil.",
            "neutral",
            gauge={"value": loop_pct, "lo": 0.0, "hi": 100.0},
        ))
    if isinstance(burial, dict) and burial:
        buried = int(burial.get("buried") or 0)
        exposed = int(burial.get("exposed") or 0)
        intermediate = int(burial.get("intermediate") or 0)
        burial_total = buried + exposed + intermediate
        if burial_total:
            cards.append(_score_card(
                "Buried residues", f"{buried} / {burial_total}",
                "Residues with relative solvent accessibility <0.25 (DSSP convention).",
                "neutral",
                gauge={"value": buried, "lo": 0.0, "hi": float(burial_total)},
            ))
    if isinstance(interface, dict) and interface:
        ligand_contacts = interface.get("ligand_contact_count")
        if ligand_contacts is not None:
            n_lc = _num(ligand_contacts) or 0.0
            cards.append(_score_card(
                "Ligand contacts", _fmt(ligand_contacts),
                "Heavy-atom contacts (<5 A) between the protein and any non-protein hetero residue.",
                "neutral",
                gauge={"value": n_lc, "lo": 0.0, "hi": max(20.0, n_lc * 2)},
            ))
        interface_residues = interface.get("interface_residue_count")
        if interface_residues:
            n_ir = _num(interface_residues) or 0.0
            cards.append(_score_card(
                "Interface residues", _fmt(interface_residues),
                "Residues with at least one inter-chain contact within 5 A.",
                "neutral",
                gauge={"value": n_ir, "lo": 0.0, "hi": max(20.0, n_ir * 2)},
            ))
    cards_html = (
        _card_grid(cards)
    )

    # ---------- sequence rulers (SS strip + burial strip) ----------
    ruler_html = _structure_ruler_html(annotations_rows, liabilities)

    # ---------- per-residue table (collapsible) ----------
    residue_table_html = _residue_annotations_table(annotations_rows)

    # ---------- downloads + empty-artifact notes ----------
    artifact_links = ""
    empty_artifact_fields: list[str] = []
    if isinstance(artifact_paths, dict) and artifact_paths:
        items = []
        for field, path in artifact_paths.items():
            if not isinstance(path, str):
                continue
            # Skip artifacts whose payload is the empty JSON literal
            # (``[]`` / ``{}``) — surface_patches_json downloads as exactly
            # this when no patches were detected. We replace that useless
            # download chip with a small "no X detected" note below.
            if _is_empty_json_artifact(path):
                empty_artifact_fields.append(field)
                continue
            items.append(_download_link(path, field))
        bits: list[str] = []
        if items:
            bits.append(
                '<div class="node-block-artifacts">'
                '<span class="artifact-section-label">Per-residue analyses</span>'
                f'<div class="artifact-row">{"".join(items)}</div></div>'
            )
        if empty_artifact_fields:
            label_map = {
                "surface_patches_json": "surface patches",
                "interface_contacts_tsv": "inter-chain contacts",
                "liabilities_json": "structural liabilities",
            }
            notes = [
                _h(label_map.get(field, field.replace("_", " ")))
                for field in empty_artifact_fields
            ]
            bits.append(
                '<p class="note">No '
                + ", ".join(notes)
                + " detected for this structure — the adapter ran cleanly but found nothing flag-worthy.</p>"
            )
        artifact_links = "".join(bits)

    explainer = (
        '<p class="widget-explainer">'
        "<strong>Structure analysis</strong> runs DSSP (or a pydssp / CA-geometry fallback) on the predicted "
        "structure, classifies secondary structure per residue, computes relative surface accessibility, and "
        "tabulates inter-chain / ligand contacts. The strip below shows SS and burial along the sequence; the "
        "table breaks every residue out for sorting and filtering."
        "</p>"
    )
    if not cards and not ruler_html and not residue_table_html:
        return (
            '<div class="widget widget-struct">'
            '<h4 class="widget-title">Structure analysis</h4>'
            f"{explainer}"
            '<p class="empty">Structure analysis emitted no per-residue data.</p>'
            f"{artifact_links}</div>"
        )

    chains_label = (
        f'<p class="note">Chains: {", ".join(_h(str(c)) for c in chains)}'
        + (f" · annotated {annotated} residues" if annotated else "")
        + "</p>"
    ) if chains else ""
    scope_note = _selected_positions_note(scope, output.get("warnings"))

    return (
        '<div class="widget widget-struct">'
        '<h4 class="widget-title">Structure analysis</h4>'
        f"{explainer}"
        f"{chains_label}{scope_note}"
        f"{cards_html}{ruler_html}{residue_table_html}{artifact_links}"
        "</div>"
    )


def _load_json_artifact(path: Any) -> Any:
    if not isinstance(path, str):
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _is_empty_json_artifact(path: Any) -> bool:
    """Return True if ``path`` is a JSON file that decodes to ``[]`` or ``{}``.

    Adapters often emit a JSON artifact even when their analysis produced
    no rows (e.g. ``surface_patches_json`` returning ``[]`` because no
    patches passed the threshold). A download chip pointing at that empty
    payload is useless noise, so the renderer filters it out.
    """
    if not isinstance(path, str):
        return False
    p = Path(path)
    if not p.is_file() or p.suffix.lower() != ".json":
        return False
    try:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return False
    return text in ("[]", "{}", "null", "")


def _parse_dssp_tsv(path: Any) -> list[tuple[str, int, str, str]]:
    if not isinstance(path, str):
        return []
    p = Path(path)
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[tuple[str, int, str, str]] = []
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    for raw in lines[1:]:
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        chain = parts[0]
        try:
            resseq = int(parts[1])
        except ValueError:
            continue
        aa = parts[3]
        ss = parts[4]
        rows.append((chain, resseq, aa, ss))
    return rows


def _parse_residue_annotations_tsv(path: Any) -> list[dict[str, Any]]:
    if not isinstance(path, str):
        return []
    p = Path(path)
    if not p.is_file():
        return []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows: list[dict[str, Any]] = []
    for raw in lines[1:]:
        parts = raw.rstrip("\n").split("\t")
        if len(parts) < len(header):
            continue
        row: dict[str, Any] = {}
        for i, key in enumerate(header):
            row[key] = parts[i]
        # Cast known numeric columns
        for k in ("resseq", "neighbor_count"):
            if k in row:
                try:
                    row[k] = int(row[k])
                except (ValueError, TypeError):
                    pass
        if "rsa" in row:
            try:
                row["rsa"] = float(row["rsa"])
            except (ValueError, TypeError):
                pass
        rows.append(row)
    return rows


def _structure_ruler_html(
    rows: list[dict[str, Any]],
    liabilities: list[Any] | None = None,
) -> str:
    """Render a per-residue SS + burial strip below the cards.

    One cell per residue, colored by SS (loop/helix/sheet) on the first row
    and by burial class (exposed/intermediate/buried) on the second. A
    liability ruler below marks positions with structural flags.
    """
    if not rows:
        return ""
    # Group by chain so multi-chain proteins render as separate rulers.
    chains_in_order: list[str] = []
    by_chain: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        c = str(r.get("chain_id") or "")
        if c not in by_chain:
            by_chain[c] = []
            chains_in_order.append(c)
        by_chain[c].append(r)

    # Build a position->liability-type map for marker rendering.
    liab_positions: dict[tuple[str, int], str] = {}
    for entry in liabilities or []:
        if not isinstance(entry, dict):
            continue
        chain = str(entry.get("chain_id") or "")
        positions = entry.get("positions")
        if isinstance(positions, list):
            for pos in positions:
                try:
                    liab_positions[(chain, int(pos))] = str(entry.get("type") or "liability")
                except (ValueError, TypeError):
                    continue

    blocks: list[str] = []
    for chain in chains_in_order:
        chain_rows = by_chain[chain]
        n = len(chain_rows)
        first_resseq = int(chain_rows[0].get("resseq") or 1)
        last_resseq = int(chain_rows[-1].get("resseq") or n)

        ss_cells: list[str] = []
        burial_cells: list[str] = []
        liab_cells: list[str] = []
        ruler_cells: list[str] = []
        for r in chain_rows:
            resseq = r.get("resseq")
            try:
                ri = int(resseq) if resseq is not None else 0
            except (ValueError, TypeError):
                ri = 0
            aa = (r.get("aa") or "?")
            ss = (r.get("secondary_structure") or "").lower()
            ss_class = "ss-helix" if ss == "helix" else ("ss-sheet" if ss == "sheet" else "ss-loop")
            burial = (r.get("burial_class") or "").lower()
            b_class = (
                "burial-buried" if burial == "buried"
                else "burial-intermediate" if burial == "intermediate"
                else "burial-exposed" if burial == "exposed"
                else "burial-na"
            )
            tip = f"{aa}{ri} · {ss or '?'} · {burial or 'n/a'}"
            ss_cells.append(f'<span class="strip-cell {ss_class}" title="{_attr(tip)}"></span>')
            burial_cells.append(f'<span class="strip-cell {b_class}" title="{_attr(tip)}"></span>')
            liab_kind = liab_positions.get((chain, ri))
            if liab_kind:
                liab_cells.append(
                    f'<span class="strip-cell strip-liab" '
                    f'title="{_attr(aa)}{ri} · {_attr(liab_kind)}"></span>'
                )
            else:
                liab_cells.append('<span class="strip-cell strip-empty"></span>')
            # Ruler labels: first, last, and every 10th residue — but skip a
            # 10-multiple when it is within 2 positions of the last residue
            # so e.g. "60" and "61" don't overlap on a 61-residue chain.
            show_label = False
            if ri == first_resseq or ri == last_resseq:
                show_label = True
            elif ri % 10 == 0 and abs(ri - last_resseq) > 2:
                show_label = True
            ruler_cells.append(
                f'<span class="strip-cell strip-ruler-cell">'
                f'{ri if show_label else "&nbsp;"}'
                "</span>"
            )

        blocks.append(
            f'<div class="strip-block">'
            f'<div class="strip-head">Chain {_h(chain) or "?"} · residues '
            f'{first_resseq}-{last_resseq}</div>'
            f'<div class="strip-scroll">'
            f'<div class="strip-row"><span class="strip-label">SS</span>'
            f'<div class="strip-cells">{"".join(ss_cells)}</div></div>'
            f'<div class="strip-row"><span class="strip-label">Burial</span>'
            f'<div class="strip-cells">{"".join(burial_cells)}</div></div>'
            + (
                f'<div class="strip-row"><span class="strip-label" '
                f'title="Structural liability candidates flagged by protein_qc / structure_analysis '
                f'(e.g. buried polar residues, aggregation-prone stretches). Hover a marker for the type.">'
                f'Liability</span>'
                f'<div class="strip-cells">{"".join(liab_cells)}</div></div>'
                if any(liab_positions.get((chain, int(r.get("resseq") or -1))) for r in chain_rows)
                else ""
            )
            + f'<div class="strip-row strip-ruler"><span class="strip-label"></span>'
            f'<div class="strip-cells">{"".join(ruler_cells)}</div></div>'
            f'</div>'  # close .strip-scroll
            f'<div class="strip-legend">'
            '<span class="legend-pip ss-helix"></span>helix'
            '<span class="legend-pip ss-sheet"></span>sheet'
            '<span class="legend-pip ss-loop"></span>loop'
            '<span class="legend-pip burial-buried"></span>buried'
            '<span class="legend-pip burial-intermediate"></span>intermediate'
            '<span class="legend-pip burial-exposed"></span>exposed'
            + (
                '<span class="legend-pip strip-liab"></span>liability flag'
                ' (e.g. buried polar / aggregation-prone residue)'
                if any(liab_positions.values())
                else ""
            )
            + "</div></div>"
        )
    return "".join(blocks)


def _residue_annotations_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    header_cols = [
        ("chain_id", "Chain"),
        ("resseq", "Pos"),
        ("aa", "AA"),
        ("resname", "Name"),
        ("rsa", "RSA"),
        ("burial_class", "Burial"),
        ("secondary_structure", "SS"),
        ("neighbor_count", "Neighbours"),
    ]
    # Drop columns entirely absent from this dataset.
    present_cols = [
        (k, name) for k, name in header_cols if any(k in r for r in rows)
    ]
    head = "".join(f"<th>{_h(name)}</th>" for _, name in present_cols)
    body_rows: list[str] = []
    for r in rows:
        cells = []
        for key, _ in present_cols:
            v = r.get(key, "")
            if key == "rsa" and isinstance(v, (int, float)):
                cells.append(f"<td>{v:.2f}</td>")
            elif key == "secondary_structure":
                s = str(v).lower()
                ss_class = "ss-helix" if s == "helix" else ("ss-sheet" if s == "sheet" else "ss-loop")
                cells.append(f'<td><span class="ss-chip {ss_class}">{_h(v)}</span></td>')
            elif key == "burial_class":
                b = str(v).lower()
                b_class = (
                    "burial-buried" if b == "buried"
                    else "burial-intermediate" if b == "intermediate"
                    else "burial-exposed" if b == "exposed"
                    else "burial-na"
                )
                cells.append(f'<td><span class="ss-chip {b_class}">{_h(v)}</span></td>')
            else:
                cells.append(f"<td>{_h(str(v))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<details class="residue-annotations">'
        f'<summary>Per-residue annotations table ({len(rows)} residues)</summary>'
        '<div class="table-wrap"><table class="residue-annotations-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div></details>"
    )


# Headline sequence-analysis properties: key -> (label, explainer, lo, hi).
# ``lo``/``hi`` None => frame ring (count / signed value with no fixed range).
_SEQANA_PROP_SPECS: dict[str, tuple[str, str, float | None, float | None]] = {
    "length": ("Length", "Total residue count.", None, None),
    "molecular_weight": (
        "Molecular weight (Da)", "Mass of the chain in daltons.", None, None,
    ),
    "pI": (
        "Isoelectric point",
        "pH of net-zero charge. Typical soluble proteins span pI 4-9.",
        0.0, 14.0,
    ),
    "gravy": (
        "GRAVY",
        "Grand average of hydropathy (Kyte-Doolittle). Negative = hydrophilic / soluble.",
        -2.0, 2.0,
    ),
    "net_charge": ("Net charge", "Net charge estimated at pH 7.", None, None),
    "instability_index": (
        "Instability index",
        "Guruprasad heuristic. <40 predicts a protein stable in vitro.",
        0.0, 80.0,
    ),
}


def _seqana_prop_card(key: str, value: Any) -> str | None:
    """One headline-property gauge card for the sequence-analysis widget."""
    num = _num(value)
    if num is None:
        return None
    spec = _SEQANA_PROP_SPECS.get(key)
    if spec is None:
        return None
    label, explainer, lo, hi = spec
    quality = _quality_class(key.lower(), num)
    gauge = (
        {"value": num, "frame": True}
        if lo is None
        else {"value": num, "lo": lo, "hi": hi}
    )
    return _score_card(label, _fmt(num), explainer, quality, gauge=gauge)


_AA_LOGO_NICE_PCT = (0.5, 1, 2, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40, 50, 75, 100)

# Amino-acid property groups, in the legend order, paired with the human
# label and the _aa_class key — used to caption the sequence-logo colours.
_AA_LOGO_LEGEND = (
    ("hydro", "Hydrophobic"),
    ("pos", "Basic"),
    ("neg", "Acidic"),
    ("polar", "Polar"),
    ("special", "Cys / Gly / Pro"),
)


def _aa_logo_ceiling_pct(max_pct: float) -> float:
    """Smallest 'nice' percentage at or above the largest composition bar."""
    for step in _AA_LOGO_NICE_PCT:
        if max_pct <= step + 1e-9:
            return float(step)
    return 100.0


def _aa_logo_svg(fractions: dict[str, float]) -> str:
    """Render the amino-acid distribution as a single-row sequence logo:
    one column per residue type, each drawn as its one-letter glyph scaled
    vertically by its composition frequency and filled with the report's
    amino-acid palette (see the .aa-logo-glyph CSS)."""
    items = [
        (str(aa), _num(f) or 0.0)
        for aa, f in fractions.items()
        if isinstance(aa, str) and aa.strip() and (_num(f) or 0.0) > 0
    ]
    if not items:
        return ""
    items.sort(key=lambda kv: (-kv[1], kv[0]))

    col_w, glyph_w = 38, 25
    plot_h, pad_t = 158, 16
    pad_l, pad_r = 58, 16
    xaxis_h, xtitle_h = 24, 22
    n = len(items)
    base_y = pad_t + plot_h
    total_w = pad_l + n * col_w + pad_r
    total_h = base_y + xaxis_h + xtitle_h

    max_f = max(f for _, f in items)
    ceil_pct = _aa_logo_ceiling_pct(max_f * 100.0)
    ceil_f = ceil_pct / 100.0
    cap = 0.72 * 100.0  # cap height of a 100px-font glyph

    parts: list[str] = [
        f'<svg class="aa-logo-svg" viewBox="0 0 {total_w} {total_h}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" '
        f'style="max-width:{total_w}px" '
        f'aria-label="Amino-acid distribution sequence logo">'
    ]
    # y-axis — gridlines, percentage ticks and rotated title.
    yc = pad_t + plot_h / 2
    for t in (0.0, 0.5, 1.0):
        gy = base_y - t * plot_h
        parts.append(
            f'<line class="aa-logo-grid" x1="{pad_l}" y1="{gy:.1f}" '
            f'x2="{total_w - pad_r}" y2="{gy:.1f}"/>'
        )
        parts.append(
            f'<text class="aa-logo-ytick" x="{pad_l - 9}" y="{gy + 3.4:.1f}" '
            f'text-anchor="end">{ceil_pct * t:g}%</text>'
        )
    parts.append(
        f'<text class="aa-logo-axis-title" x="15" y="{yc:.1f}" '
        f'text-anchor="middle" transform="rotate(-90 15 {yc:.1f})">Frequency</text>'
    )
    parts.append(
        f'<line class="aa-logo-baseline" x1="{pad_l}" y1="{base_y}" '
        f'x2="{total_w - pad_r}" y2="{base_y}"/>'
    )
    # glyph columns — each letter scaled vertically by its frequency.
    for i, (aa, frac) in enumerate(items):
        cx = pad_l + i * col_w + col_w / 2
        h = (frac / ceil_f) * plot_h if ceil_f else 0.0
        sy = h / cap if cap else 0.0
        cls = _aa_class(aa)
        parts.append(
            f'<g transform="translate({cx:.2f} {base_y}) scale(1 {sy:.4f})">'
            f"<title>{_h(aa)} — {frac * 100:.1f}%</title>"
            f'<text class="aa-logo-glyph g-{cls}" x="0" y="0" '
            f'text-anchor="middle" textLength="{glyph_w}" '
            f'lengthAdjust="spacingAndGlyphs" font-size="100">{_h(aa)}</text></g>'
        )
        parts.append(
            f'<text class="aa-logo-xtick" x="{cx:.2f}" y="{base_y + 16:.1f}" '
            f'text-anchor="middle">{_h(aa)}</text>'
        )
    parts.append(
        f'<text class="aa-logo-axis-title" '
        f'x="{pad_l + (total_w - pad_l - pad_r) / 2:.1f}" y="{total_h - 5}" '
        f'text-anchor="middle">Amino acid</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _sequence_analysis_widget(output: dict[str, Any]) -> str:
    """Sequence-analysis widget — biochemical descriptors of a residue string,
    grouped into key properties, composition, AA distribution and sampled
    per-residue annotation sections."""
    if not isinstance(output, dict):
        return ""
    metrics = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Sequence analysis</strong> derives deterministic biochemical descriptors "
        "from the residue string alone — size, charge, hydropathy, stability and "
        "composition — via BioPython's <code>ProteinAnalysis</code>."
        "</p>"
    )
    if not metrics:
        return (
            '<div class="widget widget-seqana">'
            '<h4 class="widget-title">Sequence analysis</h4>'
            f'{explainer}<p class="empty">No sequence metrics were emitted.</p></div>'
        )

    # ---- meta line ----
    meta_bits: list[str] = []
    if metrics.get("analysis_mode"):
        meta_bits.append(str(metrics["analysis_mode"]).replace("_", " "))
    if metrics.get("metric_backend"):
        meta_bits.append(f'{metrics["metric_backend"]} backend')
    length = metrics.get("length", output.get("sequence_length"))
    if length is not None:
        noncanon = _num(metrics.get("noncanonical_residue_count")) or 0
        meta_bits.append(
            f"{length} residues ({int(noncanon)} non-canonical)"
            if noncanon else f"{length} residues"
        )
    meta = f'<p class="note">{_h(" · ".join(meta_bits))}</p>' if meta_bits else ""

    # ---- key properties (gauge cards) ----
    prop_cards = [
        card
        for key in ("length", "molecular_weight", "pI", "gravy", "net_charge", "instability_index")
        if (card := _seqana_prop_card(key, metrics.get(key))) is not None
    ]
    props_html = (
        '<h5 class="widget-subhead">Key properties</h5>' + _card_grid(prop_cards)
        if prop_cards else ""
    )

    # ---- residue composition (labelled bars) ----
    comp_rows: list[str] = []
    for key, label in (
        ("hydrophobic_fraction", "Hydrophobic"),
        ("polar_fraction", "Polar"),
        ("charged_fraction", "Charged"),
        ("aromatic_fraction", "Aromatic"),
        ("cysteine_fraction", "Cysteine"),
        ("glycine_fraction", "Glycine"),
        ("proline_fraction", "Proline"),
    ):
        v = _num(metrics.get(key))
        if v is None:
            continue
        pct = max(0.0, min(100.0, v * 100))
        comp_rows.append(
            f'<div class="seqana-comp-row"><span class="seqana-comp-label">{label}</span>'
            f'<div class="seqana-comp-track"><i style="width:{pct:.1f}%"></i></div>'
            f'<span class="seqana-comp-pct">{pct:.0f}%</span></div>'
        )
    comp_html = (
        '<h5 class="widget-subhead">Residue composition</h5>'
        f'<div class="seqana-comp">{"".join(comp_rows)}</div>'
        if comp_rows else ""
    )

    # ---- amino-acid distribution (sequence logo) ----
    aa_fracs = (
        metrics.get("amino_acid_fractions")
        if isinstance(metrics.get("amino_acid_fractions"), dict) else {}
    )
    aa_counts = (
        metrics.get("amino_acid_counts")
        if isinstance(metrics.get("amino_acid_counts"), dict) else {}
    )
    fractions: dict[str, float] = {}
    if aa_fracs:
        fractions = {str(k): (_num(v) or 0.0) for k, v in aa_fracs.items()}
    elif aa_counts:
        total = sum((_num(v) or 0.0) for v in aa_counts.values()) or 1.0
        fractions = {str(k): (_num(v) or 0.0) / total for k, v in aa_counts.items()}
    aa_html = ""
    logo_svg = _aa_logo_svg(fractions)
    if logo_svg:
        legend = "".join(
            f'<span class="aa-logo-legend-item">'
            f'<span class="aa-logo-legend-pip g-{cls}"></span>{label}</span>'
            for cls, label in _AA_LOGO_LEGEND
        )
        aa_html = (
            '<h5 class="widget-subhead">Amino-acid distribution</h5>'
            f'<div class="aa-logo">{logo_svg}</div>'
            f'<div class="aa-logo-legend">{legend}</div>'
        )

    # ---- sampled per-residue annotations ----
    anns = output.get("top_annotations") if isinstance(output.get("top_annotations"), list) else []
    ann_html = ""
    ann_rows: list[str] = []
    for a in anns:
        if not isinstance(a, dict):
            continue
        flags = a.get("caution_flags") or []
        flag_html = "".join(
            f'<span class="seqana-flag">{_h(str(f).replace("_", " "))}</span>' for f in flags
        ) or '<span class="seqana-muted">—</span>'
        res = str(a.get("residue") or "")
        ann_rows.append(
            "<tr>"
            f'<td>{_h(str(a.get("position", "")))}</td>'
            f'<td><span class="aa aa-{_aa_class(res)}">{_h(res)}</span></td>'
            f'<td>{_h(str(a.get("biochemical_class", "")))}</td>'
            f'<td>{_h(str(a.get("charge_class", "")))}</td>'
            f'<td>{_h(str(a.get("hydrophobicity_class", "")))}</td>'
            f"<td>{flag_html}</td>"
            "</tr>"
        )
    if ann_rows:
        ann_html = (
            f'<h5 class="widget-subhead">Sampled residue annotations ({len(ann_rows)})</h5>'
            '<div class="table-wrap"><table><thead><tr><th>Pos</th><th>Residue</th>'
            '<th>Class</th><th>Charge</th><th>Hydrophobicity</th><th>Caution flags</th>'
            f'</tr></thead><tbody>{"".join(ann_rows)}</tbody></table></div>'
        )

    liabs = output.get("liabilities") if isinstance(output.get("liabilities"), list) else []
    liab_html = (
        f'<p class="note">{len(liabs)} sequence-liability flag(s) recorded — '
        "see the raw output below.</p>"
        if liabs else ""
    )

    scope_note = _selected_positions_note(
        output.get("annotation_scope"), output.get("warnings")
    )
    return (
        '<div class="widget widget-seqana">'
        '<h4 class="widget-title">Sequence analysis</h4>'
        f"{explainer}{meta}{scope_note}{props_html}{comp_html}{aa_html}{ann_html}{liab_html}"
        "</div>"
    )


_SEQEDIT_OP_LABELS = {
    "mutate_residues": "Mutate residues",
    "insert_residues": "Insert residues",
    "delete_residues": "Delete residues",
    "replace_region": "Replace region",
    "truncate": "Truncate",
}


def _seqedit_parse_mutation(token: Any) -> dict[str, Any]:
    """Parse a substitution token like 'F5A' into wild-type residue,
    1-based position and mutant residue. Non-substitution tokens degrade
    to a raw label."""
    m = re.match(r"^\s*([A-Za-z*])\s*(\d+)\s*([A-Za-z*])\s*$", str(token))
    if m:
        return {
            "raw": f"{m.group(1).upper()}{m.group(2)}{m.group(3).upper()}",
            "wt": m.group(1).upper(),
            "pos": int(m.group(2)),
            "mut": m.group(3).upper(),
        }
    return {"raw": str(token).strip(), "wt": None, "pos": None, "mut": None}


def _seqedit_residue_track(seq: str, diff: set[int], cls: str) -> str:
    """Monospace residue string with every changed column wrapped in a
    `cls` span; unchanged runs are emitted as a single escaped chunk."""
    parts: list[str] = []
    buf: list[str] = []

    def _flush() -> None:
        if buf:
            parts.append(_h("".join(buf)))
            buf.clear()

    for i, ch in enumerate(seq):
        if i in diff:
            _flush()
            parts.append(f'<span class="{cls}">{_h(ch)}</span>')
        else:
            buf.append(ch)
    _flush()
    return "".join(parts)


def _sequence_edit_widget(output: dict[str, Any]) -> str:
    """Widget for sequence_edit — a professional wild-type vs mutant view.

    Surfaces the edit operation, the applied mutations as colour-coded
    chips and length / mismatch stats, then an MSA-style two-row alignment
    (wild-type above, mutant below, with a column ruler) inside a single
    horizontal scrollbar. Every changed column is highlighted in a
    striking colour. When the wild-type sequence is not stored in the
    output it is reconstructed from the mutant sequence and the
    substitution tokens.
    """
    if not output:
        return ""

    mutant = output.get("mutant_sequence") or output.get("edited_sequence")
    wt = output.get("wild_type_sequence") or output.get("original_sequence")
    raw_muts = (
        output.get("mutations") if isinstance(output.get("mutations"), list) else []
    )
    parsed = [
        _seqedit_parse_mutation(m.get("change") if isinstance(m, dict) else m)
        for m in raw_muts
    ]

    # Reconstruct the wild-type sequence when only the mutant is stored:
    # revert every substitution position back to its original residue.
    wt_reconstructed = False
    if (not isinstance(wt, str) or not wt) and isinstance(mutant, str) and parsed:
        chars = list(mutant)
        ok = True
        for p in parsed:
            idx = (p["pos"] or 0) - 1
            if p["wt"] is None or not 0 <= idx < len(chars):
                ok = False
                break
            chars[idx] = p["wt"]
        if ok:
            wt = "".join(chars)
            wt_reconstructed = True

    has_wt = isinstance(wt, str) and bool(wt)
    has_mut = isinstance(mutant, str) and bool(mutant)

    # Changed columns (0-based). Prefer a direct wild-type/mutant diff;
    # fall back to the explicit `changed_positions` list otherwise.
    diff: set[int] = set()
    if has_wt and has_mut:
        diff = {i for i in range(min(len(wt), len(mutant))) if wt[i] != mutant[i]}
    if not diff:
        for p in output.get("changed_positions") or []:
            if isinstance(p, int) and p >= 1:
                diff.add(p - 1)

    # ---- summary chips -------------------------------------------------
    operation = str(output.get("operation") or "").strip()
    op_html = (
        f'<span class="seqedit-op">'
        f'{_h(_SEQEDIT_OP_LABELS.get(operation, operation.replace("_", " ").title()))}'
        "</span>"
        if operation
        else ""
    )
    wt_len = output.get("wild_type_length")
    mut_len = output.get("mutant_length")
    if wt_len is None and has_wt:
        wt_len = len(wt)
    if mut_len is None and has_mut:
        mut_len = len(mutant)
    stats: list[str] = []

    def _stat(label: str, value: str) -> None:
        stats.append(f'<span class="seqedit-stat">{label} <b>{_h(value)}</b></span>')

    if wt_len is not None:
        _stat("Wild-type", f"{int(wt_len)} aa")
    if mut_len is not None:
        _stat("Mutant", f"{int(mut_len)} aa")
    if wt_len is not None and mut_len is not None:
        delta = int(mut_len) - int(wt_len)
        _stat("Length delta", f"+{delta}" if delta >= 0 else f"-{abs(delta)}")
    mismatch = output.get("mismatch_count")
    if mismatch is not None:
        _stat("Changed residues", str(int(mismatch)))
    elif diff:
        _stat("Changed residues", str(len(diff)))
    is_consistent = output.get("is_consistent")
    if is_consistent is True:
        _stat("Consistency", "verified")
    elif is_consistent is False:
        _stat("Consistency", "mismatch")
    summary_html = (
        '<div class="seqedit-summary">'
        f'{op_html}<div class="seqedit-stats">{"".join(stats)}</div></div>'
        if op_html or stats
        else ""
    )

    # ---- mutation chips ----------------------------------------------
    muts_html = ""
    if parsed:
        chips: list[str] = []
        for p in parsed:
            if p["wt"] is not None:
                chips.append(
                    '<span class="seqedit-mut-chip">'
                    f'<b class="sm-from">{_h(p["wt"])}</b>'
                    f'<span class="sm-pos">{p["pos"]}</span>'
                    f'<b class="sm-to">{_h(p["mut"])}</b></span>'
                )
            else:
                chips.append(
                    f'<span class="seqedit-mut-chip">{_h(p["raw"])}</span>'
                )
        muts_html = (
            '<div class="seqedit-align-head">Mutations applied</div>'
            f'<div class="seqedit-muts">{"".join(chips)}</div>'
        )

    # ---- alignment ----------------------------------------------------
    alignment_html = ""
    if has_wt or has_mut:
        max_len = max(len(wt or ""), len(mutant or ""))
        rows: list[str] = [
            '<div class="seqedit-row seqedit-row-ruler">'
            '<span class="seqedit-row-label"></span>'
            f'<code class="seqedit-row-seq">{_seq_ruler(max_len)}</code></div>'
        ]
        if has_wt:
            rows.append(
                '<div class="seqedit-row seqedit-row-wt">'
                '<span class="seqedit-row-label">WT</span>'
                '<code class="seqedit-row-seq">'
                f"{_seqedit_residue_track(wt, diff, 'seqedit-diff-old')}</code></div>"
            )
        if has_mut:
            rows.append(
                '<div class="seqedit-row seqedit-row-mut">'
                '<span class="seqedit-row-label">Mutant</span>'
                '<code class="seqedit-row-seq">'
                f"{_seqedit_residue_track(mutant, diff, 'seqedit-diff-new')}</code></div>"
            )
        legend = (
            '<div class="seqedit-legend">'
            '<span><span class="seqedit-legend-pip seqedit-diff-old"></span>'
            "wild-type residue at a changed position</span>"
            '<span><span class="seqedit-legend-pip seqedit-diff-new"></span>'
            "substituted (mutant) residue</span></div>"
            if diff
            else ""
        )
        alignment_html = (
            '<div class="seqedit-align-head">Wild-type vs mutant</div>'
            f'<div class="seqedit-align-scroll">{"".join(rows)}</div>{legend}'
        )
    else:
        alignment_html = (
            '<p class="note">No sequence data was emitted for this edit.</p>'
        )

    # ---- warnings -----------------------------------------------------
    warnings = output.get("warnings") if isinstance(output.get("warnings"), list) else []
    warn_html = "".join(
        f'<div class="seqedit-warn">{_h(str(w))}</div>' for w in warnings
    )

    recon_note = (
        " The wild-type row is reconstructed from the mutant sequence and the "
        "substitution list."
        if wt_reconstructed
        else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Sequence edit</strong> applies the requested residue mutations "
        "and reports the resulting sequence. Below, the wild-type and mutant are "
        "aligned with a column ruler; every changed column is highlighted."
        f"{recon_note}"
        "</p>"
    )
    return (
        '<div class="widget widget-seqedit">'
        '<h4 class="widget-title">Sequence edit</h4>'
        f"{explainer}{summary_html}{muts_html}{alignment_html}{warn_html}</div>"
    )


def _rank_designs_widget(output: dict[str, Any]) -> str:
    candidates = output.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        return (
            '<div class="widget widget-rank">'
            '<h4 class="widget-title">Ranking</h4>'
            '<p class="empty">No ranked candidates emitted.</p></div>'
        )
    # Auto-detect interesting columns from the first non-empty row.
    candidate_keys: list[str] = []
    for c in candidates:
        if isinstance(c, dict):
            for k in c.keys():
                if k not in candidate_keys and k not in {"structure_path", "sequence"}:
                    candidate_keys.append(k)
    display_keys = [k for k in (
        "rank", "candidate_id", "score", "ranking_score", "plddt", "iptm", "ptm",
        "qc_flag_count", "liability_count",
    ) if k in candidate_keys][:8]
    header = "".join(f"<th>{_h(_pretty_metric_name(k))}</th>" for k in display_keys)
    body_rows: list[str] = []
    for c in candidates[:10]:
        if not isinstance(c, dict):
            continue
        cells = []
        for k in display_keys:
            value = c.get(k)
            quality_cls = _quality_class(k, value) if value is not None else "neutral"
            cells.append(
                f'<td class="quality-{quality_cls}">{_h(_fmt(value))}</td>'
                if isinstance(value, (int, float))
                else f"<td>{_h(_compact_inline(value))}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table = (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Ranking</strong> aggregates per-candidate metrics (confidence, interface scores, QC flags, "
        "liability counts) into a single ranking score, sorts the designs, and exposes the top candidates "
        "for the report. Sortable columns; click the header to re-order."
        "</p>"
    )
    return (
        '<div class="widget widget-rank">'
        '<h4 class="widget-title">Design ranking</h4>'
        f"{explainer}{table}</div>"
    )


def _boltz2_widget(output: dict[str, Any]) -> str:
    """Boltz-2 affinity output — predicted binding affinity + confidence."""
    aff = output.get("affinity_pred_value")
    aff_prob = output.get("affinity_probability_binary")
    conf = output.get("confidence_score")
    pae_path = output.get("pae_path")
    cif_path = output.get("predicted_cif")
    seeds = output.get("seeds_used")
    cards: list[str] = []
    if aff is not None:
        num = _num(aff) or 0.0
        # Boltz-2 pIC50 normally falls in 0-12 — gauge ring tracks that range
        # so the user gets a quick "how high is this value" sense.
        cards.append(_score_card(
            "Predicted affinity (pIC50)",
            _fmt(num),
            "Boltz-2 predicted pIC50. Higher = tighter. Calibrated only when the target/ligand pair is within training distribution.",
            "neutral",
            gauge={"value": num, "lo": 0.0, "hi": 12.0},
        ))
    if aff_prob is not None:
        num = _num(aff_prob) or 0.0
        cards.append(_score_card(
            "Binary binding probability",
            f"{num * 100:.0f}%" if num <= 1.0 else _fmt(num),
            "Probability the predicted complex binds at experimentally relevant concentrations.",
            "good" if num >= 0.7 else ("ok" if num >= 0.4 else "poor"),
            gauge={"value": num, "lo": 0.0, "hi": 1.0},
        ))
    if conf is not None:
        num = _num(conf) or 0.0
        cards.append(_score_card(
            "Model confidence",
            _fmt(num),
            "Aggregate Boltz-2 confidence (akin to AlphaFold's confidence head).",
            "good" if num >= 0.7 else ("ok" if num >= 0.4 else "poor"),
            gauge={"value": num, "lo": 0.0, "hi": 1.0},
        ))
    cards_html = (
        _card_grid(cards)
    )
    refs: list[str] = []
    if isinstance(cif_path, str):
        refs.append(("Predicted complex (CIF)", cif_path))
    if isinstance(pae_path, str):
        refs.append(("PAE matrix (NPZ)", pae_path))
    art_html = ""
    if refs:
        items = "".join(_download_link(p, label) for label, p in refs)
        art_html = (
            '<div class="node-block-artifacts">'
            '<span class="artifact-section-label">Outputs</span>'
            f'<div class="artifact-row">{items}</div></div>'
        )
    seed_html = (
        f'<p class="note">Seeds used: <code>{_h(", ".join(str(s) for s in seeds))}</code></p>'
        if isinstance(seeds, list) and seeds
        else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Boltz-2</strong> jointly predicts the 3D complex and a binding-affinity score from sequence + "
        "ligand input, optionally guided by an MSA. The cards summarise the affinity, binary binding probability, "
        "and confidence head. The PAE artifact below contains the per-token pair-error matrix."
        "</p>"
    )
    return (
        '<div class="widget widget-boltz">'
        '<h4 class="widget-title">Boltz-2 affinity</h4>'
        f"{explainer}{cards_html}{seed_html}{art_html}</div>"
    )


# ---- AutoDock Vina docking widget --------------------------------------

# AutoDock atom-type -> chemical element. PDBQT keeps only polar hydrogens
# (HD/HS) and uses aromatic-carbon (A) and H-bond-acceptor (NA/OA/SA)
# subtypes that a plain element symbol would otherwise lose.
_PDBQT_ELEMENT = {
    "A": "C", "C": "C", "CA": "C", "CG0": "C", "CG1": "C", "G0": "C", "G1": "C",
    "N": "N", "NA": "N", "NS": "N", "NX": "N",
    "O": "O", "OA": "O", "OS": "O", "OX": "O",
    "S": "S", "SA": "S",
    "H": "H", "HD": "H", "HS": "H",
    "P": "P", "F": "F", "CL": "Cl", "BR": "Br", "I": "I", "SI": "Si",
    "MG": "Mg", "MN": "Mn", "ZN": "Zn", "FE": "Fe",
}
_PDBQT_ACCEPTOR_TYPES = {"NA", "NS", "OA", "OS", "SA"}
_PDBQT_DONOR_H_TYPES = {"HD", "HS"}

# Element swatch colours for the ligand-composition bar — brand-aligned and
# mutually distinct rather than literal CPK colouring.
_VINA_ELEMENT_COLOR = {
    "C": "#4b4565", "N": "#4cc4e4", "O": "#cc0980", "H": "#c9bbed",
    "S": "#f39e3a", "P": "#6b1ea8", "F": "#2bb59a", "Cl": "#2bb59a",
    "Br": "#a3553a", "I": "#7c2d8a",
}


def _pdbqt_floats(text: str) -> list[float]:
    """Pull every numeric token out of a PDBQT REMARK line."""
    out: list[float] = []
    for tok in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _pdbqt_element(atype: str) -> str:
    """Map an AutoDock atom type to its chemical element symbol."""
    t = (atype or "").strip().upper()
    if t in _PDBQT_ELEMENT:
        return _PDBQT_ELEMENT[t]
    return (t[:1] + t[1:2].lower()) if t else "?"


def _parse_pdbqt(text: str) -> list[dict[str, Any]]:
    """Parse a Vina docked .pdbqt into an ordered list of pose dicts, each
    carrying its scoring REMARKs and ligand atom records."""
    poses: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def _fresh() -> dict[str, Any]:
        return {
            "affinity": None, "rmsd_lb": None, "rmsd_ub": None,
            "inter": None, "intra": None, "inter_intra": None,
            "unbound": None, "torsions": None, "torsdof": None, "atoms": [],
        }

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("MODEL"):
            cur = _fresh()
        elif line.startswith("ENDMDL"):
            if cur is not None:
                poses.append(cur)
            cur = None
        elif line.startswith("REMARK"):
            if cur is None:
                cur = _fresh()
            body = line[6:].strip()
            up = body.upper()
            nums = _pdbqt_floats(body)
            if up.startswith("VINA RESULT"):
                if len(nums) > 0:
                    cur["affinity"] = nums[0]
                if len(nums) > 1:
                    cur["rmsd_lb"] = nums[1]
                if len(nums) > 2:
                    cur["rmsd_ub"] = nums[2]
            elif up.startswith("INTER + INTRA"):
                cur["inter_intra"] = nums[-1] if nums else None
            elif up.startswith("INTER:"):
                cur["inter"] = nums[-1] if nums else None
            elif up.startswith("INTRA:"):
                cur["intra"] = nums[-1] if nums else None
            elif up.startswith("UNBOUND"):
                cur["unbound"] = nums[-1] if nums else None
            elif "ACTIVE TORSIONS" in up:
                cur["torsions"] = int(nums[0]) if nums else None
        elif line.startswith(("ATOM", "HETATM")):
            if cur is None:
                cur = _fresh()
            toks = line.split()
            if len(toks) >= 4:
                charge: float | None = None
                try:
                    charge = float(toks[-2])
                except (ValueError, IndexError):
                    charge = None
                cur["atoms"].append({"type": toks[-1], "charge": charge})
        elif line.startswith("TORSDOF"):
            if cur is not None:
                n = _pdbqt_floats(line)
                cur["torsdof"] = int(n[0]) if n else None
    if cur is not None and (cur["atoms"] or cur["affinity"] is not None):
        poses.append(cur)
    return poses


def _pdbqt_ligand_profile(atoms: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-atom PDBQT records into ligand-level descriptors."""
    elements: dict[str, int] = {}
    donors = acceptors = aromatic = 0
    charge_sum = 0.0
    have_charge = False
    for a in atoms:
        t = (a.get("type") or "").strip().upper()
        el = _pdbqt_element(t)
        elements[el] = elements.get(el, 0) + 1
        if t in _PDBQT_DONOR_H_TYPES:
            donors += 1
        if t in _PDBQT_ACCEPTOR_TYPES:
            acceptors += 1
        if t == "A":
            aromatic += 1
        c = a.get("charge")
        if isinstance(c, (int, float)):
            charge_sum += c
            have_charge = True
    heavy = sum(c for el, c in elements.items() if el != "H")
    return {
        "elements": elements, "donors": donors, "acceptors": acceptors,
        "aromatic": aromatic, "heavy": heavy, "total": len(atoms),
        "charge_sum": charge_sum if have_charge else None,
    }


def _vina_affinity_quality(aff: float | None) -> str:
    """Quality bucket for a Vina affinity (kcal/mol; more negative = better)."""
    if aff is None:
        return "neutral"
    if aff <= -8.0:
        return "good"
    if aff <= -6.0:
        return "ok"
    return "poor"


def _vina_formula(elements: dict[str, int]) -> str:
    """Heavy-atom molecular formula in Hill order, with HTML subscripts."""
    heavy = {el: c for el, c in elements.items() if el != "H" and c > 0}
    if not heavy:
        return "—"
    order = (["C"] if "C" in heavy else []) + sorted(e for e in heavy if e != "C")
    return "".join(
        _h(el) + (f"<sub>{heavy[el]}</sub>" if heavy[el] != 1 else "")
        for el in order
    )


def _vina_poses_html(modes: list[dict[str, Any]]) -> str:
    """Ranked binding-pose list — an affinity bar + RMSD spread per pose."""
    affs = [_num(m.get("affinity_kcal_mol")) for m in modes]
    valid = [a for a in affs if a is not None]
    best = min(valid) if valid else 0.0
    denom = abs(best) if best < 0 else 0.0
    rows: list[str] = []
    for idx, m in enumerate(modes):
        rank = m.get("rank", idx + 1)
        aff = _num(m.get("affinity_kcal_mol"))
        quality = _vina_affinity_quality(aff)
        colour = _QUALITY_COLOR.get(quality, _QUALITY_COLOR["neutral"])
        width = (abs(aff) / denom * 100.0) if (aff is not None and aff < 0 and denom) else 0.0
        lb, ub = _num(m.get("rmsd_lb")), _num(m.get("rmsd_ub"))
        rmsd = (
            f'RMSD <b>{lb:.2f}</b> / <b>{ub:.2f}</b> Å'
            if lb is not None and ub is not None else "—"
        )
        aff_txt = f"{aff:+.2f}" if aff is not None else "—"
        rows.append(
            '<div class="vina-pose">'
            f'<span class="vina-pose-rank">#{_h(str(rank))}</span>'
            '<div class="vina-pose-bar-track">'
            f'<i class="vina-pose-bar" style="width:{width:.1f}%;'
            f'background:{colour}"></i></div>'
            f'<span class="vina-pose-aff" style="color:{colour}">{aff_txt} '
            '<small>kcal/mol</small></span>'
            f'<span class="vina-pose-rmsd">{rmsd}</span></div>'
        )
    head = (
        '<div class="vina-pose vina-pose-head">'
        '<span class="vina-pose-rank">Pose</span>'
        '<span>Binding affinity</span>'
        '<span class="vina-pose-aff">score</span>'
        '<span class="vina-pose-rmsd">RMSD l.b. / u.b.</span></div>'
    )
    return f'<div class="vina-poses">{head}{"".join(rows)}</div>'


def _vina_energy_html(pose: dict[str, Any]) -> str:
    """Vina score-term breakdown — diverging bars about a zero axis."""
    terms = [
        ("Intermolecular", pose.get("inter"),
         "Receptor-ligand interaction energy - the actual binding contribution."),
        ("Intramolecular", pose.get("intra"),
         "Ligand internal (strain) energy in the bound conformation."),
        ("Unbound reference", pose.get("unbound"),
         "Ligand energy in the free state, subtracted as the scoring reference."),
    ]
    present = [(lbl, _num(v), ex) for lbl, v, ex in terms if _num(v) is not None]
    if not present:
        return ""
    maxabs = max((abs(v) for _, v, _ in present), default=1.0) or 1.0
    rows: list[str] = []
    for label, v, expl in present:
        mag = min(50.0, abs(v) / maxabs * 50.0)
        left, cls = (50.0 - mag, "neg") if v < 0 else (50.0, "pos")
        rows.append(
            '<div class="vina-ed-row">'
            f'<span class="vina-ed-label" title="{_attr(expl)}">{_h(label)}</span>'
            '<div class="vina-ed-track"><span class="vina-ed-axis"></span>'
            f'<i class="vina-ed-fill vina-ed-{cls}" '
            f'style="left:{left:.1f}%;width:{mag:.1f}%"></i></div>'
            f'<span class="vina-ed-val">{v:+.3f}</span></div>'
        )
    affinity = _num(pose.get("affinity"))
    ii, unb = _num(pose.get("inter_intra")), _num(pose.get("unbound"))
    if affinity is None and ii is not None and unb is not None:
        affinity = ii - unb
    result = ""
    if affinity is not None:
        result = (
            '<div class="vina-ed-result">'
            '<span class="vina-ed-result-label">Predicted binding free energy</span>'
            f'<span class="vina-ed-result-val">{affinity:+.2f} '
            '<small>kcal/mol</small></span></div>'
        )
    return (
        f'<div class="vina-energy">{"".join(rows)}{result}</div>'
        '<p class="vina-caption">Vina score = (intermolecular + intramolecular) '
        "- unbound reference. Negative terms (blue) are favourable; positive "
        "terms (amber) are penalties.</p>"
    )


def _vina_ligand_html(profile: dict[str, Any], rotatable: int | None) -> str:
    """Docked-ligand profile — descriptor tiles + atom-composition bar."""
    elements = profile["elements"]
    tiles: list[str] = []

    def _tile(label: str, value_html: str) -> None:
        tiles.append(
            f'<div class="vina-desc"><span class="vina-desc-val">{value_html}</span>'
            f'<span class="vina-desc-label">{_h(label)}</span></div>'
        )

    _tile("Formula (heavy atoms)", _vina_formula(elements))
    _tile("Heavy atoms", _h(str(profile["heavy"])))
    _tile("H-bond donors", _h(str(profile["donors"])))
    _tile("H-bond acceptors", _h(str(profile["acceptors"])))
    _tile("Aromatic carbons", _h(str(profile["aromatic"])))
    if rotatable is not None:
        _tile("Rotatable bonds", _h(str(rotatable)))
    if profile["charge_sum"] is not None:
        # Collapse floating-point noise (and negative zero) to a clean 0.00.
        cs = profile["charge_sum"]
        cs = 0.0 if abs(cs) < 5e-3 else cs
        _tile("Net partial charge", _h(f"{cs:+.2f} e"))
    desc = f'<div class="vina-desc-grid">{"".join(tiles)}</div>'

    total = sum(elements.values()) or 1
    order = sorted(elements.items(), key=lambda kv: (-kv[1], kv[0]))
    segs: list[str] = []
    legend: list[str] = []
    for el, count in order:
        pct = count / total * 100.0
        colour = _VINA_ELEMENT_COLOR.get(el, "#9a8fc4")
        segs.append(
            f'<span class="vina-el-seg" style="width:{pct:.2f}%;'
            f'background:{colour}" title="{_attr(f"{el}: {count}")}"></span>'
        )
        legend.append(
            '<span class="vina-el-legend-item">'
            f'<span class="vina-el-pip" style="background:{colour}"></span>'
            f'{_h(el)} <b>{count}</b></span>'
        )
    comp = (
        '<div class="vina-el-head">Atom composition</div>'
        f'<div class="vina-el-bar">{"".join(segs)}</div>'
        f'<div class="vina-el-legend">{"".join(legend)}</div>'
    )
    return desc + comp


def _vina_widget(output: dict[str, Any]) -> str:
    """AutoDock Vina docking widget — top-pose affinity cards, a ranked
    pose list, the Vina score-term breakdown and a profile of the docked
    ligand parsed from the .pdbqt artifact."""
    if not output:
        return ""
    explainer = (
        '<p class="widget-explainer">'
        "<strong>AutoDock Vina</strong> docks a small-molecule ligand into the "
        "receptor and scores each binding pose by predicted free energy "
        "(kcal/mol — more negative is tighter). The cards summarise the top "
        "pose; the sections below break down the scoring terms and profile the "
        "docked ligand."
        "</p>"
    )

    modes = output.get("modes") if isinstance(output.get("modes"), list) else []
    modes = [m for m in modes if isinstance(m, dict)]

    # Parse the docked .pdbqt artifact for per-pose energy terms + ligand atoms.
    pq_poses: list[dict[str, Any]] = []
    pdbqt_path = output.get("docked_pdbqt")
    if isinstance(pdbqt_path, str):
        p = Path(pdbqt_path)
        if p.is_file():
            try:
                pq_poses = _parse_pdbqt(
                    p.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                pq_poses = []

    # Fall back to pdbqt-derived poses when the structured `modes` list is empty.
    if not modes and pq_poses:
        modes = [
            {
                "rank": i + 1,
                "affinity_kcal_mol": pq.get("affinity"),
                "rmsd_lb": pq.get("rmsd_lb"),
                "rmsd_ub": pq.get("rmsd_ub"),
            }
            for i, pq in enumerate(pq_poses)
        ]

    best_aff = _num(output.get("best_affinity_kcal_mol"))
    if best_aff is None:
        cand = [_num(m.get("affinity_kcal_mol")) for m in modes]
        cand = [c for c in cand if c is not None]
        best_aff = min(cand) if cand else None

    best_pose = pq_poses[0] if pq_poses else None
    profile = (
        _pdbqt_ligand_profile(best_pose["atoms"])
        if best_pose and best_pose.get("atoms") else None
    )
    rotatable = None
    if best_pose:
        rotatable = best_pose.get("torsdof")
        if rotatable is None:
            rotatable = best_pose.get("torsions")

    # ---- summary cards ----
    cards: list[str] = []
    if best_aff is not None:
        cards.append(_score_card(
            "Best affinity (kcal/mol)",
            _fmt(best_aff),
            "Predicted free energy of binding for the top-ranked pose. More "
            "negative is tighter; weak / non-binders sit near 0.",
            _vina_affinity_quality(best_aff),
            gauge={"value": best_aff, "frame": True},
        ))
    if profile and best_aff is not None and profile["heavy"]:
        le = -best_aff / profile["heavy"]
        leq = "good" if le >= 0.35 else ("ok" if le >= 0.25 else "poor")
        cards.append(_score_card(
            "Ligand efficiency",
            f"{le:.3f}",
            "Binding free energy per heavy atom (-dG / N) - a medicinal-"
            "chemistry yardstick; ≈0.3 kcal/mol per atom marks a useful hit.",
            leq,
            gauge={"value": le, "lo": 0.0, "hi": 0.5},
        ))
    if modes:
        cards.append(_score_card(
            "Binding poses",
            str(len(modes)),
            "Distinct docked conformations Vina retained and scored.",
            "neutral",
            gauge={"value": len(modes), "frame": True},
        ))
    cards_html = _card_grid(cards) if cards else ""

    runtime = _num(output.get("runtime_ms"))
    meta_bits: list[str] = []
    if modes:
        meta_bits.append(f"{len(modes)} pose{'s' if len(modes) != 1 else ''}")
    if runtime is not None:
        meta_bits.append(f"docked in {_duration(int(runtime))}")
    meta = (
        f'<p class="note">{_h(" · ".join(meta_bits))}</p>' if meta_bits else ""
    )

    poses_html = (
        '<h5 class="widget-subhead">Binding poses</h5>' + _vina_poses_html(modes)
        if modes else ""
    )
    energy_html = ""
    if best_pose:
        ed = _vina_energy_html(best_pose)
        if ed:
            energy_html = (
                '<h5 class="widget-subhead">Top-pose score breakdown</h5>' + ed
            )
    ligand_html = (
        '<h5 class="widget-subhead">Docked ligand profile</h5>'
        + _vina_ligand_html(profile, rotatable)
        if profile else ""
    )
    note = ""
    if not pq_poses and isinstance(pdbqt_path, str):
        note = (
            '<p class="note">The docked .pdbqt is not embedded in this report — '
            "per-pose energy terms and the ligand profile are unavailable.</p>"
        )

    body = cards_html + poses_html + energy_html + ligand_html + note
    if not body:
        return ""
    return (
        '<div class="widget widget-vina">'
        '<h4 class="widget-title">Molecular docking · AutoDock Vina</h4>'
        f"{explainer}{meta}{body}</div>"
    )


# ---- Generative protein-design widgets ---------------------------------

# Per-tool title + explainer for the shared generative-design widget.
_DESIGN_TOOL_INFO: dict[str, tuple[str, str]] = {
    "bindcraft": (
        "BindCraft binder design",
        "<strong>BindCraft</strong> generates de-novo binders against the target "
        "and validates each with AlphaFold2. Every accepted design below carries "
        "its interface confidence (ipTM), binder pLDDT, interaction PAE and "
        "surface-aggregation propensity (SAP).",
    ),
    "pxdesign": (
        "PXDesign",
        "<strong>PXDesign</strong> samples binder backbones and sequences for the "
        "target with the Protenix design stack; each sample is emitted as a "
        "predicted structure.",
    ),
    "rfdiffusion": (
        "RFdiffusion backbones",
        "<strong>RFdiffusion</strong> generates de-novo protein backbones by "
        "diffusion. Backbones carry glycine placeholders — sequence design happens "
        "downstream (e.g. ProteinMPNN).",
    ),
    "rfdiffusion3": (
        "RFdiffusion3 backbones",
        "<strong>RFdiffusion3</strong> is the latest-generation diffusion backbone "
        "generator; each design is emitted as a predicted structure.",
    ),
    "rfpeptides": (
        "RFpeptides macrocycles",
        "<strong>RFpeptides</strong> designs cyclic-peptide backbones with "
        "RFdiffusion. Backbones carry glycine placeholders — sequences are designed "
        "downstream.",
    ),
    "boltzgen": (
        "BoltzGen designs",
        "<strong>BoltzGen</strong> generates binder backbones — and, after the "
        "full pipeline, sequences plus a refolded complex — with the Boltz model "
        "family. Affinity / confidence columns appear only once the "
        "corresponding pipeline stages have run.",
    ),
    "proteina_complexa": (
        "Proteina complex design",
        "<strong>Proteina</strong> designs protein / ligand / motif complexes "
        "(mode shown below). Validation columns — refold RMSD, designability, "
        "ipTM — populate only when the design was scored.",
    ),
}

# Optional per-design numeric columns, in display order. Only the columns a
# given tool actually emits (non-null for ≥1 design) are rendered.
_DESIGN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("plddt", "pLDDT"),
    ("scaffold_plddt", "Scaffold pLDDT"),
    ("iptm", "ipTM"),
    ("confidence", "Confidence"),
    ("pae_interaction", "Interaction PAE"),
    ("designability", "Designability"),
    ("binder_refold_rmsd", "Refold RMSD (Å)"),
    ("motif_rmsd", "Motif RMSD (Å)"),
    ("ligand_rmsd", "Ligand RMSD (Å)"),
    ("ligand_contacts", "Ligand contacts"),
    ("sap_score", "SAP"),
    ("affinity_pred_value", "Affinity (pIC50)"),
    ("affinity_probability_binary", "Binding prob."),
    ("score", "Score"),
)


def _is_placeholder_seq(seq: Any) -> bool:
    """True when a 'sequence' is absent or a single-character placeholder
    (the X / G fillers backbone-only design tools emit)."""
    if not isinstance(seq, str) or not seq.strip():
        return True
    return len(set(seq.strip().upper())) <= 1


def _design_metric_quality(key: str, num: float) -> str:
    """Quality bucket for a per-design metric cell."""
    if key in ("plddt", "scaffold_plddt"):
        v = num * 100 if num <= 1.5 else num
        return "good" if v >= 70 else ("ok" if v >= 50 else "poor")
    if key in ("iptm", "confidence", "designability", "affinity_probability_binary"):
        return "good" if num >= 0.7 else ("ok" if num >= 0.5 else "poor")
    if key == "pae_interaction":
        return "good" if num <= 10 else ("ok" if num <= 20 else "poor")
    if key in ("binder_refold_rmsd", "motif_rmsd", "ligand_rmsd"):
        return "good" if num <= 2 else ("ok" if num <= 4 else "poor")
    if key == "sap_score":
        return "good" if num <= 35 else ("ok" if num <= 50 else "poor")
    return "neutral"


def _dsn_chip(label: str, value: Any) -> str:
    return f'<span class="dsn-chip">{_h(label)} <b>{_h(str(value))}</b></span>'


def _design_widget(output: dict[str, Any], *, tool: str) -> str:
    """Shared widget for the generative structure-design tools — a summary
    chip row plus a per-design results table (length, whatever scoring
    columns the tool emits, sequence, and the design structure file)."""
    if not isinstance(output, dict):
        return ""
    designs = [d for d in (output.get("designs") or []) if isinstance(d, dict)]
    title, body = _DESIGN_TOOL_INFO.get(
        tool,
        (f"{tool} designs",
         f"<strong>{_h(tool)}</strong> emitted the protein designs below."),
    )
    explainer = f'<p class="widget-explainer">{body}</p>'

    # ---- summary chips ----
    chips: list[str] = []
    n_designs = output.get("num_designs_returned")
    if n_designs is None:
        n_designs = len(designs)
    chips.append(_dsn_chip("designs", n_designs))
    if output.get("mode"):
        chips.append(_dsn_chip("mode", output["mode"]))
    if tool == "rfdiffusion":
        contigs = output.get("contigs")
        if isinstance(contigs, list) and contigs:
            chips.append(_dsn_chip("contigs", ", ".join(str(c) for c in contigs)))
        if output.get("num_requested") is not None:
            chips.append(_dsn_chip("requested", output["num_requested"]))
    if tool == "pxdesign":
        if output.get("num_samples") is not None:
            chips.append(_dsn_chip("samples", output["num_samples"]))
        if output.get("seed") is not None:
            chips.append(_dsn_chip("seed", output["seed"]))
        if output.get("task_name"):
            chips.append(_dsn_chip("task", output["task_name"]))
    seeds = output.get("seeds_used")
    if isinstance(seeds, list) and seeds:
        chips.append(_dsn_chip(
            "seed" + ("s" if len(seeds) > 1 else ""),
            ", ".join(str(s) for s in seeds),
        ))
    rt = _num(output.get("runtime_ms"))
    if rt is not None:
        chips.append(f'<span class="dsn-chip">{_h(_duration(int(rt)))}</span>')
    chips_html = f'<div class="dsn-chips">{"".join(chips)}</div>'

    # ---- per-design table ----
    if not designs:
        return (
            '<div class="widget widget-design">'
            f'<h4 class="widget-title">{_h(title)}</h4>'
            f'{explainer}{chips_html}'
            '<p class="empty">No designs were emitted.</p></div>'
        )
    present = [
        (key, label) for key, label in _DESIGN_COLUMNS
        if any(_num(d.get(key)) is not None for d in designs)
    ]
    has_seq = any(not _is_placeholder_seq(d.get("sequence")) for d in designs)
    head = "<th>#</th><th>Length</th>"
    head += "".join(f"<th>{_h(label)}</th>" for _, label in present)
    if has_seq:
        head += "<th>Sequence</th>"
    head += "<th>Structure</th>"

    body_rows: list[str] = []
    for i, d in enumerate(designs, 1):
        length = d.get("length")
        if length is None:
            length = d.get("binder_length")
        cells = [
            f"<td>{i}</td>",
            f"<td>{_h(_fmt(length)) if length is not None else '—'} aa</td>"
            if length is not None else "<td>—</td>",
        ]
        for key, _ in present:
            num = _num(d.get(key))
            if num is None:
                cells.append("<td>—</td>")
                continue
            quality = _design_metric_quality(key, num)
            cls = f' class="quality-{quality}"' if quality in ("good", "ok", "poor") else ""
            cells.append(f"<td{cls}>{_h(_fmt(num))}</td>")
        if has_seq:
            seq = d.get("sequence")
            if _is_placeholder_seq(seq):
                cells.append('<td><span class="dsn-muted">backbone only</span></td>')
            else:
                cells.append(f'<td><code class="dsn-seq">{_h(seq)}</code></td>')
        # Show the structure file name only — the node's artifact list below
        # already carries the downloadable copy, so don't re-embed it here.
        struct = d.get("pdb_path") or d.get("cif_path")
        if isinstance(struct, str):
            cells.append(
                f'<td><span class="dsn-file">{_h(Path(struct).name)}</span></td>'
            )
        else:
            cells.append("<td>—</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (
        '<div class="table-wrap"><table><thead><tr>'
        f'{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'
    )
    return (
        '<div class="widget widget-design">'
        f'<h4 class="widget-title">{_h(title)}</h4>'
        f"{explainer}{chips_html}{table_html}</div>"
    )


_MPNN_TOOL_INFO: dict[str, tuple[str, str]] = {
    "proteinmpnn": (
        "ProteinMPNN designs",
        "<strong>ProteinMPNN</strong> samples sequences over a fixed backbone "
        "(inverse folding). A lower score is a higher likelihood under the "
        "trained model. Sequence recovery — the fraction of positions matching "
        "the native sequence — is only available when a reference sequence was "
        "supplied.",
    ),
    "ligandmpnn": (
        "LigandMPNN designs",
        "<strong>LigandMPNN</strong> redesigns a sequence over a fixed backbone "
        "while accounting for bound ligands and context. A lower score is a "
        "higher likelihood under the model. Sequence recovery — the fraction of "
        "positions matching the native sequence — is only available when a "
        "reference sequence was supplied.",
    ),
}


def _mpnn_widget(output: dict[str, Any], *, tool: str) -> str:
    """Shared widget for the MPNN inverse-folding tools (ProteinMPNN /
    LigandMPNN).

    A score / sequence-recovery table is shown only when recovery was
    actually computed — i.e. a native reference sequence was available. When
    no recovery is present the table is omitted entirely and the per-design
    score is folded into the sequence labels instead.
    """
    if not isinstance(output, dict):
        return ""
    designs = [d for d in (output.get("designs") or []) if isinstance(d, dict)]
    title, body = _MPNN_TOOL_INFO.get(
        tool,
        (f"{tool} designs",
         f"<strong>{_h(tool)}</strong> emitted the sequence designs below."),
    )
    explainer = f'<p class="widget-explainer">{body}</p>'

    chips: list[str] = [_dsn_chip("designs", len(designs))]
    if output.get("num_residues") is not None:
        chips.append(_dsn_chip("residues", output["num_residues"]))
    rt = _num(output.get("runtime_ms"))
    if rt is not None:
        chips.append(f'<span class="dsn-chip">{_h(_duration(int(rt)))}</span>')
    chips_html = f'<div class="dsn-chips">{"".join(chips)}</div>'

    if not designs:
        return (
            f'<div class="widget widget-mpnn">'
            f'<h4 class="widget-title">{_h(title)}</h4>'
            f'{explainer}{chips_html}'
            '<p class="empty">No designs were emitted.</p></div>'
        )

    # The recovery table appears only when at least one design carries a
    # non-null seq_recovery (a native reference was available to score against).
    has_recovery = any(_num(d.get("seq_recovery")) is not None for d in designs)

    table_html = ""
    if has_recovery:
        rows: list[str] = []
        for i, d in enumerate(designs, 1):
            score = _num(d.get("score"))
            rec = _num(d.get("seq_recovery"))
            if rec is not None:
                pct = max(0.0, min(100.0, rec * 100))
                quality = "good" if pct >= 50 else ("ok" if pct >= 30 else "poor")
                rec_cell = (
                    '<td><div class="dsn-bar-cell">'
                    f'<div class="dsn-bar"><i class="metric-{quality}" '
                    f'style="width:{pct:.0f}%"></i></div>'
                    f'<span>{pct:.0f}%</span></div></td>'
                )
            else:
                rec_cell = "<td>—</td>"
            rows.append(
                f"<tr><td>{i}</td>"
                f"<td>{_h(_fmt(score)) if score is not None else '—'}</td>"
                f"{rec_cell}</tr>"
            )
        table_html = (
            '<h5 class="widget-subhead">Score &amp; sequence recovery</h5>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>#</th><th>Score</th><th>Sequence recovery</th>"
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        )

    # Designed sequences, MSA-style scroll. With no recovery table the
    # per-design score rides along in the sequence label so it isn't lost.
    #
    # When two or more designs are present we compute per-column
    # conservation across them and highlight positions where every design
    # agrees on the same residue. This surfaces "the model never wanted to
    # change this position" hot-spots at a glance — typically the
    # functional / structural core residues.
    present = [(i, d) for i, d in enumerate(designs, 1) if d.get("sequence")]
    seq_html = ""
    if present:
        max_len = max(len(d["sequence"]) for _, d in present)
        conservation: list[float] = []
        conservation_caption = ""
        if len(present) >= 2:
            conservation = _conservation(
                [(f"design_{i}", d["sequence"]) for i, d in present],
                max_len,
            )
            fully_conserved = sum(1 for v in conservation if v >= 0.9999)
            if max_len:
                pct = 100.0 * fully_conserved / max_len
                conservation_caption = (
                    '<p class="widget-subexplain">'
                    f'Highlighted cells are positions where all {len(present)} '
                    'designs agree on the same residue '
                    f'(<strong>{fully_conserved}/{max_len}</strong> &middot; '
                    f'{pct:.0f}% fully conserved). Dimmed cells are positions '
                    'where the model picked different residues across designs.'
                    '</p>'
                )
        seq_rows: list[str] = []
        for i, d in present:
            badge = ""
            if not has_recovery:
                score = _num(d.get("score"))
                if score is not None:
                    badge = f'<span class="tmpl-seq-id">score {_h(_fmt(score))}</span>'
            if conservation:
                track_html = _mpnn_seq_cells(d["sequence"], max_len, conservation)
            else:
                track_html = _h(d["sequence"])
            seq_rows.append(
                '<div class="tmpl-seq-row">'
                '<span class="tmpl-seq-label">'
                f'<span class="tmpl-seq-name">Design {i}</span>{badge}</span>'
                f'<code class="tmpl-seq-track">{track_html}</code></div>'
            )
        seq_rows.append(
            '<div class="tmpl-seq-row tmpl-seq-ruler">'
            '<span class="tmpl-seq-label"></span>'
            f'<code class="tmpl-seq-track">{_seq_ruler(max_len)}</code></div>'
        )
        seq_html = (
            '<h5 class="widget-subhead">Designed sequences</h5>'
            f'{conservation_caption}'
            f'<div class="tmpl-seq-scroll">{"".join(seq_rows)}</div>'
        )
    return (
        f'<div class="widget widget-mpnn">'
        f'<h4 class="widget-title">{_h(title)}</h4>'
        f"{explainer}{chips_html}{table_html}{seq_html}</div>"
    )


def _mpnn_seq_cells(
    sequence: str, max_len: int, conservation: list[float]
) -> str:
    """Render an MPNN design sequence as per-residue spans with conservation highlight.

    Each residue becomes a ``<span>`` carrying either ``mpnn-cell-conserved``
    (bold + a single soft brand-purple background, regardless of which
    amino acid) for positions where every design agrees, or
    ``mpnn-cell-variable`` for positions where at least one design differs.
    A single calm highlight colour is intentional — letting the per-AA
    chemistry palette through here makes the strip read as a noisy mosaic
    when most positions are conserved. Layout-affecting properties (width,
    padding, inline-block) are intentionally NOT set on these classes — the
    parent ``.tmpl-seq-track`` is ``white-space:pre`` monospace with a
    fixed ``letter-spacing``, so each character still occupies exactly one
    monospace slot and stays column-aligned with the ruler below.
    """
    cells: list[str] = []
    for idx in range(max_len):
        if idx < len(sequence):
            aa = sequence[idx]
            if conservation[idx] >= 0.9999:
                cells.append(
                    f'<span class="mpnn-cell-conserved" '
                    f'title="Position {idx + 1}: conserved across all designs">'
                    f'{_h(aa)}</span>'
                )
            else:
                cells.append(
                    f'<span class="mpnn-cell-variable" '
                    f'title="Position {idx + 1}: varies across designs">'
                    f'{_h(aa)}</span>'
                )
        else:
            # Shorter design — pad with a space to keep ruler alignment.
            cells.append(" ")
    return "".join(cells)


def _lerp_hex(a: str, b: str, t: float) -> str:
    """Linear blend between two #rrggbb colours."""
    t = max(0.0, min(1.0, t))
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    return (
        f"#{round(ar + (br - ar) * t):02x}"
        f"{round(ag + (bg - ag) * t):02x}"
        f"{round(ab + (bb - ab) * t):02x}"
    )


def _embedding_strip_svg(vec: list[Any]) -> str:
    """Render an embedding vector as a 1-D diverging-colour 'fingerprint'
    strip — negative dims cool, positive dims warm, scaled robustly so a
    single outlier dimension doesn't wash the strip out."""
    nums = [float(x) for x in vec if isinstance(x, (int, float))]
    if not nums:
        return ""
    n = len(nums)
    mags = sorted(abs(x) for x in nums)
    scale = mags[int(0.95 * (n - 1))] or mags[-1] or 1.0
    width, height = 100.0, 8.0  # viewBox units; CSS stretches to full width
    bw = width / n
    rects: list[str] = []
    for i, x in enumerate(nums):
        t = max(-1.0, min(1.0, x / scale)) if scale else 0.0
        colour = (
            _lerp_hex("#ece8fb", "#cc0980", t) if t >= 0
            else _lerp_hex("#ece8fb", "#1f9bd0", -t)
        )
        rects.append(
            f'<rect x="{i * bw:.4f}" y="0" width="{bw:.4f}" '
            f'height="{height}" fill="{colour}"/>'
        )
    return (
        f'<svg class="esm-emb-svg" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" aria-hidden="true">{"".join(rects)}</svg>'
    )


_ESM_TOOL_INFO: dict[str, tuple[str, str]] = {
    "esm2": (
        "ESM-2 protein language model",
        "<strong>ESM-2</strong> is a protein language model. In embedding mode it "
        "returns a per-residue embedding tensor and a pooled whole-sequence "
        "vector; in generation mode it fills masked positions with sampled "
        "residues.",
    ),
    "esm3": (
        "ESM3 multi-track model",
        "<strong>ESM3</strong> jointly reasons over sequence, structure and "
        "function tracks. It returns per-residue embeddings and, when run as a "
        "predictor, a structure with confidence heads.",
    ),
}


def _esm_widget(output: dict[str, Any], *, tool: str) -> str:
    """Shared widget for the ESM language models (ESM-2 / ESM3) — run
    summary, an embedding-fingerprint strip for the pooled vector,
    confidence cards and any generated sequences."""
    if not isinstance(output, dict):
        return ""
    title, body = _ESM_TOOL_INFO.get(
        tool, (f"{tool} output", f"<strong>{_h(tool)}</strong> output.")
    )
    explainer = f'<p class="widget-explainer">{body}</p>'

    chips: list[str] = []
    if output.get("model_name"):
        chips.append(_dsn_chip("model", output["model_name"]))
    if output.get("mode"):
        chips.append(_dsn_chip("mode", output["mode"]))
    if output.get("num_residues") is not None:
        chips.append(_dsn_chip("residues", output["num_residues"]))
    pooled = output.get("pooled_embedding")
    if isinstance(pooled, list) and pooled:
        chips.append(_dsn_chip("embedding dim", len(pooled)))
    if output.get("generation_strategy"):
        chips.append(_dsn_chip("strategy", output["generation_strategy"]))
    if output.get("fallback_used"):
        chips.append('<span class="dsn-chip dsn-chip-warn">fallback used</span>')
    rt = _num(output.get("runtime_ms"))
    if rt is not None:
        chips.append(f'<span class="dsn-chip">{_h(_duration(int(rt)))}</span>')
    chips_html = f'<div class="dsn-chips">{"".join(chips)}</div>' if chips else ""

    # Confidence cards (ESM3 predictor mode).
    cards: list[str] = []
    cs = output.get("confidence_scores")
    if isinstance(cs, dict):
        for key, label, tip in (
            ("mean_plddt", "Mean pLDDT", "Per-residue confidence averaged over the chain."),
            ("ptm", "pTM", "Predicted TM-score."),
        ):
            value = cs.get(key)
            if value is None:
                continue
            num = _num(value) or 0.0
            gauge_hi = 1.0 if num <= 1.5 else 100.0
            cards.append(_score_card(
                label, _fmt(value), tip,
                _quality_class(key.replace("mean_", ""), value),
                gauge={"value": num, "lo": 0.0, "hi": gauge_hi},
            ))
    cards_html = _card_grid(cards) if cards else ""

    # Pooled-embedding fingerprint.
    emb_html = ""
    if isinstance(pooled, list) and pooled:
        strip = _embedding_strip_svg(pooled)
        if strip:
            emb_html = (
                f'<h5 class="widget-subhead">Pooled embedding · {len(pooled)} dims</h5>'
                f'<div class="esm-emb">{strip}</div>'
                '<div class="esm-emb-legend">'
                '<span><span class="esm-emb-pip" style="background:#1f9bd0"></span>'
                'negative</span>'
                '<span><span class="esm-emb-pip" style="background:#ece8fb"></span>'
                '~zero</span>'
                '<span><span class="esm-emb-pip" style="background:#cc0980"></span>'
                'positive</span></div>'
            )

    # Generated sequences (ESM-2 sequence-generation mode).
    gen = output.get("generated_sequences")
    gen_html = ""
    if isinstance(gen, list) and gen:
        rows = []
        max_len = max(len(str(s)) for s in gen)
        for i, s in enumerate(gen, 1):
            rows.append(
                '<div class="tmpl-seq-row">'
                f'<span class="tmpl-seq-label"><span class="tmpl-seq-name">Seq {i}</span></span>'
                f'<code class="tmpl-seq-track">{_h(str(s))}</code></div>'
            )
        rows.append(
            '<div class="tmpl-seq-row tmpl-seq-ruler">'
            '<span class="tmpl-seq-label"></span>'
            f'<code class="tmpl-seq-track">{_seq_ruler(max_len)}</code></div>'
        )
        filled = output.get("filled_positions")
        cap = ""
        if isinstance(filled, list) and filled:
            cap = (
                '<p class="note">Filled positions: '
                f'{_h(_format_position_list(filled))}</p>'
            )
        gen_html = (
            '<h5 class="widget-subhead">Generated sequences</h5>'
            f'<div class="tmpl-seq-scroll">{"".join(rows)}</div>{cap}'
        )

    return (
        '<div class="widget widget-esm">'
        f'<h4 class="widget-title">{_h(title)}</h4>'
        f"{explainer}{chips_html}{cards_html}{emb_html}{gen_html}</div>"
    )


def _protenix_widget(output: dict[str, Any]) -> str:
    """Protenix v2 structure-prediction widget — confidence gauges for the
    top-ranked prediction plus a per-seed prediction table."""
    if not isinstance(output, dict):
        return ""
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Protenix v2</strong> is an AlphaFold-3-class structure predictor "
        "for protein / complex inputs. The cards show the top-ranked prediction's "
        "confidence heads; the table breaks the result out per sampling seed."
        "</p>"
    )
    metrics = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}

    # ---- confidence cards (top-ranked prediction) ----
    cards: list[str] = []
    for key, label, tip, lo, hi in (
        ("plddt", "Mean pLDDT", "Per-residue confidence (0-100). >=70 confident.", 0.0, 100.0),
        ("ptm", "pTM", "Predicted TM-score (0-1). >0.5 = likely-correct fold.", 0.0, 1.0),
        ("iptm", "ipTM", "Interface pTM (0-1). >0.6 trustworthy for interfaces.", 0.0, 1.0),
        ("ranking_score", "Ranking score", "Composite head Protenix ranks seeds by.", 0.0, 1.0),
    ):
        value = metrics.get(key, output.get(key))
        if value is None:
            continue
        num = _num(value) or 0.0
        gauge_hi = 1.0 if (key == "plddt" and num <= 1.5) else hi
        cards.append(_score_card(
            label, _fmt(value), tip, _quality_class(key, value),
            gauge={"value": num, "lo": lo, "hi": gauge_hi},
        ))
    cards_html = _card_grid(cards)

    # ---- summary chips ----
    chips: list[str] = []
    seeds = output.get("seeds_used")
    if isinstance(seeds, list) and seeds:
        chips.append(_dsn_chip(
            "seed" + ("s" if len(seeds) > 1 else ""),
            ", ".join(str(s) for s in seeds),
        ))
    preds = [p for p in (output.get("all_predictions") or []) if isinstance(p, dict)]
    if preds:
        chips.append(_dsn_chip("predictions", len(preds)))
    rt = _num(output.get("runtime_ms"))
    if rt is not None:
        chips.append(f'<span class="dsn-chip">{_h(_duration(int(rt)))}</span>')
    chips_html = f'<div class="dsn-chips">{"".join(chips)}</div>' if chips else ""

    # ---- per-seed predictions table ----
    table_html = ""
    if preds:
        rows: list[str] = []
        for p in preds:
            cells = [
                f"<td>{_h(_fmt(p.get('seed')))}</td>"
                if p.get("seed") is not None else "<td>—</td>",
                f"<td>{_h(_fmt(p.get('rank')))}</td>"
                if p.get("rank") is not None else "<td>—</td>",
            ]
            for key in ("plddt", "iptm", "ptm", "ranking_score"):
                num = _num(p.get(key))
                if num is None:
                    cells.append("<td>—</td>")
                    continue
                quality = _quality_class(key, num)
                cls = f' class="quality-{quality}"' if quality in ("good", "ok", "poor") else ""
                cells.append(f"<td{cls}>{_h(_fmt(num))}</td>")
            cif = p.get("cif_path")
            cells.append(
                f'<td><span class="dsn-file">{_h(Path(str(cif)).name)}</span></td>'
                if isinstance(cif, str) else "<td>—</td>"
            )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        table_html = (
            '<h5 class="widget-subhead">Per-seed predictions</h5>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Seed</th><th>Rank</th><th>pLDDT</th><th>ipTM</th><th>pTM</th>"
            "<th>Ranking score</th><th>Structure</th>"
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        )
    return (
        '<div class="widget widget-protenix">'
        '<h4 class="widget-title">Protenix structure prediction</h4>'
        f"{explainer}{chips_html}{cards_html}{table_html}</div>"
    )


def _generic_metric_widget(output: dict[str, Any], *, include_shape: bool = True) -> str:
    """Last-resort widget for tools without a specialised renderer.

    Any adapter that emits scalars with recognised metric names (see
    ``_METRIC_REGISTRY``) automatically gets the CELLKEY-style donut card
    grid here — so new tools don't need bespoke widget code to look
    professional. Unknown numeric metrics still get a card (without a
    gauge); non-numeric context (model name, runtime, etc.) is shown in a
    small footer table beneath the cards.
    """
    if not output:
        return ""

    # Pass 1: flatten one level into nested metric dicts so adapters emitting
    # ``confidence_scores: {plddt: ..., ptm: ...}`` still surface metrics as
    # cards.
    flat: list[tuple[str, Any]] = []
    seen_keys: set[str] = set()

    def _push(k: str, v: Any) -> None:
        if k in seen_keys:
            return
        seen_keys.add(k)
        flat.append((k, v))

    for key, value in output.items():
        if key in _NESTED_METRIC_KEYS and isinstance(value, dict):
            for sub_k, sub_v in value.items():
                _push(str(sub_k), sub_v)
            continue
        _push(str(key), value)

    # Pass 2: classify each scalar into cards / footer rows / "shape" entries.
    #   * numeric metrics              -> donut card (or plain card)
    #   * short context (model/seed/runtime) -> footer kv-table
    #   * sequences / long strings / lists / dicts -> shape inventory
    cards: list[str] = []
    footer_rows: list[str] = []
    has_complex_output = False  # set when there's a sequence/list/dict to surface
    for key, value in flat:
        if isinstance(value, bool) or value is None:
            continue
        canonical = key.lower()
        if isinstance(value, (list, dict)):
            has_complex_output = True
            continue
        if canonical in _FOOTER_METRIC_KEYS:
            short = _compact_inline(value)
            if len(short) <= 120:
                footer_rows.append(_kv_row(_pretty_metric_name(canonical), _h(short)))
            continue
        card = _auto_score_card(key, value)
        if card is not None:
            cards.append(card)
            continue
        # Non-numeric string. Long strings (sequences, free text) belong in
        # the shape inventory — putting them in a kv-table makes the table
        # row wrap horribly. Short strings stay as context footer rows.
        if isinstance(value, str) and (
            len(value) >= 30 or canonical in _SEQUENCE_OUTPUT_KEYS or _looks_like_sequence(value)
        ):
            has_complex_output = True
            continue
        short = _compact_inline(value)
        if short and len(short) <= 120:
            footer_rows.append(_kv_row(_pretty_metric_name(canonical), _h(short)))

    # Cap to keep the grid focused — a runaway adapter shouldn't paint 80
    # cards. Cards are picked first; footer rows trim independently.
    cards = cards[:12]
    footer_rows = footer_rows[:8]

    if not cards:
        # No numeric scalars at all — fall through to the structured
        # "what did this tool produce" panel so the report still has
        # something readable. Footer rows (model name, seed, etc.) are
        # absorbed into that panel rather than shown in isolation.
        return _output_shape_widget(output)

    # Cards were produced; if there's also complex output, append a brief
    # inventory so sequences / lists don't disappear silently. Tools that opt
    # out (``include_shape=False``) skip this block entirely.
    shape_html = (
        _output_shape_widget(output) if has_complex_output and include_shape else ""
    )

    cards_html = (
        _card_grid(cards)
    )
    footer_html = (
        f'<table class="kv-table"><tbody>{"".join(footer_rows)}</tbody></table>'
        if footer_rows
        else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "Metrics extracted from the tool's output. Cards show numeric scores "
        "with reference ranges where known; the table below carries non-numeric "
        "context (model name, runtime, etc.)."
        "</p>"
    )
    return (
        '<div class="widget widget-generic">'
        '<h4 class="widget-title">Summary</h4>'
        f"{explainer}{cards_html}{footer_html}{shape_html}</div>"
    )


def _metrics_only_widget(output: dict[str, Any]) -> str:
    """Generic metric widget with the output-shape inventory suppressed.

    Used by tools (ipSAE, MMseqs2-GPU) whose payload shape — raw residue-index
    lists, hit-table file paths — is noise rather than a useful summary; only
    the metric gauges and the context footer are kept.
    """
    return _generic_metric_widget(output, include_shape=False)


# PXMeter metric specs: key -> (label, explainer, lo, hi). ``lo``/``hi`` None
# means a frame ring (unbounded distance metric — no defensible 0..max range).
_PXMETER_SPECS: dict[str, tuple[str, str, float | None, float | None]] = {
    "lddt": (
        "lDDT",
        "Local Distance Difference Test (0-1). Higher = closer to the reference.",
        0.0, 1.0,
    ),
    "bb_lddt": (
        "Backbone lDDT",
        "lDDT restricted to backbone atoms (0-1).",
        0.0, 1.0,
    ),
    "dockq": (
        "DockQ",
        "Interface quality (0-1). >0.23 acceptable, >0.49 medium, >0.80 high.",
        0.0, 1.0,
    ),
    "fnat": (
        "fnat",
        "Fraction of native interface contacts recovered (0-1).",
        0.0, 1.0,
    ),
    "f1": (
        "F1",
        "F1 score over interface contacts (0-1).",
        0.0, 1.0,
    ),
    "fnonnat": (
        "fnonnat",
        "Fraction of model interface contacts absent from the reference (0-1). "
        "Lower = better.",
        0.0, 1.0,
    ),
    "irmsd": (
        "iRMSD (Å)",
        "Interface RMSD after superposition. Lower = better; <2 A is high quality.",
        None, None,
    ),
    "lrmsd": (
        "LRMSD (Å)",
        "Ligand RMSD after receptor superposition. Lower = better.",
        None, None,
    ),
}


def _px_metric_card(key: str, value: Any) -> str | None:
    """Build one PXMeter gauge card, or ``None`` for a non-numeric / unknown key."""
    num = _num(value)
    if num is None:
        return None
    spec = _PXMETER_SPECS.get(key.lower())
    if spec is None:
        return None
    label, explainer, lo, hi = spec
    quality = _quality_class(key.lower(), num)
    gauge = (
        {"value": num, "frame": True}
        if lo is None
        else {"value": num, "lo": lo, "hi": hi}
    )
    return _score_card(label, _fmt(num), explainer, quality, gauge=gauge)


def _px_section(title: str, subtitle: str, metrics: dict[str, Any]) -> str:
    """One PXMeter section (Complex / Chain X / Interface X,Y) — a heading,
    a grid of metric gauges, and (for interfaces) a context line."""
    cards: list[str] = []
    for key in ("lddt", "bb_lddt", "dockq"):
        if key in metrics:
            card = _px_metric_card(key, metrics[key])
            if card:
                cards.append(card)
    info = metrics.get("dockq_info") if isinstance(metrics.get("dockq_info"), dict) else {}
    for key in ("fnat", "F1", "fnonnat", "iRMSD", "LRMSD"):
        if key in info:
            card = _px_metric_card(key, info[key])
            if card:
                cards.append(card)
    if not cards:
        return ""
    context = ""
    if info:
        bits: list[str] = []
        if info.get("class1") and info.get("len1") is not None:
            bits.append(f'{info["class1"]} · {info["len1"]} aa')
        if info.get("class2") and info.get("len2") is not None:
            bits.append(f'{info["class2"]} · {info["len2"]} aa')
        nat_c, nat_t = info.get("nat_correct"), info.get("nat_total")
        if nat_c is not None and nat_t:
            bits.append(f"{nat_c}/{nat_t} native contacts recovered")
        if bits:
            context = f'<p class="px-context">{_h(" · ".join(str(b) for b in bits))}</p>'
    sub_html = (
        f'<span class="px-section-sub">{_h(subtitle)}</span>' if subtitle else ""
    )
    return (
        '<section class="px-section">'
        f'<div class="px-section-head"><h5 class="px-section-title">{_h(title)}</h5>'
        f'{sub_html}</div>{_card_grid(cards)}{context}'
        "</section>"
    )


def _pxmeter_widget(output: dict[str, Any]) -> str:
    """PXMeter widget — structure-accuracy gauges grouped into Complex, per
    Chain, and per Interface sections, read from the PXMeter report JSON."""
    if not output:
        return ""
    explainer = (
        '<p class="widget-explainer">'
        "<strong>PXMeter</strong> scores a predicted structure against a reference — "
        "for the whole complex, each chain, and each chain-chain interface. Every "
        "metric below is shown as a gauge."
        "</p>"
    )
    report = _load_json_artifact(output.get("report_json_path"))
    if not isinstance(report, dict):
        # Report JSON unavailable — fall back to the flat per-run lDDT.
        metrics = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}
        card = _px_metric_card("lddt", metrics.get("lddt"))
        body = (
            _card_grid([card]) if card
            else '<p class="note">The PXMeter report is not embedded in this report.</p>'
        )
        return (
            '<div class="widget widget-pxmeter">'
            '<h4 class="widget-title">Structure accuracy (PXMeter)</h4>'
            f"{explainer}{body}</div>"
        )

    sections: list[str] = []
    cx = report.get("complex")
    if isinstance(cx, dict):
        sections.append(_px_section("Complex", "whole structure vs reference", cx))
    chains = report.get("chain")
    ref_info = report.get("ref_chain_info") if isinstance(report.get("ref_chain_info"), dict) else {}
    if isinstance(chains, dict):
        for cid, cmetrics in chains.items():
            if isinstance(cmetrics, dict):
                etype = ""
                if isinstance(ref_info.get(cid), dict):
                    etype = str(ref_info[cid].get("entity_type") or "")
                sections.append(_px_section(f"Chain {cid}", etype, cmetrics))
    ifaces = report.get("interface")
    if isinstance(ifaces, dict):
        for iid, imetrics in ifaces.items():
            if isinstance(imetrics, dict):
                sections.append(
                    _px_section(f"Interface {iid}", "chain-chain interface", imetrics)
                )
    entry = report.get("entry_id")
    header = (
        f'<p class="note">Scored against reference entry <code>{_h(str(entry))}</code>.</p>'
        if entry else ""
    )
    if not sections:
        sections.append('<p class="note">The PXMeter report carried no metrics.</p>')
    return (
        '<div class="widget widget-pxmeter">'
        '<h4 class="widget-title">Structure accuracy (PXMeter)</h4>'
        f"{explainer}{header}{''.join(sections)}"
        "</div>"
    )


def _lit_normalize(r: dict[str, Any]) -> dict[str, Any]:
    """Normalise a EuropePMC result (or a tool ``top_records`` entry) into a
    common record shape used by the literature-search cards."""
    return {
        "title": str(r.get("title") or "").strip().rstrip("."),
        "authors": str(r.get("authorString") or r.get("authors") or "").strip(),
        "journal": str(r.get("journalTitle") or r.get("journal") or "").strip(),
        "volume": str(r.get("journalVolume") or "").strip(),
        "issue": str(r.get("issue") or "").strip(),
        "page": str(r.get("pageInfo") or "").strip(),
        "year": str(r.get("pubYear") or r.get("year") or "").strip(),
        "doi": (str(r["doi"]).strip() or None) if r.get("doi") else None,
        "pmid": (str(r["pmid"]).strip() or None) if r.get("pmid") else None,
        "pmcid": (str(r["pmcid"]).strip() or None) if r.get("pmcid") else None,
        "source": (str(r.get("source")).strip() or None) if r.get("source") else None,
        "open_access": str(r.get("isOpenAccess") or "").upper() == "Y",
        "cited_by": r.get("citedByCount"),
        "date": str(r.get("firstPublicationDate") or r.get("first_publication_date") or ""),
        "snippet": str(r.get("abstract_snippet") or r.get("abstractText") or "").strip(),
    }


def _lit_records_from_raw(raw_pages: Any) -> list[dict[str, Any]]:
    """Flatten EuropePMC raw search pages into a list of normalised records."""
    out: list[dict[str, Any]] = []
    if not isinstance(raw_pages, list):
        return out
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        results = page.get("resultList")
        results = results.get("result") if isinstance(results, dict) else None
        if isinstance(results, list):
            out.extend(_lit_normalize(r) for r in results if isinstance(r, dict))
    return out


def _lit_record_link(rec: dict[str, Any]) -> str:
    """Best canonical URL for a record — DOI first, then PMC, then PubMed."""
    if rec.get("doi"):
        return f"https://doi.org/{rec['doi']}"
    if rec.get("pmcid"):
        return f"https://europepmc.org/article/PMC/{rec['pmcid']}"
    if rec.get("pmid"):
        return f"https://europepmc.org/article/MED/{rec['pmid']}"
    return ""


def _lit_card(rec: dict[str, Any], idx: int) -> str:
    """Render one literature result as a structured card."""
    title = _h(rec.get("title") or "Untitled record")
    link = _lit_record_link(rec)
    title_html = (
        f'<a class="lit-title" href="{_attr(link)}" target="_blank" '
        f'rel="noopener noreferrer">{title}</a>'
        if link else f'<span class="lit-title">{title}</span>'
    )
    oa = (
        '<span class="lit-badge lit-oa">Open access</span>'
        if rec.get("open_access") else ""
    )
    cite: list[str] = []
    if rec.get("journal"):
        cite.append(f'<em class="lit-journal">{_h(rec["journal"])}</em>')
    vip = rec.get("volume") or ""
    if rec.get("issue"):
        vip += f'({rec["issue"]})'
    if rec.get("page"):
        vip += f':{rec["page"]}'
    if vip:
        cite.append(_h(vip))
    if rec.get("year"):
        cite.append(_h(str(rec["year"])))
    cited = rec.get("cited_by")
    if isinstance(cited, int) and cited > 0:
        cite.append(f"cited by {cited}")
    chips: list[str] = []
    if rec.get("doi"):
        chips.append(
            f'<a class="lit-id" href="https://doi.org/{_attr(rec["doi"])}" '
            f'target="_blank" rel="noopener noreferrer">DOI</a>'
        )
    if rec.get("pmid"):
        chips.append(
            f'<a class="lit-id" href="https://europepmc.org/article/MED/'
            f'{_attr(rec["pmid"])}" target="_blank" rel="noopener noreferrer">'
            f'PMID {_h(rec["pmid"])}</a>'
        )
    if rec.get("pmcid"):
        chips.append(
            f'<a class="lit-id" href="https://europepmc.org/article/PMC/'
            f'{_attr(rec["pmcid"])}" target="_blank" rel="noopener noreferrer">'
            f'{_h(rec["pmcid"])}</a>'
        )
    snippet = (
        f'<p class="lit-snippet">{_h(rec["snippet"])}</p>'
        if rec.get("snippet") else ""
    )
    return (
        '<article class="lit-card">'
        f'<div class="lit-card-num">{idx}</div>'
        '<div class="lit-card-body">'
        f'<div class="lit-card-head">{title_html}{oa}</div>'
        + (f'<div class="lit-authors">{_h(rec["authors"])}</div>' if rec.get("authors") else "")
        + (f'<div class="lit-cite">{" · ".join(cite)}</div>' if cite else "")
        + snippet
        + (f'<div class="lit-ids">{"".join(chips)}</div>' if chips else "")
        + "</div></article>"
    )


def _literature_search_widget(output: dict[str, Any]) -> str:
    """Literature-search widget — query / backend summary plus a structured
    card per retrieved record (title + link + citation + snippet)."""
    if not output:
        return ""
    query = output.get("query")
    backend = output.get("backend")
    runtime = _num(output.get("runtime_ms"))
    result_count = _num(output.get("result_count"))
    date_filter = (
        output.get("date_filter") if isinstance(output.get("date_filter"), dict) else {}
    )

    # Prefer the rich raw EuropePMC pages; fall back to the curated top_records.
    raw_pages = None
    artifacts = output.get("artifacts")
    if isinstance(artifacts, dict):
        raw_pages = _load_json_artifact(artifacts.get("raw_pages_json"))
    records = _lit_records_from_raw(raw_pages)
    top = output.get("top_records") if isinstance(output.get("top_records"), list) else []
    if not records:
        records = [_lit_normalize(r) for r in top if isinstance(r, dict)]
    else:
        # Carry over abstract snippets from top_records (keyed by id).
        snip: dict[str, str] = {}
        for r in top:
            if not isinstance(r, dict):
                continue
            s = str(r.get("abstract_snippet") or "").strip()
            if not s:
                continue
            for key in (r.get("pmid"), r.get("doi"), str(r.get("title") or "").lower()):
                if key:
                    snip[str(key)] = s
        for rec in records:
            if rec["snippet"]:
                continue
            for key in (rec.get("pmid"), rec.get("doi"), rec["title"].lower()):
                if key and str(key) in snip:
                    rec["snippet"] = snip[str(key)]
                    break

    hit_count = None
    if isinstance(raw_pages, list) and raw_pages and isinstance(raw_pages[0], dict):
        hit_count = raw_pages[0].get("hitCount")

    stats: list[str] = []
    if backend:
        stats.append(f'<span class="lit-stat">backend <b>{_h(str(backend))}</b></span>')
    if hit_count is not None:
        stats.append(
            f'<span class="lit-stat"><b>{len(records)}</b> shown · '
            f'<b>{int(_num(hit_count) or 0)}</b> total hits</span>'
        )
    elif result_count is not None:
        stats.append(f'<span class="lit-stat"><b>{int(result_count)}</b> results</span>')
    if runtime is not None:
        stats.append(f'<span class="lit-stat"><b>{int(runtime)}</b> ms</span>')
    pa = date_filter.get("published_after")
    pb = date_filter.get("published_before")
    if pa or pb:
        stats.append(
            f'<span class="lit-stat">dates {_h(str(pa or "..."))} - {_h(str(pb or "..."))}</span>'
        )

    query_html = (
        '<div class="lit-query"><span class="lit-query-label">Search query</span>'
        f'<code class="lit-query-text">{_h(str(query))}</code></div>'
        if query else ""
    )
    summary_html = (
        '<div class="lit-summary">'
        f'{query_html}<div class="lit-stats">{"".join(stats)}</div>'
        '</div>'
    )
    if records:
        cards = "".join(_lit_card(rec, i + 1) for i, rec in enumerate(records))
        body = f'<div class="lit-list">{cards}</div>'
    else:
        body = '<p class="note">No literature records were returned for this query.</p>'

    explainer = (
        '<p class="widget-explainer">'
        "<strong>Literature search</strong> queries a scientific-literature backend and "
        "returns the top matching publications. Each card links out to the article; the "
        "snippet, when the backend provides one, shows the opening of the abstract."
        "</p>"
    )
    return (
        '<div class="widget widget-lit">'
        '<h4 class="widget-title">Literature search</h4>'
        f"{explainer}{summary_html}{body}"
        "</div>"
    )


# ----------------------------------------------------------- empty-output fallback


# Keys whose values are sequences (residue strings) — surfaced as a dedicated
# card so the reader sees length + first-30 aa preview rather than a 400-char
# unreadable string.
_SEQUENCE_OUTPUT_KEYS = (
    "sequence", "designed_sequence", "wild_type_sequence", "mutant_sequence",
    "query_sequence", "edited_sequence", "fasta", "a3m",
)


def _looks_like_sequence(value: str) -> bool:
    """Heuristic: a long, mostly-uppercase amino-acid string is a sequence.

    Used by the empty-output fallback widget to label string outputs with
    their plain-English description ("Sequence (123 residues)" vs
    "filepath" vs "free text").
    """
    if not isinstance(value, str) or len(value) < 6:
        return False
    sample = value[:200]
    aa_chars = set("ACDEFGHIKLMNPQRSTVWY-X")
    upper_letters = sum(1 for ch in sample if ch.isalpha())
    aa_matches = sum(1 for ch in sample.upper() if ch in aa_chars)
    if upper_letters == 0:
        return False
    # 90%+ of letters look like residues → sequence.
    return aa_matches / max(upper_letters, 1) >= 0.85


def _output_shape_widget(output: dict[str, Any]) -> str:
    """Fallback widget for tools that emit no scalar metrics.

    Many adapters legitimately produce only sequence strings, file paths or
    lists (e.g. ``sequence_edit`` returning ``mutant_sequence`` + a FASTA
    path). Rather than rendering a blank widget body, we surface a typed
    inventory of what the tool produced so the user can still navigate the
    output: sequences get a length + preview, files get a download chip
    inside the body, lists get a count.
    """
    items: list[str] = []
    for raw_key, value in output.items():
        key = str(raw_key)
        pretty = _pretty_metric_name(key)
        if value is None:
            continue
        if isinstance(value, str):
            if key.lower() in _SEQUENCE_OUTPUT_KEYS or _looks_like_sequence(value):
                preview = value[:48] + ("…" if len(value) > 48 else "")
                items.append(
                    f'<li class="output-shape-row">'
                    f'<span class="output-shape-icon" aria-hidden="true">~</span>'
                    f'<div class="output-shape-body">'
                    f'<div class="output-shape-name">{_h(pretty)}</div>'
                    f'<div class="output-shape-meta">Sequence · {len(value)} residues</div>'
                    f'<code class="output-shape-preview">{_h(preview)}</code>'
                    "</div></li>"
                )
                continue
            # Plain string scalar — short context.
            short = value if len(value) <= 160 else value[:160] + "…"
            items.append(
                f'<li class="output-shape-row">'
                f'<span class="output-shape-icon" aria-hidden="true">•</span>'
                f'<div class="output-shape-body">'
                f'<div class="output-shape-name">{_h(pretty)}</div>'
                f'<div class="output-shape-meta">Text · {len(value)} chars</div>'
                f'<code class="output-shape-preview">{_h(short)}</code>'
                "</div></li>"
            )
            continue
        if isinstance(value, list):
            count = len(value)
            kind = "items"
            if value and all(isinstance(v, str) for v in value):
                kind = "strings"
            elif value and all(isinstance(v, dict) for v in value):
                kind = "records"
            elif value and all(isinstance(v, (int, float)) for v in value):
                kind = "numbers"
            items.append(
                f'<li class="output-shape-row">'
                f'<span class="output-shape-icon" aria-hidden="true">[]</span>'
                f'<div class="output-shape-body">'
                f'<div class="output-shape-name">{_h(pretty)}</div>'
                f'<div class="output-shape-meta">List · {count} {kind}</div>'
                "</div></li>"
            )
            continue
        if isinstance(value, dict):
            count = len(value)
            items.append(
                f'<li class="output-shape-row">'
                f'<span class="output-shape-icon" aria-hidden="true">{{}}</span>'
                f'<div class="output-shape-body">'
                f'<div class="output-shape-name">{_h(pretty)}</div>'
                f'<div class="output-shape-meta">Object · {count} fields</div>'
                "</div></li>"
            )
            continue
    items = items[:10]
    if not items:
        return ""
    explainer = (
        '<p class="widget-explainer">'
        "This tool produced no numeric metrics. The inventory below shows "
        "what its output payload contains; downloadable files are listed in "
        "the Artifacts section underneath."
        "</p>"
    )
    return (
        '<div class="widget widget-generic widget-shape">'
        '<h4 class="widget-title">Output inventory</h4>'
        f"{explainer}"
        f'<ul class="output-shape-list">{"".join(items)}</ul>'
        "</div>"
    )


def _kv_row(label: str, value: str) -> str:
    return f"<tr><th>{_h(label)}</th><td>{value if _looks_html(value) else _h(value)}</td></tr>"


def _looks_html(value: str) -> bool:
    return value.startswith("<") and "</" in value


_QUALITY_COLOR = {
    # Bar/donut quality colours. Mapping kept consistent with the CSS bar
    # rule: good = sky (cool calm), ok = orange (warm caution),
    # poor = pink (warm critical), neutral = yellow/orange (informational
    # — distinct from the cool "good" sky so the reader can tell a
    # threshold-less metric apart from a passing one).
    "good": "#4cc4e4",     # matches --good
    "ok": "#f39e3a",       # matches --ok
    "poor": "#cc0980",     # matches --poor
    "neutral": "#f39e3a",  # matches --accent-4
}


def _fmt_donut(value: object) -> str:
    """Compact numeric formatting tuned for the gauge centre.

    The default report-wide :func:`_fmt` keeps 3 sig figs (so 0.628 stays
    ``0.628``), which crowds the donut interior. Inside the ring we prefer
    a fixed-decimals look that lines up across cards: pLDDT ``48.5``,
    GRAVY ``-0.63``, pTM ``0.74``, count ``0``, runtime ``4231``.
    """
    num = _num(value)
    if num is None:
        return "n/a" if value is None else str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if abs(num - round(num)) < 1e-9 and abs(num) < 1e6:
        return f"{round(num)}"
    abs_n = abs(num)
    if abs_n < 0.01:
        return f"{num:.2e}"
    if abs_n < 10:
        return f"{num:.2f}"
    if abs_n < 100:
        return f"{num:.1f}"
    if abs_n < 1e5:
        return f"{num:.0f}"
    return f"{num:.2g}"


# Short, donut-friendly labels for metric names that would otherwise wrap or
# overflow inside an 84-90px ring. Keys are case-insensitive matches against
# the display name passed to :func:`_score_card`.
_DONUT_SHORT_LABELS: dict[str, str] = {
    "mean pldd": "pLDDT",  # matches "Mean pLDDT"
    "pldd": "pLDDT",
    "instability index": "STAB",
    "isoelectric point": "pI",
    "polybasic motifs": "POLY",
    "aromatic fraction": "AROM",
    "binder probability": "PROB",
    "binary binder probability": "PROB",
    "binary binding probability": "PROB",
    "binding probability": "PROB",
    "ranking confidence": "CONF",
    "ranking score": "RANK",
    "model confidence": "CONF",
    "seq recovery": "RECOV",
    "ligand contacts": "LIG",
    "interface residues": "IFACE",
    "low complexity runs": "LCR",
    "predicted affinity": "AFFIN",
    "affinity (-log10 ic50)": "AFFIN",
    "% helix": "HELIX",
    "% sheet": "SHEET",
    "% loop": "LOOP",
    "helix": "HELIX",
    "sheet": "SHEET",
    "loop": "LOOP",
    "buried polar": "BURY",
    "buried polar candidates": "BURY",
    "dockq": "DockQ",
    "ipsae": "ipSAE",
}


def _donut_label(label: str) -> str:
    """Pick a short, all-caps label that fits inside the donut.

    Long display names (``Instability Index``) get abbreviated; short ones
    pass through uppercased. The fallback keeps the first word so unknown
    metrics still produce a readable interior label.

    Matching strategy: exact lookup first, then longest-prefix match (so
    ``"% loop"`` beats ``"loop"``-style accidental hits), then a first-word
    fallback. Bare punctuation prefixes like ``"%"`` are stripped before
    the first-word fallback so e.g. ``"% Helix"`` becomes ``HELIX`` instead
    of ``%``.
    """
    canon = label.strip().lower()
    if canon in _DONUT_SHORT_LABELS:
        return _DONUT_SHORT_LABELS[canon]
    # Greek and short tokens like "pI", "ΔG", "PAE" stay as-is.
    if len(label) <= 6 and not label.startswith("%"):
        return label
    # Longest-prefix / contains match wins (so "instability index" wins over
    # a single-word substring inside another label).
    best_match: tuple[int, str] | None = None
    for key, short in _DONUT_SHORT_LABELS.items():
        if canon.startswith(key) or (" " in key and key in canon):
            if best_match is None or len(key) > best_match[0]:
                best_match = (len(key), short)
    if best_match is not None:
        return best_match[1]
    # Strip leading punctuation so "% Helix" -> "Helix" -> "HELIX".
    cleaned = label.lstrip("% \t").strip()
    if cleaned and cleaned.lower() in _DONUT_SHORT_LABELS:
        return _DONUT_SHORT_LABELS[cleaned.lower()]
    first = cleaned.split()[0] if cleaned.split() else (cleaned or label)
    return first[:6].upper()


def _gauge_svg(
    *,
    value: float | None,
    lo: float,
    hi: float,
    quality: str,
    size: int = 88,
    stroke: int = 7,
    center_text: str | None = None,
    sub_text: str | None = None,
    frame: bool = False,
) -> str:
    """Smooth donut gauge with value (and optional sub-label) inside.

    A single continuous arc traces the value's position within ``[lo, hi]``
    against a neutral background ring. The big number sits at the centre;
    an optional short sub-label (e.g. ``pLDDT``) sits one line below so the
    user can identify the metric at a glance without reading the body text.

    With ``frame=True`` the ring is drawn as a single complete neutral
    circle instead of a proportional arc — used for numeric metrics that
    have no defensible ``[lo, hi]`` range, so they still render as a
    circular card without implying a misleading fill percentage.

    Stays offline-pure SVG (no fonts, no images, no JS).
    """
    if value is None or not isinstance(value, (int, float)):
        return ""

    cx = cy = size / 2
    r = size / 2 - stroke

    if frame:
        # Range-less metric: a complete soft-lavender ring reads as a
        # circular "medallion" around the number rather than a 0%/100% gauge.
        ring_html = (
            f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" '
            f'stroke="#c9bbed" stroke-width="{stroke}"/>'
        )
    else:
        span = (hi - lo) if hi > lo else 1.0
        pct = max(0.0, min(1.0, (float(value) - lo) / span))
        colour = _QUALITY_COLOR.get(quality, _QUALITY_COLOR["neutral"])
        bg_color = "#ede9fe"

        circ = 2 * math.pi * r
        arc_len = pct * circ
        # A single continuous ring: full background circle behind, then a
        # colour-filled foreground arc traced via stroke-dasharray. Starting
        # at 12 o'clock by rotating -90deg around the centre.
        ring_html = (
            f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" '
            f'stroke="{bg_color}" stroke-width="{stroke}"/>'
        )
        if pct > 0:
            ring_html += (
                f'<circle cx="{cx}" cy="{cy}" r="{r:.2f}" fill="none" '
                f'stroke="{colour}" stroke-width="{stroke}" stroke-linecap="round" '
                f'stroke-dasharray="{arc_len:.2f} {circ - arc_len:.2f}" '
                f'transform="rotate(-90 {cx} {cy})"/>'
            )

    text = _h(center_text) if center_text is not None else _h(_fmt_donut(value))
    # Auto-shrink the value text when it gets long enough to crowd the ring
    # interior (e.g. a 6-char "-1.23e3" or a 4-char "4231").
    char_count = len(text)
    if char_count >= 6:
        scale = 0.20
    elif char_count >= 5:
        scale = 0.22
    elif char_count >= 4:
        scale = 0.24
    else:
        scale = 0.27
    value_size = max(14, int(size * scale))

    text_blocks: list[str] = []
    if sub_text:
        sub_size = max(8, int(size * 0.105))
        # Stack value+sub roughly centred around cy with a small gap.
        value_y = cy - sub_size * 0.55
        sub_y = cy + value_size * 0.55
        text_blocks.append(
            f'<text x="{cx}" y="{value_y:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" class="gauge-value" '
            f'style="font-size:{value_size}px;">{text}</text>'
        )
        text_blocks.append(
            f'<text x="{cx}" y="{sub_y:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" class="gauge-sub" '
            f'style="font-size:{sub_size}px;">{_h(sub_text)}</text>'
        )
    else:
        text_blocks.append(
            f'<text x="{cx}" y="{cy}" text-anchor="middle" '
            f'dominant-baseline="central" class="gauge-value" '
            f'style="font-size:{value_size}px;">{text}</text>'
        )

    return (
        f'<svg class="metric-gauge" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}" aria-hidden="true">'
        + ring_html
        + "".join(text_blocks)
        + "</svg>"
    )


def _balanced_cols(n: int, max_cols: int = 4) -> int:
    """Pick a column count that fills rows evenly for ``n`` cards.

    Avoids the ugly "4 + 1 orphan" layout the default auto-fit grid produces:

    * 1 card   -> 1 col
    * 2 cards  -> 2 col
    * 3 cards  -> 3 col
    * 4 cards  -> 4 col (one row)
    * 5 cards  -> 3 col  (3 + 2, no orphan)
    * 6 cards  -> 3 col  (3 + 3)
    * 7 cards  -> 4 col  (4 + 3)
    * 8 cards  -> 4 col  (4 + 4)
    * 9 cards  -> 3 col  (3 + 3 + 3)
    * 12 cards -> 4 col  (4 + 4 + 4)

    Capped at ``max_cols`` so cards don't get crushed below the readable
    width — narrow viewports collapse further via media queries.
    """
    if n <= 1:
        return 1
    if n <= max_cols:
        return n
    best_c = max_cols
    best_score = -10_000
    for c in range(2, max_cols + 1):
        rows = math.ceil(n / c)
        leftover = rows * c - n  # empty slots in the last row (orphan count)
        # Penalise both orphan cells and excessively-tall layouts. Without
        # the rows penalty, n=10 picks 2 columns (5 rows of 2) just because
        # leftover is 0 — visually that's worse than 4+4+2.
        score = -leftover * 3 - rows * 3 + c
        if score > best_score:
            best_score = score
            best_c = c
    return best_c


def _card_grid(cards: list[str]) -> str:
    """Wrap a list of cards in a balanced-column grid.

    Every per-tool widget that builds a card grid should funnel through
    this helper instead of hard-coding ``<div class="qc-card-grid">`` so
    the report stays consistent and never produces orphan-row layouts.
    """
    if not cards:
        return ""
    cols = _balanced_cols(len(cards))
    return (
        f'<div class="qc-card-grid" style="--cols:{cols}">'
        f'{"".join(cards)}</div>'
    )


def _score_card(
    label: str,
    value: str,
    explanation: str,
    quality: str,
    *,
    gauge: dict[str, Any] | None = None,
) -> str:
    """Donut-centric metric card.

    When a ``gauge`` is provided (range + numeric value), the donut becomes
    the card's visual centrepiece on the left, with the value rendered in
    the centre of the ring. The label + explainer sit on the right. When
    no gauge data is available, the card falls back to a clean number-and-
    label layout. Either way the card stays compact and tile-shaped, which
    keeps grids of cards aligned across widgets.
    """
    quality = quality if quality in {"good", "ok", "poor", "neutral"} else "neutral"
    if gauge and gauge.get("value") is not None:
        # Re-format the centre text with the donut-tuned formatter so
        # adapters that pass ``value="-0.628"`` (raw 3-sig-fig output)
        # collapse to the cleaner ``-0.63`` shown in the ring.
        center_value = _fmt_donut(gauge["value"])
        gauge_html = _gauge_svg(
            value=gauge["value"],
            lo=gauge.get("lo", 0.0),
            hi=gauge.get("hi", 1.0),
            quality=quality,
            size=88,
            stroke=7,
            center_text=center_value,
            sub_text=_donut_label(label),
            frame=bool(gauge.get("frame", False)),
        )
        # The full label stays visible at the top of the body so the reader
        # always sees the metric's full name; the donut interior shows a
        # condensed version of the same label for at-a-glance ID.
        return (
            f'<article class="qc-card has-gauge metric-{quality}">'
            f'<div class="qc-card-viz">{gauge_html}</div>'
            f'<div class="qc-card-body">'
            f'<div class="qc-card-name">{_h(label)}</div>'
            f'<p class="qc-card-explain">{_h(explanation)}</p>'
            "</div></article>"
        )
    # Fallback: no gauge data — keep the simple value-and-label header.
    return (
        f'<article class="qc-card metric-{quality}">'
        "<header>"
        f"<div class='qc-card-figure'><strong>{_h(value)}</strong>"
        f'<span class="qc-card-name">{_h(label)}</span></div>'
        "</header>"
        f'<p class="qc-card-explain">{_h(explanation)}</p>'
        "</article>"
    )


def _compact_inline(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return _fmt(value)
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "…"
    return _short(json.dumps(value, default=str), 200)


def _repro_html(model: ReportModel) -> str:
    """Reproducibility block — show only the replay command(s).

    Per the simplified report contract, the long lists of IDs / tool versions
    / params hashes / artifact SHA-256 are noisy chrome that's already
    embedded in the per-tool widget metadata and the `report-data` JSON
    payload at the bottom of the page. The replay command is the one piece
    of information a human actually re-uses, so it's the only thing we
    surface here.
    """
    repro = model.reproducibility
    if repro is None or not repro.replay_commands:
        return "<p class='empty'>No replay command available.</p>"
    commands = "".join(
        f'<div class="cmd"><pre>{_h(cmd)}</pre>'
        f'<button class="copy-btn" data-copy="{_attr(cmd)}">Copy</button></div>'
        for cmd in repro.replay_commands
    )
    return (
        '<p class="note">Run this command in the same workspace to regenerate '
        'this report from the stored job results — no recomputation.</p>'
        f'{commands}'
    )


# Priority order for candidate score bars. Anything in this tuple lands at
# the top of the candidate card; remaining scalars from the candidate's
# ``.scores`` map appear below in the order they were emitted. Context-only
# keys (residue count, runtime) are filtered out — they're not bar-worthy.
_SCORE_BAR_PRIORITY: tuple[str, ...] = (
    "plddt",
    "iplddt",
    "ptm",
    "iptm",
    "ranking_confidence",
    "confidence_score",
    "confidence",
    "model_confidence",
    "ipsae",
    "ipsae_score",
    "dockq",
    "affinity",
    "affinity_value",
    "affinity_pred_value",
    "affinity_probability",
    "affinity_probability_binary",
    "binding_probability",
    "delta_g",
    "dg",
    "recovery",
    "seq_recovery",
    "ranking_score",
    "score",
    "instability_index",
    "gravy",
    "pi",
    "polybasic_hits",
    "aromatic_fraction",
)

# Context-only metrics that should never appear as score bars (they belong
# in the candidate header / footer instead).
_SCORE_BAR_SKIP: frozenset[str] = frozenset({
    "num_residues", "length", "runtime_ms", "n_residues",
    "elapsed_ms", "elapsed_s", "seed",
})


def _score_bars(candidate: ReportCandidate) -> str:
    """Stacked metric bars — one row per unique canonical metric.

    The candidate may carry several entries for the same metric (e.g.
    ``plddt`` from the ranker + ``protein_qc:plddt`` + ``boltz2_affinity:plddt``).
    The view shows only one bar per canonical metric so the reader isn't
    looking at 3 identical bars: the bare-key variant wins when present,
    otherwise the first prefixed variant in stable order. The label is
    rendered cleanly without any ``· tool`` suffix — provenance is kept in
    the row's ``title`` tooltip for hover detail.
    """
    if not candidate.scores and candidate.score is None:
        return ""
    combined: dict[str, Any] = {}
    if candidate.score is not None:
        combined["score"] = candidate.score
    combined.update(candidate.scores)

    # Alias collapse — ``mean_*`` flavours merge into the canonical key
    # when both exist. The provenance-prefixed variants (``tool:plddt``,
    # ``tool:mean_plddt``) collapse via the same logic per-tool below.
    def _collapse_aliases(scores: dict[str, Any], prefix: str = "") -> None:
        canonical = f"{prefix}plddt"
        mean = f"{prefix}mean_plddt"
        if canonical in scores and mean in scores:
            scores.pop(mean, None)
        elif mean in scores and canonical not in scores:
            scores[canonical] = scores.pop(mean)
        canonical = f"{prefix}pae"
        mean = f"{prefix}mean_pae"
        if canonical in scores and mean in scores:
            scores.pop(mean, None)
        elif mean in scores and canonical not in scores:
            scores[canonical] = scores.pop(mean)
        if f"{prefix}delta_g" in scores and f"{prefix}dg" in scores:
            scores.pop(f"{prefix}dg", None)
        elif f"{prefix}dg" in scores and f"{prefix}delta_g" not in scores:
            scores[f"{prefix}delta_g"] = scores.pop(f"{prefix}dg")

    _collapse_aliases(combined)
    tools_seen: set[str] = set()
    for key in list(combined.keys()):
        if ":" in key:
            tools_seen.add(key.split(":", 1)[0])
    for tool in tools_seen:
        _collapse_aliases(combined, prefix=f"{tool}:")

    # Dedup by canonical base key — keep one entry per metric. Bare keys
    # (no tool prefix) win because they came from the ranker / candidate
    # row itself; among prefixed variants we keep the alphabetically-first
    # tool for stable rendering.
    by_base: dict[str, list[str]] = {}
    for key in combined:
        base = key.split(":", 1)[-1].lower()
        by_base.setdefault(base, []).append(key)
    deduped: dict[str, Any] = {}
    for _base, keys in by_base.items():
        bare = [k for k in keys if ":" not in k]
        if bare:
            chosen = bare[0]
        else:
            chosen = sorted(keys)[0]
        deduped[chosen] = combined[chosen]

    def _priority(key: str) -> tuple[int, str]:
        base = key.split(":", 1)[-1].lower()
        try:
            base_idx = _SCORE_BAR_PRIORITY.index(base)
        except ValueError:
            base_idx = len(_SCORE_BAR_PRIORITY)
        return (base_idx, key)

    # Tool that produced this candidate — used by ``_score_bar_row`` to
    # pick the right per-tool explainer for ambiguous metric names like
    # ``score``. Falls back to "" when source is missing so the bare-metric
    # registry explainer wins.
    candidate_tool = ""
    if isinstance(candidate.source, dict):
        candidate_tool = str(candidate.source.get("tool") or "")

    # Monomer candidates can't have an interface — drop ipTM / ipSAE /
    # DockQ / binding-affinity rows entirely so the card doesn't claim a
    # "weak interface" verdict for a single-chain structure.
    monomer = _is_monomer_candidate(candidate)

    rows: list[str] = []
    for key in sorted(deduped.keys(), key=_priority):
        base_key = key.split(":", 1)[-1].lower()
        if base_key in _SCORE_BAR_SKIP:
            continue
        if monomer and _is_interface_only_metric(base_key):
            continue
        row = _score_bar_row(key, deduped[key], candidate_tool=candidate_tool)
        if row:
            rows.append(row)
    rows = rows[:14]
    if not rows:
        return ""
    return "<div class='score-bars'>" + "".join(rows) + "</div>"


def _score_bar_row(key: str, value: Any, *, candidate_tool: str = "") -> str:
    """Render a single metric row with a percentage-filled bar.

    Bar width is computed from the metric's registered range in
    :data:`_METRIC_REGISTRY` so e.g. ``delta_g = -8`` becomes 50% of a
    -16..0 bar, ``ranking_confidence = 0.7`` becomes 70% of a 0..1 bar,
    etc. Unknown metrics fall back to a 0..1 (probability) range when
    ``|value| <= 1.5`` and 0..100 otherwise, so a stray adapter scalar
    still produces a plausible bar.

    ``key`` may carry a ``tool:metric`` provenance prefix emitted by
    :func:`_attach_qc_and_analysis` when multiple tools produce the same
    metric. The prefix renders as a small source tag next to the label so
    the reader sees ``pLDDT  ·  boltz2`` instead of an ambiguous bare
    ``pLDDT``.

    Each row carries a definition tooltip — pulled from the per-tool
    override table or the bare-metric registry — and an inline ``ⓘ`` chip
    next to the label so users discover the hover. Ambiguous metrics
    (notably ``score`` for the MPNN family of inverse-folding tools) get a
    tool-specific explanation instead of the generic "composite score"
    placeholder.
    """
    num = _num(value)
    if num is None:
        return ""
    # Strip provenance prefix for registry / quality lookup.
    if ":" in key:
        tool_prefix, base_key = key.split(":", 1)
    else:
        tool_prefix, base_key = "", key
    canonical = base_key.lower()
    spec = _METRIC_REGISTRY.get(canonical)
    if spec is not None:
        label, _explain, lo, hi = spec
        # Dual-scale autoscale (pLDDT: 0-1 vs 0-100, recovery: same, etc.)
        if canonical in _DUAL_SCALE_METRICS and num <= 1.5 and hi >= 50.0:
            hi = 1.0
    else:
        label = _pretty_metric_name(canonical)
        # No registered range — guess a sensible one from the value scale.
        if abs(num) <= 1.5:
            lo, hi = 0.0, 1.0
        elif abs(num) <= 100:
            lo, hi = 0.0, 100.0
        else:
            lo, hi = 0.0, abs(num) * 1.2
    span = (hi - lo) if hi > lo else 1.0
    width = max(3.0, min(100.0, ((num - lo) / span) * 100.0))
    quality = _quality_class(canonical, num)

    # Compose the row tooltip: definition first, then source provenance.
    # Tool-prefixed metric keys (``boltz2:plddt``) win over candidate_tool
    # when picking the per-tool explainer override — the explicit prefix
    # is more authoritative.
    explainer_tool = tool_prefix or candidate_tool
    explainer = _metric_explainer(canonical, explainer_tool)
    source_tool = tool_prefix or candidate_tool
    title_parts: list[str] = []
    if explainer:
        title_parts.append(explainer)
    if source_tool:
        title_parts.append(f"Source: {source_tool}")
    title_text = "\n\n".join(title_parts)
    row_title = f' title="{_attr(title_text)}"' if title_text else ""
    # ``ⓘ`` chip beside the label so users notice the tooltip exists
    # without having to guess that the row is hoverable. Skipped when
    # there's no explainer to surface — empty info circle would mislead.
    info_chip = (
        f' <span class="score-info" title="{_attr(title_text)}" aria-label="metric definition">ⓘ</span>'
        if explainer
        else ""
    )

    return (
        f"<div class='score-row quality-{quality}'{row_title}>"
        f"<span>{_h(label)}{info_chip}</span>"
        f"<div class='bar-track'><i style='width:{width:.1f}%'></i></div>"
        f"<b>{_h(_fmt(num))}</b></div>"
    )


def _flag_badges(candidate: ReportCandidate) -> str:
    """Render QC flags + structure liabilities as deduped badge chips.

    Liabilities are grouped by ``type`` so multiple per-residue hits of the
    same kind collapse into a single chip with a count and residue list
    (``buried_polar_candidate x9 (A:15, 18, ...)``). QC flags are kept in
    order but deduplicated up-stream in ``_attach_qc_and_analysis``.
    """
    chips: list[tuple[str, str]] = []  # (label, tooltip)
    seen: set[str] = set()
    for flag in candidate.qc_flags:
        key = str(flag).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        chips.append((str(flag), str(flag)))

    # Group liabilities by ``type``
    groups: dict[str, list[Any]] = {}
    order: list[str] = []
    for entry in candidate.liabilities:
        if isinstance(entry, dict):
            t = str(entry.get("type") or "liability")
        else:
            t = str(entry)
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append(entry)
    for t in order:
        entries = groups[t]
        positions: list[str] = []
        for e in entries:
            if isinstance(e, dict):
                chain = e.get("chain_id") or e.get("chain") or ""
                pos = e.get("positions") or e.get("position")
                if isinstance(pos, list):
                    pos_label = ",".join(str(p) for p in pos)
                else:
                    pos_label = str(pos) if pos is not None else ""
                if pos_label:
                    positions.append(f"{chain}:{pos_label}" if chain else pos_label)
        count = len(entries)
        label = f"{t} x{count}" if count > 1 else t
        tooltip = (
            f"{t} at " + ", ".join(positions) if positions else t
        )
        chips.append((label, tooltip))

    if not chips:
        return ""  # Don't take screen real-estate for "no flags".
    return (
        "<div class='badges'>"
        + "".join(
            f"<span class='badge-warn' title='{_attr(tooltip)}'>{_h(label)}</span>"
            for label, tooltip in chips[:12]
        )
        + "</div>"
    )


def _warning_list(model: ReportModel) -> str:
    if not model.warnings:
        return ""
    return (
        "<div class='warnings'>"
        + "".join(
            f"<p class='{_h(w.level)}'><strong>{_h(w.level)}</strong> {_h(w.message)}"
            f"{' · ' + _h(w.source) if w.source else ''}</p>"
            for w in model.warnings
        )
        + "</div>"
    )


# --------------------------------------------------------------------------- agent transcript
#
# Three-tier panel for LLM-agent runs: original user query (You),
# plan / planning response (Agent plan), final answer (Final response),
# plus a collapsible full transcript at the bottom for users who want
# every assistant / tool turn. Renders nothing when the run wasn't
# LLM-planned (``model.agent_transcript is None``).

# Role icons for the chat-bubble transcript. Kept small and monochrome
# so the panel reads as a polished product, not a chat-app pastiche.
_AGENT_ROLE_BADGES: dict[str, tuple[str, str]] = {
    "user":      ("You",    "❝"),
    "assistant": ("Agent",  "✦"),
    "system":    ("System", "⚙"),
    "tool":      ("Tool",   "⌬"),
}


def _agent_transcript_section(transcript: AgentTranscript) -> str:
    """Top-level "Conversation" card for an LLM-agent run.

    Layout (top to bottom):
      1. Card header with eyebrow + title + meta chips
         (model, message count, tool-call count, round count).
      2. **You** — the user's original natural-language query.
      3. **Agent plan** — the plan response (structured + prose).
      4. **Final response** — the agent's wrap-up answer.
      5. **Execution timeline** — agent_summaries, one card per
         completed job/workflow, in chronological order. Only rendered
         when there's at least one step summary.
      6. **Decision audit** — agent_decisions, one row per round, only
         rendered for multi-round runs (≥1 decision recorded).
      7. **Full transcript** — collapsed ``<details>`` with every
         message AND tool call interleaved chronologically.
    """
    user_html = _agent_user_block(transcript.user_query)
    plan_html = _agent_plan_block(transcript)
    final_html = _agent_final_block(transcript.final_response)
    timeline_html = _agent_step_timeline_block(transcript.step_summaries)
    decisions_html = _agent_decisions_block(transcript.decisions)
    transcript_html = _agent_full_transcript_block(
        transcript.messages, transcript.tool_calls,
    )

    meta_chips: list[str] = []
    if transcript.model:
        meta_chips.append(
            f'<span class="conv-meta-chip">model · <b>{_h(transcript.model)}</b></span>'
        )
    if transcript.provider:
        meta_chips.append(
            f'<span class="conv-meta-chip">{_h(transcript.provider)}</span>'
        )
    msg_count = len(transcript.messages)
    if msg_count:
        meta_chips.append(
            f'<span class="conv-meta-chip">{msg_count} message{"s" if msg_count != 1 else ""}</span>'
        )
    tc_count = len(transcript.tool_calls)
    if tc_count:
        meta_chips.append(
            f'<span class="conv-meta-chip">{tc_count} tool call{"s" if tc_count != 1 else ""}</span>'
        )
    round_count = len(transcript.decisions)
    if round_count > 1:
        meta_chips.append(
            f'<span class="conv-meta-chip">{round_count} rounds</span>'
        )
    meta_html = (
        f'<div class="conv-meta">{"".join(meta_chips)}</div>'
        if meta_chips
        else ""
    )

    return f"""<section id="conversation" class="card card-conversation">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Conversation</p>
      <h2 class="card-title">Agent transcript</h2>
    </div>
    {meta_html}
  </header>
  <p class="widget-explainer">This run was orchestrated by an LLM agent — the prompt below was translated into the workflow you see in the rest of the report. The plan and final response are the agent's own narrative; the timeline, decision audit, and full transcript expose every step the agent took along the way.</p>
  {user_html}
  {plan_html}
  {final_html}
  {timeline_html}
  {decisions_html}
  {transcript_html}
</section>"""


def _agent_user_block(query: str) -> str:
    if not query:
        return ""
    return f"""<div class="conv-block conv-block-user">
  <div class="conv-block-head">
    <span class="conv-role-badge conv-role-badge-user">❝ You</span>
  </div>
  <blockquote class="conv-user-quote">{_render_agent_prose(query)}</blockquote>
</div>"""


# Friendly humanised labels for the decision_label strings emitted by the
# planner / DecisionEngine. Mapped on lookup; unknown labels fall back to
# a snake_case → "Sentence case" best effort via :func:`_pretty_field_name`.
_DECISION_LABEL_FRIENDLY: dict[str, str] = {
    "completed_no_additional_round_recommended": "Workflow completed",
    "completed_review_metrics":                   "Completed · review metrics",
    "needs_next_round":                           "Needs another round",
    "workflow_not_completed":                     "Workflow did not complete",
    "tool_selection":                             "Tool selection",
}


def _humanize_decision_label(label: str) -> str:
    if not label:
        return ""
    key = label.strip().lower()
    if key in _DECISION_LABEL_FRIENDLY:
        return _DECISION_LABEL_FRIENDLY[key]
    return _pretty_field_name(label)


def _plan_text_is_dump_of(plan_text: str, plan_structured: dict[str, Any] | None) -> bool:
    """Return True when ``plan_text`` is just a JSON dump of the same plan
    we already rendered structurally — in that case the prose render
    would be a duplicate noise block at the bottom. Tolerant to whitespace
    and key-order variations."""
    if not plan_text or not isinstance(plan_structured, dict):
        return False
    stripped = plan_text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    if not isinstance(parsed, dict):
        return False
    # If parsed shares >=2 keys with plan_structured, treat as the same payload.
    shared = set(parsed.keys()) & set(plan_structured.keys())
    return len(shared) >= 2 or parsed == plan_structured


def _agent_plan_block(transcript: AgentTranscript) -> str:
    """Render the agent's plan as a tight product-style summary card.

    Layout principle: the "core" (goal · planned steps · tool chips)
    stays prominent; everything else (assumptions, missing inputs,
    success criteria, plan rationale) collapses into one expandable
    "Plan details" element so the visible surface stays focused.

    If ``plan_text`` is a JSON dump of the same plan we already
    structured we skip the prose render entirely — otherwise we keep it
    as the agent's free-form narrative.
    """
    structured_html = ""
    if isinstance(transcript.plan_structured, dict):
        structured_html = _render_plan_structured(transcript.plan_structured)
    prose_html = ""
    if transcript.plan_text and not _plan_text_is_dump_of(
        transcript.plan_text, transcript.plan_structured,
    ):
        # Free-form narrative — keep it, but only if it doesn't start with a
        # raw JSON character (which would also be a serialised payload we
        # can't structurally lay out from plan_structured).
        stripped = transcript.plan_text.strip()
        if not stripped.startswith(("{", "[")):
            prose_html = (
                '<div class="conv-plan-prose">'
                f'{_render_agent_prose(transcript.plan_text)}'
                '</div>'
            )
    if not structured_html and not prose_html:
        return ""
    return f"""<div class="conv-block conv-block-plan">
  <div class="conv-block-head">
    <span class="conv-role-badge conv-role-badge-agent">✦ Agent plan</span>
  </div>
  {prose_html}
  {structured_html}
</div>"""


def _render_plan_structured(plan: dict[str, Any]) -> str:
    """Render the ``agent_plans.plan_json`` dict as a polished outline.

    Visual hierarchy (top → bottom):

    1. **Goal** — single lede paragraph.
    2. **Planned steps** — numbered list of ``tool · reason`` rows
       (the actual workflow the agent will execute).
    3. **Tool chips** — required + warm-in-advance tools as two
       side-by-side chip strips.
    4. **Plan details** — collapsible ``<details>`` carrying
       assumptions, missing inputs, success criteria, rationale,
       mode, and any extra fields. Default closed so the visible card
       stays compact.

    Recognised keys (with synonyms): ``goal``, ``workflow_plan``,
    ``required_tools`` / ``candidate_tools``, ``warm_tools``,
    ``assumptions``, ``missing_inputs``, ``success_criteria``,
    ``reason`` / ``rationale``, ``mode``. Anything else falls into
    the collapsible "Other fields" sublist so the structured renderer
    doesn't silently swallow data.
    """
    blocks: list[str] = []

    # ---- (1) Goal — lede paragraph ----
    goal = plan.get("goal")
    if isinstance(goal, str) and goal.strip():
        blocks.append(
            '<p class="conv-plan-goal">'
            f'{_render_agent_prose(goal.strip())}'
            '</p>'
        )

    # ---- (2) Planned steps ----
    workflow_plan = plan.get("workflow_plan")
    if isinstance(workflow_plan, list) and workflow_plan:
        steps = []
        for step in workflow_plan:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or step.get("name") or "step")
            reason = step.get("reason") or step.get("why") or step.get("purpose")
            steps.append(
                '<li>'
                f'<code class="conv-plan-tool">{_h(tool)}</code>'
                + (
                    f'<span class="conv-plan-reason">{_render_agent_prose(str(reason))}</span>'
                    if reason else ""
                )
                + '</li>'
            )
        if steps:
            blocks.append(
                '<div class="conv-plan-list conv-plan-list-steps">'
                '<h5 class="conv-plan-list-head">Planned steps</h5>'
                f'<ol>{"".join(steps)}</ol>'
                '</div>'
            )

    # ---- (3) Tool chips — required + warm in advance ----
    # ``candidate_tools`` is a synonym for ``required_tools`` emitted by the
    # tool-selection mode of the planner. Prefer the explicit required_tools
    # when both are present; otherwise candidate_tools takes its place.
    required = plan.get("required_tools")
    if not (isinstance(required, list) and required):
        required = plan.get("candidate_tools")
    warm = plan.get("warm_tools")
    tool_groups: list[tuple[str, list[Any]]] = []
    if isinstance(required, list) and required:
        tool_groups.append(("Required tools", required))
    if isinstance(warm, list) and warm and warm != required:
        tool_groups.append(("Warm in advance", warm))
    if tool_groups:
        cols = []
        for label, items in tool_groups:
            chips = "".join(
                f'<code class="conv-plan-tool-chip">{_h(str(item))}</code>'
                for item in items if item
            )
            if chips:
                cols.append(
                    '<div class="conv-plan-tools-col">'
                    f'<h5 class="conv-plan-list-head">{_h(label)}</h5>'
                    f'<div class="conv-plan-tool-chips">{chips}</div>'
                    '</div>'
                )
        if cols:
            blocks.append(
                f'<div class="conv-plan-tools-row">{"".join(cols)}</div>'
            )

    # ---- (4) Plan details — collapsed by default ----
    details_blocks: list[str] = []
    # Rationale prose (the planner's "why this plan" field).
    for key in ("reason", "rationale", "explanation"):
        v = plan.get(key)
        if isinstance(v, str) and v.strip():
            details_blocks.append(
                '<div class="conv-plan-detail-section">'
                '<h5 class="conv-plan-list-head">Plan rationale</h5>'
                f'<div class="conv-plan-rationale">{_render_agent_prose(v.strip())}</div>'
                '</div>'
            )
            break
    # Assumptions / missing inputs / success criteria as lists.
    for key, label in (
        ("assumptions",      "Assumptions"),
        ("missing_inputs",   "Missing inputs"),
        ("success_criteria", "Success criteria"),
    ):
        items = plan.get(key)
        if isinstance(items, list) and items:
            lis = "".join(
                f"<li>{_render_agent_prose(_stringify_final_item(item))}</li>"
                for item in items if item
            )
            if lis:
                details_blocks.append(
                    '<div class="conv-plan-detail-section">'
                    f'<h5 class="conv-plan-list-head">{_h(label)}</h5>'
                    f'<ul class="conv-plan-detail-list">{lis}</ul>'
                    '</div>'
                )
    # Mode (small metadata chip).
    mode = plan.get("mode")
    if isinstance(mode, str) and mode.strip():
        details_blocks.append(
            '<div class="conv-plan-detail-section">'
            '<h5 class="conv-plan-list-head">Mode</h5>'
            f'<code class="conv-plan-meta">{_h(mode.strip())}</code>'
            '</div>'
        )
    # Any other unknown fields → small extras block (collapsed as code).
    known_keys = {
        "goal", "workflow_plan", "required_tools", "candidate_tools",
        "warm_tools", "reason", "rationale", "explanation",
        "assumptions", "missing_inputs", "success_criteria", "mode",
    }
    extras_rows: list[str] = []
    for k, v in plan.items():
        if k in known_keys:
            continue
        rendered = _render_extra_value(v)
        if rendered:
            extras_rows.append(
                f'<dt>{_h(_pretty_field_name(k))}</dt><dd>{rendered}</dd>'
            )
    if extras_rows:
        details_blocks.append(
            '<div class="conv-plan-detail-section">'
            '<h5 class="conv-plan-list-head">Other fields</h5>'
            f'<dl class="conv-final-dl">{"".join(extras_rows)}</dl>'
            '</div>'
        )

    if details_blocks:
        blocks.append(
            '<details class="conv-plan-details">'
            '<summary class="conv-plan-details-summary">'
            '<span class="conv-plan-details-label">Plan details</span>'
            f'<span class="conv-plan-details-meta">{len(details_blocks)} '
            f'section{"s" if len(details_blocks) != 1 else ""}</span>'
            '</summary>'
            f'<div class="conv-plan-details-body">{"".join(details_blocks)}</div>'
            '</details>'
        )

    return "".join(blocks)


def _agent_final_block(final: str | None) -> str:
    if not final:
        return ""
    return f"""<div class="conv-block conv-block-final">
  <div class="conv-block-head">
    <span class="conv-role-badge conv-role-badge-final">✓ Final response</span>
  </div>
  <div class="conv-final-body">{_render_agent_final_payload(final)}</div>
</div>"""


# Field-name aliases the structured-final renderer recognises. Agents that
# don't follow our canonical names still land in the right section as long
# as one of these synonyms shows up. Lowercase + underscore — the renderer
# normalises keys before matching.
_FINAL_DECISION_KEYS  = ("decision", "verdict", "status", "outcome")
_FINAL_REASON_KEYS    = ("reason", "summary", "narrative", "explanation", "rationale", "message")
_FINAL_CHANGES_KEYS   = ("changes", "recommendations", "action_items", "next_steps", "actions", "todo")
_FINAL_NOTES_KEYS     = ("notes", "caveats", "warnings", "observations")
_FINAL_METRICS_KEYS   = ("metrics_used", "metrics", "metrics_evaluated", "criteria_evaluated")
_FINAL_NEXT_ROUND_KEYS = ("next_round", "recommendation", "next")


def _render_agent_final_payload(text: str) -> str:
    """Render the agent's final response with structured-JSON detection.

    When ``text`` parses as a JSON object carrying any of the recognised
    field names (decision / next_round / reason / changes / notes /
    metrics_used / ...), we lay it out as a polished structured card
    instead of dumping the raw JSON string. Free-form prose responses
    fall through to ``_render_agent_prose`` unchanged.
    """
    stripped = (text or "").strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict) and _looks_like_structured_final(data):
            return _render_structured_final(data)
    return _render_agent_prose(text)


def _looks_like_structured_final(data: dict[str, Any]) -> bool:
    """Return True iff ``data`` has at least one key the structured
    renderer knows how to lay out. Keeps free-form JSON-like blobs from
    being misidentified as structured payloads."""
    known = {
        *_FINAL_DECISION_KEYS, *_FINAL_REASON_KEYS, *_FINAL_CHANGES_KEYS,
        *_FINAL_NOTES_KEYS, *_FINAL_METRICS_KEYS, *_FINAL_NEXT_ROUND_KEYS,
    }
    return any(k in data for k in known)


def _pick_first(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return ``data[k]`` for the first ``k`` in ``keys`` that's present."""
    for k in keys:
        if k in data:
            return data[k]
    return None


def _render_structured_final(data: dict[str, Any]) -> str:
    """Lay out a structured final-response payload as readable HTML.

    Five logical sections, each rendered only when its data is present:
      1. Header bar: decision verdict + next-round action pill
      2. Reason / summary prose
      3. Recommended changes — bulleted list
      4. Notes / caveats — callout list
      5. Metrics evaluated — chip strip
      6. Anything else — collapsed key-value rows so nothing is silently dropped.
    """
    parts: list[str] = []

    # ---- Header: decision label + next-round status ----
    decision_label = _pick_first(data, _FINAL_DECISION_KEYS)
    next_round = _pick_first(data, _FINAL_NEXT_ROUND_KEYS)
    nr_pill = ""
    nr_reason = ""
    if isinstance(next_round, dict):
        recommended = bool(next_round.get("recommended", False))
        if recommended:
            nr_pill = (
                '<span class="conv-final-pill conv-final-pill-action">'
                '↻ Next round recommended</span>'
            )
        else:
            nr_pill = (
                '<span class="conv-final-pill conv-final-pill-done">'
                '✓ No further rounds needed</span>'
            )
        nr_reason_raw = next_round.get("reason")
        if isinstance(nr_reason_raw, str) and nr_reason_raw.strip():
            nr_reason = nr_reason_raw.strip()
    header_bits: list[str] = []
    if isinstance(decision_label, str) and decision_label.strip():
        human = _humanize_decision_label(decision_label.strip())
        # Humanised label as the eye-catching chip; raw underscore form
        # lives in the title= for engineers who need the audit string.
        header_bits.append(
            f'<span class="conv-final-decision" title="{_attr(decision_label.strip())}">'
            f'{_h(human)}</span>'
        )
    if nr_pill:
        header_bits.append(nr_pill)
    if header_bits:
        parts.append(f'<div class="conv-final-header">{"".join(header_bits)}</div>')

    # ---- Reason / summary prose ----
    reason = nr_reason
    if not reason:
        candidate = _pick_first(data, _FINAL_REASON_KEYS)
        if isinstance(candidate, str) and candidate.strip():
            reason = candidate.strip()
    if reason:
        parts.append(
            '<div class="conv-final-section">'
            f'<h5 class="conv-final-section-head">{"Reason" if next_round else "Summary"}</h5>'
            f'<div class="conv-final-section-body">{_render_agent_prose(reason)}</div>'
            '</div>'
        )

    # ---- Recommended changes / action items ----
    changes_value = _pick_first(data, _FINAL_CHANGES_KEYS)
    if isinstance(changes_value, list) and changes_value:
        lis = "".join(
            f'<li>{_render_agent_prose(_stringify_final_item(item))}</li>'
            for item in changes_value
            if item not in (None, "")
        )
        if lis:
            parts.append(
                '<div class="conv-final-section">'
                '<h5 class="conv-final-section-head">Recommended changes</h5>'
                f'<ul class="conv-final-list">{lis}</ul>'
                '</div>'
            )

    # ---- Notes / caveats ----
    notes_value = _pick_first(data, _FINAL_NOTES_KEYS)
    if isinstance(notes_value, list) and notes_value:
        note_lis: list[str] = []
        for n in notes_value:
            if isinstance(n, dict):
                level = str(n.get("level") or "note").lower()
                msg = str(n.get("message") or n.get("text") or "").strip()
                if not msg:
                    continue
                note_lis.append(
                    f'<li class="conv-final-note conv-final-note-{_h(level)}">'
                    f'<strong>{_h(level)}</strong> {_render_agent_prose(msg)}'
                    '</li>'
                )
            else:
                s = _stringify_final_item(n)
                if s:
                    note_lis.append(f'<li class="conv-final-note">{_render_agent_prose(s)}</li>')
        if note_lis:
            parts.append(
                '<div class="conv-final-section">'
                '<h5 class="conv-final-section-head">Notes &amp; caveats</h5>'
                f'<ul class="conv-final-list conv-final-notes">{"".join(note_lis)}</ul>'
                '</div>'
            )

    # ---- Metrics evaluated — chip strip ----
    metrics_value = _pick_first(data, _FINAL_METRICS_KEYS)
    if isinstance(metrics_value, list) and metrics_value:
        chips = "".join(
            f'<code class="conv-final-metric">{_h(str(m))}</code>'
            for m in metrics_value[:60]
            if m not in (None, "")
        )
        more = ""
        if len(metrics_value) > 60:
            more = (
                f'<span class="conv-final-metric-more">+{len(metrics_value) - 60} more</span>'
            )
        if chips:
            parts.append(
                '<div class="conv-final-section">'
                '<h5 class="conv-final-section-head">Metrics evaluated</h5>'
                f'<div class="conv-final-metrics">{chips}{more}</div>'
                '</div>'
            )

    # ---- Anything else — key/value rows so nothing is silently dropped ----
    known_keys: set[str] = set()
    for ks in (_FINAL_DECISION_KEYS, _FINAL_REASON_KEYS, _FINAL_CHANGES_KEYS,
               _FINAL_NOTES_KEYS, _FINAL_METRICS_KEYS, _FINAL_NEXT_ROUND_KEYS):
        known_keys.update(ks)
    extras_rows: list[str] = []
    for k, v in data.items():
        if k in known_keys:
            continue
        rendered = _render_extra_value(v)
        if rendered:
            extras_rows.append(
                f'<dt>{_h(_pretty_field_name(k))}</dt><dd>{rendered}</dd>'
            )
    if extras_rows:
        parts.append(
            '<div class="conv-final-section">'
            '<h5 class="conv-final-section-head">Additional</h5>'
            f'<dl class="conv-final-dl">{"".join(extras_rows)}</dl>'
            '</div>'
        )

    if not parts:
        # No recognised fields produced output — fall back to a pretty-
        # printed JSON code block so at minimum the raw payload reads
        # cleanly instead of as a single overflowing line.
        return (
            f'<code class="conv-codeblock">{_h(_pretty_json(data))}</code>'
        )
    return "".join(parts)


def _stringify_final_item(item: Any) -> str:
    """Coerce a list-item value to a short readable string."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Prefer ``message`` / ``text`` / ``description`` over JSON dump.
        for k in ("message", "text", "description", "summary", "name"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return _pretty_json(item, max_len=400)
    return str(item)


def _render_extra_value(value: Any) -> str:
    """Render an unknown extras value as HTML — short scalars inline,
    medium lists as comma-joined, anything bigger as a code block."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return _h("yes" if value else "no")
    if isinstance(value, (str, int, float)):
        s = str(value).strip()
        if not s:
            return ""
        return _h(s)
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(x, (str, int, float, bool)) for x in value) and len(value) <= 12:
            return _h(", ".join(str(x) for x in value))
        return f'<code class="conv-final-extra-payload">{_h(_pretty_json(value, max_len=600))}</code>'
    if isinstance(value, dict):
        if not value:
            return ""
        return f'<code class="conv-final-extra-payload">{_h(_pretty_json(value, max_len=600))}</code>'
    return _h(str(value))


def _pretty_field_name(key: str) -> str:
    """Convert a snake_case key into a Sentence-case label."""
    tokens = [t for t in str(key).replace("-", "_").split("_") if t]
    if not tokens:
        return str(key)
    return " ".join(
        t.upper() if t.isupper() and len(t) <= 4 else t.capitalize()
        for t in tokens
    )


def _agent_full_transcript_block(
    messages: list[AgentMessage],
    tool_calls: list[AgentToolCall] | None = None,
) -> str:
    """Collapsible transcript at the bottom: chat bubbles interleaved with tool-call chips.

    Messages and tool calls are merged into one chronological stream
    using ``timestamp_ms``; missing timestamps sort first (so they stay
    at the top in append order). Each tool call renders as a compact
    pill chip with the tool name, ok/fail badge, and a click-to-expand
    ``<details>`` showing pretty-printed args / result.
    """
    tool_calls = tool_calls or []
    if not messages and not tool_calls:
        return ""

    # Chronological merge. Use enumeration index as a tiebreaker so two
    # events sharing a timestamp keep their original order.
    events: list[tuple[int, int, str, Any]] = []
    for idx, msg in enumerate(messages):
        events.append((msg.timestamp_ms or 0, idx, "msg", msg))
    for idx, tc in enumerate(tool_calls):
        events.append((tc.timestamp_ms or 0, 10_000_000 + idx, "tool", tc))
    events.sort(key=lambda e: (e[0], e[1]))

    pieces: list[str] = []
    for _ts, _ord, kind, payload in events:
        if kind == "msg":
            pieces.append(_agent_message_bubble(payload))
        else:
            pieces.append(_agent_tool_call_chip(payload))

    return f"""<details class="conv-transcript-details">
  <summary class="conv-transcript-summary">
    <span class="conv-transcript-summary-label">Full transcript</span>
    <span class="conv-transcript-summary-meta">{len(messages)} message{"s" if len(messages) != 1 else ""}{f", {len(tool_calls)} tool call{'s' if len(tool_calls) != 1 else ''}" if tool_calls else ""}</span>
  </summary>
  <div class="conv-transcript-body">
    {"".join(pieces)}
  </div>
</details>"""


def _agent_message_bubble(msg: AgentMessage) -> str:
    """Single chat bubble for one message in the full transcript."""
    label, glyph = _AGENT_ROLE_BADGES.get(
        msg.role.lower(),
        (msg.role.title(), "•"),
    )
    role_cls = msg.role.lower().replace(" ", "-")
    model_chip = (
        f'<span class="conv-bubble-model">{_h(msg.model)}</span>'
        if msg.model and msg.role.lower() == "assistant"
        else ""
    )
    return (
        f'<article class="conv-bubble conv-bubble-{_h(role_cls)}">'
        f'<header class="conv-bubble-head">'
        f'<span class="conv-role-badge conv-role-badge-{_h(role_cls)}">'
        f'{_h(glyph)} {_h(label)}</span>'
        f'{model_chip}'
        '</header>'
        f'<div class="conv-bubble-body">{_render_agent_prose(msg.content)}</div>'
        '</article>'
    )


def _agent_tool_call_chip(tc: AgentToolCall) -> str:
    """Compact inline chip for a single tool invocation in the transcript.

    Shows the tool name + ok/fail badge by default; click-to-expand
    reveals pretty-printed args (JSON) and a compact result preview.
    Lives in the same chronological stream as message bubbles but reads
    as a single-line audit row rather than a chat bubble — that visual
    distinction is the cue "this is the agent doing, not saying."
    """
    status_cls = "ok" if tc.ok else "fail"
    status_glyph = "✓" if tc.ok else "✕"
    args_html = ""
    if tc.args:
        args_html = (
            '<div class="conv-tool-section">'
            '<div class="conv-tool-section-head">args</div>'
            f'<code class="conv-tool-payload">{_h(_pretty_json(tc.args))}</code>'
            '</div>'
        )
    result_html = ""
    if tc.result is not None and tc.result != "":
        result_html = (
            '<div class="conv-tool-section">'
            '<div class="conv-tool-section-head">result</div>'
            f'<code class="conv-tool-payload">{_h(_pretty_json(tc.result))}</code>'
            '</div>'
        )
    body_html = args_html + result_html
    summary_html = (
        f'<summary class="conv-tool-summary">'
        f'<span class="conv-tool-glyph conv-tool-glyph-{status_cls}">{status_glyph}</span>'
        f'<span class="conv-tool-label">Tool</span>'
        f'<code class="conv-tool-name">{_h(tc.tool_name)}</code>'
        + (
            '<span class="conv-tool-state conv-tool-state-fail">failed</span>'
            if not tc.ok else ""
        )
        + '</summary>'
    )
    if not body_html:
        # No payload to show — render as a plain (non-collapsible) chip.
        return (
            f'<div class="conv-tool-chip conv-tool-chip-{status_cls} conv-tool-chip-plain">'
            f'<span class="conv-tool-glyph conv-tool-glyph-{status_cls}">{status_glyph}</span>'
            f'<span class="conv-tool-label">Tool</span>'
            f'<code class="conv-tool-name">{_h(tc.tool_name)}</code>'
            + (
                '<span class="conv-tool-state conv-tool-state-fail">failed</span>'
                if not tc.ok else ""
            )
            + '</div>'
        )
    return (
        f'<details class="conv-tool-chip conv-tool-chip-{status_cls}">'
        f'{summary_html}'
        f'<div class="conv-tool-body">{body_html}</div>'
        '</details>'
    )


def _agent_step_timeline_block(steps: list[AgentStepSummary]) -> str:
    """Vertical execution timeline — one card per step the agent ran.

    Built from ``agent_summaries``; rendered as a left-rail timeline
    with an icon per step state (completed / failed / running) and a
    compact body showing the tool/workflow name + state pill + key
    metrics. The whole section is omitted when the agent didn't
    record any per-step summary (e.g. a single-shot tool call).
    """
    if not steps:
        return ""
    cards: list[str] = []
    for idx, step in enumerate(steps, 1):
        state = (step.state or "unknown").lower()
        state_class = _state_class(state)
        title_parts: list[str] = []
        if step.kind == "workflow":
            title_parts.append("Workflow")
            if step.name:
                title_parts.append(_h(step.name))
            elif step.workflow_id:
                title_parts.append(f'<code>{_h(_short(step.workflow_id, 14))}</code>')
        else:
            title_parts.append("Job")
            if step.tool:
                title_parts.append(_h(step.tool))
            elif step.job_id:
                title_parts.append(f'<code>{_h(_short(step.job_id, 14))}</code>')
        title_html = " · ".join(title_parts)
        meta_bits: list[str] = []
        if step.state:
            meta_bits.append(
                f'<span class="conv-timeline-state state-{state_class}">{_h(step.state)}</span>'
            )
        if step.metrics_used:
            shown = ", ".join(step.metrics_used[:6])
            extra = ""
            if len(step.metrics_used) > 6:
                extra = f' <span class="conv-timeline-meta-extra">(+{len(step.metrics_used) - 6})</span>'
            meta_bits.append(
                f'<span class="conv-timeline-metrics">metrics: {_h(shown)}{extra}</span>'
            )
        if step.workflow_id and step.kind != "workflow":
            meta_bits.append(
                f'<span class="conv-timeline-meta-id">in <code>{_h(_short(step.workflow_id, 14))}</code></span>'
            )
        meta_html = (
            f'<div class="conv-timeline-meta">{"  ·  ".join(meta_bits)}</div>'
            if meta_bits else ""
        )
        error_html = ""
        if step.error_message:
            error_html = (
                f'<div class="conv-timeline-error">'
                f'<strong>Error</strong> {_h(step.error_message)}'
                f'</div>'
            )
        cards.append(
            f'<li class="conv-timeline-row conv-timeline-row-{state_class}">'
            f'<div class="conv-timeline-marker"><span class="conv-timeline-bullet"></span>'
            f'<span class="conv-timeline-index">{idx}</span></div>'
            f'<div class="conv-timeline-card">'
            f'<div class="conv-timeline-title">{title_html}</div>'
            f'{meta_html}{error_html}'
            '</div>'
            '</li>'
        )
    return f"""<div class="conv-block conv-block-timeline">
  <div class="conv-block-head">
    <span class="conv-role-badge conv-role-badge-timeline">⏳ Execution timeline</span>
    <span class="conv-meta-chip">{len(steps)} step{"s" if len(steps) != 1 else ""}</span>
  </div>
  <ol class="conv-timeline">
    {"".join(cards)}
  </ol>
</div>"""


def _agent_decisions_block(decisions: list[AgentDecision]) -> str:
    """Per-round decision audit for multi-round agent runs.

    Each round renders as a compact card: round index, humanised verdict
    label, continue/stop pill, reason prose, and a *collapsed* detail
    drawer for the raw decision string + metrics + notes. The visible
    surface keeps the eye on "did the agent decide to keep going and
    why" without dumping every audit row inline.
    """
    if not decisions:
        return ""
    rows: list[str] = []
    for d in decisions:
        verdict_cls = "go" if d.next_round_recommended else "stop"
        verdict_label = "Continue → next round" if d.next_round_recommended else "Stop"
        human_label = _humanize_decision_label(d.decision_label) or d.decision_label

        # Collapsed details: raw label + metrics + notes. Only build when
        # there's at least one chunk to put inside; otherwise we skip the
        # drawer entirely and the row stays as a clean two-line summary.
        detail_chunks: list[str] = []
        if d.decision_label and d.decision_label != human_label:
            detail_chunks.append(
                '<div class="conv-decision-detail-row">'
                '<span class="conv-decision-detail-label">Raw label</span>'
                f'<code class="conv-decision-raw">{_h(d.decision_label)}</code>'
                '</div>'
            )
        if d.metrics_used:
            shown = ", ".join(d.metrics_used[:8])
            more = f" (+{len(d.metrics_used) - 8} more)" if len(d.metrics_used) > 8 else ""
            detail_chunks.append(
                '<div class="conv-decision-detail-row">'
                '<span class="conv-decision-detail-label">Metrics weighed</span>'
                f'<span class="conv-decision-metrics-list">{_h(shown)}{_h(more)}</span>'
                '</div>'
            )
        if d.notes:
            note_items: list[str] = []
            for entry in d.notes:
                if isinstance(entry, dict):
                    level = str(entry.get("level") or "note").lower()
                    msg = str(entry.get("message") or "").strip()
                    code = entry.get("code")
                    if not msg:
                        continue
                    code_html = (
                        f' <code class="conv-decision-note-code">{_h(str(code))}</code>'
                        if code else ""
                    )
                    note_items.append(
                        f'<li class="conv-decision-note conv-decision-note-{_h(level)}">'
                        f'<strong>{_h(level)}</strong> {_render_agent_prose(msg)}{code_html}</li>'
                    )
                else:
                    s = _stringify_final_item(entry)
                    if s:
                        note_items.append(
                            f'<li class="conv-decision-note">{_render_agent_prose(s)}</li>'
                        )
            if note_items:
                detail_chunks.append(
                    '<div class="conv-decision-detail-row">'
                    '<span class="conv-decision-detail-label">Notes</span>'
                    f'<ul class="conv-decision-notes">{"".join(note_items)}</ul>'
                    '</div>'
                )
        details_html = ""
        if detail_chunks:
            details_html = (
                '<details class="conv-decision-details">'
                '<summary class="conv-decision-details-summary">More detail</summary>'
                f'<div class="conv-decision-details-body">{"".join(detail_chunks)}</div>'
                '</details>'
            )

        rows.append(
            f'<li class="conv-decision-row">'
            f'<div class="conv-decision-head">'
            f'<span class="conv-decision-round">Round {d.round_index + 1}</span>'
            f'<span class="conv-decision-label conv-decision-label-friendly" title="{_attr(d.decision_label)}">{_h(human_label)}</span>'
            f'<span class="conv-decision-verdict conv-decision-verdict-{verdict_cls}">{verdict_label}</span>'
            '</div>'
            + (
                f'<p class="conv-decision-reason">{_render_agent_prose(d.reason)}</p>'
                if d.reason else ""
            )
            + details_html
            + '</li>'
        )
    return f"""<div class="conv-block conv-block-decisions">
  <div class="conv-block-head">
    <span class="conv-role-badge conv-role-badge-decisions">⚖ Decision audit</span>
    <span class="conv-meta-chip">{len(decisions)} round{"s" if len(decisions) != 1 else ""}</span>
  </div>
  <ol class="conv-decisions">
    {"".join(rows)}
  </ol>
</div>"""


def _pretty_json(value: Any, *, max_len: int = 2_000) -> str:
    """Pretty-print arbitrary JSON-compatible data with a length cap.

    Lists/dicts get 2-space indent. Strings render verbatim. The cap
    keeps massive tool results (e.g. base64 blobs) from blowing up the
    transcript size — the trailing slice is dropped with an ellipsis.
    """
    if isinstance(value, str):
        s = value
    else:
        try:
            import json as _json
            s = _json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False, default=str)
        except Exception:
            s = repr(value)
    if len(s) > max_len:
        s = s[:max_len].rstrip() + f"\n… [{len(s) - max_len} more chars truncated]"
    return s


def _render_agent_prose(text: str) -> str:
    """Escape + lightly format an agent's text content for HTML.

    Preserves paragraph breaks (double newline) and inline newlines
    (single newline → ``<br>``). Code fences ``` ``` ``` are rendered as
    monospace blocks. Nothing fancy — the goal is faithful prose
    rendering, not a markdown parser.
    """
    if not text:
        return ""
    s = str(text).strip()
    # Split out triple-backtick fenced blocks first.
    parts: list[str] = []
    buf: list[str] = []
    in_code = False
    code_buf: list[str] = []
    for line in s.split("\n"):
        if line.strip().startswith("```"):
            if in_code:
                parts.append("<para>" + "\n".join(buf) + "</para>" if buf else "")
                buf = []
                parts.append(
                    "<code class='conv-codeblock'>"
                    + _h("\n".join(code_buf))
                    + "</code>"
                )
                code_buf = []
                in_code = False
            else:
                if buf:
                    parts.append("<para>" + "\n".join(buf) + "</para>")
                    buf = []
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
        else:
            buf.append(line)
    if in_code and code_buf:
        # Unterminated fence — render what we have as a code block.
        parts.append(
            "<code class='conv-codeblock'>" + _h("\n".join(code_buf)) + "</code>"
        )
    elif buf:
        parts.append("<para>" + "\n".join(buf) + "</para>")

    html_parts: list[str] = []
    for chunk in parts:
        if not chunk:
            continue
        if chunk.startswith("<code"):
            html_parts.append(chunk)
            continue
        # Strip our <para> sentinel and split on blank lines for paragraphs.
        inner = chunk[len("<para>"):-len("</para>")]
        for paragraph in inner.split("\n\n"):
            paragraph = paragraph.strip("\n")
            if not paragraph:
                continue
            # Within a paragraph, single newlines become <br>.
            escaped = _h(paragraph).replace("\n", "<br>")
            html_parts.append(f"<p>{escaped}</p>")
    return "".join(html_parts)


def _why_candidate(candidate: ReportCandidate) -> str:
    justification = candidate.source.get("justification")
    if isinstance(justification, str) and justification.strip():
        return justification.strip()
    bits = []
    # ipTM / ipSAE are interface metrics — meaningless for a single-chain
    # candidate. Drop them from the per-card "why" prose when monomer.
    monomer = _is_monomer_candidate(candidate)
    metric_specs = [("plddt", "pLDDT"), ("ptm", "pTM")]
    if not monomer:
        metric_specs.extend([("iptm", "ipTM"), ("ipsae", "ipSAE")])
    for key, label in metric_specs:
        value = candidate.scores.get(key)
        if value is not None:
            quality = _quality_class(key, value)
            qualifier = {
                "good": "strong",
                "ok": "moderate",
                "poor": "weak",
                "neutral": "",
            }.get(quality, "")
            bit = f"{label} {_fmt(value)}"
            if qualifier:
                bit = f"{qualifier} {bit}"
            bits.append(bit)
    if candidate.score is not None:
        bits.append(f"composite score {_fmt(candidate.score)}")
    if candidate.qc_flags:
        bits.append(f"{len(candidate.qc_flags)} QC flag(s)")
    if candidate.liabilities:
        bits.append(f"{len(candidate.liabilities)} structure liability item(s)")
    if not bits:
        return "Selected as the best available output — no comparator metrics were attached."
    caution = ""
    if candidate.qc_flags or candidate.liabilities:
        caution = " Review the attached flags before experimental follow-up."
    return f"This candidate is highlighted by {', '.join(bits)}.{caution}"


# ----------------------------------------------------------------- formatting helpers


def _quality_class(key: str, value: Any) -> str:
    """Coarse good/ok/poor classification for the report's chrome.

    Used for both metric panels and table cells so confidence quality is
    immediately readable via colour. Recognises both 0-1 and 0-100 scales
    for the confidence-style metrics (pLDDT being the canonical dual-scale
    case).
    """
    num = _num(value)
    if num is None:
        return "neutral"
    k = key.lower()
    if k in {"plddt", "mean_plddt", "ptm", "iptm", "dockq", "ipsae", "seq_recovery",
             "recovery", "lddt", "bb_lddt"}:
        scaled = num * 100 if num <= 1.5 else num
        if scaled >= 70:
            return "good"
        if scaled >= 50:
            return "ok"
        return "poor"
    if k in {"pae", "mean_pae"}:
        if num <= 5:
            return "good"
        if num <= 12:
            return "ok"
        return "poor"
    if k in {"delta_g", "dg"}:
        if num <= -8:
            return "good"
        if num <= -4:
            return "ok"
        return "poor"
    if k == "instability_index":
        return "good" if num < 40 else ("ok" if num < 50 else "poor")
    if k == "polybasic_hits":
        return "good" if num == 0 else ("ok" if num <= 2 else "poor")
    if k in {"affinity_probability", "affinity_probability_binary", "binding_probability"}:
        if num >= 0.7:
            return "good"
        if num >= 0.4:
            return "ok"
        return "poor"
    if k in {"ranking_confidence", "confidence_score", "confidence", "model_confidence"}:
        if num >= 0.7:
            return "good"
        if num >= 0.5:
            return "ok"
        return "poor"
    if k == "iplddt":
        scaled = num * 100 if num <= 1.5 else num
        if scaled >= 70:
            return "good"
        if scaled >= 50:
            return "ok"
        return "poor"
    return "neutral"


# ``_pretty_metric_name`` lives in :mod:`.metrics` (as ``pretty_metric_name``)
# alongside the science-token + unit-suffix tables it consumes. Imported at
# module top so the rest of this file's call sites can keep using the same
# local name.


def _as_unit(value: Any) -> float | None:
    """Coerce to a 0..1 scaled float when possible — for narrative tests."""
    num = _num(value)
    if num is None:
        return None
    if num > 1.5:
        return num / 100.0
    return num


def _conservation(seqs: list[tuple[str, str]], width: int) -> list[float]:
    values: list[float] = []
    for idx in range(width):
        chars = [seq[idx] for _, seq in seqs if idx < len(seq) and seq[idx] not in {"-", "."}]
        if not chars:
            values.append(0.0)
            continue
        counts: dict[str, int] = {}
        for char in chars:
            counts[char] = counts.get(char, 0) + 1
        values.append(max(counts.values()) / len(chars))
    return values


def _colored_sequence(seq: str) -> str:
    return "".join(f"<span class='aa aa-{_aa_class(ch)}'>{_h(ch)}</span>" for ch in seq)


def _aa_class(ch: str) -> str:
    if ch in "AVLIMFWY":
        return "hydro"
    if ch in "KRH":
        return "pos"
    if ch in "DE":
        return "neg"
    if ch in "STNQ":
        return "polar"
    if ch in "CGP":
        return "special"
    if ch in "-.":
        return "gap"
    return "other"


def _duration(ms: int | None) -> str:
    if ms is None:
        return "runtime n/a"
    if ms < 1000:
        return f"{ms} ms"
    sec = ms / 1000
    if sec < 60:
        return f"{sec:.1f} s"
    return f"{sec / 60:.1f} min"


def _size(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"


def _fmt(value: object) -> str:
    num = _num(value)
    if num is None:
        return "n/a" if value is None else str(value)
    # Integers stay integer-formatted; floats get 3 sig figs.
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if abs(num - int(num)) < 1e-9 and abs(num) < 1e6:
        return f"{int(num)}"
    return f"{num:.3g}"


def _num(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _short(value: str, length: int = 16) -> str:
    if not value or len(value) <= length:
        return value
    return value[: length - 1] + "…"


def _short_sha(value: str | None) -> str:
    if not value:
        return "—"
    if len(value) <= 12:
        return value
    return value[:8] + "…" + value[-4:]


def _compact(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, list):
        return [_compact(v, depth=depth + 1) for v in value[:12]]
    if isinstance(value, dict):
        return {str(k): _compact(v, depth=depth + 1) for k, v in list(value.items())[:30]}
    return str(value)


def _state_class(value: str) -> str:
    return value.lower().replace("_", "-")


def _h(value: object) -> str:
    return html.escape(str(value), quote=False)


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _svg_text(value: object) -> str:
    return html.escape(str(value), quote=False)


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _public_model(model: ReportModel) -> dict[str, Any]:
    data = asdict(model)
    data["raw_summary"] = _compact(data.get("raw_summary"))
    return data


# ----------------------------------------------------------------- CSS


def _css() -> str:
    return """
:root {
  color-scheme: light;
  /* Product-grade violet/indigo palette: lavender-tinted near-white
     background, pure-white cards, hairline lavender borders, deep-indigo
     ink, and a vivid violet accent with a complementary pink secondary. */
  --bg:#f6f5fb;
  --ink:#1a1535;
  --muted:#6b6491;
  --subtle:#a09bbf;
  --line:#e8e4f2;
  --line-strong:#d4cee5;
  --panel:#ffffff;
  --panel-elev:#ffffff;
  /* Brand purple — deep ``#2d046e`` for sidebar / dark contexts, plus a
     brighter royal violet for everyday accent use (buttons, focus rings,
     active nav). The bright variant has enough contrast on white to read
     well without straining the eyes. */
  --accent:#6b1ea8;           /* royal violet — primary brand accent */
  --accent-bright:#a78bfa;    /* violet-400 — gradient stop / hover */
  --accent-deep:#2d046e;      /* user-specified deep purple — sidebar bg */
  --accent-soft:#ede9fe;      /* very light lavender — chip backgrounds */
  /* Supporting palette — exact hex values from the design reference. Each
     accent has a "deep" version (icons, strong fills) and a "soft" tint
     (chip backgrounds, large surfaces). Used across bars, diagrams and
     stateful UI for visual variety instead of a single violet wash. */
  --accent-2:#cc0980;         /* magenta-pink (warm complementary) */
  --accent-2-soft:#f4cfe6;
  --accent-3:#4cc4e4;         /* sky cyan (cool complementary) */
  --accent-3-soft:#e5f4fd;
  --accent-4:#f39e3a;         /* warm orange (tertiary highlight) */
  --accent-4-soft:#fcebdd;
  /* Semantic quality states — all drawn from the user's brand palette so
     the report never falls back to off-brand red/green/blue. Visual cues
     map: good = sky (cool calm), ok = orange (warm caution), poor = pink
     (warm critical). Lines are derived lighter shades of each accent. */
  --good:#4cc4e4;     --good-soft:#e5f4fd;  --good-line:#a5dceb;
  --ok:#f39e3a;       --ok-soft:#fcebdd;    --ok-line:#f8c98a;
  --poor:#cc0980;     --poor-soft:#f4cfe6;  --poor-line:#e89ec5;
  --warn:#92400e;
  --sidebar-w: 220px;
  --topbar-h: 60px;
  --shadow-card: 0 1px 2px rgba(76, 29, 149, 0.06);
  --shadow-card-hover: 0 4px 16px rgba(76, 29, 149, 0.10);
  /* Smaller, dual-layer shadow for nested detail blocks — same Linear /
     Vercel feel the workflow graph uses for its DAG cards, but slightly
     softer so a single card doesn't get visually heavy with 6+ tiles. */
  --shadow-tile: 0 1px 2px rgba(45, 4, 110, 0.05), 0 2px 5px rgba(45, 4, 110, 0.04);
  --radius-card: 14px;
  --radius-pill: 999px;
  --radius-sm: 8px;
}
* { box-sizing: border-box; }
body {
  margin:0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background:var(--bg); color:var(--ink);
  letter-spacing:0; line-height:1.55;
  display:grid;
  grid-template-columns: var(--sidebar-w) 1fr;
  grid-template-rows: var(--topbar-h) 1fr;
  grid-template-areas: "sidebar header" "sidebar main";
  min-height:100vh;
}

/* SIDEBAR — deep-violet brand surface (CELLKEY-style), light text on top */
.sidebar {
  grid-area:sidebar;
  position:sticky; top:0;
  background:var(--accent-deep);
  color:#e9e6ff;
  padding:22px 14px 16px;
  border-right:none;
  display:flex; flex-direction:column;
  min-height:100vh; height:100vh; overflow-y:auto;
}
.sidebar-brand {
  display:flex; align-items:center; gap:9px;
  font-size:14px; font-weight:600; margin-bottom:20px; color:#ffffff;
  letter-spacing:0.02em;
}
.brand-mark {
  width:26px; height:26px; flex-shrink:0; border-radius:6px; display:block;
  box-shadow:0 1px 4px rgba(255,255,255,0.22);
}
.sidebar-meta {
  display:flex; flex-direction:column; gap:3px;
  padding:8px 10px; margin:0 0 14px;
  background:rgba(255,255,255,0.07); border-radius:7px;
  border:1px solid rgba(255,255,255,0.12);
}
.sidebar-meta-label {
  font-size:9px; text-transform:uppercase; letter-spacing:0.08em;
  color:rgba(255,255,255,0.55); font-weight:600;
}
.sidebar-meta-value {
  font-size:11.5px; color:#f3f0ff; font-family:ui-monospace, monospace;
  background:transparent; padding:0;
  /* Long workspace names should wrap across lines instead of getting
     ellipsis-truncated — the sidebar is the one place where the user
     wants to see the full identifier. ``overflow-wrap: anywhere`` lets
     the underscore-heavy names break at any character. */
  white-space:normal; overflow-wrap:anywhere; word-break:break-word;
  line-height:1.35;
}
.side-nav { display:flex; flex-direction:column; gap:1px; flex:1; }
.side-nav-item {
  display:flex; align-items:center; gap:11px;
  padding:9px 12px;
  border-radius:7px;
  border-left:2px solid transparent;
  color:rgba(255,255,255,0.78); text-decoration:none;
  font-size:14px;
  transition:color 0.12s, background 0.12s, border-left-color 0.12s;
}
.side-nav-item:hover {
  color:#ffffff;
  background:rgba(255,255,255,0.08);
}
.side-nav-item.active {
  color:#ffffff; font-weight:600;
  background:rgba(255,255,255,0.12);
  border-left:2px solid #c4b5fd;
}
.side-nav-item.active .side-nav-glyph { color:#c4b5fd; }
.side-nav-glyph {
  width:18px; text-align:center; font-size:13px;
  color:rgba(255,255,255,0.55);
}
.sidebar-foot {
  display:flex; justify-content:space-between; align-items:center;
  padding-top:12px; border-top:1px solid rgba(255,255,255,0.10);
  font-size:10.5px; color:rgba(255,255,255,0.5);
  font-family:ui-monospace, monospace;
  letter-spacing:0.01em;
}

/* TOP BAR — flat, hairline, ample padding, just title + status + id */
.top-bar {
  grid-area:header;
  background:var(--panel);
  border-bottom:1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between;
  padding:0 28px;
  position:sticky; top:0; z-index:5;
}
.top-bar-left { display:flex; align-items:center; gap:12px; min-width:0; }
.top-bar-kind {
  font-size:11.5px; text-transform:uppercase; letter-spacing:0.09em;
  color:var(--subtle); font-weight:600;
}
.top-bar-title {
  margin:0; font-size:19px; font-weight:600; letter-spacing:-0.01em;
  color:var(--ink);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.status-chip {
  font-size:12.5px; font-weight:600; padding:3px 11px; border-radius:var(--radius-pill);
  border:1px solid var(--line); background:var(--panel);
  display:inline-flex; align-items:center; gap:5px;
}
.status-chip::before {
  content:""; display:inline-block; width:6px; height:6px; border-radius:50%;
  background:var(--subtle);
}
.status-chip.state-completed::before, .status-chip.state-cached::before { background:var(--good); }
.status-chip.state-failed::before, .status-chip.state-cancelled::before { background:var(--poor); }
.status-chip.state-running::before { background:var(--accent-2); }
.status-chip.state-completed, .status-chip.state-cached {
  border-color:var(--good-line); background:var(--good-soft); color:var(--good);
}
.status-chip.state-failed, .status-chip.state-cancelled {
  border-color:var(--poor-line); background:var(--poor-soft); color:var(--poor);
}
.status-chip.state-running { border-color:#c4b5fd; background:var(--accent-soft); color:#6b1ea8; }
.top-bar-id {
  font-family:ui-monospace, monospace; font-size:13px;
  background:var(--accent-soft); padding:4px 12px; border-radius:var(--radius-sm);
  color:var(--accent); border:1px solid var(--line);
}

/* MAIN CONTENT */
.content {
  grid-area:main;
  padding:28px 32px 56px;
  max-width:1280px;
  width:100%;
}
:where(section, .card) { scroll-margin-top: calc(var(--topbar-h) + 14px); }

/* CARDS — uniform clean look across all sections */
.card {
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:var(--radius-card);
  padding:24px 28px;
  margin-bottom:18px;
  box-shadow:var(--shadow-card);
  transition:box-shadow 0.2s;
}
.card:hover { box-shadow:var(--shadow-card-hover); }
.card-header {
  display:flex; justify-content:space-between; align-items:flex-start;
  gap:16px; margin-bottom:18px;
}
.card-eyebrow {
  margin:0 0 7px; color:var(--subtle);
  font-size:12px; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;
}
.card-title {
  margin:0; font-size:20px; font-weight:600; line-height:1.4; letter-spacing:-0.005em;
  color:var(--ink);
}
.card-meta {
  font-size:13.5px; color:var(--muted); padding-top:14px;
}
.card-header-pills { display:flex; gap:7px; flex-wrap:wrap; }

/* The summary lede headline is product-style copy — slightly larger, calmer */
.card-summary .card-title {
  font-size:19px; color:var(--ink); max-width:840px; line-height:1.55; font-weight:500;
}
/* Drop the historical accent-tinted card-candidates variant — every card is now identical. */
.card-candidates { background:var(--panel); border:1px solid var(--line); box-shadow:var(--shadow-card); }

/* ==========================================================================
   Conversation panel — three-tier transcript for LLM-agent runs.
   Mirrors the visual weight of the executive-summary card while staying
   visually distinct: the user/agent/final tiers each have a coloured left
   border + soft tint so a reader can scan the conversation at a glance.
   ========================================================================== */
.card-conversation .card-header-pills,
.conv-meta { display:flex; gap:7px; flex-wrap:wrap; }
.conv-meta-chip {
  font-size:11.5px; color:var(--muted);
  background:var(--panel); border:1px solid var(--line);
  border-radius:var(--radius-pill); padding:3px 11px;
}
.conv-meta-chip b {
  color:var(--ink); font-weight:600;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}
.conv-block {
  margin:14px 0; padding:16px 18px 14px;
  border-radius:12px; border:1px solid var(--line);
  background:#ffffff; box-shadow:var(--shadow-tile);
  position:relative; overflow:hidden;
}
.conv-block::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
}
.conv-block-user::before  { background:var(--accent); }
.conv-block-plan::before  { background:#5b8def; }   /* sky-indigo for "thinking" */
.conv-block-final::before { background:var(--good); }
.conv-block-head {
  display:flex; align-items:center; justify-content:space-between; gap:10px;
  margin-bottom:10px;
}
.conv-role-badge {
  display:inline-flex; align-items:center; gap:6px;
  font-size:11px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.07em;
  padding:4px 11px; border-radius:var(--radius-pill);
  border:1px solid var(--line); background:#fbfafe; color:var(--ink);
}
.conv-role-badge-user,
.conv-role-badge-user.conv-role-badge {
  color:var(--accent); background:var(--accent-soft); border-color:#d9cef3;
}
.conv-role-badge-agent,
.conv-role-badge-assistant,
.conv-role-badge-assistant.conv-role-badge {
  color:#1e40af; background:#e7efff; border-color:#bfd4ff;
}
.conv-role-badge-final,
.conv-role-badge-final.conv-role-badge {
  color:var(--good); background:var(--good-soft); border-color:var(--good-line);
}
.conv-role-badge-system { color:var(--muted); background:#f7f5fb; }
.conv-role-badge-tool   { color:#7c2d12; background:#fcebdd; border-color:#f3d3b8; }

/* The user quote leans on the brand serif accent so it reads as
   prose — a customer-quote treatment rather than a code block. */
.conv-user-quote {
  margin:0; padding:0 0 0 14px;
  border-left:3px solid var(--accent-soft);
  color:var(--ink); font-size:16px; line-height:1.65;
}
.conv-user-quote p { margin:0 0 8px; }
.conv-user-quote p:last-child { margin-bottom:0; }

/* Plan tier: structured outline + optional prose. */
.conv-plan-dl { margin:0 0 12px; }
.conv-plan-dl dt {
  font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;
  color:var(--subtle); margin-bottom:4px;
}
.conv-plan-dl dd {
  margin:0; padding:0; font-size:14px; color:var(--ink); line-height:1.55;
}
.conv-plan-list { margin:10px 0; }
.conv-plan-list-head {
  margin:0 0 6px; font-size:11px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.08em; color:var(--subtle);
}
.conv-plan-list ul, .conv-plan-list ol {
  margin:0; padding-left:20px; font-size:13.5px; color:var(--ink); line-height:1.6;
}
.conv-plan-list li { margin:3px 0; }
.conv-plan-list-steps ol { padding-left:22px; }
.conv-plan-list-steps li {
  display:flex; align-items:baseline; gap:10px;
  padding:5px 0;
}
.conv-plan-tool {
  flex-shrink:0;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; font-weight:700; color:var(--accent);
  background:var(--accent-soft); border-radius:5px; padding:1px 8px;
}
.conv-plan-reason { color:var(--muted); font-size:13.5px; }
.conv-plan-prose { margin-top:10px; color:var(--ink); font-size:14px; line-height:1.65; }
.conv-plan-prose p { margin:0 0 8px; }
.conv-plan-prose p:last-child { margin-bottom:0; }

/* Goal as a lede paragraph — slightly larger, calm ink colour, no
   list / dl chrome. Reads as the headline of the card. */
.conv-plan-goal {
  margin:0 0 14px;
  font-size:14.5px; line-height:1.6; color:var(--ink);
}
.conv-plan-goal p { margin:0; }

/* Required-tools + Warm chip strips in a side-by-side row. Falls back
   to stacked when narrow. */
.conv-plan-tools-row {
  display:flex; flex-wrap:wrap; gap:18px 28px; margin:12px 0;
}
.conv-plan-tools-col {
  flex:1 1 220px; min-width:0;
}
.conv-plan-tool-chips { display:flex; flex-wrap:wrap; gap:5px; }
.conv-plan-tool-chip {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px; font-weight:600; color:var(--accent);
  background:var(--accent-soft); border:1px solid #d9cef3;
  border-radius:5px; padding:2px 9px;
}

/* Collapsible "Plan details" drawer carrying assumptions, missing
   inputs, success criteria, rationale, mode, extras. Default-closed
   keeps the visible surface focused on goal + planned steps + tools. */
.conv-plan-details {
  margin-top:14px; padding-top:12px; border-top:1px dashed #f0eef8;
}
.conv-plan-details-summary {
  cursor:pointer; user-select:none;
  display:inline-flex; align-items:center; gap:10px;
  padding:5px 10px; border-radius:7px;
  font-size:12.5px; color:var(--muted);
  list-style:none;
}
.conv-plan-details-summary::-webkit-details-marker { display:none; }
.conv-plan-details-summary::before {
  content:"▸"; color:var(--subtle); font-size:10px;
  transition:transform 0.15s;
}
.conv-plan-details[open] > .conv-plan-details-summary::before {
  transform:rotate(90deg); display:inline-block;
}
.conv-plan-details-summary:hover { background:#f7f5fb; color:var(--accent); }
.conv-plan-details-label { font-weight:600; color:var(--ink); }
.conv-plan-details-meta { color:var(--subtle); font-size:11.5px; }
.conv-plan-details-body { margin-top:10px; }
.conv-plan-detail-section { margin:10px 0; }
.conv-plan-detail-section:first-child { margin-top:0; }
.conv-plan-detail-list {
  margin:0; padding-left:20px; font-size:13.5px; color:var(--ink); line-height:1.6;
}
.conv-plan-detail-list li { margin:3px 0; }
.conv-plan-rationale {
  font-size:13.5px; color:var(--ink); line-height:1.6;
}
.conv-plan-rationale p { margin:0 0 6px; }
.conv-plan-rationale p:last-child { margin-bottom:0; }
.conv-plan-meta {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px; color:var(--ink);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:5px; padding:2px 8px;
}

/* Final tier: prose, slightly heavier weight to read as the "answer". */
.conv-final-body { font-size:14.5px; line-height:1.65; color:var(--ink); }
.conv-final-body p { margin:0 0 10px; }
.conv-final-body p:last-child { margin-bottom:0; }

/* Structured final-response layout (when the agent emits a decision /
   recommendation JSON payload instead of free-form prose). Visual goal:
   read as a polished product summary — header pills for the verdict,
   section subheads for reason / changes / notes / metrics, no raw JSON
   dump. */
.conv-final-header {
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  margin:0 0 12px;
}
.conv-final-decision {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; font-weight:700;
  background:var(--good-soft); color:var(--good);
  border:1px solid var(--good-line);
  border-radius:5px; padding:2px 9px;
}
.conv-final-pill {
  font-size:11px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.06em;
  padding:3px 11px; border-radius:var(--radius-pill);
  border:1px solid transparent;
}
.conv-final-pill-action {
  color:#92400e; background:#fef3c7; border-color:#fde68a;
}
.conv-final-pill-done {
  color:var(--good); background:var(--good-soft); border-color:var(--good-line);
}
.conv-final-section { margin:12px 0; }
.conv-final-section:first-of-type { margin-top:0; }
.conv-final-section-head {
  margin:0 0 6px; font-size:10.5px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.08em;
  color:var(--subtle);
}
.conv-final-section-body {
  font-size:14px; line-height:1.65; color:var(--ink);
}
.conv-final-section-body p { margin:0 0 8px; }
.conv-final-section-body p:last-child { margin-bottom:0; }
.conv-final-list {
  margin:0; padding-left:20px; font-size:13.5px; color:var(--ink);
  line-height:1.6;
}
.conv-final-list li { margin:4px 0; }
.conv-final-list li p { margin:0; }
.conv-final-notes li { color:var(--muted); }
.conv-final-note strong {
  text-transform:uppercase; font-size:10.5px; font-weight:700;
  letter-spacing:0.06em; margin-right:5px; color:var(--subtle);
}
.conv-final-note-warning strong { color:var(--warn); }
.conv-final-note-error   strong { color:var(--poor); }
.conv-final-metrics {
  display:flex; flex-wrap:wrap; gap:5px;
}
.conv-final-metric {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11px; color:var(--ink);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:5px; padding:2px 8px;
}
.conv-final-metric-more {
  font-size:11px; color:var(--subtle); align-self:center;
  padding:2px 4px;
}
.conv-final-dl { margin:0; }
.conv-final-dl dt {
  font-size:10.5px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.06em;
  color:var(--subtle); margin-bottom:2px;
}
.conv-final-dl dt + dt { margin-top:8px; }
.conv-final-dl dd {
  margin:0 0 8px; font-size:13.5px; color:var(--ink);
  word-break:break-word;
}
.conv-final-extra-payload {
  display:block; white-space:pre-wrap; word-break:break-word;
  padding:8px 10px; max-height:200px; overflow:auto;
  background:#0f0a1f; color:#ede9fe;
  border-radius:6px; font-size:11.5px; line-height:1.5;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Inline code block inside any agent prose. */
.conv-codeblock {
  display:block; white-space:pre; overflow-x:auto;
  margin:10px 0; padding:10px 12px;
  background:#0f0a1f; color:#ede9fe;
  border-radius:8px; font-size:12.5px; line-height:1.55;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Full-transcript: collapsible chat-bubble list. */
.conv-transcript-details {
  margin-top:16px; border-top:1px solid var(--line); padding-top:14px;
}
.conv-transcript-summary {
  cursor:pointer; user-select:none; padding:6px 10px;
  display:inline-flex; align-items:center; gap:10px;
  border-radius:8px; color:var(--muted); font-size:13px;
  list-style:none;
}
.conv-transcript-summary::-webkit-details-marker { display:none; }
.conv-transcript-summary::before {
  content:"▸"; color:var(--subtle); transition:transform 0.15s;
}
.conv-transcript-details[open] > .conv-transcript-summary::before {
  transform:rotate(90deg); display:inline-block;
}
.conv-transcript-summary:hover { background:#f7f5fb; color:var(--accent); }
.conv-transcript-summary-label { font-weight:600; color:var(--ink); }
.conv-transcript-summary-meta { color:var(--subtle); font-size:12px; }
.conv-transcript-body {
  margin-top:12px;
  display:flex; flex-direction:column; gap:10px;
}
.conv-bubble {
  padding:12px 14px;
  border-radius:10px; border:1px solid var(--line);
  background:#ffffff;
}
.conv-bubble-user { background:var(--accent-soft); border-color:#d9cef3; }
.conv-bubble-assistant { background:#f4f7ff; border-color:#bfd4ff; }
.conv-bubble-system { background:#fafafa; }
.conv-bubble-tool { background:#fcebdd; border-color:#f3d3b8; }
.conv-bubble-head {
  display:flex; align-items:center; justify-content:space-between;
  gap:10px; margin-bottom:6px;
}
.conv-bubble-model {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:10.5px; color:var(--subtle);
}
.conv-bubble-body {
  font-size:13.5px; line-height:1.6; color:var(--ink);
}
.conv-bubble-body p { margin:0 0 6px; }
.conv-bubble-body p:last-child { margin-bottom:0; }

/* -------- Execution timeline -------- */
/* Distinguishes the timeline block from the user/plan/final tiers via
   a calmer slate-tinted left rail so the eye groups "what the agent
   DID" separately from "what the agent SAID". */
.conv-block-timeline::before { background:#94a3b8; }
.conv-role-badge-timeline {
  color:#475569; background:#eef2f7; border-color:#dbe3ec;
}
.conv-timeline { list-style:none; margin:8px 0 0; padding:0; }
.conv-timeline-row {
  display:grid; grid-template-columns:38px 1fr; gap:10px;
  padding:7px 0; position:relative;
}
.conv-timeline-row + .conv-timeline-row { border-top:1px dashed #f0eef8; }
.conv-timeline-marker {
  display:flex; flex-direction:column; align-items:center; gap:3px;
  padding-top:4px; color:var(--subtle);
}
.conv-timeline-bullet {
  width:9px; height:9px; border-radius:50%;
  background:var(--accent-3); box-shadow:0 0 0 3px #fbfafe;
}
.conv-timeline-row-completed .conv-timeline-bullet,
.conv-timeline-row-cached    .conv-timeline-bullet { background:var(--good); }
.conv-timeline-row-failed    .conv-timeline-bullet { background:var(--poor); }
.conv-timeline-row-running   .conv-timeline-bullet { background:var(--accent-bright); }
.conv-timeline-index {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:10.5px; color:var(--subtle); font-variant-numeric:tabular-nums;
}
.conv-timeline-card { min-width:0; }
.conv-timeline-title {
  font-size:14px; font-weight:600; color:var(--ink); line-height:1.4;
}
.conv-timeline-title code {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; color:var(--accent);
  background:var(--accent-soft); border-radius:5px; padding:1px 7px;
}
.conv-timeline-meta {
  font-size:12px; color:var(--muted); margin-top:3px;
  display:flex; flex-wrap:wrap; gap:6px 12px; align-items:center;
}
.conv-timeline-state {
  display:inline-block; font-size:10.5px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.05em;
  padding:1px 8px; border-radius:var(--radius-pill);
  background:#fbfafe; border:1px solid var(--line); color:var(--muted);
}
.conv-timeline-state.state-completed,
.conv-timeline-state.state-cached {
  background:var(--good-soft); color:var(--good); border-color:var(--good-line);
}
.conv-timeline-state.state-failed {
  background:var(--poor-soft); color:var(--poor); border-color:var(--poor-line);
}
.conv-timeline-metrics {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px;
}
.conv-timeline-meta-extra { color:var(--subtle); }
.conv-timeline-meta-id code {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11px; color:var(--subtle); background:transparent;
  border:none; padding:0;
}
.conv-timeline-error {
  margin-top:5px; padding:6px 10px; border-radius:6px;
  background:var(--poor-soft); border:1px solid var(--poor-line);
  color:var(--poor); font-size:12.5px;
}

/* -------- Decision audit -------- */
.conv-block-decisions::before { background:#d97706; }   /* amber for "the agent decided" */
.conv-role-badge-decisions {
  color:#92400e; background:#fef3c7; border-color:#fde68a;
}
.conv-decisions { list-style:none; margin:8px 0 0; padding:0; }
.conv-decision-row {
  padding:9px 0;
}
.conv-decision-row + .conv-decision-row { border-top:1px dashed #f0eef8; }
.conv-decision-head {
  display:flex; align-items:center; gap:10px; flex-wrap:wrap;
}
.conv-decision-round {
  font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.07em; color:var(--subtle);
  font-variant-numeric:tabular-nums;
}
.conv-decision-label {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; color:var(--ink);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:5px; padding:1px 8px;
}
.conv-decision-verdict {
  font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.05em; padding:2px 9px; border-radius:var(--radius-pill);
}
.conv-decision-verdict-go {
  color:#1e40af; background:#e7efff; border:1px solid #bfd4ff;
}
.conv-decision-verdict-stop {
  color:#475569; background:#eef2f7; border:1px solid #dbe3ec;
}
.conv-decision-reason {
  margin:6px 0 0; color:var(--ink); font-size:13.5px; line-height:1.55;
}
.conv-decision-reason p { margin:0 0 5px; }
.conv-decision-reason p:last-child { margin-bottom:0; }
.conv-decision-metrics {
  margin-top:5px; font-size:12px; color:var(--muted);
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}
.conv-decision-metrics-head {
  font-family:Inter, system-ui, sans-serif;
  text-transform:uppercase; font-size:10.5px; letter-spacing:0.07em;
  color:var(--subtle); margin-right:4px;
}
.conv-decision-notes {
  margin:6px 0 0; padding-left:18px; font-size:12.5px; color:var(--muted);
}
.conv-decision-note { margin:2px 0; }
.conv-decision-note strong { text-transform:uppercase; font-size:10.5px; letter-spacing:0.05em; }
.conv-decision-note-warning strong { color:var(--warn); }
.conv-decision-note-error   strong { color:var(--poor); }
.conv-decision-note-code {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11px; color:var(--subtle);
}

/* Humanised verdict label (e.g. "Workflow completed"). Sits between
   the round number and the continue/stop pill in the row head; raw
   underscore form lives in the title= tooltip + the collapsed
   "More detail" drawer. */
.conv-decision-label-friendly {
  font-size:13.5px; font-weight:600; color:var(--ink);
  letter-spacing:-0.005em;
}

/* Collapsible "More detail" drawer per decision row — carries the raw
   label, weighed metrics, and notes when present. Default-closed
   keeps the row to a clean two-line summary. */
.conv-decision-details {
  margin-top:6px;
}
.conv-decision-details-summary {
  cursor:pointer; user-select:none;
  display:inline-flex; align-items:center; gap:6px;
  padding:3px 8px; border-radius:6px;
  font-size:11.5px; color:var(--muted);
  list-style:none;
}
.conv-decision-details-summary::-webkit-details-marker { display:none; }
.conv-decision-details-summary::before {
  content:"▸"; color:var(--subtle); font-size:9px;
  transition:transform 0.15s;
}
.conv-decision-details[open] > .conv-decision-details-summary::before {
  transform:rotate(90deg); display:inline-block;
}
.conv-decision-details-summary:hover { background:#f7f5fb; color:var(--accent); }
.conv-decision-details-body {
  margin:6px 0 0; padding:6px 10px;
  background:#fbfafe; border-radius:6px;
}
.conv-decision-detail-row {
  display:flex; align-items:baseline; gap:10px;
  padding:3px 0; font-size:12.5px;
  flex-wrap:wrap;
}
.conv-decision-detail-row + .conv-decision-detail-row {
  border-top:1px dashed #f0eef8;
}
.conv-decision-detail-label {
  flex-shrink:0; min-width:110px;
  font-size:10.5px; font-weight:700;
  text-transform:uppercase; letter-spacing:0.06em;
  color:var(--subtle);
}
.conv-decision-raw {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px; color:var(--ink);
  background:#ffffff; border:1px solid var(--line);
  border-radius:5px; padding:1px 8px;
}
.conv-decision-metrics-list {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px; color:var(--muted);
  word-break:break-word;
}

/* -------- Tool-call chip (interleaved in full transcript) -------- */
.conv-tool-chip {
  display:block; margin:4px 0;
  border:1px solid var(--line); border-radius:8px;
  background:#fbfafe;
  font-size:12.5px;
}
.conv-tool-chip[open] { box-shadow:var(--shadow-tile); }
.conv-tool-chip-plain {
  padding:6px 11px;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
}
.conv-tool-summary {
  cursor:pointer; user-select:none;
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:6px 11px; border-radius:8px;
  list-style:none;
}
.conv-tool-summary::-webkit-details-marker { display:none; }
.conv-tool-summary::after {
  content:"▸"; color:var(--subtle); margin-left:auto;
  transition:transform 0.15s; font-size:10px;
}
.conv-tool-chip[open] > .conv-tool-summary::after { transform:rotate(90deg); }
.conv-tool-summary:hover { background:#f7f5fb; }
.conv-tool-chip-ok    { border-left:3px solid var(--good); }
.conv-tool-chip-fail  { border-left:3px solid var(--poor); background:var(--poor-soft); }
.conv-tool-glyph {
  display:inline-flex; align-items:center; justify-content:center;
  width:18px; height:18px; border-radius:50%; font-size:11px; font-weight:700;
}
.conv-tool-glyph-ok   { background:var(--good-soft); color:var(--good); }
.conv-tool-glyph-fail { background:#fee2e2;          color:var(--poor); }
.conv-tool-label {
  font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.07em; color:var(--subtle);
}
.conv-tool-name {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12.5px; font-weight:700; color:var(--ink);
  background:transparent;
}
.conv-tool-state {
  font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.05em; padding:1px 7px; border-radius:var(--radius-pill);
}
.conv-tool-state-fail { background:#fee2e2; color:var(--poor); }
.conv-tool-body {
  padding:0 11px 10px;
  border-top:1px solid var(--line);
}
.conv-tool-section { margin-top:9px; }
.conv-tool-section-head {
  font-size:10px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.07em; color:var(--subtle); margin-bottom:4px;
}
.conv-tool-payload {
  display:block; white-space:pre-wrap; word-break:break-word;
  padding:8px 10px; max-height:300px; overflow:auto;
  background:#0f0a1f; color:#ede9fe;
  border-radius:6px; font-size:11.5px; line-height:1.5;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}

.pill {
  border:1px solid var(--line); border-radius:var(--radius-pill); padding:4px 11px;
  font-size:12.5px; background:var(--panel); color:var(--muted); font-weight:500;
}
.pill.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.pill.state-completed, .pill.state-cached {
  border-color:var(--good-line); background:var(--good-soft); color:var(--good);
}
.pill.state-failed, .pill.state-cancelled {
  border-color:var(--poor-line); background:var(--poor-soft); color:var(--poor);
}
.pill.state-running { border-color:#c4b5fd; background:var(--accent-soft); color:#6b1ea8; }

.secondary {
  margin:14px 0 0; font-size:14px; padding:12px 16px;
  border-radius:var(--radius-sm); background:#ffffff; border:1px solid var(--line);
  box-shadow:var(--shadow-tile);
}
.secondary.ok { border-left:3px solid var(--good); }
.secondary.caution { border-left:3px solid var(--ok); background:var(--ok-soft); }

/* Headings */
h1, h2, h3, h4, h5 { letter-spacing:-0.005em; }
.widget-title { margin:0 0 10px; font-size:16px; font-weight:600; color:var(--ink); }
.widget-subhead { margin:16px 0 8px; font-size:13px; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }
.widget-explainer { margin:0 0 14px; font-size:14px; color:var(--muted); line-height:1.6; max-width:880px; }
.widget-explainer code { font-size:12.5px; background:var(--accent-soft); padding:1px 5px; border-radius:3px; }

/* Input-selection context callout — shown for tools run with an explicit
   residue-position parameter, so a short annotated subset isn't confusing. */
.scope-note {
  margin:11px 0 14px; padding:11px 14px;
  background:var(--accent-soft); border:1px solid #d9cef3;
  border-radius:9px;
}
.scope-note-label {
  font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.05em; color:var(--accent); margin-right:9px;
}
.scope-note-pos {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12.5px; font-weight:700; color:var(--ink);
}
.scope-note-missing {
  margin-left:8px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:10.5px; font-weight:700; color:var(--ok);
  background:var(--ok-soft); border:1px solid var(--ok-line);
  border-radius:999px; padding:1px 9px;
}
.scope-note-detail {
  margin:6px 0 0; font-size:12px; color:var(--muted); line-height:1.55;
}

/* HERO METRICS — tile-style stat cards (donut-less, clean) */
.metric-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
  gap:12px; margin:16px 0 0;
}
.metric-card {
  background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; text-align:left;
  position:relative; overflow:hidden;
  transition:border-color 0.15s;
}
.metric-card:hover { border-color:var(--line-strong); }
.metric-card::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width:2px;
  background:var(--line);
}
.metric-card.metric-good::before { background:var(--good); }
.metric-card.metric-ok::before { background:var(--ok); }
.metric-card.metric-poor::before { background:var(--poor); }
.metric-card.metric-neutral::before { background:var(--accent); }
.metric-card strong {
  display:block; font-size:24px; line-height:1.15; color:var(--ink); font-weight:600;
  font-variant-numeric: tabular-nums; letter-spacing:-0.01em;
}
.metric-card span {
  display:block; color:var(--muted); font-size:10.5px; margin-top:4px;
  text-transform:uppercase; letter-spacing:0.07em; font-weight:600;
}

/* ========================================================================
   EXECUTIVE DASHBOARD — CELLKEY-style 3-panel summary at the top.
   Layout: [Run vitals] [Confidence donuts] [Pipeline health]
   Each panel is its own bordered tile with title + sub + content body.
   Collapses to a single column on narrow viewports.
   ======================================================================== */
.exec-dashboard {
  display:grid;
  /* Pipeline panel needs extra width to fit long underscored tool names
     like ``structure_analysis`` on a single line next to its donut. */
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 0.95fr) minmax(0, 1.25fr);
  gap:12px; margin:6px 0 14px;
}
.exec-panel {
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px;
  display:flex; flex-direction:column; gap:12px;
  box-shadow:var(--shadow-tile);
}
.exec-panel-head { display:flex; flex-direction:column; gap:2px; }
.exec-panel-title {
  margin:0; font-size:15px; color:var(--ink); font-weight:600;
  letter-spacing:-0.005em;
}
.exec-panel-sub {
  font-size:12.5px; color:var(--muted); letter-spacing:0.02em;
}

/* KPI tile grid (Run vitals panel). 2 cols x 3 rows: each tile gets a
   generous width budget so multi-character state labels ("Completed",
   "Cancelled") never overflow. */
.kpi-grid {
  display:grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap:9px;
}
.kpi-tile {
  display:flex; align-items:center; gap:11px;
  padding:13px 14px;
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  position:relative; overflow:visible;
  box-shadow:var(--shadow-tile);
  transition:border-color 0.15s, box-shadow 0.15s;
}
.kpi-tile:hover { border-color:var(--line-strong); box-shadow:var(--shadow-card); }
.kpi-icon {
  flex-shrink:0;
  width:34px; height:34px; border-radius:8px;
  display:flex; align-items:center; justify-content:center;
  font-size:17px; font-weight:600;
  background:#ffffff; border:1px solid var(--line);
  color:var(--ink);
}
.kpi-logo {
  flex-shrink:0; width:34px; height:34px; border-radius:8px; display:block;
}
.kpi-body { min-width:0; flex:1; }
.kpi-value {
  /* 17px / 700 — comfortably readable; with the 2-col grid this fits
     "Completed", "Cancelled" etc. on a single line. */
  font-size:17px; font-weight:700; color:var(--ink);
  font-variant-numeric:tabular-nums; letter-spacing:-0.01em;
  line-height:1.2;
  white-space:nowrap;
}
.kpi-label {
  font-size:11px; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.06em;
  font-weight:600; margin-top:3px;
}
/* Accent colours per state/quality */
.kpi-tile.kpi-good   .kpi-icon { background:var(--good-soft);   color:var(--good);   border-color:var(--good-line); }
.kpi-tile.kpi-ok     .kpi-icon { background:var(--ok-soft);     color:var(--ok);     border-color:var(--ok-line); }
.kpi-tile.kpi-poor   .kpi-icon { background:var(--poor-soft);   color:var(--poor);   border-color:var(--poor-line); }
.kpi-tile.kpi-neutral .kpi-icon { background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
.kpi-tile.kpi-completed .kpi-icon { background:var(--good-soft); color:var(--good); border-color:var(--good-line); }
.kpi-tile.kpi-running .kpi-icon { background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
.kpi-tile.kpi-failed .kpi-icon { background:var(--poor-soft); color:var(--poor); border-color:var(--poor-line); }
.kpi-tile.kpi-queued .kpi-icon,
.kpi-tile.kpi-pending .kpi-icon { background:var(--bg); color:var(--muted); }

/* Confidence cards (middle panel) -------------------------------------- */
/* Stacked vertically (one card per row), each card laid out as
   [donut | label + explainer]. The list scrolls within the panel so the
   donuts never crowd each other or blow out the dashboard height. */
.exec-donut-row {
  display:flex; flex-direction:column; gap:10px;
  max-height:250px; overflow-y:auto;
  padding-right:4px;
}
.exec-donut-row::-webkit-scrollbar { width:10px; }
.exec-donut-row::-webkit-scrollbar-track { background:#e7e3f3; border-radius:999px; }
.exec-donut-row::-webkit-scrollbar-thumb {
  background:#a99cd6; border:2px solid #e7e3f3; border-radius:999px;
}
.exec-donut-row::-webkit-scrollbar-thumb:hover { background:#8574c0; }
.exec-donut {
  flex-shrink:0;
  display:flex; flex-direction:row; align-items:center; gap:14px;
  padding:12px 14px;
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  text-align:left;
  box-shadow:var(--shadow-tile);
}
.exec-donut-viz { line-height:0; flex-shrink:0; }
.exec-donut-viz svg { width:86px; height:86px; }
.exec-donut-meta { flex:1; min-width:0; }
.exec-donut-label {
  font-size:13px; color:var(--ink); font-weight:600; letter-spacing:0;
}
.exec-donut-tip {
  font-size:11.5px; color:var(--muted); line-height:1.45; margin-top:3px;
}
.exec-donut-empty { justify-content:flex-start; }
.exec-donut-id code {
  font-size:12px; font-family:ui-monospace, monospace;
  background:var(--panel); padding:2px 6px; border-radius:5px;
}

/* Pipeline health (right panel) — donut + stages list + legend chips.
   The donut sits centred on the left, the stages list fills the rest of
   the panel's horizontal space, and the legend strip lands underneath
   both so the whole panel feels "full" rather than airy. */
.exec-pipe-body {
  display:flex; gap:14px; align-items:flex-start;
  flex:1;
}
.exec-pipe-donut-wrap {
  flex-shrink:0;
  display:flex; align-items:center; justify-content:center;
}
.exec-pipe-donut { display:block; }
.exec-pipe-val {
  font-family: Inter, system-ui, sans-serif;
  font-weight:700; font-size:22px; fill:var(--ink);
  font-variant-numeric:tabular-nums; letter-spacing:-0.02em;
}
.exec-pipe-sub {
  font-family: Inter, system-ui, sans-serif;
  font-weight:500; font-size:11px; fill:var(--muted);
  letter-spacing:0.04em;
}
/* Stages list — one row per workflow node; every node is listed and the
   list scrolls within the panel with an always-visible scrollbar. */
.exec-stage-list {
  list-style:none; padding:0; margin:0;
  display:flex; flex-direction:column; gap:3px;
  flex:1; min-width:0;
  max-height:230px; overflow-y:auto;
  padding-right:4px;
}
.exec-stage-list::-webkit-scrollbar { width:10px; }
.exec-stage-list::-webkit-scrollbar-track { background:#e7e3f3; border-radius:999px; }
.exec-stage-list::-webkit-scrollbar-thumb {
  background:#a99cd6; border:2px solid #e7e3f3; border-radius:999px;
}
.exec-stage-list::-webkit-scrollbar-thumb:hover { background:#8574c0; }
.exec-stage-row {
  flex-shrink:0;
  display:flex; align-items:center; gap:9px;
  padding:6px 10px; border-radius:7px;
  font-size:12px; line-height:1.4;
  background:#ffffff; border:1px solid var(--line);
  box-shadow:var(--shadow-tile);
}
.exec-stage-glyph {
  flex-shrink:0; width:14px; text-align:center;
  font-weight:700; font-size:13px;
}
.exec-stage-name {
  flex:1; min-width:0;
  color:var(--ink); font-weight:500;
  font-family:ui-monospace, monospace;
  /* Single-line: tool names like ``structure_analysis`` stay on one row
     inside the pipeline panel. The panel column width is sized to fit
     the longest typical name; truly oversized ones still get a graceful
     hover tooltip (the row already has a ``title`` attribute). */
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
/* Compact legend chip strip below the donut + stages */
.exec-pipe-legend {
  display:flex; flex-wrap:wrap; gap:6px;
  padding-top:10px; margin-top:auto;
  border-top:1px solid var(--line);
}
.exec-pipe-legend-chip {
  display:inline-flex; align-items:center; gap:7px;
  padding:4px 11px;
  background:#ffffff; border:1px solid var(--line); border-radius:999px;
  font-size:12.5px; color:var(--ink);
  box-shadow:var(--shadow-tile);
}
.exec-pipe-legend-swatch {
  width:8px; height:8px; border-radius:50%; flex-shrink:0;
}
.exec-pipe-legend-label { text-transform:capitalize; }
.exec-pipe-legend-count {
  color:var(--muted); font-variant-numeric:tabular-nums;
  font-weight:600;
}

/* Verdict callout — high-signal "should I trust this run?" cue --------- */
.exec-verdict {
  display:flex; align-items:flex-start; gap:16px;
  padding:16px 20px; margin:2px 0 16px;
  border-radius:12px; border:1px solid var(--line); background:#ffffff;
  box-shadow:var(--shadow-tile);
}
.exec-verdict-icon {
  flex-shrink:0;
  width:36px; height:36px; border-radius:9px;
  display:flex; align-items:center; justify-content:center;
  font-size:18px; font-weight:700;
  background:#ffffff; border:1px solid var(--line);
}
.exec-verdict-body { min-width:0; flex:1; }
.exec-verdict-head {
  font-size:15.5px; color:var(--ink); font-weight:600; line-height:1.4;
}
.exec-verdict-detail {
  font-size:13.5px; color:var(--muted); line-height:1.6; margin-top:4px;
  /* Same defence as ``.warnings p`` — cautions can carry long error
     strings (subprocess paths, hashes) that would otherwise overflow
     the callout's right edge. */
  overflow-wrap:anywhere; word-break:break-word;
}
.exec-verdict-good {
  background:var(--good-soft); border-color:var(--good-line);
}
.exec-verdict-good .exec-verdict-icon {
  background:var(--good); border-color:var(--good); color:#fff;
}
.exec-verdict-ok {
  background:var(--ok-soft); border-color:var(--ok-line);
}
.exec-verdict-ok .exec-verdict-icon {
  background:var(--ok); border-color:var(--ok); color:#fff;
}
.exec-verdict-poor {
  background:var(--poor-soft); border-color:var(--poor-line);
}
.exec-verdict-poor .exec-verdict-icon {
  background:var(--poor); border-color:var(--poor); color:#fff;
}
.exec-verdict-neutral .exec-verdict-icon {
  background:var(--accent-soft); color:var(--accent); border-color:var(--accent);
}

@media (max-width: 1100px) {
  .exec-dashboard { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }
  .exec-panel-pipeline { grid-column: span 2; }
  .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 720px) {
  .exec-dashboard { grid-template-columns: 1fr; }
  .exec-panel-pipeline { grid-column: span 1; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .exec-pipe-wrap { flex-direction:column; align-items:flex-start; gap:10px; }
}

/* HERO SEQUENCE */
.hero-sequence {
  margin-top:16px; padding:14px 16px;
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  box-shadow:var(--shadow-tile);
}
.hero-sequence-head {
  display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;
}
.hero-sequence-body {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size:14.5px; line-height:1.7; overflow-x:auto; padding-bottom:4px;
  white-space:nowrap;
}
.hero-sequence-body code { background:transparent; padding:0; }

/* Workflow DAG — single status circle replaces left strip + dot redundancy.
   Card relies on a real (very subtle) drop-shadow filter rather than tinted
   background to convey state. Edges are soft Beziers with a thin chevron. */
.dag-wrap {
  overflow-x:auto;
  /* Extra bottom padding reserves room for the always-on scrollbar. */
  padding:6px 2px 14px;
}
/* Always-visible custom scrollbar so users can tell the graph scrolls
   left/right without having to click-drag first. IMPORTANT: do NOT also
   set the standard `scrollbar-width` / `scrollbar-color` on this element —
   when either is present Chromium ignores every ::-webkit-scrollbar rule
   below and falls back to a near-invisible default scrollbar. */
.dag-wrap::-webkit-scrollbar { height:13px; }
.dag-wrap::-webkit-scrollbar-track {
  background:#e7e3f3; border-radius:999px;
}
.dag-wrap::-webkit-scrollbar-thumb {
  background:#a99cd6; border:3px solid #e7e3f3; border-radius:999px;
}
.dag-wrap::-webkit-scrollbar-thumb:hover { background:#8574c0; }
/* Render the SVG at its natural pixel size (width/height attributes) so nodes
   stay legible no matter how many tools ran; .dag-wrap scrolls horizontally. */
.dag { display:block; }

/* Workflow-composition panel — explains how / why the pipeline was
   assembled: run mode, dependency stages (or independent steps) and the
   role of each tool, so the DAG above is understandable on its own. */
.wf-comp { margin-top:20px; padding-top:18px; border-top:1px solid var(--line); }
.wf-comp-title {
  margin:0 0 10px; font-size:15px; font-weight:600; color:var(--ink);
}
.wf-comp-mode {
  display:inline-block; margin-bottom:9px;
  font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.05em; border-radius:999px; padding:3px 12px;
}
.wf-mode-agent {
  color:var(--accent); background:var(--accent-soft); border:1px solid #d9cef3;
}
.wf-mode-workflow {
  color:#0c4a63; background:#e5f4fd; border:1px solid #bfe6f5;
}
.wf-mode-tool {
  color:#7c2d12; background:#fcebdd; border:1px solid #f3d3b8;
}
.wf-comp-lead {
  margin:0; font-size:13px; color:var(--muted); line-height:1.6; max-width:900px;
}
.wf-comp-stats { display:flex; flex-wrap:wrap; gap:8px; margin:13px 0; }
.wf-comp-stat {
  font-size:11.5px; color:var(--muted);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:999px; padding:3px 11px;
}
.wf-comp-stat b {
  color:var(--ink); font-weight:700; font-variant-numeric:tabular-nums;
}
/* Ordered execution list — every step in run order with its recovered
   input parameters and any upstream dependency. */
.wf-steps {
  display:flex; flex-direction:column;
  background:#fbfafe; border:1px solid var(--line); border-radius:10px;
  padding:4px 14px;
}
.wf-step {
  display:grid; grid-template-columns:28px 10px 176px 1fr;
  align-items:baseline; gap:11px; padding:9px 0;
}
.wf-step + .wf-step { border-top:1px solid #f0eef8; }
.wf-step-num {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11px; font-weight:700; color:var(--subtle);
  font-variant-numeric:tabular-nums; text-align:right;
}
.wf-step-dot {
  width:8px; height:8px; border-radius:50%; align-self:baseline;
  background:var(--subtle);
}
.wf-step-dot.state-completed, .wf-step-dot.state-cached { background:var(--good); }
.wf-step-dot.state-failed { background:var(--poor); }
.wf-step-dot.state-running { background:var(--accent-bright); }
.wf-step-tool {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12.5px; font-weight:700; color:var(--ink); word-break:break-word;
}
.wf-step-params { display:flex; flex-wrap:wrap; gap:6px; }
.wf-param {
  font-size:11px; color:var(--muted);
  background:#ffffff; border:1px solid var(--line);
  border-radius:6px; padding:2px 8px;
}
.wf-param b { color:var(--ink); font-weight:700; margin-right:5px; }
.wf-param-dep {
  color:var(--accent); background:var(--accent-soft); border-color:#d9cef3;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-weight:700;
}
.wf-step-noparam { color:var(--subtle); font-size:11.5px; font-style:italic; }
@media (max-width:640px) {
  .wf-step { grid-template-columns:28px 10px 1fr; }
  .wf-step-params { grid-column:3; }
}
/* Workflow DAG edges — bright violet line with a filled chevron at the
   target, so the directional cue reads at a glance instead of fading
   into the lavender background. */
.dag-edge { stroke:#a78bfa; stroke-width:2; fill:none; stroke-linecap:round; }
.dag-card {
  fill:#ffffff;
  stroke:var(--line); stroke-width:1;
}
.dag-card-group:hover .dag-card { stroke:var(--line-strong); }
.dag-title {
  font-size:13.5px; font-weight:600; fill:var(--ink); letter-spacing:-0.005em;
}
.dag-tool {
  font-size:11px; fill:var(--muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing:0.01em;
}
.dag-step-num {
  font-size:11px; font-weight:600; fill:var(--muted);
  font-family: Inter, system-ui, sans-serif;
}
.dag-circle.state-pending {
  fill:#ffffff; stroke:var(--line-strong); stroke-width:1.5;
}
.dag-circle.state-running   { fill:var(--accent-3); }
.dag-circle.state-completed,
.dag-circle.state-cached    { fill:var(--good); }
.dag-circle.state-failed    { fill:var(--poor); }
.dag-circle.state-cancelled { fill:var(--accent-2); }

/* Candidate card */
.candidate-card {
  border:1px solid var(--line); border-radius:12px; padding:22px 24px; margin:16px 0;
  background:var(--panel);
  box-shadow:var(--shadow-tile);
}
/* Small uppercase eyebrow ("Top #N", "Candidate", "Input sequence ...").
   Without an explicit rule the <p class="eyebrow"> inherits browser <p>
   defaults — that's why the "Candidate" subtitle was floating with a
   weird gap above the heading. Match the rest of the report's eyebrow
   styling so the alignment looks intentional. */
.eyebrow {
  margin:0 0 6px; color:var(--subtle);
  font-size:11.5px; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;
}
.candidate-head {
  display:flex; justify-content:space-between; gap:16px; align-items:flex-start;
}
.candidate-head h3 {
  margin:0; font-size:20px; font-weight:600; letter-spacing:-0.005em;
  color:var(--ink); line-height:1.25;
}
.card-score { text-align:right; flex-shrink:0; }
.card-score strong {
  display:block; font-size:30px; font-weight:700; color:var(--ink); line-height:1;
  letter-spacing:-0.015em; font-variant-numeric:tabular-nums;
}
.card-score span { color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:0.07em; font-weight:600; margin-top:3px; display:block; }
.card-score.quality-good strong { color:var(--good); }
.card-score.quality-ok strong { color:var(--ok); }
.card-score.quality-poor strong { color:var(--poor); }
.candidate-grid {
  display:grid; grid-template-columns: minmax(360px, 1.3fr) minmax(260px, 1fr);
  gap:22px; margin-top:16px;
}
/* Sequence-only tools (no structure artifact): collapse the grid to a
   single full-width info column instead of leaving an empty viewer slot
   on the left that just says "No structure artifact linked." */
.candidate-grid--solo { grid-template-columns: 1fr; }
/* CSS Grid items default to ``min-width:auto`` (= grow to fit content),
   so a long sequence span or a wide score-bar makes the whole column
   blow past its track and overflow the card. Pin both columns to
   ``min-width:0`` so the track width wins; the sequence block's own
   ``overflow-x:auto`` then kicks in and gives a horizontal scrollbar
   instead of pushing siblings (score bars, bar tracks, ...) off-card. */
.candidate-viewer-col, .candidate-info-col { min-width:0; }

/* Inline 3D viewer (3Dmol.js host) */
.structure-section {
  background:#ffffff; border-radius:10px; padding:0; position:relative; overflow:hidden;
  border:1px solid var(--line);
  box-shadow:var(--shadow-tile);
  display:flex; flex-direction:column;
  /* Taller wrapper so the WebGL canvas + sequence strip both have
     breathing room without scrolling. */
  min-height:720px;
}
.structure-section:fullscreen {
  background:#ffffff; display:flex; flex-direction:column;
}
.structure-section:fullscreen .structure-viewer { flex:1 1 auto; }

.structure-header {
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:10px 14px;
  background:#ffffff; border-bottom:1px solid var(--line);
  font-size:13px; color:var(--muted);
}
.structure-title {
  font-weight:600; font-size:12px; color:var(--ink);
  text-transform:uppercase; letter-spacing:0.06em;
  margin-right:auto;  /* spacer: pushes the control buttons to the right edge */
}
.structure-hint {
  /* Hint sits below the controls on its own line, right-aligned. */
  order:100; flex:0 0 100%;
  text-align:right;
  font-size:11px; color:var(--muted);
  margin-top:2px;
}
.structure-control {
  display:inline-flex; align-items:center; gap:6px;
  font-size:12px; color:var(--ink);
  padding:4px 9px; border:1px solid var(--line); border-radius:5px;
  background:var(--panel); cursor:pointer;
}
.structure-control:hover {
  background:var(--accent-soft); border-color:var(--accent); color:var(--accent);
}
.structure-control input[type="checkbox"] { accent-color:var(--accent); margin:0; }
.structure-control select {
  font-size:12px; border:none; background:transparent; color:inherit; cursor:pointer;
  padding:0; font-family:inherit;
}
.structure-control span {
  text-transform:uppercase; letter-spacing:0.05em; font-size:10px;
  color:var(--muted); font-weight:600;
}
.structure-fullscreen-btn { font-weight:500; }
/* RCSB-style "Pick" toggle. When ``aria-pressed=true`` the button gets a
   solid violet fill so the active picker mode is immediately readable. */
.structure-picker-btn { font-weight:500; }
.structure-picker-btn[aria-pressed="true"] {
  background:var(--accent); border-color:var(--accent); color:#ffffff;
  box-shadow:var(--shadow-tile);
}
.structure-picker-btn[aria-pressed="true"]:hover {
  background:var(--accent-deep); border-color:var(--accent-deep); color:#ffffff;
}

/* Bottom-right residue inspector. Mirrors the floating chip RCSB Mol*
   uses at the bottom-right when the picker tool is active — shows the
   structure title, source file, currently-hovered chain + residue, and
   the residue's position in the FASTA sequence. Sits absolutely inside
   ``.structure-section`` so it overlays the WebGL canvas without
   resizing it. */
.structure-residue-info {
  position:absolute; right:14px; bottom:14px;
  z-index:6;
  background:rgba(255,255,255,0.97);
  border:1px solid var(--line-strong); border-radius:10px;
  padding:10px 14px; max-width:380px;
  box-shadow:var(--shadow-card);
  pointer-events:none;
  font-size:12px; color:var(--ink);
  backdrop-filter:saturate(140%) blur(4px);
}
.structure-residue-info[hidden] { display:none; }
.structure-residue-info .sri-title {
  font-weight:600; font-size:13px; color:var(--ink);
  letter-spacing:-0.005em; line-height:1.3;
  margin-bottom:4px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.structure-residue-info .sri-detail {
  display:flex; flex-wrap:wrap; align-items:center; gap:6px;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px; color:var(--muted);
}
.structure-residue-info .sri-sep { color:var(--subtle); }
.structure-residue-info .sri-source {
  max-width:140px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.structure-residue-info .sri-chain { color:var(--accent); font-weight:600; }
.structure-residue-info .sri-resi { color:var(--ink); font-weight:700; }
.structure-residue-info .sri-resi strong { font-weight:700; }
.structure-residue-info .sri-seqpos { color:var(--accent-2); font-weight:600; }
.structure-residue-info .sri-placeholder { color:var(--subtle); font-weight:500; }

.structure-viewer {
  flex:1 1 0;
  min-height:560px;
  position:relative;
  background:#ffffff;
}
.structure-viewer canvas { outline:none; display:block; }

/* Sequence strip above the viewer (between header and 3D canvas).
   Placed up here so the bottom-right Pick info card never overlaps the
   FASTA letters. ``.sst-cell`` carries the 1-letter code; the
   picker-mode hover handler flips a single cell into ``.is-hovered``
   so the user sees exactly which FASTA position corresponds to the 3D
   residue they're inspecting. */
.structure-sequence-track {
  border-bottom:1px solid var(--line);
  padding:14px 18px;
  background:#fafaff;
  flex-shrink:0;
}
.structure-sequence-track .sst-head {
  display:flex; align-items:center; justify-content:space-between;
  margin-bottom:8px; gap:12px;
}
.structure-sequence-track .sst-label {
  font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
  color:var(--muted); font-weight:600;
}
.structure-sequence-track .sst-meta {
  font-size:11.5px; color:var(--muted);
  font-variant-numeric:tabular-nums;
}
.structure-sequence-track .sst-strip {
  display:block;
  background:#ffffff;
  border:1px solid var(--line);
  border-radius:8px;
  padding:10px 12px;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:13px; line-height:1.75;
  /* ~7-8 lines of residues fit before vertical scroll kicks in.
     A 900-aa sequence at ~75 chars/line ≈ 12 lines → user scrolls;
     ``scrollIntoView`` in the JS hover handler keeps the active
     residue inside the viewport even on long sequences. */
  max-height:160px; overflow-y:auto;
  letter-spacing:0.02em;
  word-break:break-all;
  color:var(--ink);
  box-shadow:var(--shadow-tile);
  /* Slim brand-tinted scrollbar — replaces the chunky default OS
     scrollbar so the strip doesn't feel like a stuck dialog. */
  scrollbar-width:thin;
  scrollbar-color:var(--accent-bright) transparent;
}
.structure-sequence-track .sst-strip::-webkit-scrollbar { width:7px; }
.structure-sequence-track .sst-strip::-webkit-scrollbar-track { background:transparent; }
.structure-sequence-track .sst-strip::-webkit-scrollbar-thumb {
  background:var(--accent-bright); border-radius:6px;
  border:2px solid #ffffff;
}
.structure-sequence-track .sst-strip::-webkit-scrollbar-thumb:hover {
  background:var(--accent);
}
.sst-cell {
  display:inline-block; min-width:13px; text-align:center;
  border-radius:3px; padding:0 1px;
  color:var(--ink);
  transition:background-color 0.12s, color 0.12s, transform 0.12s;
}
.sst-cell.is-hovered {
  background:#cc0980; color:#ffffff; font-weight:700;
  transform:scale(1.18);
  box-shadow:0 1px 4px rgba(204,9,128,0.35);
}
/* Position ruler tick for sequences >= 100 aa. Sits inline with the
   residues so the reader can count without leaving the strip. */
.sst-tick {
  display:inline-block;
  font-size:10px; color:var(--accent);
  font-variant-numeric:tabular-nums;
  vertical-align:baseline;
  padding:0 6px 0 2px;
  margin-right:2px;
  border-left:2px solid var(--accent-soft);
  background:var(--accent-soft);
  border-radius:3px;
  letter-spacing:0;
  user-select:none;
}
.structure-status {
  position:absolute; inset:0;
  display:flex; align-items:center; justify-content:center;
  color:var(--muted); font-style:italic; font-size:13px;
  background:rgba(255,255,255,0.92);
}

.structure-legend {
  position:absolute; right:12px; bottom:12px;
  display:flex; gap:6px; align-items:center;
  background:rgba(255,255,255,0.94); padding:5px 10px; border-radius:6px;
  border:1px solid var(--line); font-size:11px; color:var(--muted);
  pointer-events:none;
}
.legend-text { font-size:10px; }
.legend-gradient {
  width:90px; height:8px; border-radius:2px;
  background:linear-gradient(to right, #3a2bd9, #2188d4, #1ec19a, #9fc23a, #f0d537, #f08a2b, #d62e2e);
}
.structure-legend[data-legend="plddt"] .legend-gradient {
  background:linear-gradient(to right, #FF7D45 0%, #FFDB13 33%, #65CBF3 66%, #0053D6 100%);
}
.structure-legend[data-legend="spectrum"] .legend-gradient {
  background:linear-gradient(to right, #2647e3, #1ea7e0, #1ec19a, #9fc23a, #f0d537, #f08a2b, #d62e2e);
}
.structure-legend[data-legend="chain"] { display:none; }

.viewer-empty {
  color:var(--muted); border:1px dashed var(--line); padding:24px; border-radius:8px;
  text-align:center; min-height:120px; display:flex; align-items:center; justify-content:center;
}

/* Score bars */
.score-bars { margin-bottom:14px; }
/* ``auto 1fr 70px`` lets the label column grow to the widest label in
   the entire grid — every metric name renders in full on a single line
   (no ellipsis truncation), and all labels stay left-aligned because
   they share the same auto-sized column. The bar shrinks to fill
   whatever horizontal space remains. */
.score-row {
  display:grid; grid-template-columns:auto 1fr 70px;
  gap:14px; align-items:center;
  margin:7px 0; font-size:14px;
}
.score-row > span {
  color:var(--muted);
  white-space:nowrap;
}
.score-row > b { font-family: ui-monospace, monospace; font-weight:600; text-align:right; }
/* ⓘ chip beside metric labels: surfaces the per-metric definition
   tooltip (especially useful for tool-ambiguous names like ``score`` on
   MPNN inverse-folding outputs). Visually muted by default, slightly
   coloured on hover so users notice it's interactive. ``cursor:help``
   is the standard affordance for tooltip targets. */
.score-info {
  display:inline-block; margin-left:5px;
  font-size:11px; color:var(--subtle); cursor:help;
  vertical-align:1px; user-select:none;
}
.score-info:hover { color:var(--accent); }
.bar-track { height:9px; background:#ede9fe; border-radius:999px; overflow:hidden; }
/* Bar quality colour rule:
     good   → sky    (cool calm "this is fine")
     ok     → orange (warm warning, mid-range)
     poor   → pink   (warm critical)
     neutral→ yellow/orange (informational, no threshold)  */
.bar-track i { display:block; height:100%; background:var(--accent-4); }
.score-row.quality-good .bar-track i { background:var(--good); }
.score-row.quality-ok .bar-track i { background:var(--ok); }
.score-row.quality-poor .bar-track i { background:var(--poor); }
.score-row.quality-good > b { color:var(--good); }
.score-row.quality-ok > b { color:var(--ok); }
.score-row.quality-poor > b { color:var(--poor); }

.why { color:#312e81; line-height:1.6; font-size:15px; margin:12px 0; }

/* Badges (flags) */
.badges { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0; }
.badges span.badge-warn {
  background:var(--ok-soft); color:var(--warn); border:1px solid var(--ok-line);
  border-radius:999px; padding:5px 12px; font-size:13px;
}

/* Sequence block */
.sequence-block { margin:10px 0; }
.sequence-head { display:flex; align-items:center; gap:10px; margin-bottom:6px; font-size:13px; }
.sequence-block code {
  display:block; padding:10px 12px; background:#f8f5fd;
  border:1px solid var(--line); border-radius:6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size:13px; line-height:1.7; overflow-x:auto; white-space:nowrap;
}

/* Amino acid colouring */
.aa { display:inline-block; min-width:11px; text-align:center; border-radius:2px; padding:0 1px; }
/* Amino-acid pastel cells — built on the user-specified palette so the
   five property groups stay distinct while harmonising with the brand:
   hydrophobic sky, basic pink, acidic violet, polar lavender, special
   orange. Text colours pick a darker tone from the same hue family. */
.aa-hydro   { background:#e5f4fd; color:#0c4a63; }
.aa-pos     { background:#e5f4fd; color:#0c4a63; }
.aa-neg     { background:#f4cfe6; color:#831843; }
.aa-polar   { background:#ede9fe; color:#2d046e; }
.aa-special { background:#fcebdd; color:#7c2d12; }
.aa-gap     { color:#a09bbf; }

/* Downloads */
.downloads {
  margin-top:14px; padding:10px 0 0; border-top:1px dashed var(--line);
  display:flex; flex-wrap:wrap; gap:6px; align-items:center;
}
.downloads-label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; }
.dl-link {
  display:inline-flex; gap:6px; align-items:baseline;
  padding:5px 10px; background:#fbf9fe; border:1px solid var(--line);
  border-radius:6px; font-size:12px; color:var(--accent); text-decoration:none;
}
.dl-link:hover { background:var(--accent-soft); }
.dl-meta { color:var(--muted); font-size:11px; }

/* Tables */
.table-wrap { overflow:auto; border:1px solid var(--line); border-radius:10px; margin-top:10px; background:#ffffff; box-shadow:var(--shadow-tile); }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:#f5f1fa; cursor:pointer; position:sticky; top:0; font-weight:600; }
th:hover { background:#e6e0f5; }
tr:last-child td { border-bottom:none; }
td.quality-good { color:var(--good); font-weight:600; }
td.quality-ok { color:var(--ok); font-weight:600; }
td.quality-poor { color:var(--poor); font-weight:600; }

.filter {
  width:100%; max-width:320px; margin:10px 0 0;
  border:1px solid var(--line); border-radius:7px; padding:8px 11px; font-size:13px;
}

/* Plots */
.plot { border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0; background:#ffffff; box-shadow:var(--shadow-tile); }
.plot h3 { margin:0 0 8px; font-size:14px; font-weight:600; color:var(--ink); text-transform:none; letter-spacing:0; }
.plot-svg { width:100%; max-width:100%; height:auto; }
/* Legacy plots (scatter / bar) keep the flat accent fill; the redesigned
   plots (pLDDT trace, score bars) opt out and style their own elements. */
.plot-svg:not(.plddt-plot):not(.sd-plot) rect,
.plot-svg:not(.plddt-plot):not(.sd-plot) circle { fill:var(--accent); }
.plot-svg text { font-size:11px; fill:var(--muted); }
/* Per-residue pLDDT trace — product-analytics styling. */
.plddt-plot { overflow:visible; }
.plddt-field { fill:#fbfafe; stroke:var(--line); stroke-width:1; }
.plddt-line {
  fill:none; stroke:var(--accent); stroke-width:2;
  stroke-linejoin:round; stroke-linecap:round;
}
.plddt-axis-line { stroke:var(--line-strong); stroke-width:1; }
.plddt-plot .plddt-axis { font-size:10.5px; fill:var(--muted); }
.plddt-plot .plddt-axis-title { font-size:11px; fill:var(--subtle); letter-spacing:0.04em; }
.plddt-plot .plddt-subtitle { font-size:11.5px; fill:var(--muted); font-weight:500; }
.plddt-plot .plddt-ref-label { font-size:9.5px; fill:var(--subtle); letter-spacing:0.02em; }
/* Hover layer — hidden until PLDDT_HOVER_JS reveals it on mousemove. */
.plddt-hover { opacity:0; transition:opacity .08s ease; pointer-events:none; }
.plddt-crosshair {
  stroke:var(--accent); stroke-width:1; stroke-dasharray:3 3; opacity:0.5;
}
.plddt-plot circle.plddt-cursor {
  fill:var(--accent); stroke:#ffffff; stroke-width:2.5;
}
.plddt-plot rect.plddt-tip-bg { fill:var(--ink); }
.plddt-plot .plddt-tip-res { font-size:10px; fill:#b9b3d6; font-weight:500; }
.plddt-plot .plddt-tip-val { font-size:11.5px; fill:#ffffff; font-weight:700; }
/* Candidate score bar chart — product-analytics styling. */
.sd-plot { overflow:visible; }
.sd-field { fill:#fbfafe; stroke:var(--line); stroke-width:1; }
.sd-grid { stroke:#eceaf4; stroke-width:1; }
.sd-axis-line { stroke:var(--line-strong); stroke-width:1; }
.sd-plot .sd-bar { fill:url(#sdBar); }
.sd-plot .sd-bar-label { font-size:10.5px; fill:var(--ink); font-weight:600; }
.sd-plot .sd-axis { font-size:10.5px; fill:var(--muted); }
.sd-plot .sd-axis-title { font-size:11px; fill:var(--subtle); letter-spacing:0.04em; }
.sd-plot .sd-subtitle { font-size:11.5px; fill:var(--muted); font-weight:500; }
/* Second-line subtitle (numeric summary under the tool/metric caption);
   slightly smaller and lower-contrast so the eye lands on line 1 first. */
.sd-plot .sd-subtitle-meta { font-size:11px; fill:var(--subtle); font-weight:400; }
.sd-mean { stroke:var(--accent-2); stroke-width:1; stroke-dasharray:4 4; }
.sd-plot .sd-mean-label { font-size:9.5px; fill:var(--accent-2); letter-spacing:0.02em; }

/* MSA */
.msa-section { margin:18px 0; }
.msa-section h3 { margin:0 0 4px; font-size:14px; }
.msa-stats {
  color:var(--muted); font-size:13px; margin:4px 0 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}
.msa-controls {
  display:flex; gap:14px; align-items:center; flex-wrap:wrap;
  margin:10px 0 8px; padding:8px 10px;
  background:#f5f1fa; border:1px solid var(--line); border-radius:8px;
  font-size:13px;
}
.msa-controls label { display:flex; align-items:center; gap:8px; color:var(--muted); }
.msa-controls select, .msa-controls input[type=range] { font:inherit; }
.msa-controls select {
  padding:3px 6px; border:1px solid var(--line); border-radius:5px; background:#fff;
}
.msa-controls .msa-cursor, .msa-controls .msa-hover {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  background:#fff; border:1px solid var(--line); border-radius:5px;
  padding:3px 8px; color:var(--muted);
}
.msa-grid-wrap {
  overflow-x:auto; border:1px solid var(--line); border-radius:8px;
  background:#fff;
}
.msa-grid {
  display:flex; flex-direction:column;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  line-height:1.55;
  min-width:max-content;
}
.msa-row {
  display:grid; grid-template-columns:210px 50px 1fr; align-items:center;
  font-size:13px; border-bottom:1px dashed #e8e4f2;
}
.msa-row:last-child { border-bottom:none; }
.msa-row-ruler {
  background:#fbf9fe; color:var(--muted); font-size:11px;
  border-bottom:1px solid var(--line); border-top:1px solid var(--line);
}
.msa-row-query {
  background:#f3eefa; font-weight:600;
}
.msa-row-cons {
  background:#fbf9fe; border-top:1px solid var(--line);
}
.msa-label {
  padding:3px 10px; color:var(--ink); overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap;
}
.msa-row-ruler .msa-label, .msa-row-cons .msa-label { color:var(--muted); }
.msa-identity {
  text-align:right; padding:3px 10px 3px 4px; color:var(--muted);
  font-size:12px;
}
.msa-row-query .msa-identity { color:var(--accent); }
.msa-seq { white-space:nowrap; padding:0; display:inline-block; }
.msa-cell {
  display:inline-block; width:var(--msa-cell-w, 11px);
  text-align:center; box-sizing:border-box;
}
.msa-residue {
  cursor:default; border-radius:1px; transition:transform 0.08s;
}
.msa-residue:hover { outline:2px solid #6b1ea8; outline-offset:-2px; z-index:5; position:relative; }
.msa-ruler-cell {
  font-size:10px; color:var(--muted);
  border-bottom:1px solid var(--line);
}
.msa-cons-cell {
  height:24px; position:relative; vertical-align:bottom;
  background:linear-gradient(to top,
    #6b1ea8 0%, #6b1ea8 calc(var(--cons, 0) * 100%),
    transparent calc(var(--cons, 0) * 100%), transparent 100%);
  opacity:0.85;
}
.msa-cons-cell[style*="--cons:1"] { background:linear-gradient(to top, #6b1ea8 0%, #6b1ea8 100%); }

/* Scheme: Clustal-X palette — pastel rainbow with deliberately wide hue
   separation. Closest-chemistry pairs (N/Q, K/R, D/E, L/I, F/W) are
   pushed at least 40° apart on the hue wheel so adjacent cells in the
   alignment grid never blur together. Saturation hovers around 60-70%
   on L ≈ 84-87% — pastel enough to read as a soft palette, but with
   enough chroma to actually be distinguishable. */
/* Aliphatic hydrophobic — sky / cyan / lavender ramp */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="A"] { background:#d0e8fa; color:#0c4a63; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="V"] { background:#b8dfee; color:#0c4a63; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="L"] { background:#d8d0f2; color:#3b1d8a; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="I"] { background:#c4d0f0; color:#3b1d8a; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="M"] { background:#b5e5d4; color:#15524b; }
/* Aromatic hydrophobic — violet / deep violet (further from L/I lavender) */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="F"] { background:#e0c8f2; color:#2d046e; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="W"] { background:#c8b0ec; color:#2d046e; }
/* Aromatic with rings — deeper sky / teal-aqua */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="H"] { background:#a5d4ec; color:#0c4a63; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="Y"] { background:#a8d8d2; color:#15524b; }
/* Polar — N forced into clear mint green, Q into yellow (≈80° hue jump
   so they read as different colours, not "two shades of green"). S/T
   bridge into yellow-green and sage tan. */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="N"] { background:#a8eab0; color:#1f5132; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="Q"] { background:#eee898; color:#5e4a1a; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="S"] { background:#cce8a5; color:#3f5e12; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="T"] { background:#ead8a0; color:#5e4a1a; }
/* Special — peach / coral / warm yellow */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="C"] { background:#fcd0b0; color:#7c2d12; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="G"] { background:#fcc0a8; color:#7c2d12; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="P"] { background:#fde2a0; color:#92400e; }
/* Basic — pink vs coral (clear hue separation) */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="K"] { background:#f5c0d8; color:#831843; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="R"] { background:#f5c2c0; color:#7f1d1d; }
/* Acidic — orchid vs mauve (≈30° apart) */
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="D"] { background:#efb8e5; color:#76145e; }
.msa-section[data-scheme="clustal"] .msa-residue[data-aa="E"] { background:#d8b8eb; color:#5b21b6; }

/* Scheme: Hydrophobic — only colour AVLIMFWY (sky soft) */
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="A"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="V"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="L"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="I"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="M"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="F"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="W"],
.msa-section[data-scheme="hydrophobic"] .msa-residue[data-aa="Y"] { background:#e5f4fd; color:#0c4a63; }

/* Scheme: Charge — basic pink, acidic violet (Clustal category cues) */
.msa-section[data-scheme="charge"] .msa-residue[data-aa="K"],
.msa-section[data-scheme="charge"] .msa-residue[data-aa="R"],
.msa-section[data-scheme="charge"] .msa-residue[data-aa="H"] { background:#f4cfe6; color:#831843; }
.msa-section[data-scheme="charge"] .msa-residue[data-aa="D"],
.msa-section[data-scheme="charge"] .msa-residue[data-aa="E"] { background:#ede9fe; color:#2d046e; }

/* Scheme: Taylor — mirrors the Clustal rainbow so the two schemes share
   one palette family. */
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="A"] { background:#d0e8fa; color:#0c4a63; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="V"] { background:#b8dfee; color:#0c4a63; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="L"] { background:#d8d0f2; color:#3b1d8a; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="I"] { background:#c4d0f0; color:#3b1d8a; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="M"] { background:#b5e5d4; color:#15524b; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="F"] { background:#e0c8f2; color:#2d046e; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="W"] { background:#c8b0ec; color:#2d046e; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="H"] { background:#a5d4ec; color:#0c4a63; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="Y"] { background:#a8d8d2; color:#15524b; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="N"] { background:#a8eab0; color:#1f5132; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="Q"] { background:#eee898; color:#5e4a1a; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="S"] { background:#cce8a5; color:#3f5e12; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="T"] { background:#ead8a0; color:#5e4a1a; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="C"] { background:#fcd0b0; color:#7c2d12; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="G"] { background:#fcc0a8; color:#7c2d12; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="P"] { background:#fde2a0; color:#92400e; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="K"] { background:#f5c0d8; color:#831843; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="R"] { background:#f5c2c0; color:#7f1d1d; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="D"] { background:#efb8e5; color:#76145e; }
.msa-section[data-scheme="taylor"] .msa-residue[data-aa="E"] { background:#d8b8eb; color:#5b21b6; }

/* Gaps and ruler cells never get a colour */
.msa-residue[data-aa="-"], .msa-residue[data-aa="."] { color:#a09bbf; background:transparent !important; }
.msa-residue[data-aa=""], .msa-residue[data-aa="X"] { background:transparent !important; }

/* PER-TOOL WIDGETS (in Details section) */
.node-block {
  background:#ffffff; border:1px solid var(--line); border-radius:12px;
  padding:20px 22px; margin:14px 0;
  box-shadow:var(--shadow-tile);
}
.node-block-head {
  display:flex; justify-content:space-between; align-items:center;
  gap:12px; margin-bottom:14px; padding-bottom:12px;
  border-bottom:1px solid var(--line);
}
.node-block-id {
  font-weight:600; font-size:15.5px; color:var(--ink);
  margin-right:10px;
}
.node-block-tool {
  font-family:ui-monospace, monospace; font-size:12.5px;
  color:var(--accent); background:var(--accent-soft);
  padding:3px 9px; border-radius:5px; border:1px solid var(--line);
}
.node-block-state {
  font-size:12.5px; font-weight:600; padding:4px 11px;
  border-radius:var(--radius-pill); border:1px solid var(--line); background:var(--panel);
}
.node-block-state.state-completed, .node-block-state.state-cached {
  border-color:var(--good-line); background:var(--good-soft); color:var(--good);
}
.node-block-state.state-failed { border-color:var(--poor-line); background:var(--poor-soft); color:var(--poor); }

.widget {
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin:0 0 12px;
  box-shadow:var(--shadow-tile);
}
.widget:last-child { margin-bottom:0; }

/* Per-tool artifacts */
.node-block-artifacts {
  display:flex; flex-direction:column; gap:6px; margin:12px 0;
}
.artifact-section-label {
  font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:0.05em; font-weight:600;
}
.artifact-row { display:flex; flex-wrap:wrap; gap:8px; }
.artifact-link {
  display:flex; flex-direction:column; gap:3px;
  padding:9px 13px; background:#ffffff; border:1px solid var(--line);
  border-radius:7px; text-decoration:none;
  color:var(--accent); font-size:13.5px;
  box-shadow:var(--shadow-tile);
  transition:background 0.15s, border-color 0.15s, box-shadow 0.15s;
}
.artifact-link:hover { background:var(--accent-soft); border-color:var(--accent); }
.dl-disabled, .artifact-link.dl-disabled {
  color:var(--muted); background:#ffffff; border-style:dashed;
  cursor:not-allowed;
}
.dl-disabled:hover { background:#ffffff; border-color:var(--line); color:var(--muted); }
.artifact-name { font-weight:500; }
.artifact-meta { color:var(--muted); font-size:12px; }

/* Per-residue strips (structure_analysis + DSSP widgets). One white card
   per chain; all rows live in a single horizontally-scrolling container so
   the SS / burial / liability / ruler tracks always stay aligned. */
.strip-block { margin:16px 0 18px; background:#ffffff; border:1px solid var(--line); border-radius:10px; padding:12px 16px; box-shadow:var(--shadow-tile); }
.strip-head { font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:600; margin-bottom:9px; }
.strip-scroll { overflow-x:auto; padding-bottom:7px; }
.strip-scroll::-webkit-scrollbar { height:11px; }
.strip-scroll::-webkit-scrollbar-track { background:#e7e3f3; border-radius:999px; }
.strip-scroll::-webkit-scrollbar-thumb {
  background:#a99cd6; border:2px solid #e7e3f3; border-radius:999px;
}
.strip-scroll::-webkit-scrollbar-thumb:hover { background:#8574c0; }
.strip-row { display:flex; align-items:center; gap:9px; padding:2px 0; width:max-content; }
.strip-label {
  position:sticky; left:0; z-index:1; flex-shrink:0; width:84px;
  background:#ffffff; color:var(--muted); font-size:12.5px;
  text-transform:uppercase; letter-spacing:0.05em; font-weight:600; text-align:right;
}
.strip-cells { display:flex; flex-wrap:nowrap; background:#fbf9fe; border:1px solid var(--line); border-radius:5px; padding:1px; }
.strip-cell { display:inline-block; width:6px; min-width:6px; height:14px; }
.strip-cell + .strip-cell { margin-left:1px; }
.strip-row.strip-ruler .strip-cells { background:transparent; border:1px solid transparent; padding:1px; }
.strip-ruler-cell { font-size:9px; color:var(--muted); height:11px; line-height:11px; text-align:center; }
.strip-empty { background:transparent !important; }
/* Liability flag — brand orange so it grabs attention as a warning cue
   that's distinct from the pink "critical/poor" signal. */
.strip-liab { background:#f39e3a; box-shadow: 0 0 0 1px rgba(243,158,58,0.45); }
/* DSSP secondary structure — helix uses the brand magenta-pink, sheet
   uses the brand sky, loop uses lavender grey. The warm/cool split keeps
   the conventional "helix warm, sheet cool" cue while staying on-brand. */
.ss-helix { background:#cc0980; }
.ss-sheet { background:#4cc4e4; }
.ss-loop  { background:#d4cee5; }
/* Surface burial — deep brand purple → mid violet → light lavender ramp. */
.burial-buried        { background:#2d046e; }
.burial-intermediate  { background:#a78bfa; }
.burial-exposed       { background:#ede9fe; border:1px solid #d4cee5; }
.burial-na            { background:#fbfbfb; border:1px dashed #d4cee5; }
.strip-legend { display:flex; flex-wrap:wrap; gap:10px; margin-top:8px; font-size:11px; color:var(--muted); align-items:center; }
.legend-pip { display:inline-block; width:12px; height:12px; border-radius:2px; margin-right:5px; vertical-align:middle; }

/* SS / burial chip in the per-residue table */
.ss-chip { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px; color:#fff; }
.ss-chip.ss-loop, .ss-chip.burial-exposed, .ss-chip.burial-na { color:#333; }
/* Retrieved-sequences widget (go_sequences / interpro_sequences).
   Sequences render MSA-style: one pinned-label row each, fixed-pitch
   residue track, a column ruler, and a single horizontal scrollbar. */
.seqret-summary {
  display:flex; flex-wrap:wrap; align-items:center;
  gap:8px 9px; margin:11px 0 14px;
}
.seqret-query-id {
  font-family:ui-monospace, monospace; font-size:12.5px; font-weight:700;
  color:var(--accent); background:var(--accent-soft);
  padding:3px 10px; border-radius:999px;
}
.seqret-query-label { font-size:12px; color:var(--muted); margin-right:4px; }
.seqret-stats { display:flex; flex-wrap:wrap; gap:8px 9px; }
.seqret-stat {
  font-size:11.5px; color:var(--muted);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:999px; padding:3px 11px; white-space:nowrap;
}
.seqret-stat b {
  color:var(--ink); font-weight:700; font-variant-numeric:tabular-nums;
}
.seqret-align-head {
  font-size:11.5px; color:var(--subtle); margin:2px 0 7px;
  text-transform:uppercase; letter-spacing:0.04em; font-weight:600;
}
.seqret-align-scroll {
  overflow-x:auto; padding-bottom:8px;
  background:#fbfafe; border:1px solid var(--line); border-radius:9px;
}
.seqret-align-scroll::-webkit-scrollbar { height:12px; }
.seqret-align-scroll::-webkit-scrollbar-track { background:#e7e3f3; border-radius:999px; }
.seqret-align-scroll::-webkit-scrollbar-thumb {
  background:#a99cd6; border:2px solid #e7e3f3; border-radius:999px;
}
.seqret-align-scroll::-webkit-scrollbar-thumb:hover { background:#8574c0; }
.seqret-row { display:flex; align-items:stretch; width:max-content; }
.seqret-row + .seqret-row { border-top:1px solid #f0eef8; }
.seqret-row-label {
  position:sticky; left:0; z-index:1; flex-shrink:0;
  display:flex; align-items:center; gap:6px;
  width:138px; padding:5px 10px;
  background:#ffffff; border-right:1px solid var(--line);
}
.seqret-rev {
  flex-shrink:0; width:6px; height:6px; border-radius:50%;
  background:var(--good);
}
.seqret-row-acc {
  font-family:ui-monospace, monospace; font-size:11.5px;
  font-weight:700; color:var(--ink);
}
.seqret-row-len {
  margin-left:auto; font-size:10px; color:var(--muted);
  font-variant-numeric:tabular-nums;
}
.seqret-row-seq {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; line-height:2; letter-spacing:1.6px;
  white-space:pre; color:var(--ink); padding:0 10px;
}
.seqret-row-ruler .seqret-row-seq { color:var(--subtle); }
.seqret-row-ruler .seqret-row-label { background:#fbfafe; }
/* Cross-sequence overlap — a >=10-residue stretch identical between two
   retrieved sequences, i.e. a region plausibly under shared functional
   constraint. Internal repeats are intentionally not highlighted. */
.seqret-hl {
  background:#fbe98a; border-radius:2px;
  box-shadow:0 0 0 1px #fbe98a;
}
.seqret-legend {
  display:flex; flex-wrap:wrap; gap:6px 16px; margin:8px 0 0;
  font-size:11px; color:var(--muted);
}
.seqret-legend-pip {
  display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:5px; vertical-align:-1px;
}
/* Literature-search widget */
.lit-summary {
  display:flex; flex-direction:column; gap:9px;
  margin:11px 0 14px; padding:12px 14px;
  background:#fbfafe; border:1px solid var(--line); border-radius:10px;
}
.lit-query { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px; }
.lit-query-label {
  font-size:10.5px; text-transform:uppercase; letter-spacing:0.06em;
  font-weight:700; color:var(--subtle);
}
.lit-query-text {
  font-family:ui-monospace, monospace; font-size:12.5px; color:var(--ink);
  background:#ffffff; border:1px solid var(--line); border-radius:6px; padding:3px 9px;
}
.lit-stats { display:flex; flex-wrap:wrap; gap:6px 9px; }
.lit-stat {
  font-size:11.5px; color:var(--muted);
  background:#ffffff; border:1px solid var(--line);
  border-radius:999px; padding:3px 11px;
}
.lit-stat b { color:var(--ink); font-weight:700; }
.lit-list { display:flex; flex-direction:column; gap:9px; }
.lit-card {
  display:flex; gap:12px;
  border:1px solid var(--line); border-radius:10px;
  background:#ffffff; box-shadow:var(--shadow-tile); padding:13px 15px;
}
.lit-card-num {
  flex-shrink:0; width:24px; height:24px; border-radius:7px;
  display:flex; align-items:center; justify-content:center;
  background:var(--accent-soft); color:var(--accent);
  font-size:12px; font-weight:700; font-variant-numeric:tabular-nums;
}
.lit-card-body { flex:1; min-width:0; }
.lit-card-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:8px; }
.lit-title {
  font-size:14px; font-weight:600; color:var(--ink);
  line-height:1.4; text-decoration:none; letter-spacing:-0.005em;
}
a.lit-title:hover { color:var(--accent); text-decoration:underline; }
.lit-badge {
  font-size:9.5px; text-transform:uppercase; letter-spacing:0.05em;
  font-weight:700; border-radius:999px; padding:1px 8px;
}
.lit-oa { color:var(--good); background:var(--good-soft); border:1px solid var(--good-line); }
.lit-authors {
  font-size:11.5px; color:var(--muted); margin-top:4px;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.lit-cite { font-size:11.5px; color:var(--subtle); margin-top:3px; }
.lit-journal { color:var(--muted); font-style:italic; }
.lit-snippet {
  font-size:12px; color:var(--muted); line-height:1.5; margin:7px 0 0;
  display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;
}
.lit-ids { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.lit-id {
  font-size:10.5px; font-weight:600; color:var(--accent);
  background:var(--accent-soft); border-radius:6px; padding:2px 8px; text-decoration:none;
}
.lit-id:hover { background:var(--accent); color:#ffffff; }
/* PXMeter widget — accuracy sections (complex / per chain / per interface). */
.px-section { margin:16px 0 0; }
.px-section-head {
  display:flex; flex-wrap:wrap; align-items:baseline; gap:9px; margin:0 0 9px;
}
.px-section-title {
  margin:0; font-size:12px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.06em; color:var(--ink);
}
.px-section-sub { font-size:11.5px; color:var(--muted); }
.px-context { margin:8px 0 0; font-size:11.5px; color:var(--muted); }
.residue-annotations { background:#ffffff; border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:14px 0; box-shadow:var(--shadow-tile); }
.residue-annotations summary { color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; }
.residue-annotations summary:hover { color:var(--accent); }
.residue-annotations[open] { background:var(--panel); }
.residue-annotations .table-wrap { margin-top:8px; max-height:340px; }
.residue-annotations-table { font-size:12px; }
.residue-annotations-table th { background:#f5f1fa; }

/* Key-value table used by per-tool widgets */
.kv-table { width:auto; border-collapse:collapse; font-size:13px; margin:6px 0; }
.kv-table th {
  text-align:left; padding:5px 14px 5px 0;
  color:var(--muted); font-weight:500; vertical-align:top;
  font-size:12px; text-transform:none; letter-spacing:0; background:transparent;
  border-bottom:1px solid transparent;
  position:static; cursor:default;
}
.kv-table td { padding:5px 0; vertical-align:top; border-bottom:1px solid transparent; }

/* QC widget cards. ``--cols`` is injected by the Python helper
   :func:`_card_grid` so 5 cards become 3x2 (3+2) instead of 4x2 with a
   single orphan; 7 become 4+3, 9 become 3x3, etc. ``minmax(0, 1fr)`` lets
   cards shrink to whatever fits when --cols is forced, so the grid never
   overflows on narrower main columns. Legacy paths without --cols fall
   back to a responsive auto-fit with a 220px floor. */
.qc-card-grid {
  display:grid;
  grid-template-columns: repeat(var(--cols, auto-fit), minmax(0, 1fr));
  gap:10px; margin:12px 0;
}
.qc-card-grid:not([style*="--cols"]) {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}
/* QC / metric card — donut-centric layout.
   has-gauge variant: 88px segmented donut on the left, condensed label
   inside the ring, full name + plain-English explainer on the right. The
   ring colour already encodes quality, so no extra accent strip is needed.
   Plain variant (no gauge data): kept for tools that emit non-numeric metrics. */
.qc-card {
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px; position:relative; overflow:hidden;
  box-shadow:var(--shadow-tile);
  transition:border-color 0.15s, box-shadow 0.15s;
}
.qc-card:hover { border-color:var(--line-strong); box-shadow:var(--shadow-card); }
.qc-card.has-gauge {
  display:flex; align-items:center; gap:20px;
  padding:20px 22px;
}
.qc-card-viz {
  flex-shrink:0; width:88px; height:88px;
  display:flex; align-items:center; justify-content:center;
}
.qc-card-body { min-width:0; flex:1; }
/* protein_qc gauge cards stack vertically — donut centred on top, text below. */
.widget-qc .qc-card.has-gauge {
  flex-direction:column; align-items:center; text-align:center; gap:12px;
}
.widget-qc .qc-card.has-gauge .qc-card-body { flex:0 1 auto; }
/* Plain (no-gauge) variant — keep the small accent strip for state cue. */
.qc-card:not(.has-gauge)::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width:3px;
  background:var(--line);
}
.qc-card:not(.has-gauge).metric-good::before { background:var(--good); }
.qc-card:not(.has-gauge).metric-ok::before   { background:var(--ok); }
.qc-card:not(.has-gauge).metric-poor::before { background:var(--poor); }
.qc-card:not(.has-gauge).metric-neutral::before { background:var(--accent); }
.qc-card header {
  display:flex; align-items:center; gap:12px; margin-bottom:6px;
  justify-content:space-between;
}
.qc-card-figure { display:flex; align-items:baseline; gap:10px; min-width:0; }
.qc-card header strong {
  font-size:22px; line-height:1; color:var(--ink);
  font-variant-numeric:tabular-nums; letter-spacing:-0.01em;
}
/* The full metric name on the right side of has-gauge cards: subtle but
   readable, sits above the explainer text. */
.qc-card.has-gauge .qc-card-name {
  color:var(--ink); font-size:13px; font-weight:600;
  letter-spacing:0; text-transform:none; margin-bottom:4px;
}
.qc-card-name {
  color:var(--muted); font-size:11px; text-transform:uppercase;
  letter-spacing:0.06em; font-weight:600;
}
.qc-card-explain {
  margin:6px 0 0; color:var(--muted); font-size:12px; line-height:1.5;
}
.qc-card.has-gauge .qc-card-explain { margin:0; }
.metric-gauge { flex-shrink:0; }
.gauge-value {
  font-family: Inter, system-ui, sans-serif;
  font-weight:700; fill:var(--ink); letter-spacing:-0.02em;
  font-variant-numeric:tabular-nums;
}
.gauge-sub {
  font-family: Inter, system-ui, sans-serif;
  font-weight:600; fill:#a09bbf; letter-spacing:0.08em;
  text-transform:uppercase;
}

/* Output inventory — fallback for tools with no scalar metrics. Each row is
   a typed entry (sequence / string / list / object) with a label, a
   one-line meta description, and an optional preview. The layout
   intentionally echoes the donut card grid so the report stays visually
   consistent across tool kinds. */
.widget-shape .output-shape-list {
  list-style:none; padding:0; margin:0;
  display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));
  gap:10px;
}
.output-shape-row {
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px; display:flex; gap:14px; align-items:flex-start;
  box-shadow:var(--shadow-tile);
  transition:border-color 0.15s, box-shadow 0.15s;
}
.output-shape-row:hover { border-color:var(--line-strong); box-shadow:var(--shadow-card); }
.output-shape-icon {
  flex-shrink:0; width:38px; height:38px; border-radius:9px;
  background:var(--accent-soft); border:1px solid var(--line);
  display:flex; align-items:center; justify-content:center;
  font-family:ui-monospace, monospace; font-weight:700;
  color:var(--accent); font-size:15px;
}
.output-shape-body { min-width:0; flex:1; }
.output-shape-name {
  color:var(--ink); font-weight:600; font-size:14.5px; margin-bottom:3px;
}
.output-shape-meta {
  color:var(--muted); font-size:13px; letter-spacing:0.02em;
  margin-bottom:7px;
}
.output-shape-preview {
  display:block; background:#ffffff; border:1px solid var(--line);
  padding:7px 10px; border-radius:7px; font-size:13px;
  color:var(--ink); overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; max-width:100%;
}

.qc-flag-list { display:flex; flex-direction:column; gap:10px; margin-top:16px; }
.qc-flag {
  background:var(--ok-soft); border:1px solid var(--ok-line);
  padding:12px 16px; border-radius:10px; cursor:help;
  box-shadow:var(--shadow-tile);
}
.qc-flag strong { display:block; font-size:14px; color:var(--warn); margin-bottom:4px; }
.qc-flag-explain { font-size:13px; color:#92400e; line-height:1.5; }

/* sequence_edit — wild-type vs mutant comparison. A summary row, colour-
   coded mutation chips, then a two-row MSA-style alignment with a column
   ruler and a single horizontal scrollbar; changed columns highlighted. */
.widget-seqedit .seqedit-summary {
  display:flex; flex-wrap:wrap; align-items:center;
  gap:8px 9px; margin:11px 0 13px;
}
.seqedit-op {
  font-size:11px; font-weight:700; color:var(--accent);
  background:var(--accent-soft); border-radius:999px; padding:3px 11px;
  text-transform:uppercase; letter-spacing:0.04em;
}
.seqedit-stats { display:flex; flex-wrap:wrap; gap:8px 9px; }
.seqedit-stat {
  font-size:11.5px; color:var(--muted);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:999px; padding:3px 11px; white-space:nowrap;
}
.seqedit-stat b {
  color:var(--ink); font-weight:700; font-variant-numeric:tabular-nums;
}
.seqedit-align-head {
  font-size:11.5px; color:var(--subtle); margin:13px 0 7px;
  text-transform:uppercase; letter-spacing:0.04em; font-weight:600;
}
.seqedit-muts { display:flex; flex-wrap:wrap; gap:7px; margin-bottom:3px; }
.seqedit-mut-chip {
  display:inline-flex; align-items:baseline; gap:1px;
  font-family:ui-monospace, monospace; font-size:12.5px; font-weight:700;
  background:#ffffff; border:1px solid var(--line); border-radius:7px;
  padding:3px 10px;
}
.seqedit-mut-chip .sm-from { color:var(--muted); }
.seqedit-mut-chip .sm-pos {
  color:var(--subtle); font-weight:600; padding:0 2px;
  font-variant-numeric:tabular-nums;
}
.seqedit-mut-chip .sm-to { color:var(--poor); }
.seqedit-align-scroll {
  overflow-x:auto; padding-bottom:8px;
  background:#fbfafe; border:1px solid var(--line); border-radius:9px;
}
.seqedit-align-scroll::-webkit-scrollbar { height:12px; }
.seqedit-align-scroll::-webkit-scrollbar-track { background:#e7e3f3; border-radius:999px; }
.seqedit-align-scroll::-webkit-scrollbar-thumb {
  background:#a99cd6; border:2px solid #e7e3f3; border-radius:999px;
}
.seqedit-align-scroll::-webkit-scrollbar-thumb:hover { background:#8574c0; }
.seqedit-row { display:flex; align-items:stretch; width:max-content; }
.seqedit-row + .seqedit-row { border-top:1px solid #f0eef8; }
.seqedit-row-label {
  position:sticky; left:0; z-index:1; flex-shrink:0;
  display:flex; align-items:center;
  width:74px; padding:5px 10px;
  font-size:10.5px; font-weight:700; text-transform:uppercase;
  letter-spacing:0.05em; color:var(--muted);
  background:#ffffff; border-right:1px solid var(--line);
}
.seqedit-row-mut .seqedit-row-label { color:var(--accent); }
.seqedit-row-ruler .seqedit-row-label { background:#fbfafe; }
.seqedit-row-seq {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; line-height:2; letter-spacing:1.6px;
  white-space:pre; color:var(--ink); padding:0 10px;
}
.seqedit-row-ruler .seqedit-row-seq { color:var(--subtle); }
.seqedit-diff-old {
  background:#fde2ec; color:#a3076b; border-radius:2px;
  box-shadow:0 0 0 1px #f6b9d3;
}
.seqedit-diff-new {
  background:var(--poor); color:#ffffff; border-radius:2px; font-weight:700;
  box-shadow:0 0 0 1px var(--poor);
}
.seqedit-legend {
  display:flex; flex-wrap:wrap; gap:6px 16px; margin:8px 0 0;
  font-size:11px; color:var(--muted);
}
.seqedit-legend-pip {
  display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:5px; vertical-align:-1px;
}
.seqedit-warn {
  margin-top:10px; padding:9px 13px; font-size:12px;
  background:var(--ok-soft); border:1px solid var(--ok-line);
  border-radius:8px; color:#92400e;
}

/* AutoDock Vina docking widget — ranked binding poses, the Vina score-term
   breakdown and a profile of the docked ligand parsed from the .pdbqt. */
.vina-poses { margin:4px 0 2px; }
.vina-pose {
  display:grid; grid-template-columns:48px 1fr 118px 152px;
  align-items:center; gap:13px; padding:8px 2px;
}
.vina-pose + .vina-pose { border-top:1px solid #f0eef8; }
.vina-pose-head {
  padding:4px 2px 6px; font-size:10px; text-transform:uppercase;
  letter-spacing:0.05em; color:var(--subtle); font-weight:700;
  border-bottom:1px solid var(--line);
}
.vina-pose-rank {
  font-family:ui-monospace, monospace; font-weight:700; font-size:12.5px;
  color:var(--accent);
}
.vina-pose-head .vina-pose-rank { color:var(--subtle); }
.vina-pose-bar-track {
  height:14px; background:#ede9fe; border-radius:999px; overflow:hidden;
}
.vina-pose-bar { display:block; height:100%; border-radius:999px; }
.vina-pose-aff {
  font-family:ui-monospace, monospace; font-weight:700; font-size:13px;
  text-align:right; font-variant-numeric:tabular-nums;
}
.vina-pose-aff small { font-weight:500; color:var(--muted); font-size:9.5px; }
.vina-pose-head .vina-pose-aff { color:var(--subtle); }
.vina-pose-rmsd {
  font-size:11px; color:var(--muted); text-align:right;
  font-variant-numeric:tabular-nums;
}
.vina-pose-rmsd b { color:var(--ink); font-weight:600; }

.vina-energy {
  margin:4px 0 2px; padding:13px 15px;
  background:#fbfafe; border:1px solid var(--line); border-radius:10px;
}
.vina-ed-row {
  display:grid; grid-template-columns:142px 1fr 76px;
  align-items:center; gap:12px; padding:5px 0;
}
.vina-ed-label { font-size:12px; color:var(--ink); cursor:help; }
.vina-ed-track {
  position:relative; height:16px; background:#ede9fe; border-radius:5px;
}
.vina-ed-axis {
  position:absolute; left:50%; top:-2px; bottom:-2px; width:1.5px;
  background:var(--subtle); transform:translateX(-50%);
}
.vina-ed-fill { position:absolute; top:0; bottom:0; border-radius:3px; }
.vina-ed-neg { background:#4cc4e4; }
.vina-ed-pos { background:#f39e3a; }
.vina-ed-val {
  font-family:ui-monospace, monospace; font-size:12px; font-weight:700;
  text-align:right; color:var(--ink); font-variant-numeric:tabular-nums;
}
.vina-ed-result {
  display:flex; justify-content:space-between; align-items:baseline;
  margin-top:8px; padding-top:9px; border-top:1px solid var(--line);
}
.vina-ed-result-label {
  font-size:11px; text-transform:uppercase; letter-spacing:0.05em;
  font-weight:700; color:var(--subtle);
}
.vina-ed-result-val {
  font-family:ui-monospace, monospace; font-size:16px; font-weight:700;
  color:var(--accent);
}
.vina-ed-result-val small { font-size:10px; color:var(--muted); font-weight:500; }
.vina-caption { font-size:11px; color:var(--muted); margin:7px 2px 0; line-height:1.5; }

.vina-desc-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(122px, 1fr));
  gap:9px; margin:4px 0 2px;
}
.vina-desc {
  display:flex; flex-direction:column; gap:3px; padding:10px 12px;
  background:#fbfafe; border:1px solid var(--line); border-radius:9px;
}
.vina-desc-val {
  font-size:16px; font-weight:700; color:var(--ink);
  font-variant-numeric:tabular-nums;
}
.vina-desc-val sub { font-size:11px; font-weight:600; }
.vina-desc-label {
  font-size:10.5px; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.03em;
}
.vina-el-head {
  font-size:11.5px; color:var(--subtle); margin:13px 0 6px;
  text-transform:uppercase; letter-spacing:0.04em; font-weight:600;
}
.vina-el-bar {
  display:flex; height:16px; border-radius:5px; overflow:hidden;
  border:1px solid var(--line);
}
.vina-el-seg { display:block; height:100%; }
.vina-el-legend {
  display:flex; flex-wrap:wrap; gap:5px 15px; margin-top:8px;
  font-size:11.5px; color:var(--muted);
}
.vina-el-legend-item { display:inline-flex; align-items:center; }
.vina-el-pip {
  display:inline-block; width:10px; height:10px; border-radius:3px;
  margin-right:6px;
}
.vina-el-legend-item b {
  color:var(--ink); font-weight:700; margin-left:3px;
  font-variant-numeric:tabular-nums;
}

/* template_select widget — ranked template cards (identity / coverage bars
   + E-value chips) and an MSA-style template-sequence comparison strip. */
.tmpl-list { display:flex; flex-direction:column; gap:11px; margin:4px 0 6px; }
.tmpl-card {
  background:#fbfafe; border:1px solid var(--line); border-radius:10px;
  padding:13px 15px;
}
.tmpl-card-head {
  display:flex; flex-wrap:wrap; align-items:baseline; gap:8px;
  margin-bottom:10px;
}
.tmpl-rank {
  font-family:ui-monospace, monospace; font-weight:700; font-size:12px;
  color:var(--accent);
}
.tmpl-id {
  font-family:ui-monospace, monospace; font-weight:700; font-size:13.5px;
  color:var(--ink);
}
.tmpl-desc { font-size:12px; color:var(--muted); }
.tmpl-badges { margin-left:auto; display:flex; flex-wrap:wrap; gap:6px; }
.tmpl-badge {
  font-size:10.5px; color:var(--muted);
  background:#ffffff; border:1px solid var(--line);
  border-radius:999px; padding:2px 9px;
}
.tmpl-metrics { display:flex; flex-direction:column; gap:6px; }
.tmpl-bar-row { display:flex; align-items:center; gap:11px; }
.tmpl-bar-label {
  flex-shrink:0; width:66px; font-size:11.5px; color:var(--ink); font-weight:500;
}
.tmpl-bar-track {
  flex:1; height:9px; background:#ede9fe; border-radius:999px; overflow:hidden;
}
.tmpl-bar-track i { display:block; height:100%; border-radius:999px; }
.tmpl-bar-val {
  flex-shrink:0; width:52px; text-align:right; font-size:11.5px;
  font-weight:700; color:var(--ink); font-variant-numeric:tabular-nums;
}
.tmpl-chips { display:flex; flex-wrap:wrap; gap:7px; margin-top:8px; }
.tmpl-chip {
  font-size:11px; color:var(--muted);
  background:#ffffff; border:1px solid var(--line);
  border-radius:999px; padding:3px 10px;
}
.tmpl-chip b {
  color:var(--ink); font-weight:700; font-variant-numeric:tabular-nums;
  font-family:ui-monospace, monospace; margin-left:2px;
}
.tmpl-seq-scroll {
  overflow-x:auto; padding-bottom:8px; margin-top:2px;
  background:#fbfafe; border:1px solid var(--line); border-radius:9px;
}
.tmpl-seq-scroll::-webkit-scrollbar { height:12px; }
.tmpl-seq-scroll::-webkit-scrollbar-track { background:#e7e3f3; border-radius:999px; }
.tmpl-seq-scroll::-webkit-scrollbar-thumb {
  background:#a99cd6; border:2px solid #e7e3f3; border-radius:999px;
}
.tmpl-seq-scroll::-webkit-scrollbar-thumb:hover { background:#8574c0; }
.tmpl-seq-row { display:flex; align-items:stretch; width:max-content; }
.tmpl-seq-row + .tmpl-seq-row { border-top:1px solid #f0eef8; }
.tmpl-seq-label {
  position:sticky; left:0; z-index:1; flex-shrink:0;
  display:flex; align-items:center; gap:8px;
  width:200px; padding:6px 11px;
  background:#ffffff; border-right:1px solid var(--line);
}
.tmpl-seq-name {
  flex:1; min-width:0; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap;
  font-family:ui-monospace, monospace; font-size:11.5px; font-weight:700;
  color:var(--ink);
}
.tmpl-seq-id {
  flex-shrink:0; white-space:nowrap;
  font-size:10px; font-weight:700;
  color:var(--accent); background:var(--accent-soft);
  border-radius:999px; padding:1px 7px;
}
.tmpl-seq-track {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; line-height:2; letter-spacing:1.6px;
  white-space:pre; color:var(--ink); padding:0 11px;
}
.tmpl-seq-ruler .tmpl-seq-track { color:var(--subtle); }
.tmpl-seq-ruler .tmpl-seq-label { background:#fbfafe; }
/* MPNN-widget conservation highlight: positions where every design
   agrees on the same residue. A single calm brand-purple tint is used
   for ALL conserved cells regardless of amino acid — a multi-colour
   per-AA palette reads as a noisy mosaic when most positions are
   conserved (which is typical for MPNN designs against a fixed
   backbone). Only the visual weight + a soft background + rounded
   corners are added — no width / padding / margin / border, so each
   cell stays exactly one monospace slot wide and keeps column
   alignment with the ruler below. */
.mpnn-cell-conserved {
  font-weight:700;
  color:var(--accent);
  background:var(--accent-soft);
  border-radius:2px;
}
.mpnn-cell-variable {
  color:var(--subtle); opacity:0.75;
}
/* Sub-explainer below the "Designed sequences" subhead — same muted
   tone as widget-explainer but a touch tighter so it doesn't compete
   with the section heading. */
.widget-subexplain {
  margin:2px 0 8px; color:var(--muted); font-size:12px; line-height:1.5;
}

/* Generative-design / LigandMPNN / ESM widgets — summary chips, per-design
   results table cells and the ESM pooled-embedding fingerprint strip. */
.dsn-chips { display:flex; flex-wrap:wrap; gap:7px; margin:4px 0 12px; }
.dsn-chip {
  font-size:11.5px; color:var(--muted);
  background:#fbfafe; border:1px solid var(--line);
  border-radius:999px; padding:3px 11px;
}
.dsn-chip b {
  color:var(--ink); font-weight:700; margin-left:3px;
  font-variant-numeric:tabular-nums;
}
.dsn-chip-warn {
  color:var(--ok); background:var(--ok-soft); border-color:var(--ok-line);
  font-weight:700;
}
.dsn-seq {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11.5px; color:var(--ink); background:transparent;
  word-break:break-all;
}
.dsn-muted { color:var(--subtle); font-size:12px; font-style:italic; }
.dsn-file {
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:11px; color:var(--muted); word-break:break-all;
}
.dsn-bar-cell { display:flex; align-items:center; gap:8px; }
.dsn-bar {
  flex:1; min-width:54px; max-width:130px; height:8px;
  background:#ede9fe; border-radius:999px; overflow:hidden;
}
.dsn-bar i { display:block; height:100%; border-radius:999px; background:var(--accent); }
.dsn-bar i.metric-good { background:var(--good); }
.dsn-bar i.metric-ok { background:var(--ok); }
.dsn-bar i.metric-poor { background:var(--poor); }
.dsn-bar-cell span {
  font-size:11.5px; font-weight:700; color:var(--ink);
  font-variant-numeric:tabular-nums;
}
.esm-emb {
  margin:2px 0 4px; padding:6px;
  background:#ffffff; border:1px solid var(--line); border-radius:9px;
}
.esm-emb-svg { display:block; width:100%; height:34px; border-radius:4px; }
.esm-emb-legend {
  display:flex; flex-wrap:wrap; gap:5px 16px; margin:8px 2px 0;
  font-size:11px; color:var(--muted);
}
.esm-emb-legend span { display:inline-flex; align-items:center; }
.esm-emb-pip {
  display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:6px; border:1px solid var(--line);
}

/* Amino-acid distribution — single-row sequence logo. Each residue type
   is a column whose one-letter glyph is scaled vertically by its
   composition frequency; glyph fills reuse the report's .aa-* palette so
   the logo colouring stays aligned with the rest of the report. */
.aa-logo {
  margin:8px 0 6px; padding:14px 14px 8px;
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  box-shadow:var(--shadow-tile); overflow-x:auto;
}
.aa-logo-svg { display:block; width:100%; height:auto; margin:0 auto; }
.aa-logo-grid { stroke:#efe9f7; stroke-width:1; }
.aa-logo-baseline { stroke:var(--line); stroke-width:1.4; }
.aa-logo-ytick {
  font-size:10px; fill:var(--muted); font-variant-numeric:tabular-nums;
}
.aa-logo-xtick {
  font-size:12px; font-weight:700; fill:var(--muted);
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
}
.aa-logo-axis-title {
  font-size:10px; fill:var(--subtle);
  text-transform:uppercase; letter-spacing:0.07em;
}
.aa-logo-glyph {
  font-family:"Helvetica Neue", Helvetica, Arial, sans-serif;
  font-weight:700;
}
.aa-logo-glyph.g-hydro   { fill:#0c4a63; }
.aa-logo-glyph.g-pos     { fill:#0c4a63; }
.aa-logo-glyph.g-neg     { fill:#831843; }
.aa-logo-glyph.g-polar   { fill:#2d046e; }
.aa-logo-glyph.g-special { fill:#7c2d12; }
.aa-logo-glyph.g-other   { fill:#2a2440; }
.aa-logo-legend {
  display:flex; flex-wrap:wrap; gap:5px 16px; margin:9px 2px 2px;
  font-size:11px; color:var(--muted);
}
.aa-logo-legend-item { display:inline-flex; align-items:center; }
.aa-logo-legend-pip {
  display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:6px;
}
.aa-logo-legend-pip.g-hydro   { background:#0c4a63; }
.aa-logo-legend-pip.g-pos     { background:#0c4a63; }
.aa-logo-legend-pip.g-neg     { background:#831843; }
.aa-logo-legend-pip.g-polar   { background:#2d046e; }
.aa-logo-legend-pip.g-special { background:#7c2d12; }
/* sequence_analysis — residue-composition bars + caution-flag chips. */
.seqana-comp { display:flex; flex-direction:column; gap:6px; margin:8px 0 4px; }
.seqana-comp-row { display:flex; align-items:center; gap:11px; }
.seqana-comp-label {
  flex-shrink:0; width:96px; font-size:12px; color:var(--ink); font-weight:500;
}
.seqana-comp-track {
  flex:1; height:9px; background:#ede9fe; border-radius:999px; overflow:hidden;
}
.seqana-comp-track i {
  display:block; height:100%; background:var(--accent-bright); border-radius:999px;
}
.seqana-comp-pct {
  flex-shrink:0; width:38px; text-align:right; font-size:11.5px;
  color:var(--muted); font-variant-numeric:tabular-nums;
}
.seqana-flag {
  display:inline-block; font-size:10px; font-weight:600;
  color:var(--ok); background:var(--ok-soft); border:1px solid var(--ok-line);
  border-radius:999px; padding:1px 7px; margin:0 3px 2px 0;
}
.seqana-muted { color:var(--subtle); }

/* Raw output JSON (collapsed by default at bottom of node block) */
.node-block-raw {
  border:1px solid var(--line); border-radius:8px;
  margin-top:14px; padding:12px 14px; background:#ffffff;
}
.node-block-raw summary {
  cursor:pointer; font-weight:500; font-size:12px; color:var(--muted);
  text-transform:uppercase; letter-spacing:0.05em;
}
.node-block-raw summary:hover { color:var(--accent); }
.node-block-raw[open] { padding-bottom:0; }
.node-block-raw pre {
  margin:10px 0 0; background:var(--panel); border:1px solid var(--line); border-radius:6px;
  padding:11px 14px; max-height:340px; font-size:12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height:1.5;
  overflow:auto;
}

/* Repro */
.cmd { display:flex; gap:8px; align-items:center; margin:8px 0; }
.cmd pre { flex:1; margin:0; padding:11px 14px; font-size:12px;
  background:#1e1b4b; color:#c7c2e8; border:1px solid #312e81;
  font-family:ui-monospace, monospace; border-radius:var(--radius-sm); overflow-x:auto; }
/* Misc */
pre {
  white-space:pre-wrap; overflow:auto;
  background:#f7f4fc; border:1px solid var(--line); border-radius:6px;
  padding:11px 14px; max-height:340px; font-size:12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; line-height:1.5;
}
details {
  border:1px solid var(--line); border-radius:8px; margin:10px 0; padding:12px 14px;
  background:#ffffff;
}
details[open] { background:var(--panel); }
summary { cursor:pointer; font-weight:600; font-size:14px; padding:2px 0; }
summary:hover { color:var(--accent); }
dl { display:grid; grid-template-columns:130px minmax(0,1fr); gap:6px 14px; margin:10px 0; font-size:13px; }
dt { color:var(--muted); font-weight:500; }
dd { margin:0; overflow-wrap:anywhere; }

/* Code helpers */
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size:12px; background:#f1edf9; padding:1px 5px; border-radius:3px;
  color:#1a1d1f;
}
.sha-short { font-family: ui-monospace, monospace; font-size:12px; background:#f1edf9; padding:1px 6px; border-radius:3px; cursor:help; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

/* deprecated: id-chips kept only for backward compat (no longer rendered) */
.id-chips { display:none; }

/* Buttons */
.copy-btn {
  font-size:11px; padding:4px 10px; border:1px solid var(--line);
  border-radius:5px; background:#fbf9fe; color:var(--accent); cursor:pointer;
  font-family:inherit;
}
.copy-btn:hover { background:var(--accent-soft); }
.copy-btn.copied { background:var(--good-soft); border-color:var(--good-line); color:var(--good); }

/* Notes */
.note { color:var(--muted); font-size:13px; }
.note.warning { color:var(--warn); }
.note.ok { color:var(--good); }
.empty {
  color:var(--muted); border:1px dashed var(--line); padding:14px 18px;
  border-radius:7px; font-size:14px;
}
.warnings p {
  border-left:4px solid var(--ok); padding:9px 12px; background:var(--ok-soft);
  margin:6px 0; border-radius:0 5px 5px 0;
  /* Warning messages routinely embed long absolute file paths from
     adapter subprocess failures. Without an explicit wrap rule the path
     stays a single unbreakable token and overflows the callout box,
     pushing the text past the card's right edge. ``overflow-wrap:
     anywhere`` lets the browser break inside the path at any point;
     ``word-break:break-word`` is the legacy alias kept for older
     WebKit. */
  overflow-wrap:anywhere; word-break:break-word;
}
.warnings p.error, p.error {
  border-left:4px solid var(--poor); background:var(--poor-soft); color:var(--poor);
}

/* Responsive */
@media (max-width: 940px) {
  body {
    grid-template-columns: 1fr;
    grid-template-rows: auto auto 1fr;
    grid-template-areas: "header" "sidebar" "main";
  }
  .sidebar { position:static; height:auto; min-height:0; }
  .side-nav { flex-direction:row; flex-wrap:wrap; gap:4px; }
  .side-nav-item { flex:0 0 auto; }
  .top-bar { padding:0 18px; }
  .content { padding:18px; }
  .top-bar-title { font-size:16px; }
  .candidate-grid { grid-template-columns:1fr; }
  .structure-section { min-height:440px; }
  .structure-viewer { min-height:360px; }
  .metric-grid { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
  .conservation { margin-left:0; max-width:none; }
  .qc-card-grid { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
}
"""


# ----------------------------------------------------------------- JS


def _js() -> str:
    # Sentinel ``VPA_OFFLINE_VIEWER`` marks this build as fully self-contained:
    # no CDN, no LLM, no network requests. Test contracts assert its presence
    # and the absence of external viewer CDN imports. The report's narrative and visuals
    # are produced deterministically from JobStore data — re-running
    # ``vpa report <id>`` yields byte-equivalent output for the same inputs.
    return """
// VPA_OFFLINE_VIEWER · deterministic offline 3D viewer · no external CDN, no LLM calls
(function () {
  // ----- Scroll-spy nav highlight -----
  const navLinks = Array.from(document.querySelectorAll('nav a'));
  const sectionsById = new Map();
  for (const link of navLinks) {
    const id = link.getAttribute('href').slice(1);
    const el = document.getElementById(id);
    if (el) sectionsById.set(id, { el, link });
  }
  if (sectionsById.size && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const id = entry.target.id;
        const item = sectionsById.get(id);
        if (!item) continue;
        if (entry.isIntersecting) {
          navLinks.forEach((a) => a.classList.remove('active'));
          item.link.classList.add('active');
        }
      }
    }, { rootMargin: '-40% 0px -55% 0px', threshold: 0 });
    for (const { el } of sectionsById.values()) observer.observe(el);
  }

  // ----- Sortable tables -----
  document.querySelectorAll('th').forEach((th, index) => {
    th.addEventListener('click', () => {
      const table = th.closest('table');
      if (!table) return;
      const body = table.querySelector('tbody');
      if (!body) return;
      const rows = Array.from(body.querySelectorAll('tr'));
      const dir = th.dataset.dir === 'asc' ? -1 : 1;
      rows.sort((a, b) => dir * a.children[index].textContent.localeCompare(
        b.children[index].textContent, undefined, { numeric: true }
      ));
      rows.forEach((row) => body.appendChild(row));
      table.querySelectorAll('th').forEach((other) => delete other.dataset.dir);
      th.dataset.dir = dir === 1 ? 'asc' : 'desc';
    });
  });

  // ----- Candidate filter -----
  const filter = document.getElementById('candidate-filter');
  if (filter) {
    filter.addEventListener('input', () => {
      const q = filter.value.toLowerCase();
      document.querySelectorAll('#candidate-table tbody tr').forEach((row) => {
        row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
      });
    });
  }

  // ----- Copy buttons -----
  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const text = btn.dataset.copy || '';
      try {
        await navigator.clipboard.writeText(text);
        const original = btn.textContent;
        btn.textContent = 'Copied';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = original;
          btn.classList.remove('copied');
        }, 1100);
      } catch (e) {
        console.warn('Copy failed:', e);
      }
    });
  });

  // ----- MSA scheme + zoom toggles -----
  document.querySelectorAll('.msa-section').forEach((section) => {
    const grid = section.querySelector('.msa-grid');
    const schemeSel = section.querySelector('.msa-scheme');
    const zoomInput = section.querySelector('.msa-zoom');
    const cursor = section.querySelector('.msa-cursor');
    const hoverEl = section.querySelector('.msa-hover');
    if (schemeSel) {
      schemeSel.addEventListener('change', () => {
        section.dataset.scheme = schemeSel.value;
      });
    }
    if (zoomInput && grid) {
      const apply = () => { grid.style.setProperty('--msa-cell-w', zoomInput.value + 'px'); };
      zoomInput.addEventListener('input', apply);
      apply();
    }
    if (cursor && hoverEl) {
      section.querySelectorAll('.msa-residue').forEach((cell) => {
        cell.addEventListener('mouseenter', () => {
          const col = cell.dataset.col || '?';
          const aa = cell.dataset.aa || '';
          const totalCols = section.querySelectorAll('.msa-row-ruler .msa-cell').length;
          cursor.textContent = `Col ${col} / ${totalCols}`;
          hoverEl.textContent = aa === '-' ? 'gap' : `residue ${aa} @ col ${col}`;
        });
        cell.addEventListener('mouseleave', () => {
          hoverEl.textContent = 'hover a residue';
        });
      });
    }
  });

  // ----- Inline 3D protein viewer (3Dmol.js, offline vendored) -----
  // Implements the spec at docs/agent_layer.md "3D 단백질 시각화":
  // createViewer → addModel → cartoon spectrum + HET stick/sphere → click
  // toggle + label + sidechain layer + fullscreen.
  const dataEl = document.getElementById('vpa-structures');
  if (!dataEl) return;
  let structurePayloads;
  try { structurePayloads = JSON.parse(dataEl.textContent); } catch (e) { return; }
  if (!Array.isArray(structurePayloads)) return;
  if (typeof $3Dmol === 'undefined') {
    // Vendor not present — degrade gracefully per spec §1 guard.
    for (const payload of structurePayloads) {
      const el = document.getElementById(payload.id);
      if (el) el.innerHTML = '<div class="structure-status">3D viewer unavailable (3Dmol.js asset missing)</div>';
    }
    return;
  }

  // Amino-acid colour palette — same Clustal scheme used by the MSA viewer
  // so a residue clicked here has the same colour as in the alignment.
  // 3-letter → 1-letter map used by the Pick-mode info panel to surface
  // a residue's FASTA-style single-letter code alongside the 3-letter
  // residue name.
  const AA3_TO_1 = { ALA:'A', ARG:'R', ASN:'N', ASP:'D', CYS:'C', GLN:'Q', GLU:'E', GLY:'G',
                     HIS:'H', ILE:'I', LEU:'L', LYS:'K', MET:'M', PHE:'F', PRO:'P', SER:'S',
                     THR:'T', TRP:'W', TYR:'Y', VAL:'V' };

  // AlphaFold pLDDT colour function for cartoon styling.
  // Accepts either 0-1 (mean confidence) or 0-100 scale.
  function plddtForResi(b) {
    const v = b > 1.5 ? b : b * 100;
    if (v >= 90) return '#0053D6';
    if (v >= 70) return '#65CBF3';
    if (v >= 50) return '#FFDB13';
    return '#FF7D45';
  }

  // Build one viewer per candidate-structure payload. Each viewer is
  // self-contained: its container, its selection map, its style applier.
  for (const payload of structurePayloads) {
    setupViewer(payload);
  }

  function setupViewer(payload) {
    const container = document.getElementById(payload.id);
    if (!container) return;
    container.innerHTML = '<div class="structure-status">Loading structure...</div>';
    const fmtRaw = (payload.format || 'pdb').toLowerCase();
    const fmt = (fmtRaw === 'mmcif' || fmtRaw === 'cif') ? 'cif'
              : (fmtRaw === 'ent' ? 'pdb' : fmtRaw);

    let viewer;
    try {
      container.innerHTML = '';
      viewer = $3Dmol.createViewer(container, { backgroundColor: 'white' });
      viewer.addModel(payload.content, fmt);
    } catch (err) {
      console.error('3Dmol load error', payload.id, err);
      container.innerHTML = '<div class="structure-status">Failed to load structure</div>';
      return;
    }

    // Detect whether B-factor (== pLDDT for AF/Protenix/Boltz outputs) is
    // populated. Some PDBs leave it at 0 — fall back to spectrum colouring.
    let hasPlddt = false;
    try {
      const atoms = viewer.selectedAtoms({ atom: 'CA' });
      for (const a of atoms) {
        if (typeof a.b === 'number' && a.b > 0.01) { hasPlddt = true; break; }
      }
    } catch (e) { /* selectedAtoms may not be available in some versions */ }

    // Honour a per-section default colour (set by the Python renderer for
    // binder-design complexes — defaults binder/target to chain colouring so
    // the two are visually distinct from the first paint). Falls back to the
    // pLDDT-detection heuristic otherwise. pLDDT default is still gated on
    // having real B-factor data so a placeholder zero column stays muted.
    const sectionEarly = container.closest('.structure-section');
    const defaultColour = (sectionEarly && sectionEarly.dataset.defaultColour) || '';
    let initialColour;
    if (defaultColour === 'plddt' && !hasPlddt) {
      initialColour = 'spectrum';
    } else if (defaultColour) {
      initialColour = defaultColour;
    } else {
      initialColour = hasPlddt ? 'plddt' : 'spectrum';
    }

    // Per-viewer state: current colour mode + transient picker state.
    const state = {
      viewer,
      payload,
      colourMode: initialColour,
      sidechains: false,
      pickerActive: false,
      hoverKey: null,   // transient hover highlight {resi, chain}
    };

    applyStructureStyles(state);
    // Click behaviour intentionally minimal — the report's residue
    // inspection happens through Pick (hover) mode, so clicking only
    // rotates / zooms the model rather than dropping label pins.
    viewer.zoomTo();
    viewer.render();
    try { viewer.zoom(1.1, 500); } catch (e) { /* older 3Dmol versions */ }

    // ---- Toolbar wiring (one viewer's controls) ----
    const section = container.closest('.structure-section');
    if (!section) return;

    const colourSel = section.querySelector('.structure-color');
    if (colourSel) {
      colourSel.value = state.colourMode;
      colourSel.addEventListener('change', () => {
        const next = colourSel.value;
        if (next === 'plddt' && !hasPlddt) {
          colourSel.value = state.colourMode;
          return;
        }
        state.colourMode = next;
        const legend = section.querySelector('.structure-legend');
        if (legend) legend.dataset.legend = next;
        applyStructureStyles(state);
      });
    }
    const sidechainsBox = section.querySelector('.structure-sidechains');
    if (sidechainsBox) {
      sidechainsBox.addEventListener('change', () => {
        state.sidechains = sidechainsBox.checked;
        applyStructureStyles(state);
      });
    }
    const fullscreenBtn = section.querySelector('.structure-fullscreen-btn');
    if (fullscreenBtn) {
      fullscreenBtn.addEventListener('click', () => toggleFullscreen(section));
    }
    // RCSB-style picker mode: toggle hoverable on the 3Dmol viewer so the
    // bottom-right info card updates as the cursor moves across residues.
    const pickerBtn = section.querySelector('.structure-picker-btn');
    const infoPanel = section.querySelector('.structure-residue-info');
    const seqStrip = section.querySelector('.sst-strip');
    const seqAttr = section.getAttribute('data-sequence') || '';
    state.sequence = seqAttr;
    state.section = section;
    state.infoPanel = infoPanel;
    state.seqStrip = seqStrip;
    if (pickerBtn && infoPanel) {
      pickerBtn.addEventListener('click', () => togglePicker(state, section));
    }

    section.addEventListener('fullscreenchange', () => {
      // Resize after the layout settles so the canvas matches the new container.
      setTimeout(() => {
        try { viewer.resize(); viewer.render(); } catch (e) {}
      }, 60);
    });
    window.addEventListener('resize', () => {
      try { viewer.resize(); viewer.render(); } catch (e) {}
    });
  }

  // ---- Picker mode (hover-driven residue inspection) ----------------
  function togglePicker(state, section) {
    const v = state.viewer;
    const btn = section.querySelector('.structure-picker-btn');
    const panel = state.infoPanel;
    state.pickerActive = !state.pickerActive;
    if (btn) btn.setAttribute('aria-pressed', state.pickerActive ? 'true' : 'false');
    if (state.pickerActive) {
      attachHoverable(state);
      v.render();
      if (panel) {
        panel.hidden = false;
        resetInfoPanel(panel);
      }
    } else {
      try { v.setHoverable({}, false); } catch (e) {}
      clearHoverHighlight(state);
      clearSequenceHighlight(state);
      if (panel) panel.hidden = true;
    }
  }

  function onHoverResidue(state, atom) {
    if (!state.pickerActive) return;
    if (!atom || atom.resi == null) return;
    const key = (atom.chain || '') + '_' + atom.resi;
    if (state.hoverKey === key) return;
    state.hoverKey = key;
    // Re-apply persistent styles (clears any previous transient highlight),
    // then layer the hover highlight on top.
    applyStructureStyles(state);
    applyHoverHighlight(state, atom);
    updateInfoPanel(state, atom);
    state.viewer.render();
  }

  function onUnhoverResidue(state) {
    if (!state.pickerActive) return;
    if (state.hoverKey === null) return;
    state.hoverKey = null;
    applyStructureStyles(state);
    state.viewer.render();
    if (state.infoPanel) resetInfoPanel(state.infoPanel);
    clearSequenceHighlight(state);
  }

  function clearHoverHighlight(state) {
    if (state.hoverKey === null) return;
    state.hoverKey = null;
    applyStructureStyles(state);
    state.viewer.render();
  }

  function applyHoverHighlight(state, atom) {
    // Transient pink-magenta highlight identical in shape to the click
    // highlight (cartoon + thick stick + CA sphere) so the hovered
    // residue is unambiguously located even at low confidence colours.
    const v = state.viewer;
    const sel = { resi: atom.resi };
    if (atom.chain) sel.chain = atom.chain;
    const colour = '#cc0980';   // brand pink — distinct from any AA palette tint
    v.addStyle(sel, { cartoon: { color: colour } });
    v.addStyle(Object.assign({ hetflag: false }, sel),
               { stick: { color: colour, radius: 0.34 } });
    v.addStyle(Object.assign({ hetflag: false, atom: 'CA' }, sel),
               { sphere: { color: colour, radius: 1.5 } });
  }

  function resetInfoPanel(panel) {
    const chainEl = panel.querySelector('.sri-chain');
    const resiEl = panel.querySelector('.sri-resi');
    const seqEl = panel.querySelector('.sri-seqpos');
    if (chainEl) {
      chainEl.textContent = 'Chain —';
      chainEl.classList.add('sri-placeholder');
    }
    if (resiEl) {
      resiEl.innerHTML = '<strong>—</strong>';
      resiEl.classList.add('sri-placeholder');
    }
    if (seqEl) {
      seqEl.textContent = 'seq —';
      seqEl.classList.add('sri-placeholder');
    }
  }

  function updateInfoPanel(state, atom) {
    const panel = state.infoPanel;
    if (!panel) return;
    const chainEl = panel.querySelector('.sri-chain');
    const resiEl = panel.querySelector('.sri-resi');
    const seqEl = panel.querySelector('.sri-seqpos');
    const resn = (atom.resn || '?').toUpperCase();
    const one = AA3_TO_1[resn] || '';
    if (chainEl) {
      chainEl.textContent = atom.chain ? 'Chain ' + atom.chain : 'Chain —';
      chainEl.classList.remove('sri-placeholder');
    }
    if (resiEl) {
      const oneTxt = one ? ' (' + one + ')' : '';
      resiEl.innerHTML = '<strong>' + resn + ' ' + atom.resi + '</strong>' + oneTxt;
      resiEl.classList.remove('sri-placeholder');
    }
    if (seqEl) {
      // Align with the candidate's FASTA — atom.resi is 1-indexed in the
      // PDB. When the design's sequence is shorter than the structure
      // (e.g. an unmodelled tail) we still show the position but fall
      // back to "seq N/A" when out of range.
      const idx = (atom.resi | 0) - 1;
      if (state.sequence && idx >= 0 && idx < state.sequence.length) {
        const ch = state.sequence.charAt(idx);
        seqEl.textContent = 'seq ' + (idx + 1) + ' (' + ch + ')';
        seqEl.classList.remove('sri-placeholder');
      } else {
        seqEl.textContent = 'seq ' + atom.resi;
        seqEl.classList.remove('sri-placeholder');
      }
    }
    highlightSequenceCell(state, atom.resi);
  }

  function highlightSequenceCell(state, resi) {
    // Pink-magenta highlight on the matching .sst-cell so the user can
    // see exactly which FASTA position lines up with the hovered 3D
    // residue. The strip auto-scrolls so the active cell stays visible
    // even on long sequences.
    if (!state.section) return;
    const strip = state.seqStrip;
    if (!strip) return;
    const previous = strip.querySelectorAll('.sst-cell.is-hovered');
    previous.forEach((c) => c.classList.remove('is-hovered'));
    if (resi == null) return;
    const cell = strip.querySelector('.sst-cell[data-resi="' + (resi | 0) + '"]');
    if (cell) {
      cell.classList.add('is-hovered');
      try {
        cell.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' });
      } catch (e) { /* older browsers may not honour smooth scrollIntoView */ }
    }
  }

  function clearSequenceHighlight(state) {
    if (!state.section) return;
    const strip = state.seqStrip;
    if (!strip) return;
    const previous = strip.querySelectorAll('.sst-cell.is-hovered');
    previous.forEach((c) => c.classList.remove('is-hovered'));
  }

  // Apply the full style cascade. Spec §6 - call this any time toggles change.
  // Clears all styles first, then layers protein cartoon, HET stick/sphere,
  // optional sidechains, and finally re-applies user-selected highlights.
  function applyStructureStyles(state) {
    const v = state.viewer;
    v.setStyle({}, {});

    // Protein backbone cartoon
    const cartoonStyle = { cartoon: cartoonColourSpec(state.colourMode) };
    v.setStyle({ hetflag: false }, cartoonStyle);

    // Hetero atoms (ligands etc.) as ball-and-stick
    v.setStyle({ hetflag: true }, { stick: { radius: 0.22 }, sphere: { scale: 0.32 } });
    // Hide water
    v.setStyle({ resn: 'HOH' }, {});
    v.setStyle({ resn: 'WAT' }, {});

    if (state.sidechains) {
      v.addStyle(
        { hetflag: false, not: { atom: ['N', 'CA', 'C', 'O'] } },
        { stick: { radius: 0.13 } }
      );
    }

    // Re-establish hover handlers: 3Dmol re-creates atom-level state
    // when styles change, so the picker callbacks would silently stop
    // firing the second time the cartoon is re-rendered without this.
    if (state.pickerActive) {
      attachHoverable(state);
    }
    v.render();
  }

  function attachHoverable(state) {
    const v = state.viewer;
    try {
      v.setHoverable({}, true,
        (atom, viewerObj) => onHoverResidue(state, atom),
        (atom, viewerObj) => onUnhoverResidue(state));
    } catch (e) {
      // Older 3Dmol versions or rare WebGL failures — silent degrade.
    }
  }

  function cartoonColourSpec(mode) {
    if (mode === 'plddt') {
      // 3Dmol's colorfunc receives an atom; we colour by B-factor (pLDDT).
      return { colorfunc: (atom) => plddtForResi(atom.b || 0) };
    }
    if (mode === 'chain') {
      return { colorscheme: 'chain' };
    }
    return { color: 'spectrum' };
  }

  function toggleFullscreen(section) {
    if (document.fullscreenElement === section) {
      document.exitFullscreen();
    } else if (section.requestFullscreen) {
      section.requestFullscreen().catch((e) => console.warn('fullscreen denied', e));
    }
  }
})();
""" + PLDDT_HOVER_JS


__all__ = ["render_report_html", "write_report"]

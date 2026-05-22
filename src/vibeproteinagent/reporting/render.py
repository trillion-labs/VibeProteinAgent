# HTML/CSS/JS templates are intentionally kept readable as literal snippets.
# ruff: noqa: E501

from __future__ import annotations

import base64
import html
import json
import math
import mimetypes
import shutil
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vibeproteinagent._version import __version__
from vibeproteinagent.core.errors import InvalidParamsError

from ._brand import vpa_logo
from .models import (
    MSABlock,
    PlotSpec,
    ReportArtifact,
    ReportCandidate,
    ReportModel,
    WorkflowNodeSummary,
)
from .plots import build_plots


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
    top = model.candidates[:top_k]

    headline = _executive_headline(model)
    dashboard_html = _executive_dashboard(model)
    verdict_html = _exec_verdict_callout(model)

    show_workflow = len(model.graph.nodes) >= 2
    show_plots = bool(plots)
    show_msa = bool(model.msa_blocks)
    show_candidates = bool(model.candidates)

    nav_items: list[tuple[str, str, str]] = [("summary", "Summary", "◎")]
    if show_candidates:
        nav_items.append(("candidates", "Top candidates", "★"))
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

    # ----- Candidates section -----
    candidates_section = (
        f"""<section id="candidates" class="card card-candidates">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Designs</p>
      <h2 class="card-title">Top Candidates</h2>
    </div>
    <span class="card-meta">{len(model.candidates)} total</span>
  </header>
  {_candidate_cards(top, total_candidates=len(model.candidates))}
  {_all_designs_table(model.candidates)}
</section>"""
        if show_candidates else ""
    )

    workflow_section = (
        f"""<section id="workflow" class="card">
  <header class="card-header"><div><p class="card-eyebrow">Pipeline</p><h2 class="card-title">Workflow graph</h2></div></header>
  <p class="widget-explainer">Each node ran as an independent VPA job. Colour shows completion state; arrows trace data dependencies.</p>
  {_workflow_svg(model.graph.nodes)}
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
    # should never be ellipsis-clipped.
    stages_rows: list[str] = []
    for node in nodes[:8]:  # cap to 8 to keep the panel readable
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
    overflow_row = ""
    if len(nodes) > 8:
        overflow_row = (
            f'<li class="exec-stage-row exec-stage-more">'
            f'<span class="exec-stage-glyph">…</span>'
            f'<span class="exec-stage-name">{len(nodes) - 8} more node(s)</span>'
            "</li>"
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
        f'{"".join(stages_rows)}{overflow_row}'
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
    iptm = _as_unit(top.scores.get("iptm"))
    if iptm is not None and iptm < 0.5:
        cautions.append(f"ipTM {iptm:.2f} below 0.5 — interface unlikely to be trustworthy")
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


def _structure_sequence_track(sequence: str, dom_id: str) -> str:
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
    return (
        f'<div class="structure-sequence-track" data-target="{_attr(dom_id)}">'
        '<div class="sst-head">'
        '<span class="sst-label">Sequence</span>'
        f'<span class="sst-meta">{len(sequence)} residues · '
        f'hover the structure to highlight{meta_extra}</span>'
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
        viewer = _viewer_block(candidate)
        score_html = _hero_candidate_score(candidate)
        cards.append(
            f"""<article class="candidate-card">
  <div class="candidate-head">
    <div><p class="eyebrow">{_h(eyebrow)}</p><h3>{_h(candidate.candidate_id)}</h3></div>
    {score_html}
  </div>
  <div class="candidate-grid">
    <div class="candidate-viewer-col">{viewer}</div>
    <div class="candidate-info-col">
      {_score_bars(candidate)}
      {_flag_badges(candidate)}
      <p class="why"><strong>Why this one?</strong> {_h(_why_candidate(candidate))}</p>
      {_candidate_sequence_block(candidate)}
      {_candidate_downloads(candidate)}
    </div>
  </div>
</article>"""
        )
    return "\n".join(cards)


def _hero_candidate_score(candidate: ReportCandidate) -> str:
    """Top-right corner of the card — show the *most relevant* score prominently."""
    if candidate.score is not None:
        return f'<div class="card-score"><strong>{_h(_fmt(candidate.score))}</strong><span>score</span></div>'
    for key, label in (("plddt", "pLDDT"), ("ptm", "pTM"), ("iptm", "ipTM")):
        value = candidate.scores.get(key)
        if value is None:
            continue
        quality = _quality_class(key, value)
        return (
            f'<div class="card-score quality-{quality}">'
            f'<strong>{_h(_fmt(value))}</strong><span>{_h(label)}</span></div>'
        )
    return ""


def _viewer_block(candidate: ReportCandidate) -> str:
    """Inline 3Dmol.js viewer block per the per-candidate structure file.

    HTML scaffold follows the spec at docs/agent_layer.md (3D-viz section):
    header strip with sidechain toggle + fullscreen button, then a single
    empty <div> that $3Dmol.createViewer fills with its WebGL canvas, plus
    an absolutely-positioned status overlay for loading / error states.
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
    return f"""<section class="structure-section" data-target="{_attr(dom_id)}" data-format="{_attr(fmt)}"
         data-candidate-id="{_attr(candidate.candidate_id)}"
         data-structure-name="{_attr(structure_label)}"
         data-sequence="{_attr(sequence_str)}">
  <header class="structure-header">
    <span class="structure-title">3D structure</span>
    <span class="structure-hint">Turn on Pick, then hover residues for an instant residue card.</span>
    <label class="structure-control"><span>Colour</span>
      <select class="structure-color" data-target="{_attr(dom_id)}">
        <option value="plddt" selected>pLDDT (AlphaFold)</option>
        <option value="spectrum">N to C spectrum</option>
        <option value="chain">Chain</option>
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
  {_structure_sequence_track(sequence_str, dom_id)}
  <div class="structure-viewer" id="{_attr(dom_id)}" data-structure-id="{_attr(dom_id)}"></div>
  <div class="structure-legend" data-legend="plddt">
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


def _candidate_sequence_block(candidate: ReportCandidate) -> str:
    if not candidate.sequence:
        return ""
    return (
        '<div class="sequence-block">'
        f'<div class="sequence-head"><strong>Sequence</strong> · {len(candidate.sequence)} aa'
        f' <button class="copy-btn" data-copy="{_attr(candidate.sequence)}">Copy</button></div>'
        f"<code class='msa-seq'>{_colored_sequence(candidate.sequence[:240])}</code>"
        + (
            f"<p class='note'>… {len(candidate.sequence) - 240} more residues hidden</p>"
            if len(candidate.sequence) > 240
            else ""
        )
        + "</div>"
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
        f'<svg class="dag" viewBox="0 0 {width} {height}" '
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


def _node_details(nodes: list[WorkflowNodeSummary]) -> str:
    """Per-tool result widgets — each node gets a structured presentation
    chosen by tool name, with the raw JSON tucked into a collapsible
    "Raw output" details element at the bottom for reference only.
    """
    if not nodes:
        return "<p class='empty'>No node details available.</p>"
    chunks = []
    for node in nodes:
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
        tool_lc = (node.tool or "").lower()
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
# Centralised metric metadata: any tool — known or unknown — that emits a
# scalar with one of these canonical names automatically gets a gauge-centred
# card via :func:`_generic_metric_widget` and :func:`_auto_score_card`. This
# keeps the per-tool widget surface systematic: introducing a new adapter
# does *not* require teaching the report renderer about it as long as the
# adapter emits metrics with the names already understood by the rest of the
# stack.
#
# Each entry: ``key -> (display_name, explainer, gauge_lo, gauge_hi)``
_METRIC_REGISTRY: dict[str, tuple[str, str, float, float]] = {
    "plddt": (
        "Mean pLDDT",
        "Per-residue confidence averaged across the chain (0-100). >=70 = confident, >=90 = very high.",
        0.0, 100.0,
    ),
    "mean_plddt": (
        "Mean pLDDT",
        "Per-residue confidence averaged across the chain (0-100). >=70 = confident, >=90 = very high.",
        0.0, 100.0,
    ),
    "ptm": (
        "pTM",
        "Predicted TM-score (0-1). >0.5 implies a likely-correct overall fold.",
        0.0, 1.0,
    ),
    "iptm": (
        "ipTM",
        "Interface pTM (0-1). >0.6 generally trustworthy for complex / binder interfaces.",
        0.0, 1.0,
    ),
    "pae": (
        "PAE",
        "Predicted aligned error in angstroms. Lower is better; <5 indicates well-resolved geometry.",
        0.0, 30.0,
    ),
    "mean_pae": (
        "PAE",
        "Predicted aligned error in angstroms. Lower is better; <5 indicates well-resolved geometry.",
        0.0, 30.0,
    ),
    "dockq": (
        "DockQ",
        "Interface accuracy score (0-1). >0.23 acceptable, >0.49 medium, >0.80 high.",
        0.0, 1.0,
    ),
    "ipsae": (
        "ipSAE",
        "Interface predicted symmetry-aware error. Lower is better.",
        0.0, 30.0,
    ),
    "ranking_confidence": (
        "Ranking confidence",
        "Composite head used to rank candidate structures. Scale matches the predictor.",
        0.0, 1.0,
    ),
    "ranking_score": (
        "Ranking score",
        "Pipeline-specific aggregate score used to order candidates.",
        0.0, 1.0,
    ),
    "score": (
        "Score",
        "Generic composite score reported by the tool.",
        0.0, 1.0,
    ),
    "delta_g": (
        "ΔG (kcal/mol)",
        "Predicted binding free energy. More negative = stronger binding.",
        -16.0, 0.0,
    ),
    "dg": (
        "ΔG (kcal/mol)",
        "Predicted binding free energy. More negative = stronger binding.",
        -16.0, 0.0,
    ),
    "affinity": (
        "Affinity",
        "Predicted binding strength on the tool's native scale.",
        0.0, 12.0,
    ),
    "affinity_pred_value": (
        "Affinity (pred)",
        "Boltz-2 predicted affinity value (-log10 IC50). Higher = tighter.",
        0.0, 12.0,
    ),
    "affinity_probability": (
        "Binder probability",
        "Calibrated probability that the candidate binds (0-1).",
        0.0, 1.0,
    ),
    "affinity_probability_binary": (
        "Binder probability",
        "Boltz-2 binary binder probability (0-1).",
        0.0, 1.0,
    ),
    "confidence_score": (
        "Model confidence",
        "Boltz-2 composite confidence head (0-1).",
        0.0, 1.0,
    ),
    "iplddt": (
        "ipLDDT",
        "Interface pLDDT — per-residue confidence at the binding interface.",
        0.0, 100.0,
    ),
    "recovery": (
        "Recovery",
        "Fraction of native residues recovered by the design (0-1).",
        0.0, 1.0,
    ),
    "seq_recovery": (
        "Seq recovery",
        "Fraction of native residues recovered by the designed sequence (0-1).",
        0.0, 1.0,
    ),
    "instability_index": (
        "Instability index",
        "Guruprasad heuristic. <40 = predicted stable in vitro.",
        0.0, 80.0,
    ),
    "gravy": (
        "GRAVY",
        "Grand average of hydropathy (Kyte-Doolittle). Negative = hydrophilic / likely soluble.",
        -2.0, 2.0,
    ),
    "pi": (
        "Isoelectric point",
        "pH at net-zero charge. Typical soluble proteins span 4-9.",
        0.0, 14.0,
    ),
    "polybasic_hits": (
        "Polybasic motifs",
        "Count of K/R-rich runs that drive non-specific membrane / RNA binding.",
        0.0, 10.0,
    ),
    "aromatic_fraction": (
        "Aromatic fraction",
        "Fraction of F/Y/W residues. Relevant for UV absorbance / aggregation propensity.",
        0.0, 0.25,
    ),
}


# Metrics that conventionally arrive on a 0-1 scale but may also appear as
# 0-100 (pLDDT is the canonical example). Used by :func:`_auto_score_card`
# to autoscale the gauge so the donut fills correctly either way.
_DUAL_SCALE_METRICS = {"plddt", "mean_plddt", "iptm", "ptm", "recovery", "seq_recovery"}


# Keys that look like nested metric dicts inside an adapter's output. The
# generic widget recurses one level into these so a tool emitting
# ``confidence_scores: {plddt: 0.8, ptm: 0.4}`` still surfaces both as cards.
_NESTED_METRIC_KEYS = ("confidence_scores", "metrics", "scores")


# Footer-only keys: shown as kv rows under the card grid rather than cards
# themselves (they're context, not headline metrics).
_FOOTER_METRIC_KEYS = (
    "model_name", "model", "tool", "version", "device", "seed",
    "num_residues", "length", "n_residues", "n_atoms",
    "runtime_ms", "runtime_s", "elapsed_ms", "elapsed_s",
)


def _auto_score_card(key: str, value: Any) -> str | None:
    """Build a CELLKEY-style gauge card for *any* metric in the registry.

    Returns ``None`` when the value isn't numeric (so the caller can decide
    to render it as a footer row instead). For metrics not in the registry
    but still numeric, falls back to a plain card without a gauge so the
    layout stays consistent.
    """
    num = _num(value)
    if num is None:
        return None
    canonical = key.lower()
    spec = _METRIC_REGISTRY.get(canonical)
    quality = _quality_class(canonical, num)
    if spec is None:
        # Unknown numeric metric — keep it on the same card grid but without
        # a gauge (no defensible range to fill).
        return _score_card(
            _pretty_metric_name(canonical), _fmt(num),
            "Tool-emitted metric. No reference range registered — value shown as-is.",
            quality,
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
        "structure_analysis": _structure_analysis_widget,
        "sequence_analysis": _sequence_analysis_widget,
        "sequence_edit": _sequence_edit_widget,
        "rank_designs": _rank_designs_widget,
        "boltz2_affinity": _boltz2_widget,
        "proteinmpnn": _proteinmpnn_widget,
        "protenix_v2": _protenix_widget,
        "esm3": _esm3_widget,
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

    return (
        '<div class="widget widget-struct">'
        '<h4 class="widget-title">Structure analysis</h4>'
        f"{explainer}"
        f"{chains_label}"
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


def _sequence_analysis_widget(output: dict[str, Any]) -> str:
    rows: list[str] = []
    summary = output.get("summary") if isinstance(output.get("summary"), dict) else {}
    if isinstance(summary, dict):
        for key, label in (
            ("length", "Length"),
            ("molecular_weight", "Molecular weight (Da)"),
            ("isoelectric_point", "pI"),
            ("instability_index", "Instability index"),
            ("aromaticity", "Aromaticity"),
            ("gravy", "GRAVY"),
        ):
            value = summary.get(key)
            if value is None:
                continue
            rows.append(_kv_row(label, _fmt(value)))
    aa_comp = output.get("aa_composition") if isinstance(output.get("aa_composition"), dict) else None
    composition_html = ""
    if isinstance(aa_comp, dict) and aa_comp:
        # Compose a small horizontal bar chart of amino acid composition.
        total = sum(_num(v) or 0 for v in aa_comp.values()) or 1.0
        bars: list[str] = []
        for aa, count in sorted(aa_comp.items(), key=lambda kv: -(_num(kv[1]) or 0))[:20]:
            n = _num(count) or 0.0
            frac = n / total
            bars.append(
                f'<div class="aa-bar-row"><span class="aa-bar-label aa aa-{_aa_class(aa)}">{_h(aa)}</span>'
                f'<div class="aa-bar-track"><i style="width:{frac * 100:.1f}%"></i></div>'
                f'<span class="aa-bar-count">{_fmt(n)}</span></div>'
            )
        if bars:
            composition_html = (
                '<h5 class="widget-subhead">Amino-acid composition</h5>'
                f'<div class="aa-bars">{"".join(bars)}</div>'
            )
    table_html = (
        f'<table class="kv-table"><tbody>{"".join(rows)}</tbody></table>'
        if rows
        else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Sequence analysis</strong> derives compositional and biochemical descriptors from the residue "
        "string alone: length, MW, pI, instability index, aromaticity, GRAVY and amino-acid distribution. "
        "These are deterministic and computed via BioPython's <code>ProteinAnalysis</code>."
        "</p>"
    )
    return (
        '<div class="widget widget-seqana">'
        '<h4 class="widget-title">Sequence analysis</h4>'
        f"{explainer}{table_html}{composition_html}</div>"
    )


def _sequence_edit_widget(output: dict[str, Any]) -> str:
    wt = output.get("wild_type_sequence") or output.get("original_sequence")
    mutant = output.get("mutant_sequence") or output.get("edited_sequence")
    mutations = output.get("mutations")
    metrics = output.get("metrics") if isinstance(output.get("metrics"), dict) else {}
    rows: list[str] = []
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            rows.append(_kv_row(_pretty_metric_name(k), _h(_compact_inline(v))))
    if isinstance(mutations, list) and mutations:
        rows.append(
            _kv_row(
                "Mutations applied",
                ", ".join(
                    str(m.get("change") if isinstance(m, dict) else m)
                    for m in mutations[:12]
                ),
            )
        )
    alignment_html = ""
    if isinstance(wt, str) and isinstance(mutant, str) and wt and mutant:
        alignment_html = _wt_mut_alignment_html(wt, mutant)
    table_html = (
        f'<table class="kv-table"><tbody>{"".join(rows)}</tbody></table>'
        if rows
        else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Sequence edit</strong> applies the requested mutations (substitutions, insertions, deletions) "
        "and reports the resulting sequence with side-by-side diff against the wild-type. Each highlighted "
        "column below is a position where the two sequences differ."
        "</p>"
    )
    return (
        '<div class="widget widget-seqedit">'
        '<h4 class="widget-title">Sequence edit</h4>'
        f"{explainer}{table_html}{alignment_html}</div>"
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


def _proteinmpnn_widget(output: dict[str, Any]) -> str:
    designs = output.get("designs") if isinstance(output.get("designs"), list) else []
    num_residues = output.get("num_residues")
    runtime_ms = output.get("runtime_ms")
    head_rows: list[str] = []
    if num_residues is not None:
        head_rows.append(_kv_row("Designed residues", _fmt(num_residues)))
    if runtime_ms is not None:
        head_rows.append(_kv_row("Runtime", _duration(runtime_ms)))
    if designs:
        head_rows.append(_kv_row("Designs emitted", str(len(designs))))
    table_html = ""
    if designs:
        rows = []
        for idx, d in enumerate(designs[:10]):
            if not isinstance(d, dict):
                continue
            rows.append(
                "<tr>"
                f"<td>{idx + 1}</td>"
                f"<td><code>{_h((d.get('sequence') or '')[:80])}</code></td>"
                f"<td>{_h(_fmt(d.get('score')))}</td>"
                f"<td>{_h(_fmt(d.get('seq_recovery')))}</td>"
                "</tr>"
            )
        table_html = (
            '<div class="table-wrap"><table><thead><tr>'
            "<th>#</th><th>Sequence (first 80 aa)</th><th>Score</th><th>Seq recov.</th>"
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>ProteinMPNN</strong> samples sequences over a fixed backbone. Lower score = higher likelihood "
        "under the trained model. Sequence-recovery vs the WT/native sequence is a sanity proxy when redesigning "
        "an existing protein."
        "</p>"
    )
    summary_html = (
        f'<table class="kv-table"><tbody>{"".join(head_rows)}</tbody></table>' if head_rows else ""
    )
    return (
        '<div class="widget widget-mpnn">'
        '<h4 class="widget-title">ProteinMPNN designs</h4>'
        f"{explainer}{summary_html}{table_html}</div>"
    )


def _protenix_widget(output: dict[str, Any]) -> str:
    cards: list[str] = []
    for key, label, tip, lo, hi in (
        ("mean_plddt", "Mean pLDDT", "Per-residue confidence (0-1 or 0-100).", 0.0, 100.0),
        ("plddt", "Mean pLDDT", "Per-residue confidence (0-1 or 0-100).", 0.0, 100.0),
        ("ptm", "pTM", "Predicted TM-score.", 0.0, 1.0),
        ("iptm", "ipTM", "Interface pTM.", 0.0, 1.0),
        ("ranking_confidence", "Ranking confidence", "Composite confidence head.", 0.0, 1.0),
    ):
        value = output.get(key)
        if value is None:
            continue
        num = _num(value) or 0.0
        # Protenix may emit pLDDT as 0-1 — auto-scale the gauge hi to match.
        gauge_hi = hi
        if key in {"plddt", "mean_plddt"} and num <= 1.5:
            gauge_hi = 1.0
        cards.append(_score_card(
            label, _fmt(value), tip, _quality_class(key, value),
            gauge={"value": num, "lo": lo, "hi": gauge_hi},
        ))
    cards_html = _card_grid(cards)
    cif_path = output.get("predicted_cif") or output.get("predicted_pdb")
    art_html = ""
    if isinstance(cif_path, str):
        kind = Path(cif_path).suffix.lstrip(".") or "structure"
        art_html = (
            '<div class="node-block-artifacts">'
            '<span class="artifact-section-label">Predicted structure</span>'
            f'<div class="artifact-row">'
            f'{_download_link(cif_path, Path(cif_path).name, kind=kind)}</div></div>'
        )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>Protenix v2</strong> is an AlphaFold-3-class structure predictor for protein / complex inputs. "
        "The cards show the confidence heads; the structure file is linked below for inspection in PyMOL / "
        "ChimeraX."
        "</p>"
    )
    return (
        '<div class="widget widget-protenix">'
        '<h4 class="widget-title">Protenix structure prediction</h4>'
        f"{explainer}{cards_html}{art_html}</div>"
    )


def _esm3_widget(output: dict[str, Any]) -> str:
    cs = output.get("confidence_scores") if isinstance(output.get("confidence_scores"), dict) else {}
    cards: list[str] = []
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
                label, _fmt(value), tip, _quality_class(key.replace("mean_", ""), value),
                gauge={"value": num, "lo": 0.0, "hi": gauge_hi},
            ))
    rows: list[str] = []
    for k in ("model_name", "num_residues", "runtime_ms"):
        v = output.get(k)
        if v is None:
            continue
        rows.append(_kv_row(_pretty_metric_name(k), _h(_compact_inline(v))))
    table_html = (
        f'<table class="kv-table"><tbody>{"".join(rows)}</tbody></table>' if rows else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "<strong>ESM3</strong> jointly handles sequence, structure and function tracks. The card shows the "
        "confidence heads; per-residue embeddings + the predicted structure are surfaced via the artifact list."
        "</p>"
    )
    cards_html = _card_grid(cards)
    return (
        '<div class="widget widget-esm3">'
        '<h4 class="widget-title">ESM3 output</h4>'
        f"{explainer}{cards_html}{table_html}</div>"
    )


def _generic_metric_widget(output: dict[str, Any]) -> str:
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
    # inventory so sequences / lists don't disappear silently.
    shape_html = _output_shape_widget(output) if has_complex_output else ""

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
) -> str:
    """Smooth donut gauge with value (and optional sub-label) inside.

    A single continuous arc traces the value's position within ``[lo, hi]``
    against a neutral background ring. The big number sits at the centre;
    an optional short sub-label (e.g. ``pLDDT``) sits one line below so the
    user can identify the metric at a glance without reading the body text.

    Stays offline-pure SVG (no fonts, no images, no JS).
    """
    if value is None or not isinstance(value, (int, float)):
        return ""
    span = (hi - lo) if hi > lo else 1.0
    pct = max(0.0, min(1.0, (float(value) - lo) / span))

    cx = cy = size / 2
    r = size / 2 - stroke
    colour = _QUALITY_COLOR.get(quality, _QUALITY_COLOR["neutral"])
    bg_color = "#ede9fe"

    circ = 2 * math.pi * r
    arc_len = pct * circ
    # A single continuous ring: full background circle behind, then a
    # colour-filled foreground arc traced via stroke-dasharray. Starting at
    # 12 o'clock by rotating -90deg around the centre.
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


def _wt_mut_alignment_html(wt: str, mutant: str) -> str:
    n = min(len(wt), len(mutant))
    wt_cells: list[str] = []
    mut_cells: list[str] = []
    ruler_cells: list[str] = []
    for idx in range(n):
        col = idx + 1
        ruler_cells.append(
            f'<span class="aa-bar-label">{col if col % 10 == 0 or col == 1 else "·"}</span>'
        )
        differ = wt[idx] != mutant[idx]
        wt_cells.append(
            f'<span class="aa aa-{_aa_class(wt[idx])}{" diff" if differ else ""}">{_h(wt[idx])}</span>'
        )
        mut_cells.append(
            f'<span class="aa aa-{_aa_class(mutant[idx])}{" diff" if differ else ""}">{_h(mutant[idx])}</span>'
        )
    return (
        '<div class="wt-mut-align">'
        f'<div class="wt-mut-row wt-mut-ruler"><span class="wt-mut-label">col</span><code>{"".join(ruler_cells)}</code></div>'
        f'<div class="wt-mut-row"><span class="wt-mut-label">WT</span><code>{"".join(wt_cells)}</code></div>'
        f'<div class="wt-mut-row"><span class="wt-mut-label">Mut</span><code>{"".join(mut_cells)}</code></div>'
        "</div>"
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

    rows: list[str] = []
    for key in sorted(deduped.keys(), key=_priority):
        base_key = key.split(":", 1)[-1].lower()
        if base_key in _SCORE_BAR_SKIP:
            continue
        row = _score_bar_row(key, deduped[key])
        if row:
            rows.append(row)
    rows = rows[:14]
    if not rows:
        return ""
    return "<div class='score-bars'>" + "".join(rows) + "</div>"


def _score_bar_row(key: str, value: Any) -> str:
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
    # Render a clean label — no inline chip, no ``· tool`` suffix. The
    # tool of origin (when known) lives only in the row's ``title`` so
    # hovering reveals it without cluttering the bar.
    row_title = ""
    if tool_prefix:
        row_title = f' title="Source: {_attr(tool_prefix)}"'
    return (
        f"<div class='score-row quality-{quality}'{row_title}>"
        f"<span>{_h(label)}</span>"
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


def _why_candidate(candidate: ReportCandidate) -> str:
    bits = []
    for key, label in (("plddt", "pLDDT"), ("ptm", "pTM"), ("iptm", "ipTM"), ("ipsae", "ipSAE")):
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
    if k in {"plddt", "mean_plddt", "ptm", "iptm", "dockq", "ipsae", "seq_recovery", "recovery"}:
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


def _pretty_metric_name(key: str) -> str:
    return {
        "plddt": "pLDDT",
        "mean_plddt": "pLDDT",
        "iptm": "ipTM",
        "ptm": "pTM",
        "pae": "PAE",
        "mean_pae": "PAE",
        "ipsae": "ipSAE",
        "dockq": "DockQ",
        "delta_g": "ΔG",
        "seq_recovery": "Seq recov.",
        "recovery": "Recovery",
        "score": "Score",
        "ranking_score": "Rank score",
    }.get(key, key)


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

/* Confidence donut row (middle panel) ---------------------------------- */
.exec-donut-row {
  display:flex; flex-wrap:wrap; gap:10px;
  align-items:stretch;
}
.exec-donut {
  flex:1 1 0; min-width:0;
  display:flex; flex-direction:column; align-items:center; gap:8px;
  padding:12px 8px;
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  text-align:center;
  box-shadow:var(--shadow-tile);
}
.exec-donut-viz { line-height:0; }
.exec-donut-meta { width:100%; }
.exec-donut-label {
  font-size:13px; color:var(--ink); font-weight:600; letter-spacing:0;
}
.exec-donut-tip {
  font-size:11.5px; color:var(--muted); line-height:1.45; margin-top:3px;
}
.exec-donut-empty { justify-content:center; }
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
/* Stages list — one row per workflow node */
.exec-stage-list {
  list-style:none; padding:0; margin:0;
  display:flex; flex-direction:column; gap:3px;
  flex:1; min-width:0;
  max-height:170px; overflow-y:auto;
}
.exec-stage-row {
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
.exec-stage-more { opacity:0.7; }
.exec-stage-more .exec-stage-name { font-family:inherit; }
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
.dag-wrap { overflow-x:auto; padding:6px 2px 10px; }
.dag { display:block; max-width:100%; height:auto; }
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
.plot-svg rect, .plot-svg circle { fill:var(--accent); }
.plot-svg text { font-size:11px; fill:var(--muted); }

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

/* Per-residue strips (structure_analysis widget) */
.strip-block { margin:16px 0 18px; background:#ffffff; border:1px solid var(--line); border-radius:10px; padding:12px 16px; box-shadow:var(--shadow-tile); }
.strip-head { font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:0.05em; font-weight:600; margin-bottom:9px; }
.strip-row { display:grid; grid-template-columns:84px 1fr; gap:9px; align-items:center; padding:2px 0; }
.strip-label { color:var(--muted); font-size:12.5px; text-transform:uppercase; letter-spacing:0.05em; font-weight:600; text-align:right; }
.strip-cells { display:flex; flex-wrap:nowrap; overflow-x:auto; background:#fbf9fe; border:1px solid var(--line); border-radius:5px; padding:1px; min-width:0; }
.strip-cell { display:inline-block; width:6px; min-width:6px; height:14px; }
.strip-cell + .strip-cell { margin-left:1px; }
.strip-row.strip-ruler .strip-cells { background:transparent; border:none; padding:0; }
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

/* WT vs Mut alignment */
.wt-mut-align {
  background:#ffffff; border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; margin:14px 0; overflow-x:auto;
}
.wt-mut-row {
  display:flex; align-items:center; gap:10px; padding:2px 0;
  font-family:ui-monospace, monospace;
}
.wt-mut-row code { background:transparent; padding:0; }
.wt-mut-label {
  width:50px; flex-shrink:0; font-size:11px; color:var(--muted); text-transform:uppercase;
  letter-spacing:0.05em; font-weight:600;
}
.wt-mut-ruler { color:var(--muted); font-size:10px; }
.wt-mut-row .aa { width:13px; text-align:center; font-size:12px; }
.wt-mut-row .aa.diff { outline:2px solid var(--poor); outline-offset:-2px; }

/* Amino-acid composition bars */
.aa-bars { display:flex; flex-direction:column; gap:4px; margin:8px 0; }
.aa-bar-row { display:grid; grid-template-columns:30px 1fr 50px; gap:8px; align-items:center; font-size:12px; }
.aa-bar-label { text-align:center; padding:2px 0; }
.aa-bar-track { background:var(--bg); border-radius:3px; height:10px; overflow:hidden; }
.aa-bar-track i { display:block; height:100%; background:var(--accent); }
.aa-bar-count { color:var(--muted); font-family:ui-monospace, monospace; font-size:11px; text-align:right; }

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

    // Per-viewer state: current colour mode + transient picker state.
    const state = {
      viewer,
      payload,
      colourMode: hasPlddt ? 'plddt' : 'spectrum',
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
"""


__all__ = ["render_report_html", "write_report"]

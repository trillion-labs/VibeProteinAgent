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
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vibeproteinagent._version import __version__
from vibeproteinagent.core.errors import InvalidParamsError

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
from .identity import (
    candidate_material_key as _identity_candidate_material_key,
    candidate_ranked_candidate_id as _identity_candidate_ranked_candidate_id,
)
from .models import (
    BoardGroup,
    CandidateEvidenceBundle,
    CandidateStatus,
    EvidenceStep,
    MSABlock,
    PlotSpec,
    Provenance,
    ReportArtifact,
    ReportCandidate,
    ReportModel,
    ReportReviewModel,
    ScoreGroup,
    ToolRole,
    WorkflowNodeSummary,
    WorkflowType,
)
from .plots import PLDDT_HOVER_JS
from .review import build_review_model, classify_tool_role, classify_workflow_stage_key
from .view_models import (
    AgentRationaleItem,
    AgentRationaleViewModel,
    CandidateBoardRow,
    CandidateBoardViewModel,
    CandidateDossierViewModel,
    OverviewViewModel,
    WorkflowReviewPreset,
    WorkflowStageViewModel,
    WorkflowTraceViewModel,
    build_report_view_model,
)


def resolve_report_output_path(
    out_path: str | Path,
    *,
    report_format: str = "html",
) -> Path:
    if report_format not in {"html", "bundle"}:
        raise InvalidParamsError(
            f"unknown report format {report_format!r}",
            details={"format": report_format, "allowed": ["html", "bundle"]},
        )
    if report_format == "bundle":
        root = Path(out_path).expanduser().resolve()
        if root.suffix:
            root = root.with_suffix("")
        return root / "report.html"
    return Path(out_path).expanduser().resolve()


def write_report(
    model: ReportModel,
    out_path: str | Path,
    *,
    viewer_mode: str = "offline",
    report_format: str = "html",
    share: bool = False,
    top_k: int = 20,
) -> Path:
    if viewer_mode not in {"offline", "online"}:
        raise InvalidParamsError(
            f"unknown viewer mode {viewer_mode!r}",
            details={"viewer_mode": viewer_mode, "allowed": ["offline", "online"]},
        )

    html_path = resolve_report_output_path(out_path, report_format=report_format)
    if report_format == "bundle":
        root = html_path.parent
        root.mkdir(parents=True, exist_ok=True)
        (root / "assets").mkdir(exist_ok=True)
        (root / "structures").mkdir(exist_ok=True)
        (root / "plots").mkdir(exist_ok=True)
        model = _bundle_model(model, root)
    else:
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
    structure_sources: list[tuple[str, str | None, str | None]] = [
        (artifact.path, artifact.field, artifact.sha256 or artifact.artifact_id)
        for artifact in model.artifacts
    ]
    structure_sources.extend(
        (candidate.structure_path, candidate.candidate_id, None)
        for candidate in [*model.candidates, *model.final_candidates]
        if candidate.structure_path
    )
    for idx, (path, label, token_source) in enumerate(structure_sources, start=1):
        src = Path(path)
        if not src.is_file() or src.suffix.lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
            continue
        expanded = str(src.expanduser())
        if path in path_map or expanded in path_map:
            continue
        token = (token_source or str(idx))[:12]
        target = structures_dir / f"{_safe_filename(label or 'structure')}_{token}{src.suffix}"
        shutil.copyfile(src, target)
        path_map[path] = str(target)
        path_map[expanded] = str(target)

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
        replace(
            node,
            output=_rewrite_paths(node.output, path_map),
            artifacts=[rewrite_artifact(artifact) for artifact in node.artifacts],
        )
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
    final_candidates = [
        replace(
            candidate,
            structure_path=rewrite_path(candidate.structure_path),
            artifacts=[rewrite_artifact(artifact) for artifact in candidate.artifacts],
        )
        for candidate in model.final_candidates
    ]
    raw_summary = _rewrite_paths(model.raw_summary, path_map)
    final_submission = _rewrite_paths(model.final_submission, path_map)
    submissions = _rewrite_paths(model.submissions, path_map)
    return replace(
        model,
        graph=replace(model.graph, nodes=nodes),
        candidates=candidates,
        final_candidates=final_candidates,
        final_submission=(
            final_submission
            if isinstance(final_submission, dict) or final_submission is None
            else model.final_submission
        ),
        submissions=submissions if isinstance(submissions, list) else model.submissions,
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


def _is_structure_artifact_path(path_str: str | Path, kind: str | None = None) -> bool:
    if (kind or "").strip().lower() == "structure":
        return True
    suffix = Path(path_str).suffix.lower()
    return suffix in {".pdb", ".ent", ".cif", ".mmcif"}


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


# ------------------------------------------------------------- review console
#
# The candidate review console is the report's primary surface: a ranked board
# of candidates (grouped by lifecycle status) on the left, and a lazily
# activated detail dossier on the right. Each candidate's dossier lives in an
# inert <template> so the (potentially heavy) 3D viewer and per-tab markup are
# only materialised when the reader actually opens that candidate. All content
# is derived from the pure ``ReportReviewModel`` built by ``reporting.review``.


# Status → (board label, css token). Mirrors CandidateStatus; kept here so the
# render layer owns its own display vocabulary.
_STATUS_DISPLAY: dict[str, tuple[str, str]] = {
    CandidateStatus.FINAL: ("Final", "final"),
    CandidateStatus.RANKED_ALTERNATIVE: ("Ranked", "ranked"),
    CandidateStatus.EVALUATED_NOT_SELECTED: ("Evaluated", "evaluated"),
    CandidateStatus.GENERATED_NOT_EVALUATED: ("Generated only", "generated"),
    CandidateStatus.REFERENCE_GROUNDING: ("Reference", "reference"),
    CandidateStatus.FAILED_OR_REJECTED: ("Failed/rejected", "failed"),
}

_PROV_LABEL: dict[str, str] = {
    Provenance.EXACT: "recorded",
    Provenance.DERIVED: "derived",
    Provenance.INFERRED: "inferred",
    Provenance.NOT_RECORDED: "not recorded",
}


def _prov_chip(prov: str, *, title: str | None = None) -> str:
    """A small provenance pill so a derived/inferred value never reads as fact."""
    label = _PROV_LABEL.get(prov, prov)
    tip = title or {
        Provenance.EXACT: "Directly recorded in the run.",
        Provenance.DERIVED: "Computed by the report from recorded metrics.",
        Provenance.INFERRED: "Heuristic guess from a weak hint — treat with care.",
        Provenance.NOT_RECORDED: "No data was recorded for this.",
    }.get(prov, "")
    return f'<span class="prov prov-{_h(prov)}" title="{_attr(tip)}">{_h(label)}</span>'


def _board_headline_metric(bundle: CandidateEvidenceBundle) -> str:
    """The single most relevant number for a board card (or empty)."""
    cand = bundle.candidate
    if cand.score is not None:
        val = _fmt(cand.score)
        if val is not None:
            return f"score {val}"
    # Otherwise surface the first metric of the most decision-relevant family.
    priority = ("Interface & binding", "Structure confidence", "Sequence biophysics")
    by_family = {g.family: g for g in bundle.score_groups}
    for fam in priority:
        grp = by_family.get(fam)
        if grp and grp.metrics:
            label, value, unit = grp.metrics[0][:3]
            return f"{label} {value}{(' ' + unit) if unit else ''}"
    return ""


def _board_metric_parts(bundle: CandidateEvidenceBundle) -> tuple[str, str, str]:
    """Headline metric for the candidate list as ``(value, label, source)``.

    Prefers a confidence-family metric (which carries a producing-tool source)
    so the list reads like a scientific result table; falls back to the
    candidate's composite score when no scored family exists.
    """
    by_family = {g.family: g for g in bundle.score_groups}
    for fam in (
        "Structure confidence",
        "Interface & binding",
        "Sequence biophysics",
        "Novelty & similarity",
    ):
        grp = by_family.get(fam)
        if grp and grp.metrics:
            label, value, unit, source = grp.metrics[0]
            return f"{value}{(' ' + unit) if unit else ''}", label, source
    score = bundle.candidate.score
    if score is not None:
        val = _fmt(score)
        if val is not None:
            return val, "score", ""
    return "", "", ""


def _board_card(bundle: CandidateEvidenceBundle, row: CandidateBoardRow | None = None) -> str:
    """One candidate-first board row.

    The row remains a button with the legacy ``board-card`` / ``data-cand``
    hooks so the existing lazy dossier + viewer activation JS keeps working.
    """
    cand = bundle.candidate
    dom = _structure_dom_id(cand.candidate_id)
    status_label, status_cls = _STATUS_DISPLAY.get(bundle.status, (bundle.status, "other"))
    is_final = bundle.status == CandidateStatus.FINAL
    display_rank = (
        row.rank
        if row is not None
        else (bundle.display_rank if bundle.display_rank is not None else cand.rank)
    )
    if display_rank is not None:
        rank_cell = f'<span class="crow-rank-num">{display_rank}</span>'
    else:
        rank_cell = '<span class="crow-rank-num crow-rank-dash">&middot;</span>'
    row_status = row.recommendation_label if row else status_label
    verdict = row.short_verdict if row else _why_candidate(cand)
    chips = _saas_metric_chip_html(row.metric_chips[:7] if row else [])
    has_structure = bool(row.has_structure if row else cand.structure_path)
    has_downloads = bool(row.has_downloads if row else cand.artifacts)
    structure_note = "Structure" if has_structure else "No structure"
    download_note = "Downloads" if has_downloads else "No downloads"
    final_cls = " is-final" if is_final else ""
    return (
        f'<button type="button" class="board-card candidate-board-row candidate-board-card status-{status_cls}{final_cls}" '
        f'data-cand="{_attr(dom)}" role="tab" aria-selected="false">'
        f'<span class="candidate-card-top">'
        f'<span class="crow-rank">{rank_cell}</span>'
        f'<span class="candidate-card-title">'
        f'<span class="board-row-id" title="{_attr(cand.candidate_id)}">{_h(_short(cand.candidate_id, 32))}</span>'
        f'<span class="board-row-status status-badge status-badge-{status_cls}">{_h(row_status)}</span>'
        f"</span>"
        f"</span>"
        f'<span class="board-row-verdict">{_h(verdict)}</span>'
        f'<span class="board-row-metrics">{chips}</span>'
        f'<span class="board-row-footer">'
        f'<span class="candidate-assets">'
        f'<span class="asset-pill{" is-on" if has_structure else " is-missing"}">{_h(structure_note)}</span>'
        f'<span class="asset-pill{" is-on" if has_downloads else " is-missing"}">{_h(download_note)}</span>'
        f"</span>"
        f'<strong class="open-dossier-action">Open dossier</strong>'
        f"</span>"
        f"</button>"
    )


def _board_html(review: ReportReviewModel, board: CandidateBoardViewModel | None = None) -> str:
    rows_by_id = {row.candidate_id: row for row in board.rows} if board else {}
    groups: list[str] = []
    for group in review.board_groups:
        cards = "".join(
            _board_card(b, rows_by_id.get(b.candidate.candidate_id)) for b in group.bundles
        )
        # Incomplete / failed / rejected groups are marked ``collapsed`` by the
        # review layer: on a large retrieval/semantic run dozens of un-evaluated
        # rows would otherwise bury the primary (Final / Ranked) picks. Render
        # those as a native <details> the reader can expand on demand; keep the
        # evaluated groups always-open.
        if group.collapsed and group.bundles:
            groups.append(
                f'<details class="board-group board-group-collapsible" '
                f'data-status="{_attr(group.status)}">'
                f'<summary class="board-group-head">'
                f'<span class="board-group-title">{_h(_board_group_display_title(group.status, group.title, board))}</span>'
                f'<span class="board-group-count">{len(group.bundles)}</span></summary>'
                f'<div class="board-group-cards">{cards}</div>'
                f"</details>"
            )
        else:
            groups.append(
                f'<div class="board-group" data-status="{_attr(group.status)}">'
                f'<div class="board-group-head"><span class="board-group-title">{_h(_board_group_display_title(group.status, group.title, board))}</span>'
                f'<span class="board-group-count">{len(group.bundles)}</span></div>'
                f'<div class="board-group-cards">{cards}</div>'
                f"</div>"
            )
    return f'<div class="cand-list candidate-board">{"".join(groups)}</div>'


def _board_group_display_title(
    status: str,
    fallback: str,
    board: CandidateBoardViewModel | None = None,
) -> str:
    if status == CandidateStatus.FINAL:
        return "Recommended"
    if status == CandidateStatus.RANKED_ALTERNATIVE:
        return "Alternatives"
    if status == CandidateStatus.EVALUATED_NOT_SELECTED:
        if board is not None and not getattr(board, "has_objective_rank", False):
            return "Alternatives"
        return "Evaluated, Not Selected"
    if status == CandidateStatus.GENERATED_NOT_EVALUATED:
        return "Generated, Not Evaluated"
    if status == CandidateStatus.REFERENCE_GROUNDING:
        return "Reference Grounding"
    if status == CandidateStatus.FAILED_OR_REJECTED:
        return "Failed or Rejected"
    return fallback


def _detail_structure_tab(bundle: CandidateEvidenceBundle, *, active: bool) -> str:
    # Structure is the default landing tab whenever the candidate has a 3D
    # artifact: the render gets the full panel width as a hero. Structure
    # downloads are intentionally centralized in Cross-Candidate Analysis so
    # single-file reports do not repeat the same base64 payload in every view.
    cand = bundle.candidate
    chain_context = _chain_context_for(cand)
    body = _viewer_block(cand, chain_context)
    cls = "cand-tabpanel is-active" if active else "cand-tabpanel"
    return f'<div class="{cls}" data-tab="structure">{body}</div>'


def _detail_quality_tab(bundle: CandidateEvidenceBundle, *, active: bool) -> str:
    # Quality analysis combines the headline score bars with the sequence
    # track. Both are stacked full-width so there is no ragged dead space.
    # The selection rationale lives in the Pipeline & evidence tab.
    cand = bundle.candidate
    chain_context = _chain_context_for(cand)
    parts: list[str] = []
    score_summary = _score_bars(cand)
    if score_summary:
        parts.append(f'<div class="cand-quality-scores">{score_summary}</div>')
    parts.append(_candidate_sequence_block(cand, chain_context))
    body = "".join(part for part in parts if part)
    cls = "cand-tabpanel is-active" if active else "cand-tabpanel"
    return f'<div class="{cls}" data-tab="quality">{body}</div>'


def _evidence_step_chip(step: EvidenceStep) -> str:
    summary = (
        f'<span class="estep-summary">{_h(step.evidence_summary)}</span>'
        if step.evidence_summary
        else ""
    )
    return (
        f'<span class="estep estep-{_h(step.role)}" title="{_attr(step.node_id)}">'
        f'<span class="estep-tool">{_h(step.tool)}</span>{summary}'
        f"{_prov_chip(step.provenance)}</span>"
    )


def _detail_evidence_tab(bundle: CandidateEvidenceBundle) -> str:
    path = bundle.evidence_path
    parts: list[str] = []
    if path.producer is not None:
        # The producer step chip already carries its own provenance pill; a
        # second standalone ``producer_provenance`` chip here was redundant
        # (and rendered as an empty-looking floating block).
        parts.append(
            f'<div class="detail-block"><p class="detail-block-title">Producer</p>'
            f'<div class="estep-row">{_evidence_step_chip(path.producer)}</div></div>'
        )
    if path.evaluators:
        chips = "".join(_evidence_step_chip(s) for s in path.evaluators)
        parts.append(
            f'<div class="detail-block"><p class="detail-block-title">Evaluators</p>'
            f'<div class="estep-row">{chips}</div></div>'
        )
    tail: list[str] = []
    if path.ranker is not None:
        tail.append(_evidence_step_chip(path.ranker))
    if path.submission is not None:
        tail.append(_evidence_step_chip(path.submission))
    if tail:
        parts.append(
            f'<div class="detail-block"><p class="detail-block-title">Selection</p>'
            f'<div class="estep-row">{"".join(tail)}</div></div>'
        )
    if path.shared_pipeline:
        chips = "".join(
            f'<span class="pipe-step pipe-{_h(s.role)}">{_h(s.tool)}</span>'
            for s in path.shared_pipeline
        )
        parts.append(
            '<div class="detail-block"><p class="detail-block-title">Shared pipeline</p>'
            f'<p class="detail-note">Every candidate traverses this same DAG; per-candidate attribution is shown above.</p>'
            f'<div class="pipe-row">{chips}</div></div>'
        )
    body = "".join(parts) or '<p class="empty">No pipeline evidence recorded.</p>'
    return f'<div class="cand-tabpanel" data-tab="evidence">{body}</div>'


def _detail_scores_tab(bundle: CandidateEvidenceBundle) -> str:
    body = _score_family_grid(bundle)
    return f'<div class="cand-tabpanel" data-tab="scores">{body}</div>'


def _score_family_grid(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    groups = _ordered_score_groups_for_preset(bundle.score_groups, preset)
    cards: list[str] = []
    for group in groups:
        metric_count = len(group.metrics)
        rows = "".join(
            _score_family_metric_bar(group.family, label, value, unit, src)
            for label, value, unit, src in group.metrics
        )
        family_cls = _score_family_class(group.family)
        body_cls = "score-family-body"
        if metric_count >= 4:
            body_cls += " is-scrollable"
        cards.append(
            f'<div class="score-family {family_cls}"><div class="score-family-head">'
            f'<span class="score-family-name">{_h(group.family)}</span>{_prov_chip(group.provenance)}</div>'
            f'<div class="{body_cls}" data-metric-count="{metric_count}">{rows}</div></div>'
        )
    body = "".join(cards) or '<p class="empty">No scored metrics for this candidate.</p>'
    return f'<div class="score-grid">{body}</div>'


def _score_family_class(family: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", str(family).lower()).strip("-")
    return f"score-family-{key or 'other'}"


def _score_family_metric_bar(
    family: str, label: str, value: str, unit: str | None, source: str
) -> str:
    metric = (str(value), unit, source, Provenance.DERIVED)
    numeric = _metric_float(metric)
    percent = _score_family_bar_percent(label, numeric)
    missing_cls = " is-missing" if percent is None else ""
    categorical_cls = " is-categorical" if percent is None else ""
    width = 0.0 if percent is None else max(4.0, min(100.0, percent))
    display_value = f"{value}{(' ' + unit) if unit else ''}"
    value_markup = (
        f'<span class="metric-value">{_h(display_value)}</span>'
        if percent is not None
        else _categorical_metric_value_markup(display_value)
    )
    meter = (
        f'<div class="metric-bar-track" aria-hidden="true"><i style="width:{width:.0f}%"></i></div>'
        if percent is not None
        else ""
    )
    return (
        f'<div class="metric-bar-row {_score_family_class(family)}{missing_cls}{categorical_cls}">'
        '<div class="metric-bar-top">'
        f'<span class="metric-label">{_h(label)}'
        f"{f'<span class="metric-src">{_h(source)}</span>' if source else ''}</span>"
        f"{value_markup}"
        "</div>"
        f"{meter}"
        "</div>"
    )


def _categorical_metric_value_markup(value: str) -> str:
    items = [item.strip() for item in re.split(r"\s*,\s*", str(value or "")) if item.strip()]
    if not items:
        items = [str(value or "")]
    tokens = "".join(f'<span class="metric-token">{_h(item)}</span>' for item in items[:8])
    if len(items) > 8:
        tokens += f'<span class="metric-token metric-token-more">+{len(items) - 8}</span>'
    return f'<span class="metric-token-list">{tokens}</span>'


def _score_family_bar_percent(label: str, numeric: float | None) -> float | None:
    normalized = _metric_label_for_display(label)
    if normalized in {"QC", "QC filters"}:
        return None
    if numeric is None:
        return None
    direct = _snapshot_score_percent(normalized, numeric)
    if direct is not None:
        return direct
    if normalized == "Delta G":
        return max(8.0, min(100.0, abs(numeric) / 15.0 * 100.0))
    if normalized == "GRAVY":
        return max(4.0, min(100.0, (numeric + 2.0) / 4.0 * 100.0))
    if normalized == "Net charge":
        return max(4.0, min(100.0, (numeric + 5.0) / 10.0 * 100.0))
    if normalized in {"Isoelectric point", "pI"}:
        return max(4.0, min(100.0, numeric / 14.0 * 100.0))
    if numeric <= 1.5:
        return max(4.0, min(100.0, numeric * 100.0))
    return max(4.0, min(100.0, abs(numeric)))


def _ordered_score_groups_for_preset(
    groups: list[ScoreGroup], preset: WorkflowReviewPreset | None
) -> list[ScoreGroup]:
    if preset is None:
        return list(groups)
    order = {family: idx for idx, family in enumerate(preset.evidence_family_order)}
    return sorted(groups, key=lambda group: (order.get(group.family, 99), group.family))


def _detail_msa_tab(bundle: CandidateEvidenceBundle) -> str:
    return f'<div class="cand-tabpanel" data-tab="msa">{_msa_html(bundle.msa_blocks)}</div>'


def _detail_tools_tab(bundle: CandidateEvidenceBundle) -> str:
    blocks: list[str] = []
    for tool, payload in bundle.tool_evidence.items():
        blocks.append(
            f'<details class="tool-evidence"><summary>{_h(tool)}</summary>'
            f'<pre class="tool-evidence-json">{_h(json.dumps(payload, indent=2, ensure_ascii=False, default=str))}</pre>'
            f"</details>"
        )
    body = "".join(blocks) or '<p class="empty">No per-tool evidence recorded.</p>'
    return f'<div class="cand-tabpanel" data-tab="tools">{body}</div>'


def _dossier_risk_missing_grid(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    risks = _bundle_risk_notes(bundle)
    missing = _bundle_missing_evidence_labels(bundle, preset)
    risk_items = (
        "".join(f"<li>{_h(item)}</li>" for item in risks) or "<li>No recorded risk notes.</li>"
    )
    missing_items = (
        "".join(f"<li>{_h(item)}</li>" for item in missing)
        or "<li>No blocking missing evidence detected.</li>"
    )
    return (
        '<div class="dossier-alert-grid">'
        '<div class="dossier-alert dossier-risk"><h4>Risk Notes</h4>'
        f"<ul>{risk_items}</ul></div>"
        '<div class="dossier-alert dossier-missing"><h4>Missing Evidence</h4>'
        f"<ul>{missing_items}</ul></div>"
        "</div>"
    )


def _bundle_risk_notes(bundle: CandidateEvidenceBundle) -> list[str]:
    notes: list[str] = []
    for flag in bundle.candidate.qc_flags:
        notes.append(str(flag).replace("_", " "))
    for liability in bundle.candidate.liabilities:
        if isinstance(liability, dict):
            label = liability.get("label") or liability.get("type") or liability.get("name")
            if label:
                notes.append(str(label).replace("_", " "))
        elif liability:
            notes.append(str(liability).replace("_", " "))
    if bundle.status == CandidateStatus.FAILED_QC and not notes:
        notes.append("failed recorded QC checks")
    return _dedupe_text(notes)[:6]


def _bundle_missing_evidence_labels(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> list[str]:
    workflow_type = preset.workflow_type if preset else bundle.workflow_type
    families = {group.family for group in bundle.score_groups}
    missing: list[str] = []
    if workflow_type == WorkflowType.RETRIEVAL:
        if not (
            bundle.score_groups or bundle.artifacts or bundle.tool_evidence or bundle.msa_blocks
        ):
            missing.append("retrieved evidence or source coverage")
        return _dedupe_text(missing)
    if (
        workflow_type in {WorkflowType.BINDER, WorkflowType.DENOVO}
        and not bundle.candidate.structure_path
    ):
        missing.append("structure")
    if workflow_type == WorkflowType.BINDER and "Interface & binding" not in families:
        missing.append("interface and binding metrics")
    if (
        workflow_type in {WorkflowType.BINDER, WorkflowType.DENOVO}
        and "Structure confidence" not in families
    ):
        missing.append("structure confidence metrics")
    if "Sequence biophysics" not in families:
        missing.append("sequence biophysics metrics")
    if (
        workflow_type != WorkflowType.GENERIC
        and not bundle.msa_blocks
        and "Novelty & similarity" not in families
    ):
        missing.append("novelty or similarity evidence")
    return _dedupe_text(missing)


def _dedupe_text(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value or "").strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _bundle_metric(
    bundle: CandidateEvidenceBundle, target_label: str
) -> tuple[str, str | None, str, str] | None:
    target = _metric_label_for_display(target_label)
    for group in bundle.score_groups:
        for label, value, unit, source in group.metrics:
            if _metric_label_for_display(label) == target:
                return str(value), unit, source, group.provenance
    return None


def _metric_label_for_display(label: str) -> str:
    if label == "\u0394G":
        return "Delta G"
    return label


def _metric_float(metric: tuple[str, str | None, str, str] | None) -> float | None:
    if metric is None:
        return None
    try:
        return float(metric[0].replace(",", ""))
    except (TypeError, ValueError):
        return None


def _dossier_verdict(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    status_label, _status_cls = _STATUS_DISPLAY.get(bundle.status, (bundle.status, "other"))
    plddt = _metric_float(_bundle_metric(bundle, "pLDDT"))
    ipsae = _metric_float(_bundle_metric(bundle, "ipSAE"))
    iptm = _metric_float(_bundle_metric(bundle, "ipTM"))
    ptm = _metric_float(_bundle_metric(bundle, "pTM"))
    missing = _bundle_missing_evidence_labels(bundle, preset)
    if bundle.status == CandidateStatus.FINAL:
        lead = "Recommended final candidate"
    elif bundle.status == CandidateStatus.RANKED_ALTERNATIVE:
        lead = "Alternative"
    elif bundle.status == CandidateStatus.EVALUATED_NOT_SELECTED:
        lead = "Evaluated but not selected candidate"
    elif bundle.status == CandidateStatus.GENERATED_NOT_EVALUATED:
        lead = "Generated candidate without recorded downstream evaluation"
    elif bundle.status == CandidateStatus.REFERENCE_GROUNDING:
        lead = "Reference grounding record"
    elif bundle.status == CandidateStatus.FAILED_OR_REJECTED:
        lead = "Failed or rejected candidate"
    else:
        lead = f"{status_label} candidate"
    workflow_type = preset.workflow_type if preset else bundle.workflow_type
    if ipsae is not None:
        strong_interface = ipsae >= 0.50
        usable_interface = ipsae >= 0.30
    elif iptm is not None:
        strong_interface = iptm >= 0.70
        usable_interface = iptm >= 0.50
    else:
        strong_interface = False
        usable_interface = False
    if workflow_type == WorkflowType.BINDER and strong_interface:
        lead += " with strong recorded binding-interface evidence"
    elif workflow_type == WorkflowType.BINDER and usable_interface:
        lead += " with usable recorded binding-interface evidence"
    elif workflow_type == WorkflowType.RETRIEVAL and (
        bundle.score_groups or bundle.artifacts or bundle.msa_blocks
    ):
        lead += " with recorded retrieval evidence"
    elif workflow_type == WorkflowType.RETRIEVAL and bundle.status == CandidateStatus.FINAL:
        lead += " with recorded final retrieval-hit status"
    elif plddt is not None and plddt >= 90:
        lead += " with high predicted structure confidence"
    elif plddt is not None and plddt >= 70:
        lead += " with moderate predicted structure confidence"
    elif iptm is not None and iptm >= 0.70:
        lead += " with strong recorded interface confidence"
    elif ptm is not None and ptm >= 0.75:
        lead += " with strong global fold confidence"
    if missing:
        lead += f"; missing {missing[0]}"
    return lead + "."


def _dossier_decision_points(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> list[str]:
    metric_order = (
        preset.metric_priority
        if preset
        else ["pLDDT", "pTM", "ipTM", "Delta G", "Affinity", "GRAVY", "Net charge"]
    )
    points: list[str] = []
    for label in metric_order:
        if label == "QC":
            continue
        metric = _bundle_metric(bundle, label)
        if metric is None:
            continue
        value, unit, source, _prov = metric
        unit_text = f" {unit}" if unit else ""
        source_text = f" from {source}" if source else ""
        points.append(f"{label} {value}{unit_text}{source_text}")
        if len(points) >= 3:
            break
    risks = _bundle_risk_notes(bundle)
    if risks:
        points.append(f"Risk to review: {risks[0]}")
    elif bundle.status != CandidateStatus.FAILED_QC:
        points.append("QC passed; no major recorded liabilities.")
    missing = _bundle_missing_evidence_labels(bundle, preset)
    if missing:
        points.append(f"Missing evidence: {missing[0]}")
    return points[:5]


def _dossier_summary_panel(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    points = "".join(f"<li>{_h(point)}</li>" for point in _dossier_decision_points(bundle, preset))
    return (
        '<div class="dossier-summary-panel">'
        '<div><p class="saas-card-kicker">Why this candidate?</p>'
        f'<p class="dossier-verdict">{_h(_dossier_verdict(bundle, preset))}</p></div>'
        f'<ul class="dossier-decision-points">{points}</ul>'
        f"{_prov_chip(Provenance.DERIVED, title='Summary derived from recorded candidate status, metrics, risks, and missing evidence.')}"
        "</div>"
    )


def _detail_template(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    cand = bundle.candidate
    dom = _structure_dom_id(cand.candidate_id)
    has_structure = bool((cand.structure_path or "").strip())

    status_label, status_cls = _STATUS_DISPLAY.get(bundle.status, (bundle.status, "other"))
    rank_note = ""
    head = (
        f'<div class="cand-detail-head status-{status_cls}">'
        f'<div class="cand-detail-id"><span class="cand-detail-eyebrow">{_h(status_label)}</span>'
        f'<h3 title="{_attr(cand.candidate_id)}">{_h(_short(cand.candidate_id, 40))}</h3>'
        f"{rank_note}</div>"
        f"</div>"
    )
    chain_context = _chain_context_for(cand)
    if has_structure:
        primary_media = (
            f'<div class="dossier-structure-primary">{_viewer_block(cand, chain_context)}</div>'
        )
    else:
        primary_media = (
            '<div class="dossier-structure-primary dossier-no-structure">'
            '<div class="viewer-empty">No structure artifact linked for this candidate.</div>'
            f"{_candidate_sequence_block(cand, chain_context)}"
            "</div>"
        )
    evidence_primary = (
        '<aside class="dossier-evidence-primary">'
        "<h4>Evidence Families</h4>"
        f"{_score_family_grid(bundle, preset)}"
        "</aside>"
    )
    primary = f'<div class="candidate-dossier-primary">{primary_media}{evidence_primary}</div>'
    return (
        f'<template class="cand-detail-tpl" data-cand="{_attr(dom)}">'
        f'<div class="cand-detail-inner">{head}'
        f"{primary}"
        f"{_dossier_risk_missing_grid(bundle, preset)}"
        f"</div></template>"
    )


def _review_console_section(
    review: ReportReviewModel,
    board_vm: CandidateBoardViewModel | None = None,
    preset: WorkflowReviewPreset | None = None,
    display_notice: str = "",
) -> str:
    base = review.base
    final_submission = base.final_submission or {}
    no_pick = final_submission.get("outcome") == "no_acceptable_candidate"
    if not review.bundles and not no_pick:
        return ""

    banner = ""
    if no_pick:
        reason = str(
            final_submission.get("reasoning")
            or "The agent completed the search without selecting a candidate."
        )
        banner = _no_acceptable_candidate_banner(reason)

    counts: dict[str, int] = {}
    for bundle in review.bundles:
        counts[bundle.status] = counts.get(bundle.status, 0) + 1
    meta_bits: list[str] = []
    for status, (label, _cls) in _STATUS_DISPLAY.items():
        if counts.get(status):
            meta_bits.append(f"{counts[status]} {label.lower()}")
    meta = " · ".join(meta_bits) or f"{len(review.bundles)} candidates"
    board = _board_html(review, board_vm)
    templates = "".join(_detail_template(b, preset) for b in review.bundles)
    ranking_caption_html = _candidate_ranking_basis_html(board_vm)
    override_note = getattr(board_vm, "objective_override_note", "") if board_vm is not None else ""
    override_note_html = (
        f'<p class="saas-section-subtitle candidate-ranking-caption">{_h(override_note)}</p>'
        if override_note
        else ""
    )

    console_body = (
        f"""<div class="console candidate-review-board-layout">
    <div class="console-board candidate-review-board" role="tablist" aria-label="Candidate board">{board}</div>
    <div class="console-detail" id="cand-detail-host" aria-live="polite"></div>
  </div>
  {templates}"""
        if review.bundles
        else ""
    )
    return f"""<section id="candidates" class="card card-console">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Designs</p>
      <h2 class="card-title">Candidate review</h2>
      {ranking_caption_html}
      {override_note_html}
    </div>
    <span class="card-meta">{_h(meta)}</span>
  </header>
  {display_notice}
  {banner}
  {console_body}
</section>"""


def _candidate_ranking_basis_html(board_vm: CandidateBoardViewModel | None) -> str:
    if board_vm is None:
        return ""
    basis = getattr(board_vm, "ranking_basis", None)
    if basis is None:
        caption = str(getattr(board_vm, "ranking_basis_caption", "") or "").strip()
        return (
            f'<p class="saas-section-subtitle candidate-ranking-caption">{_h(caption)}</p>'
            if caption
            else ""
        )
    fallback = str(getattr(basis, "fallback_text", "") or "").strip()
    if fallback:
        return f'<p class="saas-section-subtitle candidate-ranking-caption">{_h(fallback)}</p>'
    objective = str(getattr(basis, "objective_name", "") or "objective").strip()
    method = str(getattr(basis, "method", "") or "objective_rank").strip()
    formula = str(getattr(basis, "score_formula", "") or "recorded formula").strip()
    direction = str(getattr(basis, "score_direction", "") or "higher").strip()
    gates = [str(gate).strip() for gate in getattr(basis, "gates", []) if str(gate).strip()]
    gate_text = "; ".join(gates) if gates else "No gates recorded"
    label = (
        f"Ranking basis: {method} for {objective}. Gates: {gate_text}. "
        f"Score: {formula}; {direction} is better."
    )
    return f"""<div class="candidate-ranking-panel" aria-label="{_attr(label)}">
      <div class="candidate-ranking-head">
        <span class="candidate-ranking-label">Ranking basis</span>
        <span class="candidate-ranking-meta"><strong>{_h(method)}</strong><span>{_h(objective)}</span><span>{_h(direction)} is better</span></span>
      </div>
      <p class="candidate-ranking-line"><b>Gates</b><span>{_h(gate_text)}</span></p>
      <p class="candidate-ranking-line"><b>Score</b><code>{_h(formula)}</code></p>
    </div>"""


def _no_acceptable_candidate_banner(reason: str) -> str:
    summary = _submission_reason_summary(reason)
    bullets = _submission_reason_bullets(reason)
    bullets_html = (
        '<ul class="console-banner-list">'
        + "".join(f"<li>{_h(item)}</li>" for item in bullets)
        + "</ul>"
        if bullets
        else ""
    )
    details = ""
    if reason.strip() and reason.strip() != summary:
        details = (
            '<details class="console-banner-details">'
            "<summary>Full recorded submission reasoning</summary>"
            f"<p>{_h(reason.strip())}</p>"
            "</details>"
        )
    return (
        '<div class="console-banner">'
        "<strong>No acceptable final candidate was submitted.</strong>"
        f"<p>{_h(summary)}</p>"
        f"{bullets_html}"
        f"{details}"
        "</div>"
    )


def _submission_reason_summary(reason: str) -> str:
    text = _normalize_reason_text(reason)
    if not text:
        return "The search completed without a candidate that met the recorded acceptance criteria."
    for heading in ("CONCLUSION", "RESULT", "KEY FINDING"):
        section = _reason_section(text, heading)
        if section:
            return _clip_reason_sentence(section, 260)
    return _clip_reason_sentence(text, 260)


def _submission_reason_bullets(reason: str) -> list[str]:
    text = _normalize_reason_text(reason)
    bullets: list[str] = []
    for heading, label in (
        ("OBJECTIVE", "Objective"),
        ("KEY FINDING", "Key finding"),
        ("RESULT", "Result"),
    ):
        section = _reason_section(text, heading)
        if section:
            bullets.append(f"{label}: {_clip_reason_sentence(section, 180)}")
    return bullets[:3]


def _normalize_reason_text(reason: str) -> str:
    return "\n".join(line.strip() for line in str(reason or "").splitlines()).strip()


def _reason_section(text: str, heading: str) -> str:
    pattern = re.compile(
        rf"(?is)\b{re.escape(heading)}(?:\s*&\s*[A-Z ]+)?\s*[-:]?\s*(.+?)(?=\n\s*\n\s*[A-Z][A-Z &()0-9/-]{{3,}}\s*[-:]|\Z)"
    )
    match = pattern.search(text)
    return " ".join(match.group(1).split()) if match else ""


def _clip_reason_sentence(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return ""
    sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", normalized)
    candidate = sentence_match.group(1) if sentence_match else normalized
    if len(candidate) <= limit:
        return candidate
    return candidate[: max(0, limit - 1)].rstrip(" ,;:") + "..."


def _saas_overview_section(vm: OverviewViewModel) -> str:
    counts = "".join(
        f'<div class="saas-stat">'
        f'<span class="saas-stat-value">{_h(count.value)}</span>'
        f'<span class="saas-stat-label">{_h(count.label)}</span>'
        f'<span class="saas-stat-detail">{_h(count.detail)}</span>'
        f"{_prov_chip(count.provenance)}"
        f"</div>"
        for count in vm.counts
    )
    caveat = (
        f'<div class="saas-card saas-caveat"><p class="saas-card-kicker">Main Caveat</p>'
        f"<p>{_h(vm.main_caveat)}</p>{_prov_chip(vm.caveat_provenance)}</div>"
        if vm.main_caveat
        else '<div class="saas-card saas-caveat saas-caveat-ok"><p class="saas-card-kicker">Main Caveat</p><p>No blocking caveat recorded.</p></div>'
    )
    count_warning = (
        f'<p class="saas-count-warning">{_h(vm.count_warning)}</p>' if vm.count_warning else ""
    )
    return f"""<section id="overview" class="card saas-section saas-overview">
  <header class="card-header saas-title-row">
    <div>
      <p class="card-eyebrow">Executive Summary</p>
      <h2 class="card-title">Design review summary</h2>
      <p class="saas-section-subtitle">High-level overview of the design run and key outcomes.</p>
    </div>
  </header>
  <div class="saas-overview-grid">
    {_overview_summary_card("User Query", vm.design_objective, "quote")}
    {_overview_summary_card("Agent Objective", vm.agent_objective, "spark")}
    <div class="saas-card saas-stats">{counts}</div>
    {caveat}
  </div>
  {count_warning}
</section>"""


def _overview_summary_card(title: str, body: str, icon_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-") or "summary"
    return (
        f'<div class="saas-card saas-icon-card overview-summary-card overview-summary-card-{_attr(slug)}">'
        '<div class="saas-icon-card-head">'
        f'<span class="overview-card-icon">{_icon_svg(icon_name)}</span>'
        f'<p class="saas-card-kicker">{_h(title)}</p>'
        "</div>"
        f"{_overview_summary_body_html(title, body)}"
        "</div>"
    )


def _overview_summary_body_html(title: str, body: str) -> str:
    text = re.sub(r"\s+", " ", str(body or "").strip())
    if not text:
        return '<div class="overview-summary-body"><p>No text recorded.</p></div>'
    if title.strip().lower() == "agent objective":
        objective_sections = _overview_objective_sections(text)
        if objective_sections:
            blocks = "".join(
                f'<div class="overview-objective-block"><b>{_h(label)}</b><p>{_h(value)}</p></div>'
                for label, value in objective_sections
                if value
            )
            return f'<div class="overview-summary-body overview-objective-body">{blocks}</div>'
    segments = _overview_summary_segments(text)
    if len(segments) <= 1:
        return f'<div class="overview-summary-body"><p>{_h(text)}</p></div>'
    items = "".join(f"<li>{_h(segment)}</li>" for segment in segments)
    return (
        f'<div class="overview-summary-body"><ul class="overview-summary-list">{items}</ul></div>'
    )


def _overview_objective_sections(text: str) -> list[tuple[str, str]]:
    objective, criteria = _split_overview_labeled_pair(text)
    if not objective and not criteria:
        return []
    sections: list[tuple[str, str]] = []
    if objective:
        sections.append(("Objective", objective))
    if criteria:
        sections.append(("Acceptance criteria", criteria))
    return sections


def _split_overview_labeled_pair(text: str) -> tuple[str, str]:
    labels = [
        ("Objective", re.compile(r"(?i)(?<!\w)objective\s*:\s*")),
        (
            "Acceptance criteria",
            re.compile(
                r"(?i)(?<!\w)(acceptance criteria|acceptance|thresholds/criteria|"
                r"thresholds|criteria)\s*:\s*"
            ),
        ),
    ]
    matches: list[tuple[str, re.Match[str]]] = []
    for label, pattern in labels:
        match = pattern.search(text)
        if match:
            matches.append((label, match))
    if not matches:
        return "", ""
    matches.sort(key=lambda item: item[1].start())
    values: dict[str, str] = {}
    for idx, (label, match) in enumerate(matches):
        end = matches[idx + 1][1].start() if idx + 1 < len(matches) else len(text)
        values[label] = text[match.end() : end].strip(" ;,")
    return values.get("Objective", ""), values.get("Acceptance criteria", "")


def _overview_summary_segments(text: str) -> list[str]:
    labeled_parts = _split_labeled_summary_segments(text)
    if len(labeled_parts) > 1:
        return labeled_parts
    sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentence_parts) > 1:
        return sentence_parts
    clause_parts = [
        part.strip(" ;,") for part in re.split(r"\s*;\s+|\s+-\s+", text) if part.strip(" ;,")
    ]
    if len(clause_parts) > 1:
        return clause_parts
    return [text]


def _split_labeled_summary_segments(text: str) -> list[str]:
    marker = re.compile(
        r"(?i)(?<!\w)("
        r"objective|acceptance|accept if|thresholds/criteria|thresholds|criteria|"
        r"approach|result|key finding|constraints|evidence|caveat|"
        r"wt folds confidently only with msa|per-position single-mutant rosetta"
        r")\s*:?"
    )
    matches = list(marker.finditer(text))
    if not matches:
        return [text]
    parts: list[str] = []
    prefix = text[: matches[0].start()].strip(" ;,")
    if prefix:
        parts.append(prefix)
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[start:end].strip(" ;,")
        if segment:
            parts.append(segment)
    return parts or [text]


def _report_brand_mark() -> str:
    logo_path = Path(__file__).resolve().parents[3] / "assets" / "vpa_text_logo.png"
    embedded = _data_uri(logo_path)
    if embedded is None:
        return '<span class="brand-fallback">VibeProteinAgent</span>'
    src, _ = embedded
    return (
        '<span class="brand-lockup" role="img" aria-label="VibeProteinAgent">'
        f'<img src="{_attr(src)}" alt="" aria-hidden="true" loading="eager" decoding="async">'
        "</span>"
    )


def _saas_agent_rationale_section(vm: AgentRationaleViewModel) -> str:
    decisions = (
        "".join(_agent_decision_item_html(item) for item in vm.major_decisions)
        or '<li><span class="decision-label">Decision rationale</span><p>No structured agent decisions were recorded.</p></li>'
    )
    stages = (
        "".join(f"<span>{_h(stage)}</span>" for stage in vm.stage_plan)
        or "<span>No stage plan recorded</span>"
    )
    query = vm.user_query or "No agent user query was recorded for this run."
    return f"""<section id="agent-rationale" class="card saas-section agent-rationale-section">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Agent Rationale</p>
      <h2 class="card-title">Agent interpretation and decisions</h2>
      <p class="saas-section-subtitle">Recorded intent, review framing, plan, and decision rationale.</p>
    </div>
    <span class="card-meta">{_h(vm.tool_call_count)} tool calls</span>
  </header>
  <div class="saas-rationale-grid">
    <div class="saas-card agent-rationale-card">
      <p class="saas-card-kicker">User Intent</p>
      <div class="agent-rationale-card-body"><p>{_h(query)}</p></div>
    </div>
    <div class="saas-card agent-rationale-card">
      <p class="saas-card-kicker">Review Framing</p>
      <div class="agent-rationale-card-body"><p>{_h(vm.workflow_selection or "Review framing was derived from recorded workflow evidence.")}</p></div>
    </div>
    <div class="saas-card saas-stage-card agent-rationale-card">
      <p class="saas-card-kicker">Workflow Stages</p>
      <div class="agent-rationale-card-body"><div class="saas-stage-chips">{stages}</div></div>
    </div>
    <div class="saas-card saas-decisions agent-rationale-card">
      <p class="saas-card-kicker">Major Decisions</p>
      <div class="agent-rationale-card-body"><ul>{decisions}</ul></div>
    </div>
  </div>
</section>"""


def _agent_decision_item_html(item: AgentRationaleItem) -> str:
    return (
        "<li>"
        f'<span class="decision-label">{_h(item.label)}</span>'
        f"{_agent_decision_value_html(item.label, item.value)}"
        "</li>"
    )


def _agent_decision_value_html(label: str, value: str) -> str:
    text = " ".join(str(value or "").split())
    sections = _structured_decision_sections(text)
    if len(sections) > 1:
        blocks = "".join(
            '<div class="decision-section">'
            f"<b>{_h(label)}</b>"
            f"<p>{_h(_capitalize_first_alpha(body))}</p>"
            "</div>"
            for label, body in sections
            if body
        )
        return f'<div class="decision-text decision-text-structured">{blocks}</div>'
    if label.strip().lower() == "candidate justification":
        return f'<div class="decision-text decision-justification-card"><p>{_h(text)}</p></div>'
    if _is_rationale_decision_label(label):
        return f'<div class="decision-text decision-rationale-card"><p>{_h(text)}</p></div>'
    return f'<p class="decision-text">{_h(text)}</p>'


def _is_rationale_decision_label(label: str) -> bool:
    normalized = str(label or "").strip().lower()
    return "rationale" in normalized or normalized in {"decision", "decision rationale"}


def _capitalize_first_alpha(value: str) -> str:
    text = str(value or "")
    for idx, char in enumerate(text):
        if char.isalpha():
            return f"{text[:idx]}{char.upper()}{text[idx + 1 :]}"
    return text


def _structured_decision_sections(text: str) -> list[tuple[str, str]]:
    heading = _decision_heading_pattern()
    matches = list(heading.finditer(text))
    if len(matches) >= 2:
        return _sections_from_decision_heading_matches(text, matches)
    return _sections_from_decision_sentence_cues(text, matches)


def _decision_heading_pattern() -> re.Pattern[str]:
    return re.compile(
        r"(?i)(?:^|(?<=[.;!?]\s))("
        r"OBJECTIVE|APPROACH|RESULT|KEY FINDING|ACCEPTANCE(?: CRITERIA)?|"
        r"ACCEPTANCE GATES|PIPELINE|THRESHOLDS/CRITERIA|THRESHOLDS|"
        r"CRITERIA|EVIDENCE|FINAL DECISION|BACKBONE VALIDATION|"
        r"VALIDATION|DESIGN|RANKING|SELECTED|VALIDATED ALTERNATIVES|"
        r"ALTERNATIVES|CAVEATS|CAVEAT|LIMITATION|WHY SELECTED"
        r")(?:\s*\([^:]{1,120}\))?\s*:"
    )


def _sections_from_decision_heading_matches(
    text: str,
    matches: list[re.Match[str]],
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        label = _decision_heading_label(match.group(1))
        body = text[start:end].strip(" .;:-")
        if body:
            sections.append((label, body))
    return sections


def _sections_from_decision_sentence_cues(
    text: str,
    matches: list[re.Match[str]],
) -> list[tuple[str, str]]:
    sentences = _decision_sentences(text)
    if len(sentences) < 2:
        return []
    sections: list[tuple[str, str]] = []
    active_label = ""
    active_parts: list[str] = []

    def flush() -> None:
        nonlocal active_label, active_parts
        body = " ".join(part.strip() for part in active_parts if part.strip()).strip()
        if active_label and body:
            sections.append((active_label, body))
        active_label = ""
        active_parts = []

    for idx, sentence in enumerate(sentences):
        label, body = _decision_sentence_label_and_body(sentence, first_sentence=idx == 0)
        if not label:
            if active_parts:
                active_parts.append(sentence)
            continue
        if label != active_label:
            flush()
            active_label = label
        active_parts.append(body)
    flush()

    labels = {label for label, _body in sections}
    if len(sections) < 2 or not (labels & {"Objective", "Selection", "Result", "Ranking"}):
        return []
    if matches and sections[0][0] != _decision_heading_label(matches[0].group(1)):
        return []
    return sections


def _decision_sentences(text: str) -> list[str]:
    compact = " ".join(str(text or "").split())
    if not compact:
        return []
    return [
        part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Za-z_])", compact) if part.strip()
    ]


def _decision_sentence_label_and_body(
    sentence: str,
    *,
    first_sentence: bool,
) -> tuple[str, str]:
    stripped = sentence.strip()
    heading_match = _decision_heading_pattern().match(stripped)
    if heading_match:
        return (
            _decision_heading_label(heading_match.group(1)),
            stripped[heading_match.end() :].strip(" .;:-"),
        )
    lowered = stripped.lower()
    if first_sentence and len(stripped) > 80:
        return "Objective", stripped.strip(" .;:-")
    if lowered.startswith(("explored ", "tested ", "generated ", "screened ", "evaluated ")):
        return "Exploration", stripped
    if lowered.startswith(("ranked ", "ranking ", "objective_rank ", "score ", "scored ")):
        return "Ranking", stripped
    if lowered.startswith(("folded ", "validated ", "tmalign ", "tm-align ", "compared ")):
        return "Validation", stripped
    if re.match(r"(?i)^[A-Za-z0-9_.-]+ selected\b", stripped) or lowered.startswith(
        ("selected ", "final selection ", "submitted ")
    ):
        return "Selection", stripped
    if lowered.startswith(("runner-up", "alternative", "validated alternative")) or re.match(
        r"(?i)^[A-Za-z0-9_.-]+ is the .*runner-up\b",
        stripped,
    ):
        return "Alternative", stripped
    if lowered.startswith(("caveat", "limitation", "however ", "note ")):
        return "Caveat", stripped
    return "", stripped


def _decision_heading_label(value: str) -> str:
    normalized = " ".join(str(value or "").replace("/", " / ").split()).lower()
    labels = {
        "objective": "Objective",
        "approach": "Approach",
        "result": "Result",
        "key finding": "Key Finding",
        "acceptance": "Acceptance",
        "acceptance criteria": "Acceptance Criteria",
        "acceptance gates": "Acceptance Gates",
        "pipeline": "Pipeline",
        "thresholds / criteria": "Thresholds / Criteria",
        "thresholds": "Thresholds",
        "criteria": "Criteria",
        "evidence": "Evidence",
        "final decision": "Final Decision",
        "backbone validation": "Backbone Validation",
        "validation": "Validation",
        "design": "Design",
        "ranking": "Ranking",
        "selected": "Selected",
        "validated alternatives": "Validated Alternatives",
        "alternatives": "Alternatives",
        "caveats": "Caveats",
        "caveat": "Caveat",
        "limitation": "Limitation",
        "why selected": "Why Selected",
    }
    return labels.get(normalized, normalized.title())


def _saas_candidate_comparison_section(
    board: CandidateBoardViewModel,
    preset: WorkflowReviewPreset | None = None,
    review: ReportReviewModel | None = None,
) -> str:
    metric_cols = _comparison_metric_columns(preset)
    bundles_by_id = (
        {bundle.candidate.candidate_id: bundle for bundle in review.bundles}
        if review is not None
        else {}
    )
    rows = "".join(
        _comparison_row(row, metric_cols, bundles_by_id.get(row.candidate_id)) for row in board.rows
    )
    if not rows:
        rows = (
            '<tr><td colspan="12" class="empty">No candidates available for comparison.</td></tr>'
        )
    status_counts: dict[str, int] = {}
    for row in board.rows:
        status_counts[row.recommendation_label] = status_counts.get(row.recommendation_label, 0) + 1
    summary = "".join(
        f'<div class="comparison-summary-card"><span>{_h(label)}</span><strong>{_h(count)}</strong></div>'
        for label, count in _ordered_status_counts(status_counts)
    )
    headers = "".join(f"<th>{_h(label)}</th>" for label in metric_cols)
    return f"""<section id="candidate-comparison" class="card saas-section">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Candidate Comparison</p>
      <h2 class="card-title">Decision matrix</h2>
      <p class="saas-section-subtitle">{_h(preset.comparison_note if preset else "Compare candidates by recommendation status, core metrics, evidence gaps, and actionability. Missing values are shown explicitly.")}</p>
    </div>
  </header>
  <div class="comparison-summary-row">{summary}</div>
  <div class="saas-table-wrap comparison-matrix-wrap">
    <table class="saas-table candidate-comparison-matrix">
      <thead><tr><th>Rank</th><th>Candidate</th><th>Status</th><th>Verdict</th>{headers}<th>Assets</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
  <p class="comparison-note">{_h(preset.scientific_question if preset else "Use this as a selection panel view.")} Diversity and clustering are only shown when supported by recorded evidence.</p>
</section>"""


def _comparison_metric_columns(preset: WorkflowReviewPreset | None) -> list[str]:
    defaults = ["pLDDT", "pTM", "ipTM", "Delta G", "Affinity", "GRAVY", "Net charge", "QC"]
    if preset is None:
        return defaults
    columns: list[str] = []
    for label in preset.metric_priority:
        shown = _metric_label_for_display(label)
        if shown not in columns:
            columns.append(shown)
        if len(columns) >= 8:
            break
    for label in defaults:
        if label not in columns and len(columns) < 8:
            columns.append(label)
    return columns


def _ordered_status_counts(counts: dict[str, int]) -> list[tuple[str, int]]:
    order = {
        "Recommended": 0,
        "Alternative": 1,
        "Ranked Alternative": 1,
        "Evaluated": 2,
        "Generated Only": 3,
        "Reference": 4,
        "Failed/Rejected": 5,
    }
    return sorted(counts.items(), key=lambda item: (order.get(item[0], 99), item[0]))


def _comparison_row(
    row: CandidateBoardRow,
    metric_cols: list[str],
    bundle: CandidateEvidenceBundle | None = None,
) -> str:
    metric_map = {_metric_label_for_display(chip.label): chip for chip in row.metric_chips}
    cells = "".join(
        _comparison_metric_cell(metric_map.get(label), label=label) for label in metric_cols
    )
    rank = row.rank if row.rank is not None else "-"
    asset_html = _comparison_asset_actions(row, bundle)
    candidate_cell = (
        f'<span class="comparison-candidate-id">{_h(row.candidate_id)}</span>'
        f'<span class="comparison-candidate-sub">{_h(row.recommendation_label)}</span>'
    )
    return (
        f'<tr class="comparison-row status-{_attr(row.status)}">'
        f'<td><span class="comparison-rank">{_h(rank)}</span></td>'
        f"<td>{candidate_cell}</td>"
        f'<td><span class="status-badge">{_h(row.recommendation_label)}</span></td>'
        f'<td class="comparison-verdict">{_h(row.short_verdict)}</td>'
        f'{cells}<td><span class="comparison-assets">{asset_html}</span></td></tr>'
    )


def _comparison_asset_actions(
    row: CandidateBoardRow, bundle: CandidateEvidenceBundle | None
) -> str:
    candidate = bundle.candidate if bundle is not None else None
    actions: list[str] = []
    if row.has_structure and candidate is not None and candidate.structure_path:
        target = _structure_dom_id(candidate.candidate_id)
        actions.append(
            f'<a class="asset-pill is-on asset-action-link" href="#candidates" '
            f'data-open-cand="{_attr(target)}" data-scroll-target="{_attr(target)}" '
            f'title="Open the candidate dossier structure viewer">Structure</a>'
        )
    else:
        actions.append(
            '<span class="asset-pill is-missing" aria-disabled="true">No structure</span>'
        )

    if row.has_downloads:
        actions.append(
            '<span class="asset-pill is-on" '
            'title="Download links are centralized in Cross-Candidate Analysis">'
            "Downloads below</span>"
        )
    else:
        actions.append(
            '<span class="asset-pill is-missing" aria-disabled="true">No downloads</span>'
        )
    return "".join(actions)


def _comparison_metric_cell(chip: Any | None, *, label: str = "") -> str:
    if chip is None:
        missing_title = _comparison_missing_metric_title(label)
        return (
            '<td class="metric-cell-missing">'
            f'<span class="metric-missing" title="{_attr(missing_title)}">—</span>'
            "</td>"
        )
    unit = f" {chip.unit}" if chip.unit else ""
    title = _comparison_recorded_metric_title(label, chip)
    return (
        f'<td><span class="matrix-chip" title="{_attr(title)}">'
        f"{_h(chip.value)}{_h(unit)}</span></td>"
    )


def _comparison_missing_metric_title(label: str) -> str:
    if _metric_label_for_display(label) in {"ddG", "Mean ddG"}:
        return "No paired ddG"
    return "Metric not recorded"


def _comparison_recorded_metric_title(label: str, chip: Any) -> str:
    title = chip.source or chip.family or chip.provenance
    if _metric_label_for_display(label) == "Rosetta total score":
        suffix = "absolute Rosetta total score; compare only within the same protocol"
        return f"{title} · {suffix}" if title else suffix
    return title


def _structure_comparison_bundles(
    review: ReportReviewModel,
    *,
    limit: int | None = None,
) -> list[CandidateEvidenceBundle]:
    bundles: list[CandidateEvidenceBundle] = []
    seen: set[str] = set()
    for bundle in _review_ordered_bundles(review):
        candidate = bundle.candidate
        if not candidate.structure_path:
            continue
        if not Path(candidate.structure_path).is_file():
            continue
        key = candidate.candidate_id or str(candidate.structure_path)
        if key in seen:
            continue
        seen.add(key)
        bundles.append(bundle)
        if limit is not None and len(bundles) >= limit:
            break
    return bundles


@dataclass(frozen=True)
class _DisplayLimit:
    requested: int
    total: int
    displayed: int
    final_count: int

    @property
    def truncated(self) -> bool:
        return self.displayed < self.total


def _display_limited_review(
    review: ReportReviewModel,
    top_k: int,
) -> tuple[ReportReviewModel, _DisplayLimit]:
    requested = max(1, int(top_k or 1))
    ordered = _review_ordered_bundles(review)
    total = len(ordered)
    selected: list[CandidateEvidenceBundle] = []
    seen: set[str] = set()

    def add(bundle: CandidateEvidenceBundle) -> None:
        key = bundle.candidate.candidate_id or str(id(bundle))
        if key in seen:
            return
        seen.add(key)
        selected.append(bundle)

    for bundle in ordered:
        if bundle.status == CandidateStatus.FINAL:
            add(bundle)

    for bundle in ordered:
        if bundle.status == CandidateStatus.FINAL:
            continue
        if len(selected) >= requested:
            break
        add(bundle)

    selected_ids = {id(bundle) for bundle in selected}
    groups: list[BoardGroup] = []
    for group in review.board_groups:
        bundles = [bundle for bundle in group.bundles if id(bundle) in selected_ids]
        if bundles:
            groups.append(replace(group, bundles=bundles))
    limited = replace(review, board_groups=groups, bundles=selected)
    meta = _DisplayLimit(
        requested=requested,
        total=total,
        displayed=len(selected),
        final_count=sum(1 for bundle in selected if bundle.status == CandidateStatus.FINAL),
    )
    return limited, meta


def _display_limited_board(
    board: CandidateBoardViewModel,
    review: ReportReviewModel,
) -> CandidateBoardViewModel:
    order = [bundle.candidate.candidate_id for bundle in _review_ordered_bundles(review)]
    allowed = set(order)
    by_id = {row.candidate_id: row for row in board.rows if row.candidate_id in allowed}
    rows = [by_id[candidate_id] for candidate_id in order if candidate_id in by_id]
    return replace(
        board,
        rows=rows,
        total_count=len(rows),
        recommended_count=sum(1 for row in rows if row.status == CandidateStatus.FINAL),
    )


def _display_limited_dossiers(
    dossiers: list[CandidateDossierViewModel],
    review: ReportReviewModel,
) -> list[CandidateDossierViewModel]:
    order = [bundle.candidate.candidate_id for bundle in _review_ordered_bundles(review)]
    by_id = {dossier.candidate_id: dossier for dossier in dossiers}
    return [by_id[candidate_id] for candidate_id in order if candidate_id in by_id]


def _display_limit_notice(meta: _DisplayLimit) -> str:
    if not meta.truncated:
        return ""
    effective = (
        f" Showing {meta.displayed} candidates because {meta.final_count} final "
        "submitted candidates are always included."
        if meta.displayed != meta.requested
        else ""
    )
    return (
        '<div class="display-limit-notice" data-display-limit-notice>'
        f"Showing top {meta.requested} of {meta.total} candidates; final submitted "
        f"candidates are always included.{_h(effective)}"
        "</div>"
    )


def _review_ordered_bundles(review: ReportReviewModel) -> list[CandidateEvidenceBundle]:
    bundles: list[CandidateEvidenceBundle] = []
    for group in review.board_groups:
        bundles.extend(group.bundles)
    represented = {id(bundle) for bundle in bundles}
    bundles.extend(bundle for bundle in review.bundles if id(bundle) not in represented)
    return bundles


def _comparison_structure_dom_id(candidate_id: str, index: int) -> str:
    return f"{_structure_dom_id(candidate_id)}__compare_{index}"


def _structure_comparison_section(
    review: ReportReviewModel,
    preset: WorkflowReviewPreset | None = None,
) -> str:
    bundles = _structure_comparison_bundles(review)
    if not bundles:
        return ""
    cards = "".join(
        _structure_comparison_card(bundle, index, preset)
        for index, bundle in enumerate(bundles, start=1)
    )
    return f"""<section id="structure-comparison" class="card saas-section structure-comparison-panel">
  <header class="structure-comparison-head">
    <div>
      <p class="card-eyebrow">Structure Comparison</p>
      <h2 class="card-title">Compare candidate structures side by side</h2>
      <div class="structure-comparison-subtitle-row">
        <p class="saas-section-subtitle">Drag candidate panels to reorder them for direct visual comparison.</p>
        <button type="button" class="structure-comparison-fullscreen-btn"
                data-structure-comparison-fullscreen aria-label="Expand structure comparison"
                title="Expand all candidate structures (press Esc to exit)">&#x2922; Expand comparison</button>
      </div>
    </div>
    <div class="structure-comparison-actions">
      <span>{len(bundles)} structures · 3 visible</span>
    </div>
  </header>
  <div class="structure-comparison-intro">
    <p>Drag candidate panels into the order that makes the most useful structural comparison.</p>
    <span>{_h(preset.scientific_question if preset else "Compare candidates by structure, confidence, and recorded metrics.")}</span>
  </div>
  <div class="structure-comparison-grid" data-structure-compare-grid>{cards}</div>
</section>"""


def _structure_comparison_card(
    bundle: CandidateEvidenceBundle,
    index: int,
    preset: WorkflowReviewPreset | None,
) -> str:
    candidate = bundle.candidate
    dom_id = _comparison_structure_dom_id(candidate.candidate_id, index)
    metrics = "".join(
        _structure_comparison_metric(bundle, label)
        for label in _structure_comparison_metric_labels(preset)
    )
    metrics = (
        metrics or '<span class="structure-compare-metric is-missing">No scalar metrics</span>'
    )
    status_label, status_cls = _STATUS_DISPLAY.get(bundle.status, (bundle.status, "other"))
    open_target = _structure_dom_id(candidate.candidate_id)
    return (
        f'<article class="structure-compare-card" draggable="true" data-structure-compare-card data-compare-index="{index}">'
        '<header class="structure-compare-head">'
        f'<div><span class="structure-compare-rank">#{_h(index)}</span>'
        f"<h3>{_h(candidate.candidate_id)}</h3></div>"
        f'<span class="status-badge status-{_attr(status_cls)}">{_h(status_label)}</span>'
        "</header>"
        f'<div class="structure-compare-metrics">{metrics}</div>'
        f"{_viewer_block(candidate, _chain_context_for(candidate), dom_id=dom_id, compact=True)}"
        '<footer class="structure-compare-foot">'
        f'<a href="#candidates" data-open-cand="{_attr(open_target)}">Open full dossier</a>'
        "</footer>"
        "</article>"
    )


def _structure_comparison_metric_labels(
    preset: WorkflowReviewPreset | None,
) -> list[str]:
    if preset is not None:
        labels = [
            _metric_label_for_display(label)
            for label in preset.metric_priority
            if _metric_label_for_display(label) != "QC"
        ]
        labels = list(dict.fromkeys(labels))
        if labels:
            return labels[:4]
    return ["pLDDT", "pTM", "ipTM", "GRAVY"]


def _structure_comparison_metric(
    bundle: CandidateEvidenceBundle,
    label: str,
) -> str:
    metric = _bundle_metric(bundle, label)
    if metric is None:
        return ""
    value, unit, source, _provenance = metric
    unit_text = f" {unit}" if unit else ""
    return (
        '<span class="structure-compare-metric">'
        f"<b>{_h(_metric_label_for_display(label))}</b>"
        f"<strong>{_h(value)}{_h(unit_text)}</strong>"
        f"<em>{_h(source)}</em>"
        "</span>"
    )


def _saas_candidate_dossiers_section(
    dossiers: list[CandidateDossierViewModel],
    preset: WorkflowReviewPreset | None = None,
    final_submission: dict[str, Any] | None = None,
) -> str:
    selection_summary = _selection_summary_from_submission(final_submission)
    cards = (
        "".join(
            _evidence_snapshot_card(
                d,
                preset,
                selection_summary=selection_summary if idx == 0 else None,
            )
            for idx, d in enumerate(dossiers[:1])
        )
        or '<p class="empty">No candidate dossiers are available.</p>'
    )
    top_msa = _top_candidate_msa_block(dossiers)
    return f"""<section id="candidate-dossiers" class="card saas-section">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Evidence Summary</p>
      <h2 class="card-title">Top candidate evidence at a glance</h2>
      <p class="saas-section-subtitle">Overview of the leading candidate evidence package before opening the full candidate review board.</p>
    </div>
  </header>
  {top_msa}
  <div class="evidence-snapshot-grid">{cards}</div>
</section>"""


def _selection_summary_from_submission(
    final_submission: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(final_submission, dict):
        return None
    summary = final_submission.get("selection_summary")
    if not isinstance(summary, dict):
        return None
    headline = str(summary.get("headline") or "").strip()
    why_selected = str(summary.get("why_selected") or "").strip()
    key_points = [
        str(item).strip() for item in (summary.get("key_points") or []) if str(item).strip()
    ]
    if not headline and not why_selected and not key_points:
        return None
    return {
        "headline": headline,
        "why_selected": why_selected,
        "key_points": key_points[:4],
    }


def _selection_summary_card(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    headline = str(summary.get("headline") or "").strip()
    why_selected = str(summary.get("why_selected") or "").strip()
    key_points = [
        str(item).strip() for item in (summary.get("key_points") or []) if str(item).strip()
    ][:4]
    bullets = "".join(f"<li>{_h(point)}</li>" for point in key_points)
    bullets_html = f"<ul>{bullets}</ul>" if bullets else ""
    return (
        '<div class="selection-summary-card">'
        '<p class="selection-summary-kicker">Why this candidate</p>'
        f"<h4>{_h(headline or 'Final selection rationale')}</h4>"
        f"{f'<p>{_h(why_selected)}</p>' if why_selected else ''}"
        f"{bullets_html}"
        "</div>"
    )


def _evidence_snapshot_card(
    dossier: CandidateDossierViewModel,
    preset: WorkflowReviewPreset | None = None,
    selection_summary: dict[str, Any] | None = None,
) -> str:
    primary_label = _snapshot_primary_metric_label(dossier, preset)
    primary = _dossier_metric_chip(dossier, primary_label) or (
        dossier.metric_chips[0] if dossier.metric_chips else None
    )
    primary_value = _chip_numeric_value(primary)
    score = _snapshot_score_percent(primary.label if primary else "", primary_value)
    score_text = str(primary.value) if primary is not None and primary_value is not None else "--"
    score_style = f"--score:{score:.0f}" if score is not None else "--score:0"
    score_class = "snapshot-score is-compact" if len(score_text) >= 6 else "snapshot-score"
    bars = "".join(
        bar
        for label in _snapshot_metric_labels(dossier, preset)
        if (bar := _evidence_metric_bar(dossier, label, preset))
    )
    missing_label = (
        f"{len(dossier.missing_evidence)} missing"
        if dossier.missing_evidence
        else "Evidence complete"
    )
    return (
        '<article class="evidence-snapshot-card">'
        '<div class="evidence-snapshot-head">'
        f'<div><p class="saas-card-kicker">{_h(dossier.status_label)}</p>'
        f"<h3>{_h(dossier.candidate_id)}</h3></div>"
        f'<span class="status-badge">{_h(dossier.status_label)}</span>'
        "</div>"
        '<div class="evidence-snapshot-body">'
        f'<div class="{score_class}" style="{_attr(score_style)}">'
        f"<strong>{_h(score_text)}</strong><span>{_h(primary.label if primary else 'Score')}</span></div>"
        f'<div class="snapshot-bars">{bars}</div>'
        "</div>"
        f"{_selection_summary_card(selection_summary)}"
        '<div class="evidence-snapshot-foot">'
        f'<span class="asset-pill{" is-missing" if dossier.missing_evidence else " is-on"}">{_h(missing_label)}</span>'
        f"<span>{_h('Structure' if dossier.structure_path else 'No structure')}</span>"
        f'<a href="#candidates" data-open-cand="{_attr("viewer-" + dossier.candidate_id)}">Open dossier</a>'
        "</div>"
        "</article>"
    )


def _top_candidate_msa_block(dossiers: list[CandidateDossierViewModel]) -> str:
    if not dossiers or not dossiers[0].msa_blocks:
        return ""
    top = dossiers[0]
    block = top.msa_blocks[0]
    stats = f"{block.sequence_count} sequences"
    if block.mean_identity:
        stats += f" · {block.mean_identity * 100:.0f}% mean identity"
    return (
        '<div class="top-candidate-msa-block">'
        "<h3>Top candidate MSA context</h3>"
        '<details class="evidence-snapshot-msa" open>'
        f"<summary><span>{_h(top.candidate_id)} MSA context</span><em>{_h(stats)}</em></summary>"
        f"{_msa_html([block])}"
        "</details>"
        "</div>"
    )


def _snapshot_primary_metric_label(
    dossier: CandidateDossierViewModel,
    preset: WorkflowReviewPreset | None = None,
) -> str:
    if preset is not None:
        for label in preset.metric_priority:
            if _metric_label_for_display(label) == "QC":
                continue
            chip = _dossier_metric_chip(dossier, label)
            if chip is not None and _chip_numeric_value(chip) is not None:
                return _metric_label_for_display(label)
    if preset is not None and preset.workflow_type == WorkflowType.RETRIEVAL:
        return "QC"
    return "pLDDT"


def _snapshot_metric_labels(
    dossier: CandidateDossierViewModel,
    preset: WorkflowReviewPreset | None = None,
) -> list[str]:
    if preset is not None:
        preferred = [
            _metric_label_for_display(label)
            for label in preset.metric_priority
            if _metric_label_for_display(label) != "QC"
        ]
        if preset.workflow_type == WorkflowType.RETRIEVAL:
            preferred = ["MSA mean identity", "Seq recovery", *preferred]
    else:
        preferred = ["pLDDT", "pTM", "ipTM", "ipSAE", "GRAVY", "Net charge", "Instability", "pI"]
    observed = [
        chip.label
        for chip in dossier.metric_chips
        if _metric_label_for_display(chip.label) != "QC" and _chip_numeric_value(chip) is not None
    ]
    return list(dict.fromkeys([*preferred, *observed]))[:7]


def _dossier_metric_chip(dossier: CandidateDossierViewModel, label: str) -> Any | None:
    target = _metric_label_for_display(label)
    for chip in dossier.metric_chips:
        if _metric_label_for_display(chip.label) == target:
            return chip
    return None


def _chip_numeric_value(chip: Any | None) -> float | None:
    if chip is None:
        return None
    try:
        return float(str(chip.value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _snapshot_score_percent(label: str, value: float | None) -> float | None:
    if value is None:
        return None
    normalized = _metric_label_for_display(label)
    if normalized == "pLDDT":
        return max(0.0, min(100.0, value))
    if normalized in {"ipTM", "pTM", "ipSAE", "Affinity", "MSA mean identity", "Seq recovery"}:
        return max(0.0, min(100.0, value * 100.0 if value <= 1.5 else value))
    if normalized == "Instability":
        return max(0.0, min(100.0, 100.0 - value))
    if normalized == "GRAVY":
        return max(0.0, min(100.0, (value + 2.0) / 4.0 * 100.0))
    if normalized == "Net charge":
        return max(0.0, min(100.0, (value + 10.0) / 20.0 * 100.0))
    if normalized == "pI":
        return max(0.0, min(100.0, value / 14.0 * 100.0))
    if normalized in {"Delta G", "Interface contacts"}:
        return max(0.0, min(100.0, abs(value)))
    if normalized in {"ddG", "Mean ddG"}:
        return max(4.0, min(100.0, abs(value) / 100.0 * 100.0))
    if normalized == "Rosetta total score":
        return max(4.0, min(100.0, abs(value) / 1000.0 * 100.0))
    return None


def _evidence_metric_bar(
    dossier: CandidateDossierViewModel,
    label: str,
    preset: WorkflowReviewPreset | None = None,
) -> str:
    chip = _dossier_metric_chip(dossier, label)
    value = _chip_numeric_value(chip)
    pct = _snapshot_score_percent(label, value)
    title = _cross_candidate_metric_title(label, preset)
    if chip is None or pct is None:
        return ""
    unit = chip.unit or ""
    return (
        '<div class="snapshot-bar">'
        f'<span>{_h(title)}</span><em><i style="width:{max(2.0, min(100.0, pct)):.0f}%"></i></em>'
        f"<b>{_h(str(chip.value))}{_h((' ' + unit) if unit else '')}</b></div>"
    )


def _saas_cross_candidate_section(
    dossiers: list[CandidateDossierViewModel], preset: WorkflowReviewPreset | None = None
) -> str:
    length_title, lengths = _cross_candidate_length_values(dossiers, preset)
    status_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    qc_counts: dict[str, int] = {}
    for d in dossiers:
        status_counts[d.status_label] = status_counts.get(d.status_label, 0) + 1
        qc_label = _dossier_qc_label(d)
        qc_counts[qc_label] = qc_counts.get(qc_label, 0) + 1
        for item in d.missing_evidence:
            missing_counts[item.label] = missing_counts.get(item.label, 0) + 1
    length_card = _saas_filter_metric_card(
        "length",
        length_title,
        [float(length) for length in lengths],
    )
    status_card = _saas_count_card("Review Status", status_counts)
    missing_card = _saas_count_card("Missing Evidence", missing_counts)
    qc_card = _saas_count_card("QC Summary", qc_counts)
    metric_cards: list[str] = []
    missing_metric_labels: list[str] = []
    filter_metric_labels: list[str] = []
    for label in (
        preset.cross_candidate_metrics
        if preset
        else ("pLDDT", "pTM", "ipTM", "GRAVY", "Net charge")
    ):
        values = _metric_values_from_dossiers(dossiers, label)
        if values:
            title = _cross_candidate_metric_title(label, preset)
            metric_cards.append(_saas_filter_metric_card(label, title, values))
            filter_metric_labels.append(title)
        else:
            missing_metric_labels.append(_cross_candidate_metric_title(label, preset))
    missing_metrics_card = (
        _saas_missing_metrics_card(missing_metric_labels) if missing_metric_labels else ""
    )
    diversity = _diversity_notice(dossiers)
    filter_panel = _cross_candidate_filter_panel(
        dossiers,
        [*filter_metric_labels, length_title],
        preset,
        length_title=length_title,
    )
    return f"""<section id="cross-candidate-analysis" class="card saas-section">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Cross-Candidate Analysis</p>
      <h2 class="card-title">Patterns across candidates</h2>
      <p class="saas-section-subtitle">{_h(preset.cross_candidate_focus if preset else "Adjust the filters to narrow candidates by recorded criteria; matching candidates and downloads update with the current view.")}</p>
    </div>
  </header>
  <div class="saas-pattern-grid">{length_card}{status_card}{qc_card}{missing_card}{"".join(metric_cards)}{missing_metrics_card}</div>
  {filter_panel}
  <div class="saas-insight-note">{diversity}{_prov_chip(Provenance.DERIVED)}</div>
</section>"""


def _cross_candidate_length_values(
    dossiers: list[CandidateDossierViewModel],
    preset: WorkflowReviewPreset | None = None,
) -> tuple[str, list[int]]:
    lengths = [len(d.sequence or "") for d in dossiers if d.sequence]
    if preset is None or preset.workflow_type != WorkflowType.BINDER or not lengths:
        return "Length (aa)", lengths
    ordered = sorted(lengths)
    median = ordered[(len(ordered) - 1) // 2]
    binder_lengths = [length for length in lengths if length <= median * 1.5]
    return "Binder length (aa)", binder_lengths or lengths


def _workflow_audit_section(model: ReportModel, vm: WorkflowTraceViewModel) -> str:
    final_submission = model.final_submission or {}
    has_final = bool(
        model.final_candidates
        or final_submission.get("candidate_id")
        or final_submission.get("submitted_candidates")
    )
    stages = _workflow_audit_stage_flow_html(
        model.graph.nodes,
        workflow_type=vm.workflow_type,
        has_final=has_final,
    )
    completed = str(model.summary.state or "completed").strip().lower()
    status_label = (
        "Completed"
        if completed in {"", "completed", "success"}
        else completed.replace("_", " ").title()
    )
    status_cls = "is-complete" if completed in {"", "completed", "success"} else "is-recorded"
    node_count = len(model.graph.nodes)
    tool_calls = _workflow_audit_tool_call_list(
        model.graph.nodes,
        final_submission=final_submission,
    )
    selection_events = _workflow_audit_selection_events(model)
    return f"""<section id="workflow-audit" class="card saas-section workflow-audit-section">
  <header class="workflow-audit-header">
    <div>
      <p class="card-eyebrow">Secondary Audit</p>
      <h2 class="card-title">Workflow execution graph</h2>
      <p class="saas-section-subtitle">A compact left-to-right audit view of the recorded process for this user query.</p>
    </div>
    <div class="workflow-audit-meta">
      <span class="workflow-status-pill {status_cls}">{_h(status_label)}</span>
      <span>{_h(str(node_count))} nodes</span>
    </div>
  </header>
  <div class="workflow-graph-panel">
    <ol class="audit-stage-strip audit-stage-flow">{stages}</ol>
    {tool_calls}
    {selection_events}
    <div class="workflow-graph-legend">
      <span><i class="legend-dot is-complete"></i>Completed</span>
      <span><i class="legend-dot is-skipped"></i>Not recorded</span>
      <span><i class="legend-chip"></i>Recorded tool</span>
    </div>
  </div>
</section>"""


def _workflow_audit_selection_events(model: ReportModel) -> str:
    events = model.candidate_capability_filter_events
    if not events:
        return ""
    rows = "".join(_candidate_capability_filter_event_html(event) for event in events[:12])
    return (
        '<details class="workflow-tool-call-list" open>'
        "<summary><strong>Candidate selection and filters</strong>"
        "<em>Recorded subset coverage before downstream tools</em></summary>"
        f'<div class="artifact-list">{rows}</div>'
        "</details>"
    )


def _workflow_audit_tool_call_list(
    nodes: list[WorkflowNodeSummary],
    *,
    final_submission: dict[str, Any] | None = None,
) -> str:
    if not nodes:
        return ""
    groups = _workflow_audit_tool_call_groups(nodes)
    rows = "".join(
        _workflow_audit_tool_call_group_row(
            idx,
            group,
            final_submission=final_submission,
        )
        for idx, group in enumerate(groups, start=1)
    )
    return (
        '<details class="workflow-tool-call-list" open>'
        "<summary><strong>Recorded tool calls</strong>"
        "<em>Grouped by tool and outcome</em></summary>"
        f"<ol>{rows}</ol>"
        "</details>"
    )


def _workflow_audit_tool_call_groups(
    nodes: list[WorkflowNodeSummary],
) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for idx, node in enumerate(nodes, start=1):
        state_cls = _state_class(node.state)
        failed = state_cls == "failed"
        error_text = _workflow_audit_error_text(node) if failed else ""
        tool_name = _compact_tool_name(node.tool)
        key = (tool_name, state_cls, error_text)
        group = by_key.get(key)
        if group is None:
            group = {
                "tool_name": tool_name,
                "state_cls": state_cls,
                "state_label": str(node.state or "recorded"),
                "error_text": error_text,
                "calls": [],
            }
            by_key[key] = group
            groups.append(group)
        calls = group["calls"]
        if isinstance(calls, list):
            calls.append((idx, node))
    return groups


def _workflow_audit_tool_call_group_row(
    idx: int,
    group: dict[str, object],
    *,
    final_submission: dict[str, Any] | None = None,
) -> str:
    calls = group.get("calls")
    call_items = calls if isinstance(calls, list) else []
    count = len(call_items)
    state_cls = str(group.get("state_cls") or "recorded")
    failed = state_cls == "failed"
    error_text = str(group.get("error_text") or "")
    class_attr = ' class="tool-call-group has-failure"' if failed else ' class="tool-call-group"'
    title_attr = f' title="{_attr(error_text)}"' if error_text else ""
    state_label = str(group.get("state_label") or "recorded")
    tool_name = str(group.get("tool_name") or "Tool")
    raw_tool = _workflow_audit_group_raw_tool(call_items)
    outcome = _workflow_audit_group_outcome_caption(
        raw_tool,
        count,
        final_submission=final_submission,
    )
    error_html = f'<small class="tool-call-error">{_h(error_text)}</small>' if error_text else ""
    return (
        f"<li{class_attr}{title_attr}>"
        f'<span class="tool-call-group-index">{idx}</span>'
        '<div class="tool-call-group-main">'
        f"<strong>{_h(tool_name)}</strong>"
        f'<em class="state-{state_cls}">{_h(state_label)}</em>'
        "</div>"
        f'<span class="tool-call-count">{_h(str(count))} call{"s" if count != 1 else ""}</span>'
        f'<p class="tool-call-outcome">{_h(outcome)}</p>'
        f"{error_html}"
        "</li>"
    )


def _workflow_audit_group_raw_tool(call_items: list[Any]) -> str:
    for _, node in call_items:
        if isinstance(node, WorkflowNodeSummary) and node.tool:
            return str(node.tool)
    return ""


def _workflow_audit_group_outcome_caption(
    tool: str,
    count: int,
    *,
    final_submission: dict[str, Any] | None = None,
) -> str:
    normalized = str(tool or "").strip().lower()
    if normalized == "esm2" and _final_rationale_marks_tool_unused("esm2", final_submission):
        return "Exploratory sequence model output; final rationale says it was not used."
    captions = {
        "protenix_v2": "Predicted candidate structures and confidence metrics.",
        "esm2": "Generated or evaluated sequence-level mutation suggestions.",
        "sequence_edit": "Created edited sequence candidates.",
        "pyrosetta": "Computed Rosetta energy and stability evidence.",
        "tmalign": "Compared candidate structures against the reference fold.",
        "structure_analysis": "Annotated structural features and liabilities.",
        "protein_qc": "Checked sequence biophysics and liability flags.",
        "objective_rank": "Ranked candidates with explicit objective gates and formula.",
        "submit_candidate": "Recorded the final submitted candidate decision.",
    }
    caption = captions.get(normalized)
    if caption:
        return caption
    return f"Recorded {count} supporting workflow call{'s' if count != 1 else ''}."


def _final_rationale_marks_tool_unused(
    tool: str,
    final_submission: dict[str, Any] | None,
) -> bool:
    if not isinstance(final_submission, dict):
        return False
    text = " ".join(
        str(value)
        for value in (
            final_submission.get("reasoning"),
            *(
                row.get("justification")
                for row in final_submission.get("candidates_resolved") or []
                if isinstance(row, dict)
            ),
        )
        if isinstance(value, str)
    ).lower()
    normalized = str(tool or "").lower().replace("_", "").replace("-", "")
    compact_text = text.replace("_", "").replace("-", "")
    return normalized in compact_text and (
        "not used" in text or "not selected" in text or "did not use" in text or "blind to" in text
    )


def _workflow_audit_error_text(node: WorkflowNodeSummary) -> str:
    bits: list[str] = []
    if node.error_code:
        bits.append(str(node.error_code))
    message = str(node.error_message or "").strip()
    if message:
        first_line = message.splitlines()[0].strip()
        if first_line:
            bits.append(first_line)
    if not bits and isinstance(node.error_details, dict):
        detail = node.error_details.get("message") or node.error_details.get("error")
        if detail:
            bits.append(str(detail))
    return _short(": ".join(bits) if bits else "Failed without a recorded reason.", 150)


def _workflow_audit_stage_flow_html(
    nodes: list[WorkflowNodeSummary],
    *,
    workflow_type: str,
    has_final: bool,
) -> str:
    pieces = [
        _workflow_audit_endpoint_html(
            1,
            key="start",
            title="Start",
            icon="play",
            state_label="Completed",
            tool_label="User request",
        )
    ]
    step_index = 2
    band_html, used_steps = _workflow_audit_stage_band_html(
        step_index,
        nodes,
        workflow_type=workflow_type,
        has_final=has_final,
    )
    pieces.append(band_html)
    step_index += used_steps
    pieces.append(
        _workflow_audit_endpoint_html(
            step_index,
            key="completed",
            title="Completed",
            icon="flag",
            state_label="Success" if has_final or nodes else "Recorded",
            tool_label="Report ready",
        )
    )
    return "".join(pieces)


def _workflow_audit_stage_band_html(
    start_index: int,
    nodes: list[WorkflowNodeSummary],
    *,
    workflow_type: str,
    has_final: bool,
) -> tuple[str, int]:
    stage_map = _workflow_audit_stage_map(nodes)
    stage_specs = _workflow_trace_stage_specs(workflow_type, stage_map)
    if "other" in stage_map and not any(spec["key"] == "other" for spec in stage_specs):
        stage_specs = [
            *stage_specs[:-1],
            {
                "key": "other",
                "title": "Supporting Evidence",
                "icon": "search",
                "purpose": "Run recorded supporting tools.",
                "evidence": "Recorded tool output",
            },
            stage_specs[-1],
        ]
    cards: list[str] = []
    local_index = start_index
    for spec in stage_specs:
        key = spec["key"]
        if key == "intent":
            continue
        stage = stage_map.get(key)
        include_synthetic_final = (
            key == "recommendation" and has_final and (stage is None or stage.node_count == 0)
        )
        if stage is None and not include_synthetic_final:
            continue
        cards.append(
            _workflow_audit_recorded_stage_html(
                local_index,
                spec,
                stage,
                synthetic_tool="Final selection" if include_synthetic_final else "",
            )
        )
        local_index += 1
    if not cards:
        cards.append(
            _workflow_audit_empty_stage_html(
                local_index,
                workflow_type=workflow_type,
            )
        )
        local_index += 1
    step_count = local_index - start_index
    return (
        '<li class="audit-stage-band">'
        '<div class="audit-stage-head">'
        "<span>Recorded steps</span>"
        f"<em>{_h(str(step_count))} step{'s' if step_count != 1 else ''}</em>"
        "</div>"
        f'<ol class="audit-stage-group">{"".join(cards)}</ol>'
        "</li>",
        step_count,
    )


def _workflow_audit_stage_map(
    nodes: list[WorkflowNodeSummary],
) -> dict[str, WorkflowStageViewModel]:
    grouped: dict[str, list[WorkflowNodeSummary]] = {}
    for node in nodes:
        grouped.setdefault(_workflow_audit_node_stage_key(node), []).append(node)
    return {
        key: WorkflowStageViewModel(
            key=key,
            label=key,
            tools=_dedupe_text([node.tool for node in stage_nodes if node.tool]),
            node_count=len(stage_nodes),
            completed_count=sum(
                1 for node in stage_nodes if str(node.state).lower() == "completed"
            ),
            failed_count=sum(1 for node in stage_nodes if str(node.state).lower() == "failed"),
            evidence_label=_workflow_audit_evidence_label(key),
            provenance=Provenance.DERIVED,
        )
        for key, stage_nodes in grouped.items()
    }


def _workflow_audit_node_stage_key(node: WorkflowNodeSummary) -> str:
    return classify_workflow_stage_key(node, include_intent=True)


def _workflow_audit_evidence_label(key: str) -> str:
    return {
        "engineering": "Edited sequence variants",
        "backbone_generation": "Generated backbone scaffolds",
        "binder_generation": "Generated target-conditioned binder candidates",
        "generation": "Sequence model suggestions",
        "qc": "QC state and flags",
        "structure": "Structure confidence and structure artifacts",
        "structure_comparison": "Fold preservation and structural similarity",
        "energy": "Rosetta energy and stability evidence",
        "interface": "Interface and binding evidence",
        "ranking": "Ranked candidate scores",
        "recommendation": "Final submitted candidates",
        "other": "Additional recorded evidence",
    }.get(key, "Recorded evidence")


def _workflow_audit_stage_html(
    index: int,
    spec: dict[str, str],
    stage_map: dict[str, WorkflowStageViewModel],
    *,
    has_final: bool = False,
) -> str:
    key = spec["key"]
    stage = stage_map.get(key)
    if key == "intent":
        state_cls = "is-recorded"
        count = "Objective recorded"
        evidence = spec["evidence"]
    elif stage is None or stage.node_count == 0:
        if key == "recommendation" and has_final:
            state_cls = "is-complete"
            count = "Final candidates recorded"
            evidence = "Final submitted candidates"
        else:
            state_cls = "is-skipped"
            count = "Skipped / not recorded"
            evidence = spec["evidence"]
    else:
        state_cls = "is-complete" if stage.failed_count == 0 else "has-failures"
        count = f"{stage.node_count} nodes - {stage.completed_count} completed" + (
            f" - {stage.failed_count} failed" if stage.failed_count else ""
        )
        evidence = stage.evidence_label or spec["evidence"]
    tool_label = ""
    if stage and stage.tools:
        tool_label = ", ".join(_compact_tool_name(tool) for tool in stage.tools[:3])
    icon = _workflow_stage_icon_name(key)
    return (
        f'<li class="audit-stage {state_cls}" data-stage="{_attr(key)}">'
        '<div class="audit-stage-top">'
        f'<span class="audit-stage-icon">{_icon_svg(icon)}</span>'
        f'<span class="audit-stage-line"></span>'
        "</div>"
        f"<strong>{index}. {_h(spec['title'])}</strong>"
        f"<p>{_h(spec['purpose'])}</p>"
        f"<small><b>Evidence:</b> {_h(evidence)}</small>"
        f"<small>{_h(count)}</small>"
        f"{f'<em>{_h(tool_label)}</em>' if tool_label else ''}"
        "</li>"
    )


def _workflow_audit_endpoint_html(
    index: int,
    *,
    key: str,
    title: str,
    icon: str,
    state_label: str,
    tool_label: str,
) -> str:
    return (
        f'<li class="audit-stage audit-flow-endpoint is-complete" data-stage="{_attr(key)}">'
        f'<span class="audit-stage-number">{index}</span>'
        f'<span class="audit-stage-icon">{_icon_svg(icon)}</span>'
        f"<strong>{_h(title)}</strong>"
        f'<span class="audit-stage-status">{_h(state_label)}</span>'
        f'<div class="audit-tool-row"><span class="audit-tool-chip">{_h(tool_label)}</span></div>'
        "</li>"
    )


def _workflow_audit_recorded_stage_html(
    index: int,
    spec: dict[str, str],
    stage: WorkflowStageViewModel | None,
    *,
    synthetic_tool: str = "",
) -> str:
    key = spec["key"]
    if stage is None:
        state_cls = "is-complete"
        state_label = "Completed"
        tools = [synthetic_tool] if synthetic_tool else []
    else:
        state_cls = "is-complete" if stage.failed_count == 0 else "has-failures"
        state_label = "Completed" if stage.failed_count == 0 else "Needs review"
        compact_tools = [_compact_tool_name(tool) for tool in stage.tools if tool]
        tools = compact_tools[:6]
        if len(compact_tools) > len(tools):
            tools.append(f"+{len(compact_tools) - len(tools)} more")
    tool_html = "".join(
        f'<span class="audit-tool-chip">{_h(tool)}</span>' for tool in tools if tool
    )
    if not tool_html:
        tool_html = '<span class="audit-tool-chip is-muted">Recorded</span>'
    return (
        f'<li class="audit-stage {state_cls}" data-stage="{_attr(key)}">'
        f'<span class="audit-stage-number">{index}</span>'
        f'<span class="audit-stage-icon">{_icon_svg(_workflow_stage_icon_name(key))}</span>'
        f"<strong>{_h(spec['title'])}</strong>"
        f'<span class="audit-stage-status">{_h(state_label)}</span>'
        f'<div class="audit-tool-row">{tool_html}</div>'
        "</li>"
    )


def _workflow_audit_empty_stage_html(index: int, *, workflow_type: str) -> str:
    label = (
        "Evidence not recorded" if workflow_type == WorkflowType.RETRIEVAL else "No tools recorded"
    )
    return (
        f'<li class="audit-stage is-skipped" data-stage="empty-stage">'
        f'<span class="audit-stage-number">{index}</span>'
        f'<span class="audit-stage-icon">{_icon_svg("search")}</span>'
        "<strong>Recorded step</strong>"
        '<span class="audit-stage-status">Not recorded</span>'
        f'<div class="audit-tool-row"><span class="audit-tool-chip is-muted">{_h(label)}</span></div>'
        "</li>"
    )


def _workflow_stage_icon_name(key: str) -> str:
    return {
        "intent": "intent",
        "engineering": "tools",
        "backbone_generation": "cube",
        "binder_generation": "bind",
        "generation": "spark",
        "qc": "shield",
        "structure": "cube",
        "structure_comparison": "compare",
        "energy": "analytics",
        "interface": "bind",
        "ranking": "rank",
        "recommendation": "check",
        "other": "search",
    }.get(key, "workflow")


def _icon_svg(name: str) -> str:
    paths = {
        "overview": '<rect x="4" y="5" width="7" height="7" rx="2"/><rect x="13" y="5" width="7" height="7" rx="2"/><rect x="4" y="14" width="7" height="5" rx="1.5"/><rect x="13" y="14" width="7" height="5" rx="1.5"/>',
        "dossier": '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5"/><path d="M10 13h6"/><path d="M10 17h4"/>',
        "candidate": '<path d="M8 7a4 4 0 1 0 8 0a4 4 0 0 0-8 0"/><path d="M4 21a8 8 0 0 1 16 0"/>',
        "compare": '<path d="M6 6h11"/><path d="M6 12h15"/><path d="M6 18h8"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
        "analytics": '<path d="M4 19V5"/><path d="M4 19h17"/><path d="M8 16v-5"/><path d="M13 16V8"/><path d="M18 16v-9"/>',
        "workflow": '<circle cx="6" cy="6" r="2"/><circle cx="18" cy="18" r="2"/><circle cx="18" cy="6" r="2"/><path d="M8 6h6"/><path d="M16.7 7.4L7.3 16.6"/><path d="M8 18h6"/><path d="M6 8v8"/>',
        "agent": '<path d="M12 3v3"/><path d="M7 6h10a3 3 0 0 1 3 3v7a3 3 0 0 1-3 3H7a3 3 0 0 1-3-3V9a3 3 0 0 1 3-3z"/><path d="M9 12h.01"/><path d="M15 12h.01"/><path d="M9 16h6"/>',
        "repro": '<path d="M4 4v6h6"/><path d="M20 20v-6h-6"/><path d="M20 10a8 8 0 0 0-14-4L4 10"/><path d="M4 14a8 8 0 0 0 14 4l2-4"/>',
        "plot": '<path d="M4 19h17"/><path d="M4 19V5"/><path d="M7 15l4-4l3 3l5-7"/>',
        "msa": '<path d="M5 6h14"/><path d="M5 10h10"/><path d="M5 14h14"/><path d="M5 18h7"/>',
        "tools": '<path d="M14 7l3-3l3 3l-3 3z"/><path d="M4 20l7-7"/><path d="M8 4l12 12"/><path d="M4 8l4-4"/>',
        "availability": '<path d="M12 3l8 4v6c0 5-3.5 8-8 9c-4.5-1-8-4-8-9V7z"/><path d="M9 12l2 2l4-4"/>',
        "merge": '<path d="M5 5v4a4 4 0 0 0 4 4h6"/><path d="M5 19v-4a4 4 0 0 1 4-4h6"/><path d="M15 9l4 4l-4 4"/>',
        "transcript": '<path d="M5 4h14v12H8l-3 3z"/><path d="M8 8h8"/><path d="M8 12h6"/>',
        "intent": '<path d="M12 21a9 9 0 1 0 0-18a9 9 0 0 0 0 18z"/><path d="M12 7v5l3 2"/>',
        "spark": '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
        "quote": '<path d="M8 7H5.5A2.5 2.5 0 0 0 3 9.5V13h4v4H3v-7.5A5.5 5.5 0 0 1 8.5 4H10"/><path d="M18 7h-2.5A2.5 2.5 0 0 0 13 9.5V13h4v4h-4v-7.5A5.5 5.5 0 0 1 18.5 4H20"/>',
        "shield": '<path d="M12 3l8 4v6c0 5-3.5 8-8 9c-4.5-1-8-4-8-9V7z"/><path d="M9 12l2 2l4-4"/>',
        "cube": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5"/><path d="M12 12v9"/><path d="M12 12L4 7.5"/>',
        "bind": '<path d="M8 8a4 4 0 0 1 6 0l1 1"/><path d="M16 16a4 4 0 0 1-6 0l-1-1"/><path d="M7 13l-2-2a3 3 0 0 1 4-4l2 2"/><path d="M13 15l2 2a3 3 0 0 0 4-4l-2-2"/>',
        "rank": '<path d="M5 19V9"/><path d="M12 19V5"/><path d="M19 19v-7"/><path d="M3 19h20"/>',
        "check": '<path d="M20 6L9 17l-5-5"/>',
        "search": '<path d="M11 18a7 7 0 1 0 0-14a7 7 0 0 0 0 14z"/><path d="M20 20l-4-4"/>',
        "play": '<path d="M8 5v14l11-7z"/>',
        "database": '<ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.7 3.1 3 7 3s7-1.3 7-3V5"/><path d="M5 11v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/>',
        "flag": '<path d="M6 21V4"/><path d="M6 5h11l-1.5 4L17 13H6"/>',
        "chevron-left": '<path d="M15 18l-6-6l6-6"/>',
    }
    body = paths.get(name, paths["overview"])
    return f'<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">{body}</svg>'


def _saas_workflow_trace_section(vm: WorkflowTraceViewModel) -> str:
    stage_map = {stage.key: stage for stage in vm.stages}
    stage_specs = _workflow_trace_stage_specs(vm.workflow_type, stage_map)
    stages = "".join(
        _workflow_trace_stage_html(index, spec, stage_map)
        for index, spec in enumerate(stage_specs, start=1)
    )
    return f"""<section id="workflow-trace" class="card saas-section">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Workflow Trace</p>
      <h2 class="card-title">Why this workflow?</h2>
      <p class="saas-section-subtitle">Product-stage explanation derived from the recorded workflow graph. Each stage shows what evidence it contributed to the final review.</p>
    </div>
    <span class="card-meta">{_h(vm.workflow_type)}</span>
  </header>
  <ol class="workflow-stage-strip workflow-stage-count-{len(stage_specs)}">{stages}</ol>
</section>"""


def _workflow_trace_stage_specs(
    workflow_type: str, stage_map: dict[str, WorkflowStageViewModel]
) -> list[dict[str, str]]:
    if workflow_type == WorkflowType.RETRIEVAL:
        return [
            {
                "key": "intent",
                "title": "User Intent",
                "icon": "?",
                "purpose": "Understand the evidence question and constraints.",
                "evidence": "Recorded objective",
            },
            {
                "key": "other",
                "title": "Evidence Retrieval",
                "icon": "R",
                "purpose": "Search or retrieve relevant sequence/source evidence.",
                "evidence": "Retrieved hits and source context",
            },
            {
                "key": "ranking",
                "title": "Evidence Synthesis",
                "icon": "S",
                "purpose": "Prioritize records by available support.",
                "evidence": "Ranked evidence records",
            },
            {
                "key": "recommendation",
                "title": "Final Recommendation",
                "icon": "F",
                "purpose": "Select the strongest supported answer.",
                "evidence": "Final recommendation",
            },
        ]
    specs = [
        {
            "key": "intent",
            "title": "User Intent",
            "icon": "?",
            "purpose": "Understand design goal and constraints.",
            "evidence": "Recorded objective",
        },
        {
            "key": "engineering",
            "title": "Engineering",
            "icon": "E",
            "purpose": "Create constrained sequence edits and variants.",
            "evidence": "Edited candidate sequences",
        },
        {
            "key": "backbone_generation",
            "title": "Backbone Generation",
            "icon": "3D",
            "purpose": "Generate backbone scaffolds before sequence design.",
            "evidence": "Backbone structures",
        },
        {
            "key": "binder_generation",
            "title": "Binder Generation",
            "icon": "B",
            "purpose": "Generate target-conditioned binder candidates.",
            "evidence": "Binder candidates and complex models",
        },
        {
            "key": "generation",
            "title": "Sequence Generation",
            "icon": "G",
            "purpose": "Explore sequence-model suggestions.",
            "evidence": "Sequence model output",
        },
        {
            "key": "qc",
            "title": "Sequence QC",
            "icon": "Q",
            "purpose": "Filter liabilities and basic biophysical issues.",
            "evidence": "QC state and flags",
        },
        {
            "key": "structure",
            "title": "Structure Prediction",
            "icon": "3D",
            "purpose": "Predict 3D structures and confidence.",
            "evidence": "pLDDT, pTM, PAE, structures",
        },
        {
            "key": "structure_comparison",
            "title": "Structure Comparison",
            "icon": "C",
            "purpose": "Compare candidate folds against the reference.",
            "evidence": "TM-score, RMSD, alignment",
        },
        {
            "key": "energy",
            "title": "Energy Analysis",
            "icon": "ΔG",
            "purpose": "Score stability with molecular energy functions.",
            "evidence": "Rosetta total score, ddG",
        },
        {
            "key": "interface",
            "title": "Interface Scoring",
            "icon": "I",
            "purpose": "Assess target binding and interface quality.",
            "evidence": "ipTM, affinity, contacts",
        },
        {
            "key": "ranking",
            "title": "Ranking",
            "icon": "R",
            "purpose": "Integrate evidence and rank candidates.",
            "evidence": "Composite review order",
        },
        {
            "key": "recommendation",
            "title": "Final Recommendation",
            "icon": "F",
            "purpose": "Select final candidates for experimental follow-up.",
            "evidence": "Final submitted candidates",
        },
    ]
    if workflow_type == WorkflowType.DENOVO and not stage_map.get("interface", None):
        # Keep interface visible but skipped so users do not read it as a
        # required de novo stage.
        return specs
    return specs


def _workflow_trace_stage_html(
    index: int,
    spec: dict[str, str],
    stage_map: dict[str, WorkflowStageViewModel],
) -> str:
    key = spec["key"]
    stage = stage_map.get(key)
    if key == "intent":
        state_cls = "is-recorded"
        count = "Objective recorded"
        evidence = spec["evidence"]
    elif stage is None or stage.node_count == 0:
        state_cls = "is-skipped"
        count = "Skipped / not recorded"
        evidence = spec["evidence"]
    else:
        state_cls = "is-complete" if stage.failed_count == 0 else "has-failures"
        count = f"{stage.node_count} nodes - {stage.completed_count} completed" + (
            f" - {stage.failed_count} failed" if stage.failed_count else ""
        )
        evidence = stage.evidence_label or spec["evidence"]
    marker_label, marker_title = _workflow_stage_marker_label(key, spec, stage)
    return (
        f'<li class="workflow-stage {state_cls}" data-stage="{_attr(key)}">'
        f'<div class="workflow-stage-marker"><span class="workflow-stage-index">{index}</span>'
        f'<span class="workflow-stage-icon" title="{_attr(marker_title)}">{_h(marker_label)}</span></div>'
        f'<div class="workflow-stage-body">'
        f"<h3>{_h(spec['title'])}</h3>"
        f"<p>{_h(spec['purpose'])}</p>"
        f"<small><strong>Evidence:</strong> {_h(evidence)}</small>"
        f"<small>{_h(count)}</small>"
        f"</div></li>"
    )


def _workflow_stage_marker_label(
    key: str,
    spec: dict[str, str],
    stage: WorkflowStageViewModel | None,
) -> tuple[str, str]:
    if key == "intent":
        return ("intent", "intent")
    if stage and stage.tools:
        tools = [_compact_tool_name(tool) for tool in stage.tools if tool]
        tools = [tool for tool in tools if tool]
        if len(tools) > 1:
            filtered = [tool for tool in tools if tool != "agent_intent"]
            if filtered:
                tools = filtered
        if tools:
            return (tools[0], ", ".join(tools))
    if stage is None or stage.node_count == 0:
        return ("not recorded", "not recorded")
    return (spec["icon"], spec["icon"])


def _compact_tool_name(tool: str) -> str:
    name = str(tool).strip()
    if not name:
        return ""
    aliases = {
        "boltz2_affinity": "Boltz-2",
        "boltz2": "Boltz-2",
        "boltzgen": "BoltzGen",
        "esm3": "ESM-3",
        "esm2": "ESM-2",
        "protenix_v2": "Protenix v2",
        "proteinmpnn": "ProteinMPNN",
        "pxdesign": "PXDesign",
        "pxmeter": "PXMeter",
        "ipsae": "ipSAE",
        "prodigy": "PRODIGY",
        "vina": "Vina",
        "sequence_edit": "Sequence edit",
        "sequence_analysis": "Sequence analysis",
        "protein_qc": "Protein QC",
        "structure_analysis": "Structure analysis",
        "objective_rank": "Objective rank",
        "submit_candidate": "Submit candidate",
    }
    return aliases.get(name.lower(), name)


def _saas_metric_chip_html(chips: list[Any]) -> str:
    if not chips:
        return '<span class="saas-chip saas-chip-muted">No metrics</span>'
    return "".join(
        f'<span class="saas-chip" title="{_attr(chip.source)}">'
        f"<b>{_h(chip.label)}</b> {_h(chip.value)}{(' ' + _h(chip.unit)) if chip.unit else ''}"
        f"</span>"
        for chip in chips
    )


def _dossier_summary(dossier: CandidateDossierViewModel) -> str:
    if dossier.risk_notes:
        return f"Risks to review: {', '.join(dossier.risk_notes[:2])}."
    if dossier.missing_evidence:
        return (
            f"Missing evidence: {', '.join(item.label for item in dossier.missing_evidence[:2])}."
        )
    return "Evidence package is available for candidate review."


def _saas_hist_card(title: str, values: list[int]) -> str:
    if not values:
        return _saas_empty_pattern_card(title, "No values available.")
    return _saas_distribution_card(
        title, [float(v) for v in values], value_formatter=lambda v: str(int(v))
    )


def _saas_count_card(title: str, counts: dict[str, int]) -> str:
    if not counts:
        return _saas_empty_pattern_card(title, "No recorded values.")
    total = sum(counts.values()) or 1
    rows = "".join(
        f'<div class="saas-count-row"><span>{_h(label)}</span>'
        f'<em><i style="width:{max(6, int(count / total * 100))}%"></i></em>'
        f"<b>{_h(count)}</b></div>"
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:6]
    )
    return f'<div class="saas-pattern-card saas-count-card"><h3>{_h(title)}</h3>{rows}</div>'


def _metric_values_from_dossiers(
    dossiers: list[CandidateDossierViewModel], label: str
) -> list[float]:
    values: list[float] = []
    target = _metric_label_for_display(label)
    for dossier in dossiers:
        for chip in dossier.metric_chips:
            if _metric_label_for_display(chip.label) != target:
                continue
            try:
                values.append(float(str(chip.value).replace(",", "")))
            except (TypeError, ValueError):
                pass
            break
    return values


def _saas_range_card(title: str, values: list[float]) -> str:
    if not values:
        return _saas_empty_pattern_card(title, "Metric not recorded.")
    return _saas_distribution_card(title, values, value_formatter=_fmt_range_value)


def _saas_filter_metric_card(label: str, title: str, values: list[float]) -> str:
    if not values:
        return _saas_empty_pattern_card(title, "Metric not recorded.")
    lo = min(values)
    hi = max(values)
    avg = sum(values) / len(values)
    bins = _distribution_bins(values, bucket_count=8)
    max_count = max(bins) or 1
    bars = "".join(
        f'<span style="height:{max(8, int(count / max_count * 64))}px" title="{_attr(count)} candidate(s)"></span>'
        for count in bins
    )
    integer_only = title.endswith("(aa)") or "length" in title.lower()
    step = _metric_filter_step(lo, hi, values, integer_only=integer_only)
    disabled = " disabled" if hi <= lo else ""
    return (
        f'<div class="saas-pattern-card metric-range-card metric-filter-card" '
        f'data-filter-card data-metric="{_attr(title)}" '
        f'data-default-min="{_attr(_fmt_filter_value(lo))}" '
        f'data-default-max="{_attr(_fmt_filter_value(hi))}">'
        f"<h3>{_h(title)}</h3>"
        f'<div class="saas-bars saas-bars-distribution">{bars}</div>'
        '<div class="range-filter-control">'
        '<div class="range-filter-track"><span class="range-filter-window"></span></div>'
        f'<input class="metric-filter-min" type="range" min="{_attr(_fmt_filter_value(lo))}" max="{_attr(_fmt_filter_value(hi))}" step="{_attr(step)}" value="{_attr(_fmt_filter_value(lo))}" aria-label="{_attr(title)} minimum"{disabled}>'
        f'<input class="metric-filter-max" type="range" min="{_attr(_fmt_filter_value(lo))}" max="{_attr(_fmt_filter_value(hi))}" step="{_attr(step)}" value="{_attr(_fmt_filter_value(hi))}" aria-label="{_attr(title)} maximum"{disabled}>'
        "</div>"
        f'<p><span class="range-filter-label"><b>{_h(_fmt_range_value(lo))}</b> to <b>{_h(_fmt_range_value(hi))}</b></span> · avg {_h(_fmt_range_value(avg))}</p>'
        "</div>"
    )


def _cross_candidate_filter_panel(
    dossiers: list[CandidateDossierViewModel],
    metric_labels: list[str],
    preset: WorkflowReviewPreset | None = None,
    *,
    length_title: str | None = None,
) -> str:
    metric_labels = list(dict.fromkeys(metric_labels))
    result_dossiers = [dossier for dossier in dossiers if dossier.structure_path]
    result_dossiers.sort(
        key=lambda dossier: _filter_result_sort_key(dossier, metric_labels, preset),
        reverse=True,
    )
    result_cards = "".join(
        _cross_candidate_filter_result(dossier, metric_labels, preset, length_title=length_title)
        for dossier in result_dossiers
    )
    if not metric_labels or not result_cards:
        return ""
    return (
        '<div class="candidate-range-filter" data-candidate-range-filter>'
        '<div class="candidate-range-filter-head">'
        "<div><h3>Filter candidates by metric ranges</h3>"
        "<p>Move the min and max handles for any metric. Matching candidates must satisfy every adjusted range, and structure downloads appear below.</p></div>"
        '<span class="candidate-filter-count" data-filter-count></span>'
        "</div>"
        f'<div class="candidate-filter-results" data-filter-results>{result_cards}</div>'
        '<p class="candidate-filter-empty" data-filter-empty hidden>No structure-linked candidates match the current metric ranges.</p>'
        "</div>"
    )


def _cross_candidate_filter_result(
    dossier: CandidateDossierViewModel,
    metric_labels: list[str],
    preset: WorkflowReviewPreset | None = None,
    *,
    length_title: str | None = None,
) -> str:
    metrics: dict[str, float] = {}
    for chip in dossier.metric_chips:
        value = _chip_numeric_value(chip)
        if value is None:
            continue
        title = _cross_candidate_metric_title(chip.label, preset)
        if title in metric_labels:
            metrics[title] = value
    if length_title and length_title in metric_labels:
        length_value = _dossier_length_value(dossier)
        if length_value is not None:
            metrics[length_title] = float(length_value)
    metric_summary = "".join(
        '<span class="candidate-filter-primary-metric">'
        f"{_h(metric_labels[0])} {_h(_fmt_range_value(metrics[metric_labels[0]]))}</span>"
        if metric_labels and label == metric_labels[0]
        else f"<span>{_h(label)} {_h(_fmt_range_value(value))}</span>"
        for label, value in metrics.items()
    )
    download = (
        _download_link(
            dossier.structure_path,
            "structure",
            kind=Path(dossier.structure_path).suffix.lstrip(".") or "structure",
            css="asset-pill is-on asset-download-link",
        )
        if dossier.structure_path
        else ""
    )
    return (
        f'<article class="candidate-filter-result" data-filter-result '
        f'data-metrics="{_attr(json.dumps(metrics, separators=(",", ":")))}">'
        f"<div><strong>{_h(dossier.candidate_id)}</strong>"
        f"<span>{_h(dossier.status_label)}</span></div>"
        f'<div class="candidate-filter-result-metrics">{metric_summary}</div>'
        f"{download}"
        "</article>"
    )


def _filter_result_sort_key(
    dossier: CandidateDossierViewModel,
    metric_labels: list[str],
    preset: WorkflowReviewPreset | None = None,
) -> tuple[int, float, int, str]:
    primary = metric_labels[0] if metric_labels else ""
    value = _dossier_metric_value_by_title(dossier, primary, preset)
    is_recommended = 1 if dossier.status_label == "Recommended" else 0
    return (
        1 if value is not None else 0,
        value if value is not None else float("-inf"),
        is_recommended,
        dossier.candidate_id,
    )


def _dossier_metric_value_by_title(
    dossier: CandidateDossierViewModel,
    title: str,
    preset: WorkflowReviewPreset | None = None,
) -> float | None:
    if title in {"Length (aa)", "Binder length (aa)"}:
        length_value = _dossier_length_value(dossier)
        return float(length_value) if length_value is not None else None
    for chip in dossier.metric_chips:
        if _cross_candidate_metric_title(chip.label, preset) != title:
            continue
        return _chip_numeric_value(chip)
    return None


def _dossier_length_value(dossier: CandidateDossierViewModel) -> int | None:
    if not dossier.sequence:
        return None
    return len(dossier.sequence)


def _metric_filter_step(
    lo: float,
    hi: float,
    values: list[float] | None = None,
    *,
    integer_only: bool = False,
) -> str:
    span = abs(hi - lo)
    if span <= 0:
        return "1"
    if integer_only:
        return "1"
    endpoint_places = max(
        [_filter_value_decimal_places(value) for value in (values or [lo, hi])],
        default=0,
    )
    endpoint_step = 10 ** (-endpoint_places) if endpoint_places else 1
    if span <= 1:
        step = 0.001
    elif span <= 10:
        step = 0.01
    elif span <= 100:
        step = 0.01
    elif span <= 1000:
        step = 0.1
    else:
        step = 1
    step = min(step, endpoint_step)
    return _fmt_filter_value(step)


def _filter_value_decimal_places(value: float) -> int:
    rendered = _fmt_filter_value(value)
    if "e" in rendered.lower() or "." not in rendered:
        return 0
    return len(rendered.rsplit(".", 1)[1])


def _fmt_filter_value(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.4g}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _saas_empty_pattern_card(title: str, message: str) -> str:
    return (
        f'<div class="saas-pattern-card saas-pattern-card-empty">'
        f'<h3>{_h(title)}</h3><p class="empty">{_h(message)}</p></div>'
    )


def _cross_candidate_metric_title(label: str, preset: WorkflowReviewPreset | None = None) -> str:
    if (
        preset is not None
        and preset.workflow_type == WorkflowType.BINDER
        and _metric_label_for_display(label) == "pLDDT"
    ):
        return "Complex pLDDT"
    return label


def _saas_missing_metrics_card(labels: list[str]) -> str:
    unique = list(dict.fromkeys(label for label in labels if label))
    chips = "".join(
        f'<span class="missing-metric-token">{_h(label)}</span>' for label in unique[:8]
    )
    more = (
        f'<span class="missing-metric-token">+{len(unique) - 8} more</span>'
        if len(unique) > 8
        else ""
    )
    return (
        '<div class="saas-pattern-card saas-missing-metrics-card">'
        "<h3>Not recorded in this run</h3>"
        f'<div class="missing-metric-list">{chips}{more}</div>'
        "<p>Report rendering does not run enrichment, MSA, structure prediction, "
        "or binding tools by default.</p>"
        "</div>"
    )


def _saas_distribution_card(
    title: str,
    values: list[float],
    *,
    value_formatter: Callable[[float], str],
) -> str:
    lo = min(values)
    hi = max(values)
    avg = sum(values) / len(values)
    bins = _distribution_bins(values, bucket_count=8)
    max_count = max(bins) or 1
    bars = "".join(
        f'<span style="height:{max(8, int(count / max_count * 64))}px" title="{_attr(count)} candidate(s)"></span>'
        for count in bins
    )
    return (
        f'<div class="saas-pattern-card metric-range-card"><h3>{_h(title)}</h3>'
        f'<div class="saas-bars saas-bars-distribution">{bars}</div>'
        f'<div class="range-line"><span style="left:{_range_pct(lo, lo, hi)}%"></span>'
        f'<b style="left:{_range_pct(avg, lo, hi)}%"></b>'
        f'<span style="left:{_range_pct(hi, lo, hi)}%"></span></div>'
        f"<p>{_h(value_formatter(lo))} to {_h(value_formatter(hi))} · avg {_h(value_formatter(avg))}</p></div>"
    )


def _distribution_bins(values: list[float], bucket_count: int = 8) -> list[int]:
    lo = min(values)
    hi = max(values)
    span = max(hi - lo, 1e-9)
    bins = [0 for _ in range(bucket_count)]
    for value in values:
        idx = min(bucket_count - 1, int((value - lo) / span * bucket_count))
        bins[idx] += 1
    return bins


def _range_pct(value: float, lo: float, hi: float) -> int:
    if hi <= lo:
        return 50
    return max(0, min(100, int((value - lo) / (hi - lo) * 100)))


def _fmt_range_value(value: float) -> str:
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _dossier_qc_label(dossier: CandidateDossierViewModel) -> str:
    if dossier.status == CandidateStatus.FAILED_QC:
        return "Failed QC"
    if any(chip.label == "QC" and chip.value == "Review" for chip in dossier.metric_chips):
        return "Review QC"
    if any(chip.label == "QC" and chip.value == "Pass" for chip in dossier.metric_chips):
        return "Passed QC"
    return "QC not recorded"


def _diversity_notice(dossiers: list[CandidateDossierViewModel]) -> str:
    has_similarity = any(
        any(group.family == "Novelty & similarity" for group in dossier.score_groups)
        for dossier in dossiers
    )
    if has_similarity:
        return "Similarity evidence is present for at least one candidate; inspect the dossiers before selecting a diverse panel. "
    return "Diversity clustering is not recorded in this report; use length, charge, metric ranges, and any available MSA/similarity evidence as lightweight panel-selection signals. "


def _timestamp_label(value: int | None) -> str:
    if value is None:
        return "Timestamp not recorded"
    try:
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%d %H:%M UTC")
    except (OSError, OverflowError, ValueError, TypeError):
        return str(value)


# ----------------------------------------------------------------- main render


def render_report_html(
    model: ReportModel,
    *,
    viewer_mode: str = "offline",
    report_format: str = "html",
    share: bool = False,
    top_k: int = 20,
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
    # Candidate review console — the report's primary surface. When
    # objective_rank rows exist, build their manifest from the raw collected
    # model *before* display dedupe; otherwise a same-sequence final/support row
    # can be removed before the manifest attaches structure/evaluator metrics.
    # Without objective_rank, keep the legacy display dedupe first: it prevents
    # same-material monomer/QC support rows from appearing as independent
    # alternatives beside an accepted final submission.
    review_input = model if _has_objective_rank_manifest(model) else _dedup_candidate_view(model)
    review = build_review_model(review_input)
    manifest_model = review.base
    saas_vm = build_report_view_model(manifest_model, review)
    display_review, display_limit = _display_limited_review(review, top_k)
    display_board = _display_limited_board(saas_vm.candidate_board, display_review)
    display_dossiers = _display_limited_dossiers(
        saas_vm.candidate_dossiers,
        display_review,
    )
    display_notice = _display_limit_notice(display_limit)
    # Legacy score bars/tables still expect a flattened display model. Run the
    # display-only duplicate collapse on the manifest-prepared model, not on the
    # raw candidate stream, so objective-rank support evidence has already been
    # attached to the canonical rows.
    model = _flatten_candidate_view(_dedup_candidate_view(manifest_model))
    title = f"VPA report · {model.summary.id}"

    # Keep the removed product-stage "Why this workflow?" trace out of the
    # report, but restore the raw DAG as a secondary audit section outside the
    # Appendix so node-level execution remains inspectable without making the
    # appendix feel like a workflow debugger.
    show_workflow_audit = bool(model.graph.nodes)
    show_candidates = bool(model.final_submission or model.final_candidates or model.candidates)
    # The "Candidate Merge Audit" section is suppressed entirely per user
    # request — it duplicated information already conveyed by the Candidates
    # section and read as redundant noise.
    show_merge_audit = False
    # The "Candidate Availability" / "Empty and Filtered Candidate Streams"
    # section is suppressed entirely per user request — its content was mostly
    # no-op parameter-default audit noise and added little to the report.
    show_availability = False

    # The report follows a candidate-first SaaS hierarchy. Low-level per-tool
    # widgets and appendix panels are intentionally suppressed; the raw
    # workflow DAG remains as the audit surface.
    nav_items: list[tuple[str | None, str, str]] = [("overview", "Overview", "overview")]
    if show_candidates:
        nav_items.append(("candidate-dossiers", "Evidence Summary", "dossier"))
        nav_items.append(("candidates", "Candidate Review", "candidate"))
        if _structure_comparison_bundles(display_review):
            nav_items.append(("structure-comparison", "Structure Comparison", "structure"))
        nav_items.append(("candidate-comparison", "Candidate Comparison", "compare"))
        nav_items.append(("cross-candidate-analysis", "Cross-Candidate Analysis", "analytics"))
    if show_workflow_audit:
        nav_items.append(("workflow-audit", "Workflow Audit", "workflow"))
    nav_items.append(("agent-rationale", "Agent Rationale", "agent"))

    def _nav_item_html(anchor: str | None, label: str, glyph: str) -> str:
        if anchor is None:
            return f'<span class="side-nav-divider">{_h(label)}</span>'
        return (
            f'<a href="#{anchor}" class="side-nav-item">'
            f'<span class="side-nav-glyph">{_icon_svg(glyph)}</span>'
            f'<span class="side-nav-label">{_h(label)}</span></a>'
        )

    nav_html = "".join(_nav_item_html(*item) for item in nav_items)

    # Load structures only for the display-limited candidate surface. The full
    # evidence model remains counted above, but heavyweight inline structure
    # payloads follow the report's top-k contract.
    _viewer_entries: list[tuple[ReportCandidate, str]] = []
    _seen_viewer_ids: set[str] = set()
    for _bundle in display_review.bundles:
        _c = _bundle.candidate
        if _c.candidate_id in _seen_viewer_ids:
            continue
        _seen_viewer_ids.add(_c.candidate_id)
        _viewer_entries.append((_c, _structure_dom_id(_c.candidate_id)))
    for _index, _bundle in enumerate(_structure_comparison_bundles(display_review), start=1):
        _c = _bundle.candidate
        _viewer_entries.append((_c, _comparison_structure_dom_id(_c.candidate_id, _index)))
    viewer_payloads = _collect_viewer_payloads(_viewer_entries)
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
        f"<script>/* 3Dmol-min.js (vendored) */{threedmol_src}</script>" if threedmol_src else ""
    )

    overview_section = _saas_overview_section(saas_vm.overview)
    agent_rationale_section = _saas_agent_rationale_section(saas_vm.agent_rationale)

    availability_section = _candidate_availability_section(model) if show_availability else ""
    merge_audit_section = _candidate_merge_audit_section(model) if show_merge_audit else ""

    # ----- Candidate review console (primary surface) -----
    candidate_dossiers_section = (
        _saas_candidate_dossiers_section(
            display_dossiers,
            saas_vm.review_preset,
            manifest_model.final_submission,
        )
        if show_candidates
        else ""
    )
    candidates_section = (
        _review_console_section(
            display_review,
            display_board,
            saas_vm.review_preset,
            display_notice=display_notice,
        )
        if show_candidates
        else ""
    )
    candidate_comparison_section = (
        _saas_candidate_comparison_section(display_board, saas_vm.review_preset, display_review)
        if show_candidates
        else ""
    )
    structure_comparison_section = (
        _structure_comparison_section(display_review, saas_vm.review_preset)
        if show_candidates
        else ""
    )
    cross_candidate_section = (
        _saas_cross_candidate_section(display_dossiers, saas_vm.review_preset)
        if show_candidates
        else ""
    )
    workflow_audit_section = (
        _workflow_audit_section(model, saas_vm.workflow_trace) if show_workflow_audit else ""
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
      <div class="sidebar-brand-main">
        {_report_brand_mark()}
      </div>
      <button class="sidebar-toggle" type="button" aria-label="Collapse sidebar" aria-expanded="true" data-sidebar-toggle>{_icon_svg("chevron-left")}</button>
    </div>
    <div class="sidebar-meta">
      <span class="sidebar-meta-label">Workspace</span>
      <code class="sidebar-meta-value" title="{_attr(model.summary.workspace)}">{_h(model.summary.workspace)}</code>
    </div>
    <nav class="side-nav">{nav_html}</nav>
    <div class="sidebar-foot">
      <span>v{_h(__version__)}</span>
    </div>
  </aside>
  <header class="top-bar">
    <div class="top-bar-left">
      <span class="status-chip state-{_state_class(model.summary.state)}">{_h(saas_vm.overview.display_status)}</span>
      <span class="top-bar-kind">Workflow ID:</span>
      <code class="top-bar-id" title="{_attr(saas_vm.overview.workflow_id)}">{_h(_short(saas_vm.overview.workflow_id, 28))}</code>
      <span class="top-bar-date">{_h(_timestamp_label(saas_vm.overview.timestamp_ms))}</span>
    </div>
    <div class="top-bar-right">
      <button class="top-bar-action top-bar-action-primary" type="button" data-export-report>Export Report</button>
    </div>
  </header>
  <main class="content">
    {overview_section}
    {candidate_dossiers_section}
    {candidates_section}
    {structure_comparison_section}
    {candidate_comparison_section}
    {cross_candidate_section}
    {workflow_audit_section}
    {agent_rationale_section}
    {availability_section}
    {merge_audit_section}
  </main>
  {threedmol_script}
  {viewer_data_script}
  <script>{_js()}</script>
</body>
</html>
"""


# ----------------------------------------------------------------- hero / summary


def _candidate_availability_section(model: ReportModel) -> str:
    satisfied_items = "".join(
        _satisfied_semantic_facet_html(facet) for facet in model.satisfied_semantic_facets[:12]
    )
    empty_items = "".join(
        _empty_candidate_event_html(event) for event in model.empty_candidate_events[:12]
    )
    filter_items = "".join(
        _candidate_capability_filter_event_html(event)
        for event in model.candidate_capability_filter_events[:12]
    )
    # Drop no-op default-fallback events (e.g. rfdiffusion's dozen optional
    # params each logging "adapter schema default will be used") — they bury
    # any real resolution under noise. Keep only meaningful source changes.
    meaningful_param_events = [
        event
        for event in model.parameter_resolution_events
        if not _is_trivial_param_resolution(event)
    ]
    parameter_items = "".join(
        _parameter_resolution_event_html(event) for event in meaningful_param_events[:12]
    )
    empty_block = (
        f"""
  <div class="candidate-group">
    <div class="candidate-group-head">
      <p class="card-eyebrow">Empty facets</p>
      <span class="card-meta">{len(model.empty_candidate_events)} event(s)</span>
    </div>
    <div class="artifact-list">{empty_items}</div>
  </div>"""
        if empty_items
        else ""
    )
    satisfied_block = (
        f"""
  <div class="candidate-group">
    <div class="candidate-group-head">
      <p class="card-eyebrow">Satisfied facets</p>
      <span class="card-meta">{len(model.satisfied_semantic_facets)} facet(s)</span>
    </div>
    <div class="artifact-list">{satisfied_items}</div>
  </div>"""
        if satisfied_items
        else ""
    )
    filter_block = (
        f"""
  <div class="candidate-group">
    <div class="candidate-group-head">
      <p class="card-eyebrow">Capability partitions</p>
      <span class="card-meta">{len(model.candidate_capability_filter_events)} event(s)</span>
    </div>
    <div class="artifact-list">{filter_items}</div>
  </div>"""
        if filter_items
        else ""
    )
    parameter_block = (
        f"""
  <div class="candidate-group">
    <div class="candidate-group-head">
      <p class="card-eyebrow">Parameter resolution</p>
      <span class="card-meta">{len(meaningful_param_events)} event(s)</span>
    </div>
    <div class="artifact-list">{parameter_items}</div>
  </div>"""
        if parameter_items
        else ""
    )
    return f"""<section id="availability" class="card">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Candidate Availability</p>
      <h2 class="card-title">Empty and Filtered Candidate Streams</h2>
    </div>
  </header>
  <p class="widget-explainer">Bounded runtime events showing which retrieval facets returned no candidates and which candidate streams were partitioned before tool execution.</p>
  {satisfied_block}
  {empty_block}
  {filter_block}
  {parameter_block}
</section>"""


def _candidate_merge_audit_section(model: ReportModel) -> str:
    items = "".join(
        _candidate_merge_event_html(event)
        for event in model.candidate_merge_events[:12]
        if isinstance(event, dict)
    )
    return f"""<section id="merge-audit" class="card">
  <header class="card-header">
    <div>
      <p class="card-eyebrow">Candidate Merge Audit</p>
      <h2 class="card-title">Duplicate Enrichment Merges</h2>
    </div>
    <span class="card-meta">{len(model.candidate_merge_events)} event(s)</span>
  </header>
  <p class="widget-explainer">Compact audit entries for duplicate candidate versions that were merged before reporting. Candidate cards show only the retained representative.</p>
  <div class="artifact-list">{items}</div>
</section>"""


def _candidate_merge_event_html(event: dict[str, Any]) -> str:
    groups = event.get("groups") if isinstance(event.get("groups"), list) else []
    group_html = "".join(
        _candidate_merge_group_pill_html(group) for group in groups[:6] if isinstance(group, dict)
    )
    counts = (
        f"{event.get('duplicates_removed') or 0} merged / "
        f"{event.get('duplicate_group_count') or 0} group(s) / "
        f"{event.get('conflict_count') or 0} conflict(s)"
    )
    truncated = " · truncated" if event.get("truncated") else ""
    return f"""
      <div class="artifact-row">
        <div>
          <strong>{_h(str(event.get("node_id") or "merge_candidates"))}</strong>
          <div class="artifact-meta">
            <span class="artifact-pill"><strong>counts</strong> {_h(counts + truncated)}</span>
            {group_html}
          </div>
        </div>
        <code>{_h(str(event.get("tool") or ""))}</code>
      </div>"""


def _candidate_merge_group_pill_html(group: dict[str, Any]) -> str:
    label = str(group.get("retained_candidate_id") or group.get("identity_key") or "candidate")
    details = [
        f"type {group.get('identity_type')}" if group.get("identity_type") else "",
        f"versions {group.get('merged_count')}"
        if group.get("merged_count") not in (None, "")
        else "",
    ]
    tools = ", ".join(_bounded_str_list(group.get("source_tools"), limit=3))
    if tools:
        details.append(f"tools {tools}")
    scores = ", ".join(_bounded_str_list(group.get("score_namespaces"), limit=3))
    if scores:
        details.append(f"scores {scores}")
    return f'<span class="artifact-pill">{_h(label)}: {_h("; ".join(item for item in details if item))}</span>'


def _satisfied_semantic_facet_html(facet: dict[str, Any]) -> str:
    label = facet.get("label") or facet.get("intent_id") or facet.get("query") or "satisfied facet"
    facts = [
        ("intent", facet.get("intent_id")),
        ("query", facet.get("query")),
        ("selected", ", ".join(_bounded_str_list(facet.get("selected_terms"), limit=3))),
        ("candidates", facet.get("candidate_count")),
        (
            "best score",
            _fmt(facet.get("best_sequence_relevance_score"))
            if facet.get("best_sequence_relevance_score") is not None
            else None,
        ),
        ("examples", ", ".join(_bounded_str_list(facet.get("candidate_ids"), limit=3))),
    ]
    facts_html = "".join(
        f'<span class="artifact-pill"><strong>{_h(name)}</strong> {_h(str(value))}</span>'
        for name, value in facts
        if value not in (None, "")
    )
    tools = ", ".join(_bounded_str_list(facet.get("tools"), limit=3))
    return f"""
      <div class="artifact-row">
        <div>
          <strong>{_h(str(label))}</strong>
          <div class="artifact-meta">{facts_html}</div>
        </div>
        <code>{_h(tools)}</code>
      </div>"""


def _empty_candidate_event_html(event: dict[str, Any]) -> str:
    label = event.get("label") or event.get("intent_id") or event.get("query") or "empty facet"
    selected = event.get("selected_term")
    facts = [
        ("intent", event.get("intent_id")),
        ("query", event.get("query")),
        ("selected", selected),
        ("reason", event.get("selected_reason")),
        ("candidates", event.get("candidate_count")),
    ]
    facts_html = "".join(
        f'<span class="artifact-pill"><strong>{_h(name)}</strong> {_h(str(value))}</span>'
        for name, value in facts
        if value not in (None, "")
    )
    return f"""
      <div class="artifact-row">
        <div>
          <strong>{_h(str(label))}</strong>
          <div class="artifact-meta">{facts_html}</div>
        </div>
        <code>{_h(str(event.get("tool") or ""))}</code>
      </div>"""


def _candidate_capability_filter_event_html(event: dict[str, Any]) -> str:
    filtered = (
        event.get("filtered_candidates")
        if isinstance(event.get("filtered_candidates"), list)
        else []
    )
    filtered_html = "".join(
        _filtered_candidate_pill_html(item) for item in filtered[:5] if isinstance(item, dict)
    )
    selected = (
        event.get("selected_candidate_ids")
        if isinstance(event.get("selected_candidate_ids"), list)
        else []
    )
    skipped = (
        event.get("skipped_candidate_ids")
        if isinstance(event.get("skipped_candidate_ids"), list)
        else []
    )
    selected_html = (
        f'<span class="artifact-pill"><strong>selected</strong> {_h(", ".join(map(str, selected)))}</span>'
        if selected
        else ""
    )
    skipped_html = (
        f'<span class="artifact-pill"><strong>skipped sample</strong> {_h(", ".join(map(str, skipped)))}</span>'
        if skipped
        else ""
    )
    counts = (
        f"{event.get('compatible_candidate_count') or 0} compatible / "
        f"{event.get('skipped_candidate_count') or 0} skipped"
    )
    label = (
        "candidate subset selection"
        if event.get("event_type") == "candidate_selection_event"
        else "candidate consumer"
    )
    return f"""
      <div class="artifact-row">
        <div>
          <strong>{_h(str(event.get("node_id") or label))}</strong>
          <div class="artifact-meta">
            <span class="artifact-pill"><strong>status</strong> {_h(str(event.get("status") or ""))}</span>
            <span class="artifact-pill"><strong>counts</strong> {_h(counts)}</span>
            <span class="artifact-pill"><strong>reason</strong> {_h(str(event.get("reason") or ""))}</span>
            <span class="artifact-pill"><strong>downstream</strong> {_h(str(event.get("downstream_tool") or ""))}</span>
            {selected_html}
            {skipped_html}
            {filtered_html}
          </div>
        </div>
        <code>{_h(str(event.get("tool") or ""))}</code>
      </div>"""


def _is_trivial_param_resolution(event: dict[str, Any]) -> bool:
    """True when an event is a no-op default fallback (display noise).

    A parameter-resolution event where the source didn't actually change
    (e.g. ``adapter_default -> adapter_default``, reason "adapter schema
    default will be used") records nothing the reader can act on — it just
    means an optional parameter the planner left unset kept its schema
    default. Adapters with many optional fields (rfdiffusion) emit a dozen
    of these per node, burying any *real* resolution (an override, an
    upstream-derived value, an inferred default) under noise. We drop the
    no-ops from the report and keep only events where the source meaningfully
    moved.
    """
    before = event.get("source_before") or event.get("source")
    after = event.get("source_after")
    if not after:
        # No "after" recorded — keep; this is a differently-shaped event.
        return False
    return before == after


def _parameter_resolution_event_html(event: dict[str, Any]) -> str:
    node = str(event.get("node_id") or "?")
    tool = str(event.get("tool") or "?")
    field = str(event.get("field") or "?")
    before = str(event.get("source_before") or event.get("source") or "?")
    after = str(event.get("source_after") or "?")
    reason = str(event.get("reason") or event.get("status") or "")
    value = event.get("value_summary")
    facts = [
        ("field", field),
        ("source", f"{before} -> {after}"),
        ("reason", reason),
    ]
    if value not in (None, ""):
        facts.append(("value", _short(str(value), 80)))
    facts_html = "".join(
        f'<span class="artifact-pill"><strong>{_h(name)}</strong> {_h(str(val))}</span>'
        for name, val in facts
        if val not in (None, "")
    )
    return f"""
      <div class="artifact-row">
        <div>
          <strong>{_h(node)}</strong>
          <div class="artifact-meta">{facts_html}</div>
        </div>
        <code>{_h(tool)}</code>
      </div>"""


def _filtered_candidate_pill_html(item: dict[str, Any]) -> str:
    label = str(item.get("candidate_id") or "candidate")
    parts: list[str] = []
    source = _source_label(item.get("source"))
    if source:
        parts.append(f"source {source}")
    missing = _bounded_str_list(item.get("missing_capabilities"), limit=3)
    if missing:
        parts.append(f"missing {', '.join(missing)}")
    capabilities = _bounded_str_list(item.get("capabilities"), limit=3)
    if capabilities:
        parts.append(f"caps {', '.join(capabilities)}")
    details = "; ".join(parts) if parts else "filtered"
    return f'<span class="artifact-pill">{_h(label)}: {_h(details)}</span>'


def _source_label(source: Any) -> str:
    if isinstance(source, dict):
        tool = source.get("tool")
        local_id = source.get("local_id") or source.get("native_id") or source.get("node_id")
        if tool and local_id:
            return f"{tool}:{local_id}"
        if tool:
            return str(tool)
        if local_id:
            return str(local_id)
        return ""
    if isinstance(source, str):
        return source
    return ""


def _bounded_str_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = [str(item) for item in value[:limit] if item not in (None, "")]
    if len(value) > limit:
        out.append(f"+{len(value) - limit} more")
    return out


# Aliases that map a canonical metric key to the metric-name suffixes it can
# appear under once a candidate's scores have been flattened to ``tool:metric``
# keys (see ``_flatten_scores``). Lets the executive summary find e.g. pLDDT
# whether it's stored as ``plddt``, ``mean_plddt``, or ``esm3:mean_plddt``.
_SUMMARY_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "plddt": ("plddt", "mean_plddt"),
    "iptm": ("iptm",),
    "ptm": ("ptm",),
    "dockq": ("dockq",),
    "delta_g": ("delta_g", "dg", "binding_energy"),
    "ranking_confidence": ("ranking_confidence",),
}


def _summary_top_candidate(model: ReportModel) -> ReportCandidate | None:
    """The candidate the executive summary should headline.

    Prefers the submitted ``final_candidates`` (the run's actual answer) and
    falls back to the explored pool. Before the candidate-view dedup the
    summary read ``candidates[0]`` directly, but that list is now explored-only
    and can be empty for runs whose explored entries all collapsed into the
    final pick — so reading it alone blanked the whole Confidence panel.
    """
    if model.final_candidates:
        return model.final_candidates[0]
    if model.candidates:
        return model.candidates[0]
    return None


def _summary_candidate_count(model: ReportModel) -> int:
    """Distinct candidate count for KPI tiles / headline.

    ``final_candidates`` and ``candidates`` are disjoint after the view dedup
    (explored entries matching a final pick are dropped), so the run's distinct
    candidate population is simply their sum.
    """
    return len(model.final_candidates) + len(model.candidates)


def _lookup_candidate_metric(candidate: ReportCandidate, base_key: str) -> Any:
    """Resolve a canonical metric for a candidate across flat / nested keys.

    Handles both legacy flat scores (``{"plddt": 0.98}``) and the flattened
    ``tool:metric`` form (``{"esm3:mean_plddt": 0.98}``) produced for enriched
    final picks, plus name aliases (pLDDT ↔ mean_plddt). Returns the first
    numeric match, or ``None``.
    """
    scores = candidate.scores or {}
    aliases = _SUMMARY_METRIC_ALIASES.get(base_key, (base_key,))
    for alias in aliases:
        val = scores.get(alias)
        if val is not None and not isinstance(val, dict):
            return val
    for key, val in scores.items():
        if isinstance(val, dict):
            continue
        metric = str(key).split(":", 1)[-1].lower()
        if metric in aliases:
            return val
    return None


def _lookup_candidate_metric_source(
    candidate: ReportCandidate, base_key: str
) -> tuple[Any, str | None]:
    """Like :func:`_lookup_candidate_metric` but also returns the source tool.

    Flattened scores key on ``tool:metric`` (e.g. ``esm3:mean_plddt``), so the
    prefix names the model that produced the value. Returns ``(value, tool)``;
    ``tool`` is ``None`` for legacy flat keys that carry no prefix.
    """
    scores = candidate.scores or {}
    aliases = _SUMMARY_METRIC_ALIASES.get(base_key, (base_key,))
    # Prefer a ``tool:metric`` key first so the value carries a source; only
    # fall back to an un-prefixed legacy key (no attributable tool) afterwards.
    fallback: Any = None
    found_fallback = False
    for key, val in scores.items():
        if isinstance(val, dict):
            continue
        parts = str(key).split(":", 1)
        if parts[-1].lower() not in aliases:
            continue
        if len(parts) == 2 and parts[0]:
            return val, parts[0]
        if not found_fallback:
            fallback, found_fallback = val, True
    if found_fallback:
        return fallback, None
    return None, None


def _executive_dashboard(model: ReportModel) -> str:
    """Compact, typography-driven run summary (deployment-summary style).

    Replaces the old boxed three-panel dashboard with a restrained layout:
    a single grouped *stat strip* (status / runtime / candidates / nodes /
    artifacts / warnings), then a two-column band — top-candidate *confidence*
    metrics (each tagged with its producing model) and *pipeline* node health.
    No metric cards, no donuts, no decorative colour: structure and type carry
    the hierarchy; colour appears only as a small status dot.
    """
    return f'<div class="exec-summary">{_exec_stat_strip(model)}{_exec_signal_band(model)}</div>'


def _exec_stat_strip(model: ReportModel) -> str:
    """Grouped run statistics as a single typographic strip (no boxes)."""
    state = (model.summary.state or "unknown").lower()
    state_cls = _state_class(model.summary.state)
    n_warn = len(model.warnings)
    n_err = sum(1 for w in model.warnings if w.level == "error")
    if n_err:
        warn_label, warn_val, warn_cls = "Errors", str(n_err), "poor"
    elif n_warn:
        warn_label, warn_val, warn_cls = "Warnings", str(n_warn), "ok"
    else:
        warn_label, warn_val, warn_cls = "Warnings", "0", None
    # (label, value, status_dot_class|None)
    stats: list[tuple[str, str, str | None]] = [
        ("Status", state.title(), state_cls),
        ("Runtime", _duration(model.summary.runtime_ms), None),
        ("Candidates", str(_summary_candidate_count(model)), None),
        ("Pipeline nodes", str(len(model.graph.nodes)), None),
        ("Artifacts", str(len(model.artifacts)), None),
        (warn_label, warn_val, warn_cls),
    ]
    items = []
    for label, value, dot_cls in stats:
        dot = f'<span class="exec-stat-dot dot-{dot_cls}"></span>' if dot_cls else ""
        items.append(
            f'<div class="exec-stat"><div class="exec-stat-val">{dot}{_h(value)}</div>'
            f'<div class="exec-stat-lbl">{_h(label)}</div></div>'
        )
    return f'<div class="exec-stats">{"".join(items)}</div>'


def _exec_signal_band(model: ReportModel) -> str:
    """Two columns: top-candidate confidence (with sources) + pipeline health."""
    cols: list[str] = []
    conf = _exec_confidence_col(model)
    if conf:
        cols.append(conf)
    pipe = _exec_pipeline_col(model)
    if pipe:
        cols.append(pipe)
    if not cols:
        return ""
    return f'<div class="exec-band">{"".join(cols)}</div>'


def _exec_confidence_col(model: ReportModel) -> str:
    top = _summary_top_candidate(model)
    if top is None:
        return ""
    rows: list[str] = []
    for key, label, tip, _lo, _hi in _EXEC_CONFIDENCE_SPECS:
        if len(rows) >= 4:
            break
        src_tool: str | None = None
        if key == "score":
            value = top.score
        else:
            value, src_tool = _lookup_candidate_metric_source(top, key)
        value = _plddt_on_100(key, value)
        num = _num(value)
        if num is None:
            continue
        quality = _quality_class(key, num)
        src = (
            f'<span class="exec-cm-src">&middot; {_h(_tool_display_name(src_tool))}</span>'
            if src_tool
            else ""
        )
        rows.append(
            f'<div class="exec-cm" title="{_attr(tip)}">'
            f'<span class="exec-cm-dot dot-{quality}"></span>'
            f'<span class="exec-cm-val">{_h(_fmt_donut(num))}</span>'
            f'<span class="exec-cm-lbl">{_h(label)}{src}</span>'
            f"</div>"
        )
    if not rows:
        rows.append(
            f'<div class="exec-cm exec-cm-empty"><span class="exec-cm-lbl">'
            f"Top candidate <code>{_h(top.candidate_id)}</code> &middot; no scalar confidence metrics</span></div>"
        )
    return (
        '<div class="exec-col">'
        '<div class="exec-col-head">Confidence <span class="exec-col-sub">top candidate</span></div>'
        f'<div class="exec-cm-list">{"".join(rows)}</div>'
        "</div>"
    )


def _exec_pipeline_col(model: ReportModel) -> str:
    nodes = model.graph.nodes
    if len(nodes) < 2:
        return ""
    counts: dict[str, int] = {}
    for node in nodes:
        counts[(node.state or "unknown").lower()] = (
            counts.get((node.state or "unknown").lower(), 0) + 1
        )
    total = len(nodes)
    n_done = counts.get("completed", 0) + counts.get("cached", 0)
    chips: list[str] = []
    for node in nodes:
        state = (node.state or "unknown").lower()
        chips.append(
            f'<span class="exec-node" title="{_attr(state)}">'
            f'<span class="exec-node-dot dot-{_state_class(node.state)}"></span>'
            f"{_h(node.tool or node.node_id or 'tool')}</span>"
        )
    return (
        '<div class="exec-col">'
        f'<div class="exec-col-head">Pipeline <span class="exec-col-sub">{n_done}/{total} completed</span></div>'
        f'<div class="exec-node-list">{"".join(chips)}</div>'
        "</div>"
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


def _exec_verdict_callout(model: ReportModel) -> str:
    """Single-line plain-English verdict beneath the dashboard.

    Distils ``_executive_secondary`` into a more visually prominent callout
    so the reader has a clear "should I trust this?" cue right below the
    numbers. Icon + colour reflect the verdict severity.
    """
    top = _summary_top_candidate(model)
    if top is None:
        return (
            '<div class="exec-verdict exec-verdict-neutral">'
            '<span class="exec-verdict-icon" aria-hidden="true">•</span>'
            '<div class="exec-verdict-body">'
            '<div class="exec-verdict-head">No candidate designs produced</div>'
            '<div class="exec-verdict-detail">Review the per-tool widgets and artifact list to inspect inputs and intermediate outputs.</div>'
            "</div></div>"
        )
    cautions: list[str] = []
    plddt = _as_unit(_lookup_candidate_metric(top, "plddt"))
    if plddt is not None and plddt < 0.7:
        cautions.append(f"mean pLDDT {plddt * 100:.0f} suggests low structural confidence")
    ptm = _as_unit(_lookup_candidate_metric(top, "ptm"))
    if ptm is not None and ptm < 0.5:
        cautions.append(f"pTM {ptm:.2f} below 0.5 — overall fold may be unreliable")
    # ipTM is the "interface trustworthiness" call — only meaningful when
    # the structure actually HAS an interface. Skip the caution on a
    # single-chain output so we don't issue an "interface unlikely to be
    # trustworthy" warning for a monomer that has no interface to begin
    # with.
    if not _is_monomer_candidate(top):
        iptm = _as_unit(_lookup_candidate_metric(top, "iptm"))
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
            "</div></div>"
        )
    severity = "poor" if (n_err or len(cautions) >= 3) else "ok"
    icon = "✕" if severity == "poor" else "⚠"
    return (
        f'<div class="exec-verdict exec-verdict-{severity}">'
        f'<span class="exec-verdict-icon" aria-hidden="true">{icon}</span>'
        '<div class="exec-verdict-body">'
        '<div class="exec-verdict-head">Review before downstream use</div>'
        f'<div class="exec-verdict-detail">{_h("; ".join(cautions))}.</div>'
        "</div></div>"
    )


def _hero_sequence_block(model: ReportModel) -> str:
    top = _summary_top_candidate(model)
    if top is None:
        return ""
    seq = top.sequence
    if not seq:
        return ""
    seq_html = _colored_sequence(seq)
    return (
        f'<div class="hero-sequence">'
        f'<div class="hero-sequence-head">'
        f'<span class="eyebrow">Candidate sequence · {len(seq)} residues</span>'
        f'<button class="copy-btn" data-copy="{_attr(seq)}">Copy</button>'
        f"</div>"
        f'<div class="hero-sequence-body"><code>{seq_html}</code></div>'
        "</div>"
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
        f"{state_phrase} in {runtime}" if runtime != "runtime n/a" else state_phrase
    ]

    n_candidates = _summary_candidate_count(model)
    if n_candidates:
        pieces.append(
            f"{n_candidates} candidate" if n_candidates == 1 else f"{n_candidates} candidates"
        )

    top = _summary_top_candidate(model)
    if top is not None:
        bits: list[str] = []
        for key, label_short in (
            ("plddt", "pLDDT"),
            ("iptm", "ipTM"),
            ("ptm", "pTM"),
            ("delta_g", "ΔG"),
        ):
            value = _lookup_candidate_metric(top, key)
            if value is not None and _num(value) is not None:
                value = _plddt_on_100(key, value)
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
        worst = max(
            (w.level for w in model.warnings),
            default="warning",
            key=lambda level: {"error": 2, "warning": 1}.get(level, 0),
        )
        return f"{len(model.warnings)} {worst}(s) need review"
    if state == "completed":
        return "no warnings"
    return f"state: {state}"


# ----------------------------------------------------------------- candidates


def _collect_viewer_payloads(entries: list[tuple[ReportCandidate, str]]) -> list[dict[str, Any]]:
    """Build inline structure payloads — once per top candidate, JSON-embedded.

    Each payload includes PDB/CIF text content plus pre-computed pLDDT
    per-residue (B-factor) for client-side colouring. No iframe, no escaped
    HTML attribute — the viewer JS reads from a single <script type="json">
    block.
    """
    payloads: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: dict[str, str] = {}
    for candidate, dom_id in entries:
        if dom_id in seen_ids:
            continue
        seen_ids.add(dom_id)
        if not candidate.structure_path:
            continue
        path = Path(candidate.structure_path)
        if not path.is_file():
            continue
        path_key = str(path.expanduser().resolve())
        if path_key in seen_paths:
            payloads.append(
                {
                    "id": dom_id,
                    "candidate_id": candidate.candidate_id,
                    "filename": path.name,
                    "content_ref": seen_paths[path_key],
                }
            )
            continue
        try:
            content = path.read_text(errors="replace")
        except OSError:
            continue
        if len(content) > 2_000_000:  # 2 MB hard cap to avoid blowing up the report
            content = content[:2_000_000]
        fmt = path.suffix.lower().lstrip(".")
        # Mark standard-AA atoms as ATOM so 3Dmol cartoons the protein and the
        # Sidechains toggle works even for PXDesign/Protenix HETATM-only CIFs.
        content = _normalize_structure_for_viewer(content, fmt)
        seen_paths[path_key] = dom_id
        payloads.append(
            {
                "id": dom_id,
                "candidate_id": candidate.candidate_id,
                "format": fmt,
                "filename": path.name,
                "content": content,
                # Authoritative sidechain check done here in Python by scanning
                # the atom records — more reliable than the browser-side 3Dmol
                # atom scan, which silently misses sidechains for some CIF
                # inputs. PXDesign binder outputs may legitimately be
                # backbone-only, so this flag reflects the actual atom records
                # in the rendered structure rather than the producing tool.
                "has_sidechains": _structure_has_sidechains(content, fmt),
                # Whether the B-factor column carries real per-residue pLDDT.
                # Coordinate-only generators (PXDesign / rfdiffusion) leave it
                # flat-zero, so pLDDT colouring would paint one flat colour —
                # the viewer disables that mode and falls back to a spectrum.
                "has_plddt": _structure_has_plddt(content, fmt),
            }
        )
    return payloads


# Heavy backbone atoms present in every residue. Anything outside this set
# (CB, CG, OD1, NE2, …) proves real sidechain geometry was written out.
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


def _chain_sidechain_flags(content: str, fmt: str) -> list[tuple[str, bool]]:
    """Per-chain ``(chain_id, has_sidechain_atoms)`` in source order.

    A chain counts as carrying sidechains if any *standard amino-acid* residue
    in it has a non-backbone heavy atom (CB, CG, …). The check is keyed off the
    residue (comp) name, not the PDB record type: some writers (PXDesign's
    mmCIF) mark every protein atom as ``HETATM``, so filtering on ``ATOM`` would
    wrongly call a full-atom complex backbone-only, and keying off comp_id also
    avoids counting ligand / cofactor atoms as "sidechains". Hydrogens ignored.

    This is the single primitive behind both the whole-structure
    :func:`_structure_has_sidechains` flag and the per-chain coverage note.
    """
    flags: dict[str, bool] = {}
    order: list[str] = []

    def _note(chain: str, sidechain: bool) -> None:
        if chain not in flags:
            flags[chain] = False
            order.append(chain)
        if sidechain:
            flags[chain] = True

    fmt = (fmt or "").lower()
    try:
        if fmt in {"cif", "mmcif"}:
            lines = content.splitlines()
            cols: list[str] = []
            in_header = False
            start = -1
            for idx, raw in enumerate(lines):
                stripped = raw.strip()
                if stripped == "loop_":
                    in_header, cols = True, []
                    continue
                if in_header and stripped.startswith("_atom_site."):
                    cols.append(stripped.split(".", 1)[1].strip())
                    continue
                if (
                    in_header
                    and stripped.startswith("_")
                    and not stripped.startswith("_atom_site.")
                ):
                    in_header, cols = False, []
                    continue
                if in_header and cols and not stripped.startswith("_"):
                    start = idx
                    break
            if not (cols and start >= 0):
                return []
            atom_idx = (
                cols.index("label_atom_id")
                if "label_atom_id" in cols
                else (cols.index("auth_atom_id") if "auth_atom_id" in cols else -1)
            )
            comp_idx = (
                cols.index("label_comp_id")
                if "label_comp_id" in cols
                else (cols.index("auth_comp_id") if "auth_comp_id" in cols else -1)
            )
            chain_idx = (
                cols.index("label_asym_id")
                if "label_asym_id" in cols
                else (cols.index("auth_asym_id") if "auth_asym_id" in cols else -1)
            )
            elem_idx = cols.index("type_symbol") if "type_symbol" in cols else -1
            if atom_idx < 0 or comp_idx < 0:
                return []
            for raw in lines[start:]:
                stripped = raw.strip()
                if not stripped or stripped.startswith(("_", "#", "loop_")):
                    break
                parts = stripped.split()
                if atom_idx >= len(parts) or comp_idx >= len(parts):
                    continue
                if parts[comp_idx].strip('"').upper() not in _AA3_TO_1:
                    continue
                if 0 <= elem_idx < len(parts) and parts[elem_idx].strip('"').upper() in {"H", "D"}:
                    continue
                chain = parts[chain_idx].strip('"') if 0 <= chain_idx < len(parts) else "?"
                atom_name = parts[atom_idx].strip('"').upper()
                _note(
                    chain,
                    bool(atom_name)
                    and atom_name not in _BACKBONE_ATOMS
                    and not atom_name.startswith(("H", "D")),
                )
        else:
            # PDB / ent: fixed-column ATOM/HETATM records.
            for line in content.splitlines():
                if not line.startswith(("ATOM", "HETATM")) or len(line) < 20:
                    continue
                if line[17:20].strip().upper() not in _AA3_TO_1:
                    continue
                element = line[76:78].strip().upper() if len(line) >= 78 else ""
                if element in {"H", "D"}:
                    continue
                chain = line[21].strip() or "?"
                atom_name = line[12:16].strip().upper()
                _note(
                    chain,
                    bool(atom_name)
                    and atom_name not in _BACKBONE_ATOMS
                    and not atom_name.startswith(("H", "D")),
                )
    except Exception:
        # Never let a parsing hiccup break report generation.
        return []
    return [(chain, flags[chain]) for chain in order]


def _structure_has_sidechains(content: str, fmt: str) -> bool:
    """True if the structure carries any non-backbone heavy atom.

    Backbone-only outputs (rfdiffusion scaffolds, some esm3 predictions, and —
    for the binder chain — PXDesign designs) hold just N/CA/C/O per residue, so
    the viewer's "Sidechains" toggle has nothing to draw. Decided here rather
    than in JS so the answer is stable across 3Dmol parser quirks. An empty
    scan (parse failure) falls back to ``True`` so the control stays usable.
    """
    flags = _chain_sidechain_flags(content, fmt)
    if not flags:
        return True
    return any(has for _chain, has in flags)


def _sidechain_coverage_note(content: str, fmt: str) -> str:
    """Small caption explaining a backbone-only or partially-modelled structure.

    Generalises across monomers and complexes: when one or more chains were
    produced as backbone-only (no sidechain coordinates), a missing-sidechain
    view can read as a rendering bug. This explains the gap. Returns "" when
    every chain is full-atom (nothing to explain).
    """
    flags = _chain_sidechain_flags(content, fmt)
    if not flags:
        return ""
    backbone = [chain for chain, has in flags if not has]
    if not backbone:
        return ""
    full = [chain for chain, has in flags if has]
    if not full:
        msg = (
            "Backbone-only model — no sidechain atoms were generated for this "
            "structure, so the Sidechains view is unavailable."
        )
    elif len(flags) > 1:
        names = ", ".join(f"chain {c}" for c in backbone)
        msg = (
            f"Partial atoms — {names} {'was' if len(backbone) == 1 else 'were'} "
            "produced as backbone-only in this artifact, so sidechains can only "
            "be shown for the rest of the model."
        )
    else:
        msg = (
            "Backbone-only model — no sidechain atoms were generated for this "
            "structure, so the Sidechains view is unavailable."
        )
    return f'<p class="structure-coverage-note">{_h(msg)}</p>'


def _ca_bfactors(content: str, fmt: str) -> list[float]:
    """B-factor of every CA atom, in source order.

    For AlphaFold / ESM / Boltz / Protenix predictions the B-factor column holds
    per-residue pLDDT. Generators that emit only coordinates (PXDesign binder
    CIFs, rfdiffusion scaffolds) leave it at a flat 0.0.
    """
    vals: list[float] = []
    if fmt in {"pdb", "ent"}:
        for line in content.splitlines():
            if line[:6].strip() not in ("ATOM", "HETATM") or len(line) < 66:
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                vals.append(float(line[60:66]))
            except ValueError:
                continue
        return vals
    if fmt == "cif":
        cols = [
            line.strip() for line in content.splitlines() if line.strip().startswith("_atom_site.")
        ]
        if not cols:
            return vals
        try:
            ai = next(i for i, c in enumerate(cols) if c.endswith(".label_atom_id"))
            bi = next(i for i, c in enumerate(cols) if "B_iso_or_equiv" in c)
        except StopIteration:
            return vals
        ncol = len(cols)
        for line in content.splitlines():
            if not line.startswith(("ATOM", "HETATM")):
                continue
            parts = line.split()
            if len(parts) < ncol:
                continue
            if parts[ai].strip().strip('"') != "CA":
                continue
            try:
                vals.append(float(parts[bi]))
            except ValueError:
                continue
    return vals


def _structure_has_plddt(content: str, fmt: str) -> bool:
    """True when the structure carries usable per-residue pLDDT in its B-factors.

    Mirrors the browser-side check (any CA B-factor above ~0): a flat-zero column
    (PXDesign / rfdiffusion coordinate-only outputs) means pLDDT colouring would
    paint the whole chain one colour, so the viewer must not offer it. A parse
    failure falls back to ``True`` so the control stays available.
    """
    vals = _ca_bfactors(content, fmt)
    if not vals:
        return True
    return any(v > 0.01 for v in vals)


def _normalize_structure_for_viewer(content: str, fmt: str) -> str:
    """Mark standard amino-acid atoms as ``ATOM`` for the embedded viewer copy.

    PXDesign / Protenix mmCIF writes *every* protein atom as ``HETATM``. 3Dmol's
    cartoon builder and our backbone / sidechain style selectors both key off
    the het flag (``hetflag = group_PDB == "HETATM"``), so a full-atom design
    wrongly renders as a hetero blob with a Sidechains toggle that draws
    nothing. Flipping the record type for standard residues — keyed off the
    residue name, so genuine ligands / waters are never touched — lets the
    existing polymer-vs-hetero styling work (cartoon + working sidechain
    toggle). Only the *embedded display copy* is rewritten; the artifact on
    disk is untouched.
    """
    fmt = (fmt or "").lower()
    try:
        if fmt in {"cif", "mmcif"}:
            return _cif_mark_protein_as_atom(content)
        # PDB / ent: fixed-column records — group_PDB in cols 1-6, resn 18-20.
        out: list[str] = []
        changed = False
        for line in content.splitlines():
            if (
                line.startswith("HETATM")
                and len(line) >= 20
                and line[17:20].strip().upper() in _AA3_TO_1
            ):
                line = "ATOM  " + line[6:]
                changed = True
            out.append(line)
        return "\n".join(out) if changed else content
    except Exception:
        # Never let a normalisation hiccup break report generation.
        return content


def _cif_mark_protein_as_atom(content: str) -> str:
    """mmCIF variant of :func:`_normalize_structure_for_viewer`."""
    lines = content.splitlines()
    cols: list[str] = []
    in_header = False
    start = -1
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if stripped == "loop_":
            in_header = True
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
            start = idx
            break
    if not (cols and start >= 0 and "group_PDB" in cols):
        return content
    g_idx = cols.index("group_PDB")
    comp_idx = (
        cols.index("label_comp_id")
        if "label_comp_id" in cols
        else (cols.index("auth_comp_id") if "auth_comp_id" in cols else -1)
    )
    if comp_idx < 0:
        return content
    changed = False
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith(("_", "#", "loop_")):
            break
        parts = stripped.split()
        if g_idx >= len(parts) or comp_idx >= len(parts):
            continue
        if parts[g_idx] == "HETATM" and parts[comp_idx].strip('"').upper() in _AA3_TO_1:
            parts[g_idx] = "ATOM"
            lines[i] = " ".join(parts)
            changed = True
    return "\n".join(lines) if changed else content


def _structure_dom_id(candidate_id: str) -> str:
    return "viewer-" + "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in candidate_id)


# 3-letter → 1-letter map for structure-file → FASTA fallback. Mirrors the
# JS-side AA3_TO_1 used by the Pick-mode info card.
_AA3_TO_1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
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
            comp_idx = (
                cols.index("label_comp_id")
                if "label_comp_id" in cols
                else cols.index("auth_comp_id")
            )
            atom_idx = (
                cols.index("label_atom_id")
                if "label_atom_id" in cols
                else cols.index("auth_atom_id")
            )
            asym_idx = (
                cols.index("label_asym_id")
                if "label_asym_id" in cols
                else cols.index("auth_asym_id")
            )
            seq_idx = (
                cols.index("label_seq_id") if "label_seq_id" in cols else cols.index("auth_seq_id")
            )
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
    parse failure or unsupported format. Standard amino-acid CA records are
    read even when a tool marks them HETATM — pxdesign / Protenix flag the
    target protein as a non-polymer (group_PDB=HETATM), and dropping it would
    hide the target chain of a binder complex. Genuine heteroatoms
    (ligands / water / ions) are still excluded so cofactors don't pollute
    chain sequences.
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
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM") or len(line) < 27:
                continue
            if line[12:16].strip() != "CA":
                continue
            resn = line[17:20].strip().upper()
            # Keep standard amino acids even on HETATM rows (protein marked
            # non-polymer); a non-standard HETATM residue is a real heteroatom.
            if rec != "ATOM" and resn not in _AA3_TO_1:
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
            comp_idx = (
                cols.index("label_comp_id")
                if "label_comp_id" in cols
                else cols.index("auth_comp_id")
            )
            atom_idx = (
                cols.index("label_atom_id")
                if "label_atom_id" in cols
                else cols.index("auth_atom_id")
            )
            asym_idx = (
                cols.index("label_asym_id")
                if "label_asym_id" in cols
                else cols.index("auth_asym_id")
            )
            seq_idx = (
                cols.index("label_seq_id") if "label_seq_id" in cols else cols.index("auth_seq_id")
            )
        except ValueError:
            return []
        for raw in lines[start:]:
            row = raw.strip()
            if not row or row.startswith("#") or row.startswith("loop_") or row.startswith("_"):
                break
            parts = row.split()
            if len(parts) < len(cols):
                continue
            if parts[atom_idx].strip('"').strip("'") != "CA":
                continue
            resn = parts[comp_idx].strip('"').strip("'").upper()
            # Keep standard amino acids even when group_PDB marks them HETATM
            # (pxdesign / Protenix mark the target protein as non-polymer);
            # genuine heteroatoms (ligands / water / ions) stay excluded.
            is_atom = grp < 0 or parts[grp] == "ATOM"
            if not is_atom and resn not in _AA3_TO_1:
                continue
            chain = parts[asym_idx]
            resi = parts[seq_idx]
            key = (chain, resi)
            if key in seen:
                continue
            seen.add(key)
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


def _candidate_artifact_metadata(candidate: ReportCandidate) -> dict[str, Any]:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    artifacts = source.get("candidate_artifacts")
    return artifacts if isinstance(artifacts, dict) else {}


def _complex_chain_rows(candidate: ReportCandidate) -> list[dict[str, Any]]:
    artifacts = _candidate_artifact_metadata(candidate)
    for payload in artifacts.values():
        if not isinstance(payload, dict):
            continue
        rows = payload.get("complex_chains")
        if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
            return rows
    return []


def _chain_context_from_metadata(
    candidate: ReportCandidate,
    chains: list[tuple[str, str]],
) -> tuple[_ChainContext, bool]:
    rows = _complex_chain_rows(candidate)
    if not rows or len(chains) <= 1:
        return _ChainContext(), False
    chain_by_order = {
        index: (chain_id, sequence) for index, (chain_id, sequence) in enumerate(chains)
    }

    def chain_for(row: dict[str, Any]) -> tuple[str, int] | None:
        def valid_chain(chain_id: str) -> tuple[str, int] | None:
            expected_length = row.get("sequence_length")
            for observed_chain_id, sequence in chains:
                if observed_chain_id != chain_id:
                    continue
                if (
                    isinstance(expected_length, int)
                    and expected_length > 0
                    and len(sequence) != expected_length
                ):
                    return None
                return observed_chain_id, len(sequence)
            return None

        observed = row.get("observed_chain_id")
        if isinstance(observed, str) and observed:
            chain = valid_chain(observed)
            if chain is not None:
                return chain
        explicit = row.get("chain_id")
        if isinstance(explicit, str) and explicit:
            chain = valid_chain(explicit)
            if chain is not None:
                return chain
        order = row.get("input_order")
        if isinstance(order, int) and order in chain_by_order:
            chain_id, sequence = chain_by_order[order]
            expected_length = row.get("sequence_length")
            if (
                isinstance(expected_length, int)
                and expected_length > 0
                and len(sequence) != expected_length
            ):
                return None
            return chain_id, len(sequence)
        length = row.get("sequence_length")
        if isinstance(length, int) and length > 0:
            candidates = [
                (chain_id, len(sequence))
                for chain_id, sequence in chains
                if len(sequence) == length
            ]
            if len(candidates) == 1:
                return candidates[0]
        return None

    binder_roles = {"binder", "design", "query", "peptide"}
    binder = next(
        (chain_for(row) for row in rows if str(row.get("role") or "").lower() in binder_roles),
        None,
    )
    if binder is None:
        return _ChainContext(), False
    targets: list[tuple[str, int]] = []
    for row in rows:
        if str(row.get("role") or "").lower() in binder_roles:
            continue
        chain = chain_for(row)
        if chain is not None and chain[0] != binder[0]:
            targets.append(chain)
        elif chain is not None and chain[0] == binder[0]:
            return _ChainContext(), True
    if not targets:
        targets = [
            (chain_id, len(sequence)) for chain_id, sequence in chains if chain_id != binder[0]
        ]
    return _ChainContext(binder_chain_id=binder[0], target_chains=tuple(targets)), False


def _chain_context_for(candidate: ReportCandidate) -> _ChainContext:
    """Detect binder-in-complex layout for ``candidate``.

    Returns an empty (non-complex) context unless the structure has 2+
    chains AND exactly one of them is an exact sequence match for
    ``candidate.sequence`` — the unambiguous "this chain is the
    designed binder, the rest are target context" signal.
    """
    if not candidate.structure_path:
        return _ChainContext()
    chains = _extract_chains_from_structure(Path(candidate.structure_path))
    if len(chains) <= 1:
        return _ChainContext()
    metadata_context, block_fallback = _chain_context_from_metadata(candidate, chains)
    if metadata_context.is_complex:
        return metadata_context
    if block_fallback:
        return _ChainContext()
    if not candidate.sequence:
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
_INTERFACE_ONLY_METRICS: frozenset[str] = frozenset(
    {
        "iptm",
        "mean_iptm",
        "ipsae",
        "dockq",
        "iplddt",
        "pae_interaction",
        "sap_score",
        "interface_residues",
        "interface_contacts",
        "interface_size",
        "binding_probability",
        "binder_probability",
        "affinity",
        "affinity_value",
        "affinity_pred_value",
        "affinity_probability",
        "affinity_probability_binary",
        "confidence_score",  # Boltz-2 affinity composite head, complex-only
        "dg",
        "delta_g",
    }
)


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
        cells.append(f'<span class="sst-cell" data-resi="{idx}" data-aa="{_h(aa)}">{_h(aa)}</span>')
    meta_extra = f" · ruler every {ruler_every} residues" if long_seq else ""
    # For binder-design complex outputs, prepend the binder-chain id and
    # follow with a target-context note so the sequence track stops looking
    # like it disagrees with the rendered structure size.
    if chain_context and chain_context.is_complex:
        label_html = (
            f'<span class="sst-label">Binder · chain {_h(chain_context.binder_chain_id)}</span>'
        )
        meta_html = (
            f'<span class="sst-meta">{len(sequence)} residues · '
            f"hover the structure to highlight · "
            f"complex also shows {_h(chain_context.target_summary())}{meta_extra}</span>"
        )
    else:
        label_html = '<span class="sst-label">Sequence</span>'
        meta_html = (
            f'<span class="sst-meta">{len(sequence)} residues · '
            f"hover the structure to highlight{meta_extra}</span>"
        )
    return (
        f'<div class="structure-sequence-track" data-target="{_attr(dom_id)}">'
        '<div class="sst-head">'
        f"{label_html}"
        f"{meta_html}"
        "</div>"
        f'<code class="sst-strip">{"".join(cells)}</code>'
        "</div>"
    )


def _flatten_scores(scores: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten a candidate ``scores`` map to numeric leaves keyed for bars.

    Candidate records from the archive nest their metrics under the
    producing tool (``{"esm3": {"mean_plddt": .98, "ptm": .86}, ...}``),
    which the flat ``tool:metric`` form the score-bar/table renderers
    expect can't read — so those cards come up blank. This flattens one
    level of nesting into ``"esm3:mean_plddt"`` keys and keeps only
    numeric (non-bool) leaves. Already-flat numeric entries pass through
    unchanged, so the function is idempotent on a flat map.
    """
    flat: dict[str, Any] = {}
    for key, value in (scores or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[key] = value
        elif isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, bool) or not isinstance(sub_val, (int, float)):
                    continue
                flat[f"{key}:{sub_key}"] = sub_val
        # strings / lists / None are not bar-renderable — dropped.
    return flat


def _candidate_metric_richness(candidate: ReportCandidate) -> int:
    """Count a candidate's distinct numeric metric leaves.

    Used to pick the "richest" representative when several candidate
    records describe the same protein — the one carrying the most metric
    results wins. A present structure breaks ties toward the viewable one.
    """
    n = len(_flatten_scores(candidate.scores))
    if candidate.score is not None:
        n += 1
    if candidate.structure_path:
        n += 1
    return n


@functools.lru_cache(maxsize=2048)
def _structure_chain_count(path: str) -> int:
    try:
        return len(_extract_chains_from_structure(Path(path)))
    except Exception:
        return 0


def _is_complex_candidate_record(candidate: ReportCandidate) -> bool:
    if _complex_chain_rows(candidate):
        return True
    if candidate.structure_path:
        return _structure_chain_count(candidate.structure_path) > 1
    return False


def _has_structure_evidence(candidate: ReportCandidate) -> bool:
    return bool(candidate.structure_path)


def _has_interface_evidence(candidate: ReportCandidate) -> bool:
    scores = candidate.scores or {}
    for key in ("iptm", "ipsae", "ipsae_score"):
        if key in scores:
            return True
    for namespace in ("protenix_v2", "boltz2", "boltz2_affinity", "ipsae"):
        block = scores.get(namespace)
        if not isinstance(block, dict):
            continue
        if any(key in block for key in ("iptm", "ipTM", "ipsae", "ipsae_score")):
            return True
    return False


def _is_sequence_level_support(candidate: ReportCandidate) -> bool:
    return (
        bool(candidate.sequence)
        and not _has_structure_evidence(candidate)
        and not _is_complex_candidate_record(candidate)
        and not _has_interface_evidence(candidate)
    )


def _can_sequence_dedupe(candidate: ReportCandidate) -> bool:
    return _is_sequence_level_support(candidate)


def _degapped_sequence(value: str | None) -> str:
    # Sequence equality is a material-identity check, not a sanitizer. Only
    # remove common alignment gap/whitespace characters; stripping every
    # non-letter can collapse distinct upstream IDs or synthetic fixture
    # sequences such as ACDE0/ACDE1 into the same candidate.
    return re.sub(r"[-.\s]", "", str(value or "")).upper()


def _same_material_sequence(left: ReportCandidate, right: ReportCandidate) -> bool:
    left_seq = _degapped_sequence(left.sequence)
    right_seq = _degapped_sequence(right.sequence)
    return bool(left_seq and right_seq and left_seq == right_seq)


def _final_sequence_is_safe_material_key(candidate: ReportCandidate) -> bool:
    sequence = _degapped_sequence(candidate.sequence)
    if not sequence:
        return False
    rows = _complex_chain_rows(candidate)
    if not rows:
        # Sequence-only or monomer rows are already material sequence records.
        # Multi-chain structures without role metadata are ambiguous: the stored
        # sequence could be the target, binder, or concatenated context.
        return not (
            candidate.structure_path and _structure_chain_count(candidate.structure_path) > 1
        )
    binder_lengths = [
        int(row.get("sequence_length"))
        for row in rows
        if str(row.get("role") or "").lower() == "binder"
        and isinstance(row.get("sequence_length"), int)
    ]
    return len(binder_lengths) == 1 and binder_lengths[0] == len(sequence)


def _same_sequence_final_support(
    final_candidate: ReportCandidate,
    candidate: ReportCandidate,
) -> bool:
    return _final_sequence_is_safe_material_key(final_candidate) and _same_material_sequence(
        final_candidate, candidate
    )


def _normalized_structure_identity(path: str | None) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return str(Path(raw).expanduser())


def _same_structure_final_support(
    final_candidate: ReportCandidate,
    candidate: ReportCandidate,
) -> bool:
    left = _normalized_structure_identity(final_candidate.structure_path)
    right = _normalized_structure_identity(candidate.structure_path)
    return bool(left and right and left == right)


def _same_material_final_support(
    final_candidate: ReportCandidate,
    candidate: ReportCandidate,
) -> bool:
    return _same_sequence_final_support(
        final_candidate,
        candidate,
    ) or _same_structure_final_support(final_candidate, candidate)


def _candidate_ranked_candidate_id(candidate: ReportCandidate) -> str | None:
    return _identity_candidate_ranked_candidate_id(candidate, include_selection=True)


def _candidate_material_key(candidate: ReportCandidate) -> str | None:
    return _identity_candidate_material_key(candidate, include_selection=True)


def _candidate_view_identity(candidate: ReportCandidate) -> str:
    ranked_id = _candidate_ranked_candidate_id(candidate)
    if ranked_id:
        return f"ranked:{ranked_id}"
    material_key = _candidate_material_key(candidate)
    if material_key:
        return f"material:{material_key}"
    if candidate.candidate_id:
        return f"candidate:{candidate.candidate_id}"
    return ""


def _final_support_for_display_merge(
    final_candidate: ReportCandidate,
    candidate: ReportCandidate,
) -> bool:
    final_ranked_id = _candidate_ranked_candidate_id(final_candidate)
    if final_ranked_id:
        return _candidate_ranked_candidate_id(candidate) == final_ranked_id
    final_material_key = _candidate_material_key(final_candidate)
    if final_material_key:
        return _candidate_material_key(candidate) == final_material_key
    return (
        candidate.candidate_id == final_candidate.candidate_id
        or _same_material_final_support(final_candidate, candidate)
    )


def _has_objective_rank_evidence(candidate: ReportCandidate) -> bool:
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    if isinstance(scores.get("objective_rank"), dict):
        return True
    source = candidate.source if isinstance(candidate.source, dict) else {}
    if source.get("ranked_candidate_id") or source.get("objective_rank_evidence"):
        return True
    return False


def _sequence_support_for_final(
    final_candidate: ReportCandidate,
    candidate: ReportCandidate,
) -> bool:
    return _same_sequence_final_support(
        final_candidate,
        candidate,
    ) and _is_sequence_level_support(candidate)


def _merge_score_namespaces(
    base: dict[str, Any] | None, extra: dict[str, Any] | None
) -> dict[str, Any]:
    """Union two candidate ``scores`` maps, nested tool dicts merged, extra wins.

    Used when enriching a final pick with the matching explored record's
    fuller evaluation suite. Nested tool namespaces (``{"protein_qc": {...}}``)
    are merged key-by-key rather than wholesale-replaced so a partial QC block
    on the final record doesn't shadow extra metrics carried by the explored
    one. Crucially this preserves *non-numeric* leaves (``passes_filters``,
    ``flags``, ``liability_types``) that the review layer relies on — unlike
    :func:`_flatten_scores`, which drops them.
    """
    out: dict[str, Any] = dict(base or {})
    for key, val in (extra or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(val)
            out[key] = merged
        else:
            out[key] = val
    return out


def _merge_candidate_sources_for_display(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    out = dict(base)
    base_artifacts = out.get("candidate_artifacts")
    extra_artifacts = extra.get("candidate_artifacts")
    if not isinstance(base_artifacts, dict) and isinstance(extra_artifacts, dict):
        out["candidate_artifacts"] = extra_artifacts
    return out


def _merge_duplicate_candidate_for_display(
    base: ReportCandidate,
    extra: ReportCandidate,
) -> ReportCandidate:
    base_source = base.source if isinstance(base.source, dict) else {}
    extra_source = extra.source if isinstance(extra.source, dict) else {}
    source = _merge_candidate_sources_for_display(base_source, extra_source)
    sequence = base.sequence
    if _complex_chain_rows(extra) and extra.sequence:
        sequence = extra.sequence
    elif (
        _is_complex_candidate_record(base)
        and base.sequence
        and extra.sequence
        and len(extra.sequence) < len(base.sequence)
    ):
        sequence = extra.sequence
    return replace(
        base,
        sequence=sequence,
        structure_path=base.structure_path or extra.structure_path,
        scores=_merge_score_namespaces(extra.scores, base.scores),
        source=source,
        qc_flags=base.qc_flags or extra.qc_flags,
        liabilities=base.liabilities or extra.liabilities,
        artifacts=base.artifacts or extra.artifacts,
    )


def _dedup_candidate_view(model: ReportModel) -> ReportModel:
    """Deduplicate / enrich candidate lists, keeping ``scores`` *nested*.

    Three display-only transforms (the underlying run record is untouched):

    1. **Enrich final picks.** A ranked/submitted record often carries only
       a couple of headline scores while the matching explored record holds
       the full evaluation suite (sequence_analysis, protein_qc, ...). We
       merge support rows by ranked_candidate_id/material_key first, then by
       safe legacy material checks.
    2. **Drop submitted material duplicates from the explored list.** Explored
       rows with the same ranked/material identity as the final, or rows whose
       sequence safely matches the submitted material, are removed so a complex
       final is not replaced by a monomer/QC variant in the review board.
    3. **Collapse remaining sequence-only duplicates.** Among non-final
       alternatives, keep distinct structured/interface rows separate, but
       merge duplicate sequence-only support rows into the richest
       representative. Objective-ranked rows are not sequence-collapsed. Then
       restore rank order for display.

    Scores are left in their original nested ``{tool: {metric: value}}`` form
    so the review console (which classifies status from ``passes_filters`` /
    evaluator-tool namespaces and renders per-tool evidence) sees the full
    record. :func:`_flatten_candidate_view` performs the numeric-leaf
    flattening the legacy score-bar / table renderers need, as a separate
    step.
    """
    final = list(model.final_candidates)
    explored = list(model.candidates)
    if not final and not explored:
        return model

    final_ids = {c.candidate_id for c in final if not _candidate_view_identity(c)}
    final_identities = {_candidate_view_identity(c) for c in final} - {""}

    enriched_final: list[ReportCandidate] = []
    for fc in final:
        merged: dict[str, Any] = dict(fc.scores or {})
        for ec in explored:
            if _final_support_for_display_merge(fc, ec) and ec.scores:
                merged = _merge_score_namespaces(merged, ec.scores)
        enriched_final.append(replace(fc, scores=merged))

    rank_index = {id(ec): i for i, ec in enumerate(explored)}
    kept: list[ReportCandidate] = []
    kept_by_id: dict[str, int] = {}
    kept_by_seq: dict[str, int] = {}
    kept_rank_index: dict[str, int] = {}
    seen_ids: set[str] = set()
    seen_seqs: set[str] = set()

    def explored_sort_key(candidate: ReportCandidate) -> tuple[int, int, int]:
        final_structure = any(_same_structure_final_support(fc, candidate) for fc in final)
        final_sequence = any(_same_sequence_final_support(fc, candidate) for fc in final)
        return (
            0 if final_structure else (1 if final_sequence else 2),
            0 if _has_objective_rank_evidence(candidate) else 1,
            -_candidate_metric_richness(candidate),
        )

    # Prefer the objective-ranked structure row for submitted material, then
    # fall back to the richest representative for ordinary duplicate support.
    for ec in sorted(explored, key=explored_sort_key):
        view_identity = _candidate_view_identity(ec)
        id_key = view_identity or f"candidate:{ec.candidate_id}"
        if view_identity and view_identity in final_identities:
            continue
        if not view_identity and ec.candidate_id in final_ids:
            continue
        if not _has_objective_rank_evidence(ec) and any(
            _same_material_final_support(fc, ec) for fc in final
        ):
            continue
        can_sequence_dedupe = _can_sequence_dedupe(ec) and not _has_objective_rank_evidence(ec)
        if id_key in seen_ids:
            kept_index = kept_by_id.get(id_key)
            if kept_index is not None:
                kept[kept_index] = _merge_duplicate_candidate_for_display(
                    kept[kept_index],
                    ec,
                )
            continue
        sequence_key = _degapped_sequence(ec.sequence)
        if can_sequence_dedupe and sequence_key in seen_seqs:
            kept_index = kept_by_seq.get(sequence_key)
            if kept_index is not None:
                kept[kept_index] = _merge_duplicate_candidate_for_display(
                    kept[kept_index],
                    ec,
                )
            continue
        seen_ids.add(id_key)
        kept_by_id[id_key] = len(kept)
        if can_sequence_dedupe and sequence_key:
            seen_seqs.add(sequence_key)
            kept_by_seq[sequence_key] = len(kept)
        kept.append(ec)
        kept_rank_index[id_key] = rank_index[id(ec)]
    kept.sort(
        key=lambda ec: kept_rank_index.get(
            _candidate_view_identity(ec) or f"candidate:{ec.candidate_id}",
            rank_index.get(id(ec), 999999),
        )
    )

    return replace(model, final_candidates=enriched_final, candidates=kept)


def _flatten_candidate_view(model: ReportModel) -> ReportModel:
    """Flatten every candidate's nested ``scores`` to numeric ``tool:metric``
    leaves for the legacy score-bar / table renderers."""
    final = [replace(fc, scores=_flatten_scores(fc.scores)) for fc in model.final_candidates]
    explored = [replace(ec, scores=_flatten_scores(ec.scores)) for ec in model.candidates]
    return replace(model, final_candidates=final, candidates=explored)


def _prepare_candidate_view(model: ReportModel) -> ReportModel:
    """Deduplicated + flattened candidate view (legacy convenience).

    Equivalent to :func:`_flatten_candidate_view` applied after
    :func:`_dedup_candidate_view`. ``_render_html`` keeps the two stages
    separate so the review console can consume the nested (un-flattened)
    deduplicated model; this composition is retained for any caller that
    wants the flat view in one shot.
    """
    return _flatten_candidate_view(_dedup_candidate_view(model))


def _has_objective_rank_manifest(model: ReportModel) -> bool:
    for node in model.graph.nodes:
        if (node.tool or "").strip().lower() != "objective_rank":
            continue
        output = node.output if isinstance(node.output, dict) else {}
        if isinstance(output.get("ranked_candidates"), list):
            return True
    return False


# Structure-folding tools whose predictions write per-residue pLDDT into the
# B-factor column, in the order we prefer to credit them. ``protein_qc`` is
# deliberately excluded — it reports a single chain-level pLDDT scalar, it does
# not fold the structure, so it must never be named as the source of the
# per-residue colouring.
_PLDDT_SOURCE_TOOLS: tuple[str, ...] = (
    "boltzgen",
    "protenix_v2",
    "boltz2_affinity",
    "esm3",
    "esm2",
)


def _structure_confidence_tool(candidate: ReportCandidate) -> str | None:
    """Best guess at the tool that produced this structure's per-residue pLDDT.

    Used only to label the pLDDT colour option honestly (we are not using
    AlphaFold). Reads the candidate's recorded ``scores`` namespaces — the
    folding tool always leaves a scores block — and returns the first known
    pLDDT-source tool present, or ``None`` when none is identifiable.
    """
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    present = {str(k).strip().lower() for k in scores}
    for tool in _PLDDT_SOURCE_TOOLS:
        if tool in present:
            return tool
    for key in present:
        prefix, sep, metric = key.partition(":")
        if sep and metric.lower() in {"plddt", "mean_plddt", "atom_plddts", "atom_plddt"}:
            prefix = prefix.strip().lower()
            if prefix in _PLDDT_SOURCE_TOOLS:
                return prefix
    source = candidate.source if isinstance(candidate.source, dict) else {}
    source_tool = str(source.get("tool") or "").strip().lower()
    if source_tool in _PLDDT_SOURCE_TOOLS:
        return source_tool
    return None


def _viewer_block(
    candidate: ReportCandidate,
    chain_context: _ChainContext | None = None,
    *,
    dom_id: str | None = None,
    compact: bool = False,
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
    dom_id = dom_id or _structure_dom_id(candidate.candidate_id)
    fmt = path.suffix.lower().lstrip(".")
    structure_label = path.name
    # Explain (generalisably) when some or all chains are backbone-only, so a
    # missing-sidechain view doesn't read as a rendering bug.
    try:
        _struct_text = path.read_text(errors="replace")[:2_000_000]
    except OSError:
        _struct_text = ""
    coverage_note = _sidechain_coverage_note(_struct_text, fmt) if _struct_text else ""
    is_complex = bool(chain_context and chain_context.is_complex)
    # Prefer the binder chain sequence for explicit binder-target complexes.
    # This protects the viewer even for older rows whose candidate.sequence
    # accidentally held the target chain.
    sequence_str = _display_sequence_for_candidate(candidate, chain_context)
    # Otherwise prefer the candidate's declared sequence; fall back to
    # extracting one from the structure file itself so the sequence track still
    # renders for adapters that don't echo the input sequence.
    sequence_str = sequence_str or _extract_sequence_from_structure(path)
    # Does the B-factor column carry real per-residue pLDDT? Coordinate-only
    # generators (PXDesign binder CIFs, rfdiffusion scaffolds) leave it flat
    # zero, so pLDDT colouring would paint the whole chain one flat colour.
    # In that case the viewer must not default to — or offer — pLDDT.
    has_plddt = _structure_has_plddt(_struct_text, fmt) if _struct_text else True
    # Confidence colouring is the most informative default whenever the artifact
    # carries per-residue pLDDT — for monomers AND complexes. Fall back to the
    # N→C spectrum (orientation cue) only when there is no per-residue
    # confidence. ``chain`` stays a selectable option (it separates binder /
    # target on a complex), but it is never the default: a flat grey-ish chain
    # paint on first load reads as a broken render.
    if has_plddt:
        default_colour = "plddt"
    else:
        default_colour = "spectrum"
    # Name the tool that produced the pLDDT so the label is honest (we are not
    # using AlphaFold). Falls back to a generic label when the folder is unknown.
    plddt_tool = _structure_confidence_tool(candidate)
    plddt_label = (
        f"pLDDT ({_tool_display_name(plddt_tool)})"
        if plddt_tool
        else "pLDDT (per-residue confidence)"
    )
    # Order options so the default one is the first (and only) ``selected``.
    spectrum_option = '<option value="spectrum">N to C spectrum</option>'
    chain_option = '<option value="chain">Chain</option>'
    if not has_plddt:
        # No per-residue confidence in this artifact: keep the option visible
        # for orientation but disabled, and relabel so the reason is explicit.
        plddt_option = '<option value="plddt" disabled>pLDDT (not in this model)</option>'
    else:
        plddt_option = f'<option value="plddt">{_h(plddt_label)}</option>'
    if default_colour == "spectrum":
        spectrum_option = '<option value="spectrum" selected>N to C spectrum</option>'
    else:
        plddt_option = f'<option value="plddt" selected>{_h(plddt_label)}</option>'
    if compact:
        hint_text = "Compare orientation, confidence colour, chains, and gross interface geometry."
    elif is_complex:
        # Explicit complex-composition hint that subsumes the generic
        # Pick-mode pointer. Reads more naturally than two separate hints.
        hint_text = (
            f"Binder chain {chain_context.binder_chain_id} in complex with "
            f"{chain_context.target_summary()}. Turn on Pick, then hover "
            f"residues for an instant residue card."
        )
    else:
        hint_text = "Turn on Pick, then hover residues for an instant residue card."
    # For binder-design complexes the sequence strip belongs to the binder
    # chain ONLY; expose its id so the picker restricts FASTA highlighting to
    # that chain (hovering the target must not light up the binder strip).
    binder_chain_attr = chain_context.binder_chain_id if is_complex and chain_context else ""
    # When the artifact carries no per-residue confidence, say so once so the
    # absence of a pLDDT view doesn't read as a broken viewer. A chain-level QC
    # mean pLDDT may still be reported elsewhere in the report (protein_qc emits
    # a single scalar), which otherwise contradicts the disabled pLDDT view —
    # so when that scalar exists we surface it here regardless of complex/monomer
    # to resolve the apparent "mean pLDDT shown but pLDDT colouring greyed out".
    scalar_plddt = _as_unit(_lookup_candidate_metric(candidate, "plddt"))
    if not has_plddt and scalar_plddt is not None:
        plddt_note = (
            '<p class="structure-coverage-note">This model carries no '
            "per-residue pLDDT — only a chain-level QC mean "
            f"(&asymp;{scalar_plddt * 100:.0f}). The 3D view colours residues "
            "by the structure file's per-residue confidence, which is absent "
            "here, so the pLDDT colour option is disabled.</p>"
        )
    elif not has_plddt and not is_complex:
        plddt_note = (
            '<p class="structure-coverage-note">No per-residue pLDDT in this '
            "artifact — this structure was produced by a coordinate-only generator "
            "(e.g. PXDesign / RFdiffusion), so it is coloured N&rarr;C spectrum "
            "instead of by confidence.</p>"
        )
    else:
        plddt_note = ""
    sequence_track = (
        "" if compact else _structure_sequence_track(sequence_str, dom_id, chain_context)
    )
    residue_info = (
        ""
        if compact
        else f"""<aside class="structure-residue-info" data-target="{_attr(dom_id)}" hidden>
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
  </aside>"""
    )
    pick_button = (
        ""
        if compact
        else f"""<button type="button" class="structure-control structure-picker-btn"
            data-target="{_attr(dom_id)}" aria-pressed="false"
            title="Pick mode: hover residues for an instant residue card (RCSB-style)">&#x2316; Pick</button>"""
    )
    return f"""<section class="structure-section{" structure-section-compact" if compact else ""}" data-target="{_attr(dom_id)}" data-format="{_attr(fmt)}"
         data-candidate-id="{_attr(candidate.candidate_id)}"
         data-structure-name="{_attr(structure_label)}"
         data-sequence="{_attr(sequence_str)}"
         data-binder-chain="{_attr(binder_chain_attr)}"
         data-default-colour="{_attr(default_colour)}">
  <header class="structure-header">
    <span class="structure-title">{_h("Compare structure" if compact else "3D structure")}</span>
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
    {pick_button}
    <button type="button" class="structure-control structure-fullscreen-btn"
            data-target="{_attr(dom_id)}" aria-label="Toggle fullscreen"
            title="Expand (press Esc to exit)">&#x2922; Expand</button>
  </header>
  {coverage_note}
  {plddt_note}
  {sequence_track}
  <div class="structure-viewer" id="{_attr(dom_id)}" data-structure-id="{_attr(dom_id)}"></div>
  <div class="structure-health" data-target="{_attr(dom_id)}" hidden>
    <strong>Structure viewer unavailable.</strong>
    <span>The viewer could not display this structure. Reload the page, close other GPU-heavy tabs, or reopen this report in a browser with WebGL enabled. Structure downloads remain available below.</span>
  </div>
  <div class="structure-legend" data-legend="{_attr(default_colour)}">
    <span class="legend-text">Very low</span>
    <span class="legend-gradient"></span>
    <span class="legend-text">Very high</span>
  </div>
  {residue_info}
</section>"""


def _display_sequence_for_candidate(
    candidate: ReportCandidate,
    chain_context: _ChainContext | None,
) -> str:
    if (
        chain_context
        and chain_context.is_complex
        and chain_context.binder_chain_id
        and candidate.structure_path
    ):
        chain_sequences = dict(_extract_chains_from_structure(Path(candidate.structure_path)))
        sequence = chain_sequences.get(chain_context.binder_chain_id)
        if sequence:
            return sequence
    return candidate.sequence or ""


def _candidate_sequence_block(
    candidate: ReportCandidate,
    chain_context: _ChainContext | None = None,
) -> str:
    sequence = _display_sequence_for_candidate(candidate, chain_context)
    if not sequence:
        return ""
    is_complex = bool(chain_context and chain_context.is_complex)
    # When the structure is a binder-design complex, surface the binder
    # chain id in the head and add a small caption pointing at the target
    # so the reader sees the sequence-vs-structure size gap is intentional.
    head_suffix = f" · binder chain {_h(chain_context.binder_chain_id)}" if is_complex else ""
    target_note = (
        f"<p class='note'>Structure also shows {_h(chain_context.target_summary())}.</p>"
        if is_complex
        else ""
    )
    truncation_note = (
        f"<p class='note'>… {len(sequence) - 240} more residues hidden</p>"
        if len(sequence) > 240
        else ""
    )
    return (
        '<div class="sequence-block">'
        f'<div class="sequence-head"><strong>Sequence</strong> · {len(sequence)} aa'
        f"{head_suffix}"
        f' <button class="copy-btn" data-copy="{_attr(sequence)}">Copy</button></div>'
        f"<code class='msa-seq'>{_colored_sequence(sequence[:240])}</code>"
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
        links.append(
            _download_link(artifact.path, artifact.field, kind=artifact.kind, css="dl-link")
        )
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


# ----------------------------------------------------------------- workflow


def _topo_sort_nodes(
    nodes: list[WorkflowNodeSummary],
) -> list[WorkflowNodeSummary]:
    """Return ``nodes`` in dependency (topological) order for display.

    The graph's stored order is *insertion* order, which need not match
    the data-flow order — e.g. ``protein_qc`` can be inserted ahead of the
    ``esm3`` node it consumes. Ordering by upstream dependencies makes the
    DAG's left-to-right layout (and the numbered composition list) agree
    with the "consumes" edges, removing the apparent contradiction between
    the visual order and the dependency text.

    Kahn's algorithm with an insertion-order-stable ready queue, so
    independent steps keep their original relative order. Falls back to
    the original order on a cycle or when nothing resolves.
    """
    if len(nodes) < 2:
        return list(nodes)
    by_id = {n.node_id: n for n in nodes}
    order_index = {n.node_id: i for i, n in enumerate(nodes)}
    indeg: dict[str, int] = {n.node_id: 0 for n in nodes}
    deps: dict[str, list[str]] = {n.node_id: [] for n in nodes}
    for n in nodes:
        for u in n.upstream or []:
            if u in by_id and u != n.node_id:
                indeg[n.node_id] += 1
                deps[u].append(n.node_id)
    ready = sorted(
        (nid for nid, d in indeg.items() if d == 0),
        key=lambda nid: order_index[nid],
    )
    result: list[str] = []
    while ready:
        nid = ready.pop(0)
        result.append(nid)
        newly = False
        for v in deps.get(nid, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                newly = True
        if newly:
            ready.sort(key=lambda x: order_index[x])
    if len(result) != len(nodes):
        return list(nodes)  # cycle / unresolved — keep stored order
    return [by_id[nid] for nid in result]


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

    # Lay the cards out in data-flow order, not insertion order, so the
    # arrows read left-to-right and match the composition text below.
    nodes = _topo_sort_nodes(nodes)

    NODE_W = 190
    NODE_H = 64
    H_GAP = 44
    PAD_X = 28
    PAD_Y = 32
    cols = len(nodes)
    width = PAD_X * 2 + cols * NODE_W + (cols - 1) * H_GAP
    height = PAD_Y * 2 + NODE_H

    positions = {
        node.node_id: (PAD_X + idx * (NODE_W + H_GAP), PAD_Y) for idx, node in enumerate(nodes)
    }

    parts: list[str] = []
    parts.append(
        "<defs>"
        # Real, very subtle dual-layer drop shadow (Linear / Vercel style).
        '<filter id="dag-shadow" x="-20%" y="-20%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="1" stdDeviation="2" flood-opacity="0.05"/>'
        '<feDropShadow dx="0" dy="3" stdDeviation="6" flood-opacity="0.04"/>'
        "</filter>"
        # Solid filled chevron — bold enough to read as a directional cue
        # without dominating the diagram. ``refX="10"`` lands the chevron
        # tip exactly at the line's endpoint, so the arrow visually touches
        # the destination card. Colour matches the line for a unified feel.
        '<marker id="dag-chev" viewBox="0 0 12 12" refX="10" refY="6" '
        'markerWidth="9" markerHeight="9" orient="auto-start-reverse">'
        '<path d="M1,1 L10,6 L1,11 Z" fill="#a78bfa" stroke="none"/>'
        "</marker>"
        "</defs>"
    )

    # ---------- edges (drawn behind cards) ----------
    seen_edges: set[tuple[str, str]] = set()
    for idx, node in enumerate(nodes):
        x, _ = positions[node.node_id]
        # Use declared upstreams when present, else fall back to sequential.
        upstreams = (
            list(node.upstream) if node.upstream else ([nodes[idx - 1].node_id] if idx > 0 else [])
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

        # Title + subtitle to the right of the circle. The *tool* name is the
        # stable, human-legible label (e.g. "sequence_analysis"); the planner-
        # assigned node_id (e.g. "nadphx_sequence_analysis" or
        # "merge_rank_nadphx_representatives_candidates") can be long and is
        # only useful to disambiguate repeats, so it goes underneath, truncated
        # with a real ellipsis instead of a mid-word cut.
        title_text = node.tool or node.node_id or "step"
        # Text area = NODE_W - left offset - right padding. Limits are tuned to
        # that width (13.5px sans title ≈ 7.3px/char, 11px mono subtitle ≈
        # 6.6px/char); a clip-path below is the hard safety net so even a wide
        # glyph run can never spill past the card edge.
        if len(title_text) > 17:
            title_text = title_text[:16] + "…"
        sub_text = node.node_id or ""
        if len(sub_text) > 19:
            sub_text = sub_text[:18] + "…"
        title_x = x + 46
        text_clip = f"dagtxt{idx}"
        clip_w = NODE_W - (title_x - x) - 12
        parts.append(
            f'<clipPath id="{text_clip}">'
            f'<rect x="{title_x - 2}" y="{y}" width="{clip_w}" height="{NODE_H}"/></clipPath>'
            f'<g clip-path="url(#{text_clip})">'
            f'<text x="{title_x}" y="{y + 26}" class="dag-title">'
            f"{_svg_text(title_text)}</text>"
            f'<text x="{title_x}" y="{y + 44}" class="dag-tool">'
            f"{_svg_text(sub_text)}</text>"
            f"</g>"
        )
        parts.append("</g>")

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
            f'<circle cx="{cx}" cy="{cy}" r="{r}" class="dag-circle state-{state_class}"/>' + check
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
        f"{step_num}</text>"
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
_WF_MSA_KEYS = frozenset(
    {
        "use_msa",
        "msa_backend",
        "use_rna_msa",
        "use_template",
        "msa_paths",
        "msa_index_csv",
        "msa",
    }
)
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
    if tool not in _WF_MSA_AWARE_TOOLS and not any(k in params for k in _WF_MSA_DEFAULTS):
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
            items.append(("MSA", f"{len(mp)} alignment{'s' if len(mp) != 1 else ''} used"))
        elif isinstance(mp, list) or out.get("msa_index_csv"):
            # boltz2_affinity echoed its MSA inputs — empty means single-sequence.
            items.append(
                (
                    "MSA",
                    "MSA index supplied" if out.get("msa_index_csv") else "off — single-sequence",
                )
            )
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
            add(
                "dates",
                f"{df.get('published_after') or '...'} - {df.get('published_before') or '...'}",
            )
    elif tool == "template_select":
        add("engine", out.get("engine"))
        add("query kind", out.get("query_kind"))
    elif tool == "sequence_analysis":
        m = out.get("metrics")
        if isinstance(m, dict):
            add("mode", m.get("analysis_mode"))
    elif tool == "tmalign":
        if out.get("length_a") is not None:
            add("structure A", f"{out['length_a']} aa")
        if out.get("length_b") is not None:
            add("structure B", f"{out['length_b']} aa")
    elif tool == "objective_rank":
        if out.get("candidate_count") is not None:
            add("candidates", out.get("candidate_count"))
        add("objective", out.get("objective_name"))
    elif tool == "submit_candidate":
        # Surface the terminal-submission shape so the workflow composition
        # row reads as "step N: submit_candidate (outcome=accepted, 2
        # candidates, 'top ranked binder')" instead of an unannotated
        # marker. The full resolved snapshot lives on the candidates
        # section; here we only show the at-a-glance signals.
        outcome = out.get("outcome")
        add("outcome", outcome)
        if out.get("auto_selected") is True:
            add("auto-selected", "yes")
        resolved = out.get("candidates_resolved")
        if isinstance(resolved, list) and outcome == "accepted":
            add("candidates", len(resolved))
        reasoning = out.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            add("reasoning", _short(reasoning.strip(), 80))
        if out.get("supersedes"):
            add("supersedes", _short(str(out["supersedes"]), 24))

    # Generic fallback: surface any unambiguous run-config keys the tool
    # echoes (model name, mode, seed, sample count, …) when no tool-specific
    # extractor matched.
    if not items:
        seen: set[str] = set()
        for key, label in (
            ("model_name", "model"),
            ("mode", "mode"),
            ("task_name", "task"),
            ("num_samples", "samples"),
            ("seed", "seed"),
            ("seed_used", "seed"),
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
    nodes = _topo_sort_nodes(model.graph.nodes)
    if len(nodes) < 2:
        return ""
    summary = model.summary
    kind = (summary.kind or "").lower()
    by_id = {n.node_id: n for n in nodes}
    edges = sum(len([u for u in (n.upstream or []) if u in by_id]) for n in nodes)

    if kind == "tool":
        mode_key, mode_label = "tool", "Single tool run"
        lead = "This run executed a single tool."
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
            f"dependenc{'y' if edges == 1 else 'ies'}</span>"
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
                f'<span class="wf-param wf-param-dep">consumes {_h(", ".join(ups))}</span>'
            )
        for label, value in _wf_node_inputs(n):
            chips.append(f'<span class="wf-param"><b>{_h(label)}</b>{_h(value)}</span>')
        params_html = (
            f'<span class="wf-step-params">{"".join(chips)}</span>'
            if chips
            else '<span class="wf-step-params wf-step-noparam">no input parameters recorded</span>'
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
    return "".join(f"<div class='plot'><h3>{_h(plot.title)}</h3>{plot.svg}</div>" for plot in plots)


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
        column_count = block.column_count or (max((len(seq) for _, seq in seqs), default=0))
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
            f"<h3>{_h(block.label)}</h3>"
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
            f"</select></label>"
            f'<label>Zoom <input type="range" class="msa-zoom" min="6" max="18" value="11"></label>'
            f'<span class="msa-cursor">Col - / {column_count}</span>'
            f'<span class="msa-hover">hover a residue</span>'
            f"</div>"
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
_WIDGETS_OWN_ARTIFACTS = frozenset(
    {
        "structure_analysis",
        "boltz2_affinity",
        "protenix_v2",
    }
)


# Tools whose job is purely to *transform* one tool's output into another
# tool's input (format / schema bridges). They don't carry a user-visible
# deliverable of their own — the output is a derivative file that the
# downstream tool consumes — so the per-tool widget would render an empty
# card with just the tool name + state. Skip those entirely. They still
# appear in the Workflow DAG so the run's structure stays auditable.
#
# Add a new tool here when its only output is "convert X to Y format" —
# i.e. nothing a human reviewer would want to inspect in the report.
_TOOLS_WITHOUT_DELIVERABLE = frozenset(
    {
        "mpnn_to_protenix_input",
        "sequence_to_protenix_input",
        "noop",
        # Candidate-stream plumbing. ``merge_candidates`` only unions/dedupes
        # candidate lists between steps (node ids like
        # "merge_rank_<x>_candidates") — it carries no deliverable a reviewer
        # needs to inspect, and the merged pool is already shown in the
        # Candidates section. Keep it in the DAG, drop the per-tool widget.
        "merge_candidates",
        # The explicit termination tool — its deliverable is already shown as
        # the prominent "Final picks" / "Final outcome" panel at the top of
        # the candidates section, so a second copy of the resolved-candidates
        # JSON inside Per-tool widgets would be pure noise. Failures still
        # surface because ``_node_details`` keeps failed passthrough nodes.
        "submit_candidate",
    }
)


def _node_candidate_ids(output: dict[str, Any]) -> list[str]:
    """Candidate ids a node's output covers, in order, de-duplicated.

    Reads the standard ``ranked_candidates`` / ``candidates`` lists every tool
    emits. This is what lets the per-tool panel say *which* samples its result
    spans, so the reader isn't left guessing whether a widget is the final pick
    or the whole batch.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for key in ("ranked_candidates", "candidates"):
        items = output.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                cid = item.get("candidate_id")
                if isinstance(cid, str) and cid and cid not in seen:
                    seen.add(cid)
                    ids.append(cid)
        if ids:
            break
    return ids


@dataclass(frozen=True)
class _PerToolCand:
    """One entry in the per-tool sample selector, mirroring a board card."""

    candidate_id: str
    short_id: str
    status_label: str
    status_cls: str
    is_final: bool
    rank: int | None
    metric: str  # headline metric line, may be empty


def _per_tool_cand_order(review: ReportReviewModel) -> list[_PerToolCand]:
    """Candidate selector order for the per-tool section.

    Flattens the review board groups (final pick first, then ranked
    alternatives, etc.) into :class:`_PerToolCand` entries so the per-tool
    sample selector mirrors the candidate console's order, ranks, and headline
    metrics, with the final pick selected by default.
    """
    order: list[_PerToolCand] = []
    seen: set[str] = set()
    for group in review.board_groups:
        for bundle in group.bundles:
            cid = bundle.candidate.candidate_id
            if not cid or cid in seen:
                continue
            seen.add(cid)
            label, cls = _STATUS_DISPLAY.get(bundle.status, (bundle.status, "other"))
            rank = bundle.display_rank if bundle.display_rank is not None else bundle.candidate.rank
            order.append(
                _PerToolCand(
                    candidate_id=cid,
                    short_id=_short(cid, 16),
                    status_label=label,
                    status_cls=cls,
                    is_final=bundle.status == CandidateStatus.FINAL,
                    rank=rank,
                    metric=_board_headline_metric(bundle),
                )
            )
    return order


def _node_candidate_map(output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map ``candidate_id -> candidate record`` for a node's output.

    Reads the ``ranked_candidates`` / ``candidates`` lists (first non-empty
    wins, matching :func:`_node_candidate_ids`) so each per-candidate widget
    variant can be built from that design's own record.
    """
    for key in ("ranked_candidates", "candidates"):
        items = output.get(key)
        if not isinstance(items, list):
            continue
        out: dict[str, dict[str, Any]] = {}
        for item in items:
            if isinstance(item, dict):
                cid = item.get("candidate_id")
                if isinstance(cid, str) and cid and cid not in out:
                    out[cid] = item
        if out:
            return out
    return {}


def _slice_node_output_for_candidate(
    output: dict[str, Any], cand: dict[str, Any]
) -> dict[str, Any]:
    """Project a node's output down to a single candidate's result.

    Per-tool widgets read several top-level shapes — ``metrics`` / ``flags``
    for QC, ``burial`` / ``interface`` for structure analysis, and the
    ``candidates`` / ``ranked_candidates`` lists for the design + ranking
    widgets. Every candidate carries its own copy of those values under
    ``scores[<tool>]``, so lifting that copy to the top level and filtering the
    candidate lists to this one design lets the *same* widget render a
    candidate-specific panel — no bespoke per-tool branching, no engine change.

    Artifact-derived sub-sections that are genuinely node-level (e.g. the
    per-residue structure-analysis strip parsed from a shared TSV) are left untouched; they
    describe the node's annotation product, not one candidate's metric.
    """
    sliced = dict(output)
    cid = cand.get("candidate_id")
    for key in ("candidates", "ranked_candidates"):
        items = output.get(key)
        if isinstance(items, list):
            match = [c for c in items if isinstance(c, dict) and c.get("candidate_id") == cid]
            sliced[key] = match or [cand]
    scores = cand.get("scores") if isinstance(cand.get("scores"), dict) else {}
    pq = scores.get("protein_qc")
    if isinstance(pq, dict):
        sliced["metrics"] = {k: v for k, v in pq.items() if k not in ("flags", "passes_filters")}
        sliced["flags"] = pq.get("flags") or []
    sa = scores.get("structure_analysis")
    if isinstance(sa, dict):
        for key in ("burial", "interface", "site_contacts", "ligand_pocket"):
            if isinstance(sa.get(key), dict):
                sliced[key] = sa[key]
        # Swap the node-level artifact bundle for THIS candidate's own. The
        # widget loads ``artifacts.summary_json`` (its burial / interface /
        # liabilities / ligand contacts / secondary-structure) and the per-
        # residue TSVs (its SS strip, burial strip, residue table, downloads).
        # Crucially the loaded summary_json *wins* over the top-level shapes, so
        # without this swap every candidate panel would re-render the node's
        # shared aggregate regardless of the burial/interface promotion above.
        cand_arts = cand.get("artifacts")
        sa_arts = cand_arts.get("structure_analysis") if isinstance(cand_arts, dict) else None
        if isinstance(sa_arts, dict):
            sliced["artifacts"] = sa_arts
            # The node-level liabilities list (and SS / ligand-contact
            # aggregates) would otherwise gap-fill into this candidate's panel
            # when its own summary_json omits a key. Drop them so the per-
            # candidate bundle is the single source of truth.
            for leak_key in (
                "liabilities",
                "secondary_structure",
                "ligand_contacts",
                "site_contacts",
                "ligand_pocket",
            ):
                if leak_key in {"site_contacts", "ligand_pocket"} and isinstance(
                    sa.get(leak_key), dict
                ):
                    continue
                sliced.pop(leak_key, None)
    seq_an = scores.get("sequence_analysis")
    if isinstance(seq_an, dict):
        # The node stores a multi-FASTA *pool aggregate* at the top level
        # (``sequence_length`` is the SUM across every design, the AA-composition
        # is the pooled distribution). Re-frame the panel as this single design:
        # use its own scored descriptors and its own residue count, and drop the
        # pool framing so the widget renders one protein, not the batch blob.
        per_metrics = {
            k: v for k, v in seq_an.items() if k not in ("liability_count", "liability_types")
        }
        per_metrics["analysis_mode"] = "single"
        seq_str = cand.get("sequence")
        if isinstance(seq_str, str) and seq_str:
            per_metrics["length"] = len(seq_str)
            sliced["sequence_length"] = len(seq_str)
        sliced["metrics"] = per_metrics
        liab_types = seq_an.get("liability_types")
        sliced["liabilities"] = liab_types if isinstance(liab_types, list) else []
        sliced.pop("representative_sequences", None)
    struct_path = cand.get("structure_path")
    if isinstance(struct_path, str) and struct_path:
        sliced["structure_path"] = struct_path
    return sliced


# Sentinel data-cand value for a node whose result is NOT per-candidate (a
# retrieval / input step). Its single variant stays visible for every
# selection so the selector never blanks it out.
_NODE_SHARED_CAND = "__shared__"


def _node_cand_toolbar(cand_order: list[_PerToolCand]) -> str:
    """Sample selector for the per-tool section, as segmented candidate cards.

    Each card mirrors a candidate-review board card — rank badge (★ for the
    final pick), short id, headline metric, status colour accent — so the two
    surfaces read as one product. Rendered only when 2+ samples exist; a
    single-sample run needs no switch. The ``.node-cand-btn`` / ``data-pt-cand``
    contract is preserved so the existing switching JS keeps working.
    """
    if len(cand_order) < 2:
        return ""
    cards: list[str] = []
    for idx, c in enumerate(cand_order):
        active = " is-active" if idx == 0 else ""
        final_cls = " is-final" if c.is_final else ""
        # Rank token: a star medal for the final pick, a big numeral otherwise.
        if c.is_final:
            token = '<span class="cbc-token-final" title="Final / submitted">&#10003;</span>'
        elif c.rank is not None:
            token = f'<span class="cbc-token-num">{c.rank}</span>'
        else:
            token = '<span class="cbc-token-num cbc-token-dash">&middot;</span>'
        eyebrow = "Final pick" if c.is_final else _h(c.status_label)
        metric_html = f'<span class="cbc-metric">{_h(c.metric)}</span>' if c.metric else ""
        cards.append(
            f'<button type="button" class="node-cand-btn cbrowse-card status-{_h(c.status_cls)}{final_cls}{active}" '
            f'data-pt-cand="{_attr(c.candidate_id)}" role="tab" '
            f'aria-selected="{"true" if idx == 0 else "false"}" '
            f'title="{_attr(c.candidate_id)}">'
            f'<span class="cbc-token">{token}</span>'
            f'<span class="cbc-body">'
            f'<span class="cbc-eyebrow">{eyebrow}</span>'
            f'<span class="cbc-name">{_h(c.short_id)}</span>'
            f"{metric_html}"
            f"</span>"
            f'<span class="cbc-check" aria-hidden="true">&#10003;</span>'
            f"</button>"
        )
    n = len(cand_order)
    return (
        '<div class="node-cand-toolbar" role="region" aria-label="Design selector">'
        '<div class="cbrowse-head">'
        '<span class="cbrowse-title">View results for design</span>'
        f'<span class="cbrowse-count">{n} designs</span>'
        "</div>"
        f'<div class="node-cand-btnrow" role="tablist">{"".join(cards)}</div>'
        "</div>"
    )


def _node_cand_variants(
    node: WorkflowNodeSummary,
    output: dict[str, Any],
    cand_order: list[_PerToolCand],
    cand_map: dict[str, dict[str, Any]],
) -> str:
    """The per-candidate widget variants for one node block.

    When the node carries no candidate scope (or the run has no candidate
    list), a single shared variant is emitted that stays visible for every
    selection. Otherwise one variant per sample is produced by re-running the
    tool widget on a candidate-sliced output; samples the node never processed
    get a muted "didn't pass through this tool" note.
    """
    tool_lc = (node.tool or "").lower()
    # No candidate selector, or node has no per-candidate result -> one shared
    # variant, always shown.
    if not cand_order or not cand_map:
        widget = _render_tool_widget(tool_lc, output)
        return (
            '<div class="node-cand-variants">'
            f'<div class="node-cand-variant is-active" data-cand="{_NODE_SHARED_CAND}">'
            f"{widget}</div></div>"
        )
    variants: list[str] = []
    for idx, c in enumerate(cand_order):
        cid = c.candidate_id
        active = " is-active" if idx == 0 else ""
        cand = cand_map.get(cid)
        if cand is not None:
            body = _render_tool_widget(tool_lc, _slice_node_output_for_candidate(output, cand))
        else:
            body = (
                '<p class="node-cand-na-note">This sample did not pass through '
                "this tool, so it has no result here.</p>"
            )
        variants.append(
            f'<div class="node-cand-variant{active}" data-cand="{_attr(cid)}">{body}</div>'
        )
    return f'<div class="node-cand-variants">{"".join(variants)}</div>'


# Friendly, human-readable name for each tool, used as the prominent per-tool
# block title. Keeps the appendix readable when the planner gives nodes opaque
# ids (e.g. "nadphx_sequence_qc", "rank_nadphx_representatives"): the node id is
# the planner's private label, not something a reader should have to decode.
# Tools missing here fall back to a Title-Cased version of the tool name.
_TOOL_DISPLAY_NAMES: dict[str, str] = {
    "protein_qc": "Protein QC",
    "search_sequence": "Sequence retrieval",
    "go_sequences": "GO sequence retrieval",
    "interpro_sequences": "InterPro sequence retrieval",
    "literature_search": "Literature search",
    "structure_analysis": "Structure analysis",
    "sequence_analysis": "Sequence analysis",
    "sequence_edit": "Sequence edit",
    "objective_rank": "Objective ranking",
    "template_select": "Template selection",
    "boltz2_affinity": "Boltz-2 affinity",
    "protenix_v2": "Protenix structure prediction",
    "pxmeter": "Structure accuracy (PXMeter)",
    "ipsae": "ipSAE interface score",
    "tmalign": "TM-align",
    "mmseqs2_gpu": "MMseqs2",
    "vina": "AutoDock Vina docking",
    "proteinmpnn": "ProteinMPNN",
    "ligandmpnn": "LigandMPNN",
    "bindcraft": "BindCraft",
    "pxdesign": "PXDesign",
    "rfdiffusion": "RFdiffusion",
    "rfdiffusion3": "RFdiffusion3",
    "rfpeptides": "RFpeptides",
    "boltzgen": "BoltzGen",
    "proteina_complexa": "Proteina Complexa",
    "esm2": "ESM-2",
    "esm3": "ESM-3",
}


def _tool_display_name(tool: str | None) -> str:
    """Human-readable title for a tool, for the per-tool block header.

    Falls back to a Title-Cased version of the raw tool id so a tool we don't
    have an explicit label for still reads cleanly rather than as snake_case.
    """
    tool_lc = (tool or "").lower()
    explicit = _TOOL_DISPLAY_NAMES.get(tool_lc)
    if explicit:
        return explicit
    if not tool_lc:
        return "Tool"
    return tool_lc.replace("_", " ").title()


def _node_details(
    nodes: list[WorkflowNodeSummary],
    cand_order: list[_PerToolCand] | tuple = (),
) -> str:
    """Per-tool result widgets, candidate-filterable.

    Each node gets a structured presentation chosen by tool name. A single
    sample selector at the top filters *every* panel to one candidate's result:
    the cards inside each block are rebuilt from that design's own
    ``scores`` (see :func:`_slice_node_output_for_candidate`), so the section
    mirrors the candidate review console rather than showing a node-level
    aggregate. Samples a node never processed are called out explicitly, and
    retrieval / input steps with no per-sample result stay visible for all.

    Node-level context that isn't candidate-specific — the error banner, the
    artifacts footer, and the raw-JSON dump — sits outside the per-candidate
    variants so it reads once per node.

    Tools listed in :data:`_TOOLS_WITHOUT_DELIVERABLE` (format/input bridges)
    are omitted — their output is a derivative file the next tool consumes,
    not something a human reviewer needs to inspect. Errors on those nodes
    DO still surface (we don't want to swallow real failures), but a
    successful pass-through stays out of the report.
    """
    if not nodes:
        return "<p class='empty'>No node details available.</p>"
    cand_order = list(cand_order)
    toolbar = _node_cand_toolbar(cand_order)
    # Two layouts can't coexist cleanly in one stream: evaluators yield one
    # result *per design* (they track the sample selector), while generators /
    # rankers yield a single *cohort* table covering every design at once (a
    # BindCraft batch, a ranking) — slicing those to one candidate is wrong, so
    # they stay invariant. Rather than let that read as a broken selector, split
    # the section into two labelled bands: the selector scopes the per-design
    # band only; the cohort band is rendered once and flagged "not filtered".
    # Banding only kicks in with a real selector (2+ samples); a single-sample
    # run keeps the original flat stream.
    banded = len(cand_order) >= 2
    per_design: list[str] = []
    cohort: list[str] = []
    flat: list[str] = []
    for node in nodes:
        tool_lc = (node.tool or "").lower()
        is_passthrough = tool_lc in _TOOLS_WITHOUT_DELIVERABLE
        node_failed = bool(node.error_code or node.error_message) or node.state in {
            "failed",
            "skipped",
        }
        # Skip passthrough nodes that succeeded — nothing to show. If they
        # failed, render them so the operator can see why the bridge step
        # blocked the rest of the workflow.
        if is_passthrough and not node_failed:
            continue
        output_obj = node.output if isinstance(node.output, dict) else {}
        # Evaluators carry board-keyed per-candidate scores → slice to the
        # selected design (always, even single-sample, so the panel shows that
        # design's own numbers and not the node-level aggregate). Everything
        # else (generators, rankers, other) is a cohort result → rendered once.
        is_evaluator = classify_tool_role(node.tool) == ToolRole.EVALUATOR
        if is_evaluator:
            cand_map = _node_candidate_map(output_obj)
            variants = _node_cand_variants(node, output_obj, cand_order, cand_map)
            cohort_badge = ""
        else:
            # Force a single shared render (empty cand_order/map → shared
            # variant, always visible regardless of the selector).
            variants = _node_cand_variants(node, output_obj, [], {})
            cohort_badge = _cohort_badge(output_obj) if banded else ""
        article = _node_block_article(node, variants, cohort_badge=cohort_badge)
        if not banded:
            flat.append(article)
        elif is_evaluator:
            per_design.append(article)
        else:
            cohort.append(article)

    if not banded:
        if not flat:
            return "<p class='empty'>No tool results to display for this run.</p>"
        return f"{toolbar}{''.join(flat)}"

    if not per_design and not cohort:
        return "<p class='empty'>No tool results to display for this run.</p>"
    sections: list[str] = []
    if per_design:
        sections.append(
            '<section class="nodeband nodeband-perdesign">'
            '<header class="nodeband-head"><span class="nb-glyph" aria-hidden="true">&#9678;</span>'
            '<span class="nodeband-title">Per-design results</span>'
            '<span class="nodeband-sub">Filtered to the design selected above</span></header>'
            f"{toolbar}{''.join(per_design)}</section>"
        )
    if cohort:
        sections.append(
            '<section class="nodeband nodeband-cohort">'
            '<header class="nodeband-head"><span class="nb-glyph" aria-hidden="true">&#9707;</span>'
            '<span class="nodeband-title">All-designs results</span>'
            '<span class="nodeband-sub">Generators &amp; rankers — these span every design, '
            "so they are not filtered by the selector</span></header>"
            f"{''.join(cohort)}</section>"
        )
    return "".join(sections)


def _cohort_badge(output: dict[str, Any]) -> str:
    """A 'shows all N designs' badge for a cohort (non-per-design) node block.

    ``N`` is the node's own candidate count when it exposes one; otherwise the
    badge drops the count and just states it spans every design.
    """
    n = len(_node_candidate_ids(output))
    label = f"Shows all {n} designs · not filtered" if n else "Shows all designs · not filtered"
    return f'<span class="node-block-cohort" title="This result covers every design at once and does not change with the sample selector.">{_h(label)}</span>'


def _node_block_article(
    node: WorkflowNodeSummary,
    variants: str,
    *,
    cohort_badge: str = "",
) -> str:
    """One ``<article class="node-block">`` for a tool node.

    ``variants`` is the per-candidate (or single shared) body produced by
    :func:`_node_cand_variants`; ``cohort_badge`` is the optional "shows all N
    designs" chip shown for cohort nodes in the all-designs band.
    """
    tool_lc = (node.tool or "").lower()
    output_obj = node.output if isinstance(node.output, dict) else {}
    raw_json = _h(json.dumps(_compact(output_obj), indent=2, sort_keys=True))
    error = (
        f"<p class='error'><strong>{_h(node.error_code or 'error')}:</strong> "
        f"{_h(node.error_message or '')}</p>"
        if node.error_code or node.error_message
        else ""
    )
    # Skip the generic "Artifacts" footer for tools whose own widget already
    # surfaces those files with friendlier labels — duplicating them under both
    # "Per-residue analyses" and "Artifacts" is just visual noise.
    artifacts = "" if tool_lc in _WIDGETS_OWN_ARTIFACTS else _node_artifacts_list(node)
    return (
        f'<article class="node-block" data-tool="{_attr(node.tool)}">'
        f'<header class="node-block-head">'
        f'<div><span class="node-block-title">{_h(_tool_display_name(node.tool))}</span>'
        f'<span class="node-block-meta">{_h(node.tool)}@{_h(node.version)} '
        f"· step <code>{_h(node.node_id)}</code></span></div>"
        f'<span class="node-block-headright">{cohort_badge}'
        f'<span class="node-block-state state-{_state_class(node.state)}">{_h(node.state)}</span></span>'
        "</header>"
        f"{error}{variants}{artifacts}"
        f'<details class="node-block-raw"><summary>Raw output (JSON)</summary>'
        f"<pre>{raw_json}</pre></details>"
        "</article>"
    )


def _node_artifacts_list(node: WorkflowNodeSummary) -> str:
    if not node.artifacts:
        return ""
    items = "".join(_download_link(a.path, a.field, kind=a.kind) for a in node.artifacts)
    return (
        '<div class="node-block-artifacts">'
        '<span class="artifact-section-label">Artifacts</span>'
        f'<div class="artifact-row">{items}</div>'
        "</div>"
    )


# ----------------------------------------------------------------- per-tool widgets


# Deterministic per-metric specs for the QC widget. Each entry:
#   (key, display_name, one-sentence_explanation, quality_fn)
# ``quality_fn(value) -> "good"|"ok"|"poor"|"neutral"`` keeps the coloring
# rule explicit (not LLM-generated, fully reproducible across runs).
_QC_METRIC_SPECS: tuple[tuple[str, str, str, Callable[[float], str]], ...] = (
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
    "model_name",
    "model",
    "tool",
    "version",
    "device",
    "seed",
    "seed_used",
    "seeds_used",
    "num_residues",
    "length",
    "n_residues",
    "n_atoms",
    "runtime_ms",
    "runtime_s",
    "elapsed_ms",
    "elapsed_s",
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
        return (
            pretty,
            "Sequence identity — fraction of identical aligned residues (0-1).",
            0.0,
            1.0,
        )
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
            _pretty_metric_name(canonical),
            _fmt(num),
            "Tool-emitted metric. No reference range registered — value shown as-is.",
            quality,
            gauge={"value": num, "frame": True},
        )
    label, explainer, lo, hi = spec
    # Autoscale gauges that may arrive on either 0-1 or 0-100.
    if canonical in _DUAL_SCALE_METRICS and num <= 1.5 and hi >= 50.0:
        hi = 1.0
    return _score_card(
        label,
        _fmt(num),
        explainer,
        quality,
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
    return _render_tool_widget((node.tool or "").lower(), output)


def _render_tool_widget(tool: str, output: dict[str, Any]) -> str:
    """Render a single tool's widget from a (possibly candidate-sliced) output.

    Split out from :func:`_per_tool_widget` so the per-candidate variants in
    :func:`_node_cand_variants` can re-run the exact same dispatch on a sliced
    output dict without needing a ``WorkflowNodeSummary`` instance.
    """
    specialised: Callable[[dict[str, Any]], str] | None = {
        "protein_qc": _qc_widget,
        "ipsae": _metrics_only_widget,
        "mmseqs2_gpu": _metrics_only_widget,
        "tmalign": _metrics_only_widget,
        "vina": _vina_widget,
        "template_select": _template_select_widget,
        "pxmeter": _pxmeter_widget,
        "search_sequence": _search_sequence_widget,
        "go_sequences": _sequence_retrieval_widget,
        "interpro_sequences": _sequence_retrieval_widget,
        "literature_search": _literature_search_widget,
        "structure_analysis": _structure_analysis_widget,
        "sequence_analysis": _sequence_analysis_widget,
        "sequence_edit": _sequence_edit_widget,
        "objective_rank": _objective_rank_widget,
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
        value = _plddt_on_100(key, metrics.get(key))
        num = _num(value)
        if num is None:
            continue
        quality = qual_fn(num)
        if quality not in {"good", "ok", "poor", "neutral"}:
            quality = "neutral"
        gauge_range = _QC_GAUGE_RANGES.get(key)
        gauge = {"value": num, "lo": gauge_range[0], "hi": gauge_range[1]} if gauge_range else None
        # Tag the confidence metrics with the tool that reported them so a
        # pLDDT / pTM / ipTM number here is traceable, not free-floating.
        src = "Protein QC" if key in {"plddt", "ptm", "iptm"} else ""
        cards.append(_score_card(name, _fmt(num), explanation, quality, gauge=gauge, source=src))
    cards_html = _card_grid(cards) if cards else '<p class="empty">No QC metrics emitted.</p>'
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
                f"<strong>⚠️ {_h(flag)}</strong>"
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
        parts.append(f'<span class="seqret-hl">{chunk}</span>' if highlight else chunk)
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
            f'<span class="seqret-query-label">{_h(str(query_label))}</span>' if query_label else ""
        )
        query_html = (
            f'<div class="seqret-query"><span class="seqret-query-id">'
            f"{_h(str(query_id))}</span>{label}</div>"
        )
    summary_html = (
        '<div class="seqret-summary">'
        f'{query_html}<div class="seqret-stats">{"".join(stats)}</div>'
        "</div>"
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
                if meta.get("reviewed") is True
                else ""
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
            "cross-sequence overlap — a stretch of ≥10 residues identical "
            "in another retrieved sequence</span></div>"
            if any(masks)
            else ""
        )
        list_html = (
            f'<div class="seqret-align-head">Retrieved sequences · {len(records)} shown</div>'
            f'<div class="seqret-align-scroll">{"".join(rows)}</div>{legend}'
        )
    elif kept == 0:
        dropped = summary.get("dropped") if isinstance(summary.get("dropped"), dict) else {}
        reason = (
            " — " + ", ".join(f"{v} {str(k).replace('_', ' ')}" for k, v in dropped.items())
            if dropped
            else ""
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


def _search_sequence_widget(output: dict[str, Any]) -> str:
    if not output:
        return ""
    summary = output.get("summary") if isinstance(output.get("summary"), dict) else {}
    source_summaries = output.get("source_summaries")
    if not isinstance(source_summaries, dict):
        source_summaries = (
            summary.get("source_summaries")
            if isinstance(summary.get("source_summaries"), dict)
            else {}
        )
    retrieval_sources = output.get("retrieval_sources")
    if not isinstance(retrieval_sources, list):
        retrieval_sources = (
            summary.get("retrieval_sources")
            if isinstance(summary.get("retrieval_sources"), list)
            else []
        )
    empty_facets = output.get("empty_facets")
    if not isinstance(empty_facets, list):
        empty_facets = (
            summary.get("empty_facets") if isinstance(summary.get("empty_facets"), list) else []
        )

    resolved_terms = output.get("resolved_terms")
    if not isinstance(resolved_terms, dict):
        resolved_terms = {}

    status = str(output.get("retrieval_status") or summary.get("retrieval_status") or "unknown")
    query = str(output.get("query") or "")
    downstream_count = _num(summary.get("downstream_candidate_count"))
    max_downstream = _num(summary.get("max_downstream_candidates"))
    runtime = _num(output.get("runtime_ms"))

    stats = [
        f'<span class="seqret-stat"><b>{_h(status)}</b> status</span>',
    ]
    if downstream_count is not None:
        cap = f" / {int(max_downstream)}" if max_downstream is not None else ""
        stats.append(
            f'<span class="seqret-stat"><b>{int(downstream_count)}{cap}</b> sequences retrieved</span>'
        )
    if runtime is not None:
        stats.append(f'<span class="seqret-stat"><b>{int(runtime)}</b> ms</span>')

    # ---- Resolved terms with a relevance-score bar (the "how relevant was each
    # term the resolver picked" diagram). One row per GO / InterPro term. ----
    term_rows: list[str] = []
    for source_label, term_key in (("GO", "go_terms"), ("InterPro", "interpro_terms")):
        terms = resolved_terms.get(term_key)
        if not isinstance(terms, list):
            continue
        for term in terms:
            if not isinstance(term, dict):
                continue
            tid = str(term.get("term_id") or term.get("id") or "?")
            label = str(term.get("label") or "")
            reason = str(term.get("selected_reason") or "").replace("_", " ")
            score = _num(term.get("term_relevance_score"))
            pct = max(0.0, min(1.0, score)) * 100 if score is not None else 0.0
            score_txt = f"{score:.2f}" if score is not None else "—"
            term_rows.append(
                '<div class="seqret-term">'
                '<div class="seqret-term-head">'
                f'<span class="seqret-term-src">{_h(source_label)}</span>'
                f'<span class="seqret-term-id">{_h(tid)}</span>'
                f'<span class="seqret-term-label">{_h(label)}</span>'
                "</div>"
                '<div class="seqret-relrow">'
                '<span class="seqret-relbar">'
                f'<span class="seqret-relbar-fill" style="width:{pct:.0f}%"></span></span>'
                f'<span class="seqret-relval">{_h(score_txt)}</span>'
                f'<span class="seqret-term-reason">{_h(reason)}</span>'
                "</div></div>"
            )
    terms_html = (
        '<div class="seqret-align-head">Resolved terms · relevance score</div>'
        f'<div class="seqret-terms">{"".join(term_rows)}</div>'
        if term_rows
        else ""
    )

    # ---- Per-source yield: how many sequences each retrieval stream returned,
    # drawn as proportional bars so the contribution is read at a glance. ----
    yields: list[tuple[str, int, str, int]] = []
    for source in ("go", "interpro"):
        row = source_summaries.get(source) if isinstance(source_summaries, dict) else None
        if not isinstance(row, dict):
            continue
        count = int(_num(row.get("candidate_count")) or 0)
        term_count = int(_num(row.get("term_count")) or 0)
        yields.append((source.upper(), count, str(row.get("status") or "unknown"), term_count))
    max_count = max((c for _, c, _, _ in yields), default=0)
    yield_rows: list[str] = []
    for src_label, count, src_status, term_count in yields:
        width = (count / max_count * 100) if max_count > 0 else 0.0
        empty_cls = " is-empty" if count == 0 else ""
        note = (
            f" · {src_status}"
            if count == 0
            else f" · {term_count} term{'s' if term_count != 1 else ''}"
        )
        yield_rows.append(
            f'<div class="seqret-yield{empty_cls}">'
            f'<span class="seqret-yield-label">{_h(src_label)}</span>'
            '<span class="seqret-yield-track">'
            f'<span class="seqret-yield-fill" style="width:{width:.0f}%"></span></span>'
            f'<span class="seqret-yield-val">{count}<span class="muted">{_h(note)}</span></span>'
            "</div>"
        )
    yield_html = (
        '<div class="seqret-align-head">Sequences retrieved per source</div>'
        f'<div class="seqret-yields">{"".join(yield_rows)}</div>'
        if yield_rows
        else ""
    )

    # ---- Concise note for sources that resolved no term (replaces the raw
    # empty-facet dump). Surfaces the first resolver error so a silent
    # InterPro/API failure is explainable, not noise. ----
    unresolved: list[str] = []
    for facet in empty_facets:
        if not isinstance(facet, dict):
            continue
        stage = str(facet.get("stage") or "").replace("_sequences", "").upper() or "source"
        reason = str(facet.get("selected_reason") or "no term resolved")
        evidence = facet.get("resolver_evidence")
        errs = evidence.get("errors") if isinstance(evidence, dict) else None
        err_txt = ""
        if isinstance(errs, list) and errs:
            err_txt = f" ({_short(str(errs[0]), 140)})"
        unresolved.append(f"<li><b>{_h(stage)}</b> — {_h(reason)}{_h(err_txt)}</li>")
    unresolved_html = (
        '<div class="seqret-align-head">Unresolved sources</div>'
        f'<ul class="seqret-unresolved">{"".join(unresolved)}</ul>'
        if unresolved
        else ""
    )

    query_html = (
        f'<div class="seqret-query"><span class="seqret-query-id">{_h(query)}</span></div>'
        if query
        else ""
    )
    explainer = (
        '<p class="widget-explainer">'
        "Resolves the semantic sequence query into GO and InterPro terms (scored for "
        "relevance), retrieves UniProt sequences linked to each resolved term, then "
        "deduplicates and bounds the downstream sequence stream. The bars below show how "
        "relevant each resolved term was and how many sequences each source contributed."
        "</p>"
    )
    return (
        '<div class="widget widget-seqret">'
        '<h4 class="widget-title">Sequence retrieval</h4>'
        f'{explainer}<div class="seqret-summary">{query_html}'
        f'<div class="seqret-stats">{"".join(stats)}</div></div>'
        f"{terms_html}{yield_html}{unresolved_html}</div>"
    )


# DSSP 8-state secondary-structure codes -> 3-state buckets for the SS bar.
# H/G/I = helices, E/B = strands; turn / bend / polyproline / coil -> loop.
_DSSP_SS_3STATE: dict[str, str] = {
    "H": "helix",
    "G": "helix",
    "I": "helix",
    "E": "sheet",
    "B": "sheet",
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
        ss_cells.append(f'<span class="strip-cell ss-{bucket}" title="{_attr(tip)}"></span>')
        show = ri in (first_resi, last) or (ri % 10 == 0 and abs(ri - last) > 2)
        ruler_cells.append(
            f'<span class="strip-cell strip-ruler-cell">{ri if show else "&nbsp;"}</span>'
        )
    head = (
        f"Chain {_h(chain_label)} · residues {first_resi}-{last}"
        if chain_label
        else f"Residues {first_resi}-{last}"
    )
    return (
        '<div class="strip-block">'
        f'<div class="strip-head">{head}</div>'
        '<div class="strip-scroll">'
        '<div class="strip-row"><span class="strip-label">SS</span>'
        f'<div class="strip-cells">{"".join(ss_cells)}</div></div>'
        '<div class="strip-row strip-ruler"><span class="strip-label"></span>'
        f'<div class="strip-cells">{"".join(ruler_cells)}</div></div>'
        "</div>"
        '<div class="strip-legend">'
        '<span class="legend-pip ss-helix"></span>helix'
        '<span class="legend-pip ss-sheet"></span>sheet'
        '<span class="legend-pip ss-loop"></span>loop'
        "</div>"
        "</div>"
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
        if missing_txt
        else ""
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
    badge_html = f'<span class="tmpl-badges">{"".join(badges)}</span>' if badges else ""
    desc_html = f'<span class="tmpl-desc">{_h(desc)}</span>' if desc and desc != tid else ""
    head = (
        '<div class="tmpl-card-head">'
        f'<span class="tmpl-rank">#{rank}</span>'
        f'<span class="tmpl-id">{_h(tid)}</span>'
        f"{desc_html}{badge_html}</div>"
    )

    bars: list[str] = []
    for label, frac in (
        ("Identity", _num(t.get("identity"))),
        ("Coverage", _num(t.get("coverage"))),
    ):
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
        chips.append(f'<span class="tmpl-chip">Length <b>{_h(str(t.get("length")))} aa</b></span>')
    chip_html = f'<div class="tmpl-chips">{"".join(chips)}</div>' if chips else ""
    metrics = (
        f'<div class="tmpl-metrics">{"".join(bars)}{chip_html}</div>' if bars or chip_html else ""
    )
    return f'<div class="tmpl-card">{head}{metrics}</div>'


def _template_sequences_html(templates: list[dict[str, Any]], seqs: dict[str, str]) -> str:
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
            f'<span class="tmpl-seq-id">{ident * 100:.0f}% id</span>' if ident is not None else ""
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
        meta_bits.append(f"{output['engine']} engine")
    if output.get("query_kind"):
        meta_bits.append(f"{output['query_kind']} query")
    meta_bits.append(f"{len(templates)} template{'s' if len(templates) != 1 else ''}")
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
        note = '<p class="note">Template sequences are not embedded in this report.</p>'
    return (
        '<div class="widget widget-tmpl">'
        '<h4 class="widget-title">Template selection</h4>'
        f'{explainer}{meta}<div class="tmpl-list">{cards}</div>{seq_html}{note}</div>'
    )


def _structure_analysis_widget(output: dict[str, Any]) -> str:
    """Rich Structure-analysis widget.

    Pulls together: burial / SS summary cards (from output.summary or output
    itself), per-residue secondary-structure + RSA strips parsed from the
    structure annotation TSV artifacts, liability markers placed on the
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
    for key in (
        "burial",
        "interface",
        "site_contacts",
        "ligand_pocket",
        "chains",
        "liabilities",
        "annotation_scope",
    ):
        if key not in summary and key in output:
            summary[key] = output[key]

    burial = summary.get("burial") if isinstance(summary.get("burial"), dict) else {}
    interface = summary.get("interface") if isinstance(summary.get("interface"), dict) else {}
    site_contacts = (
        summary.get("site_contacts") if isinstance(summary.get("site_contacts"), dict) else {}
    )
    ligand_pocket = (
        summary.get("ligand_pocket") if isinstance(summary.get("ligand_pocket"), dict) else {}
    )
    chains = summary.get("chains") if isinstance(summary.get("chains"), list) else []
    liabilities = summary.get("liabilities") if isinstance(summary.get("liabilities"), list) else []
    scope = (
        summary.get("annotation_scope") if isinstance(summary.get("annotation_scope"), dict) else {}
    )
    annotated = scope.get("annotated_residue_count") if isinstance(scope, dict) else None

    # Per-residue parsed rows (chain, resi, aa, rsa, burial_class, ss, neighbours)
    annotations_rows = (
        _parse_residue_annotations_tsv(artifact_paths.get("residue_annotations_tsv"))
        if isinstance(artifact_paths, dict)
        else []
    )
    # If annotations TSV missing, fall back to secondary-structure rows so the SS strip
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
        cards.append(
            _score_card(
                "% helix",
                f"{helix_pct:.0f}%",
                "Fraction of residues classified as alpha-helix by DSSP / pydssp.",
                "neutral",
                gauge={"value": helix_pct, "lo": 0.0, "hi": 100.0},
            )
        )
        cards.append(
            _score_card(
                "% sheet",
                f"{sheet_pct:.0f}%",
                "Fraction of residues classified as beta-sheet / strand.",
                "neutral",
                gauge={"value": sheet_pct, "lo": 0.0, "hi": 100.0},
            )
        )
        cards.append(
            _score_card(
                "% loop",
                f"{loop_pct:.0f}%",
                "Fraction of residues classified as loop / coil.",
                "neutral",
                gauge={"value": loop_pct, "lo": 0.0, "hi": 100.0},
            )
        )
    if isinstance(burial, dict) and burial:
        buried = int(burial.get("buried") or 0)
        exposed = int(burial.get("exposed") or 0)
        intermediate = int(burial.get("intermediate") or 0)
        burial_total = buried + exposed + intermediate
        if burial_total:
            cards.append(
                _score_card(
                    "Buried residues",
                    f"{buried} / {burial_total}",
                    "Residues with relative solvent accessibility <0.25 (DSSP convention).",
                    "neutral",
                    gauge={"value": buried, "lo": 0.0, "hi": float(burial_total)},
                )
            )
    if isinstance(interface, dict) and interface:
        ligand_contacts = interface.get("ligand_contact_count")
        if ligand_contacts is not None:
            n_lc = _num(ligand_contacts) or 0.0
            cards.append(
                _score_card(
                    "Ligand contacts",
                    _fmt(ligand_contacts),
                    "Heavy-atom contacts (<5 A) between the protein and any non-protein hetero residue.",
                    "neutral",
                    gauge={"value": n_lc, "lo": 0.0, "hi": max(20.0, n_lc * 2)},
                )
            )
        buried_fraction = (
            ligand_pocket.get("ligand_buried_fraction") if isinstance(ligand_pocket, dict) else None
        )
        if buried_fraction is not None:
            n_bf = _num(buried_fraction)
            if n_bf is not None:
                cards.append(
                    _score_card(
                        "Ligand buried",
                        f"{n_bf * 100:.0f}%",
                        "Fraction of ligand heavy atoms with at least one protein heavy-atom contact.",
                        "neutral",
                        gauge={"value": n_bf * 100, "lo": 0.0, "hi": 100.0},
                    )
                )
        interface_residues = interface.get("interface_residue_count")
        if interface_residues:
            n_ir = _num(interface_residues) or 0.0
            cards.append(
                _score_card(
                    "Interface residues",
                    _fmt(interface_residues),
                    "Residues with at least one inter-chain contact within 5 A.",
                    "neutral",
                    gauge={"value": n_ir, "lo": 0.0, "hi": max(20.0, n_ir * 2)},
                )
            )
    if isinstance(site_contacts, dict) and site_contacts:
        site_contact_count = site_contacts.get("contact_count")
        mapped_count = site_contacts.get("mapped_position_count")
        if site_contact_count is not None and mapped_count:
            n_sc = _num(site_contact_count) or 0.0
            n_mapped = _num(mapped_count) or 0.0
            cards.append(
                _score_card(
                    "Site contacts",
                    f"{_fmt(site_contact_count)} / {_fmt(mapped_count)}",
                    "Requested target-site positions with at least one partner-chain heavy-atom contact.",
                    "neutral",
                    gauge={"value": n_sc, "lo": 0.0, "hi": max(1.0, n_mapped)},
                )
            )
    cards_html = _card_grid(cards)

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
                _h(label_map.get(field, field.replace("_", " "))) for field in empty_artifact_fields
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
        (
            f'<p class="note">Chains: {", ".join(_h(str(c)) for c in chains)}'
            + (f" · annotated {annotated} residues" if annotated else "")
            + "</p>"
        )
        if chains
        else ""
    )
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
            aa = r.get("aa") or "?"
            ss = (r.get("secondary_structure") or "").lower()
            ss_class = "ss-helix" if ss == "helix" else ("ss-sheet" if ss == "sheet" else "ss-loop")
            burial = (r.get("burial_class") or "").lower()
            b_class = (
                "burial-buried"
                if burial == "buried"
                else "burial-intermediate"
                if burial == "intermediate"
                else "burial-exposed"
                if burial == "exposed"
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
                f'<span class="strip-cell strip-ruler-cell">{ri if show_label else "&nbsp;"}</span>'
            )

        blocks.append(
            f'<div class="strip-block">'
            f'<div class="strip-head">Chain {_h(chain) or "?"} · residues '
            f"{first_resseq}-{last_resseq}</div>"
            f'<div class="strip-scroll">'
            f'<div class="strip-row"><span class="strip-label">SS</span>'
            f'<div class="strip-cells">{"".join(ss_cells)}</div></div>'
            f'<div class="strip-row"><span class="strip-label">Burial</span>'
            f'<div class="strip-cells">{"".join(burial_cells)}</div></div>'
            + (
                f'<div class="strip-row"><span class="strip-label" '
                f'title="Structural liability candidates flagged by protein_qc / structure_analysis '
                f'(e.g. buried polar residues, aggregation-prone stretches). Hover a marker for the type.">'
                f"Liability</span>"
                f'<div class="strip-cells">{"".join(liab_cells)}</div></div>'
                if any(liab_positions.get((chain, int(r.get("resseq") or -1))) for r in chain_rows)
                else ""
            )
            + f'<div class="strip-row strip-ruler"><span class="strip-label"></span>'
            f'<div class="strip-cells">{"".join(ruler_cells)}</div></div>'
            f"</div>"  # close .strip-scroll
            f'<div class="strip-legend">'
            '<span class="legend-pip ss-helix"></span>helix'
            '<span class="legend-pip ss-sheet"></span>sheet'
            '<span class="legend-pip ss-loop"></span>loop'
            '<span class="legend-pip burial-buried"></span>buried'
            '<span class="legend-pip burial-intermediate"></span>intermediate'
            '<span class="legend-pip burial-exposed"></span>exposed'
            + (
                '<span class="legend-pip strip-liab"></span>liability flag'
                " (e.g. buried polar / aggregation-prone residue)"
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
    present_cols = [(k, name) for k, name in header_cols if any(k in r for r in rows)]
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
                ss_class = (
                    "ss-helix" if s == "helix" else ("ss-sheet" if s == "sheet" else "ss-loop")
                )
                cells.append(f'<td><span class="ss-chip {ss_class}">{_h(v)}</span></td>')
            elif key == "burial_class":
                b = str(v).lower()
                b_class = (
                    "burial-buried"
                    if b == "buried"
                    else "burial-intermediate"
                    if b == "intermediate"
                    else "burial-exposed"
                    if b == "exposed"
                    else "burial-na"
                )
                cells.append(f'<td><span class="ss-chip {b_class}">{_h(v)}</span></td>')
            else:
                cells.append(f"<td>{_h(str(v))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<details class="residue-annotations">'
        f"<summary>Per-residue annotations table ({len(rows)} residues)</summary>"
        '<div class="table-wrap"><table class="residue-annotations-table">'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div></details>"
    )


# Headline sequence-analysis properties: key -> (label, explainer, lo, hi).
# ``lo``/``hi`` None => frame ring (count / signed value with no fixed range).
_SEQANA_PROP_SPECS: dict[str, tuple[str, str, float | None, float | None]] = {
    "length": ("Length", "Total residue count.", None, None),
    "molecular_weight": (
        "Molecular weight (Da)",
        "Mass of the chain in daltons.",
        None,
        None,
    ),
    "pI": (
        "Isoelectric point",
        "pH of net-zero charge. Typical soluble proteins span pI 4-9.",
        0.0,
        14.0,
    ),
    "gravy": (
        "GRAVY",
        "Grand average of hydropathy (Kyte-Doolittle). Negative = hydrophilic / soluble.",
        -2.0,
        2.0,
    ),
    "net_charge": ("Net charge", "Net charge estimated at pH 7.", None, None),
    "instability_index": (
        "Instability index",
        "Guruprasad heuristic. <40 predicts a protein stable in vitro.",
        0.0,
        80.0,
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
    gauge = {"value": num, "frame": True} if lo is None else {"value": num, "lo": lo, "hi": hi}
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

    # When the tool ran over a *multi-FASTA* (a whole pool of retrieved
    # sequences, not one protein), ``length`` is the SUM of residues across
    # every sequence — showing it as a single "12,859 residues" protein is
    # misleading. Count the sequences so we can label the aggregate honestly
    # and drop the single-protein length gauge below.
    is_multi = str(metrics.get("analysis_mode") or "").lower() == "multi_fasta"
    n_seqs = 0
    if is_multi:
        for key in ("candidates", "representative_sequences", "sequences"):
            v = output.get(key)
            if isinstance(v, list):
                n_seqs = len(v)
                break
        if not n_seqs:
            n_seqs = int(_num(metrics.get("sequence_count")) or 0)

    # ---- meta line ----
    meta_bits: list[str] = []
    if metrics.get("analysis_mode"):
        meta_bits.append(str(metrics["analysis_mode"]).replace("_", " "))
    if metrics.get("metric_backend"):
        meta_bits.append(f"{metrics['metric_backend']} backend")
    length = metrics.get("length", output.get("sequence_length"))
    if length is not None:
        noncanon = _num(metrics.get("noncanonical_residue_count")) or 0
        if is_multi:
            seq_part = f"{n_seqs} sequences · " if n_seqs else ""
            meta_bits.append(
                f"{seq_part}{int(_num(length) or 0):,} residues total"
                + (f" ({int(noncanon)} non-canonical)" if noncanon else "")
            )
        else:
            meta_bits.append(
                f"{length} residues ({int(noncanon)} non-canonical)"
                if noncanon
                else f"{length} residues"
            )
    meta = f'<p class="note">{_h(" · ".join(meta_bits))}</p>' if meta_bits else ""

    # ---- key properties (gauge cards) ----
    # For a multi-FASTA pool the per-protein gauges (length / MW / pI / charge)
    # are aggregates over many sequences and read as nonsense for "one"
    # protein, so we suppress the misleading single-protein length gauge.
    prop_keys = (
        ("molecular_weight", "pI", "gravy", "net_charge", "instability_index")
        if is_multi
        else ("length", "molecular_weight", "pI", "gravy", "net_charge", "instability_index")
    )
    prop_cards = [
        card for key in prop_keys if (card := _seqana_prop_card(key, metrics.get(key))) is not None
    ]
    props_html = (
        '<h5 class="widget-subhead">Key properties</h5>' + _card_grid(prop_cards)
        if prop_cards
        else ""
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
        if comp_rows
        else ""
    )

    # ---- amino-acid distribution (sequence logo) ----
    aa_fracs = (
        metrics.get("amino_acid_fractions")
        if isinstance(metrics.get("amino_acid_fractions"), dict)
        else {}
    )
    aa_counts = (
        metrics.get("amino_acid_counts")
        if isinstance(metrics.get("amino_acid_counts"), dict)
        else {}
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

    # The "Sampled residue annotations" table was dropped per user request —
    # it was long and largely redundant with the composition / distribution
    # views above. The per-residue detail still lives in the raw output below.
    ann_html = ""

    liabs = output.get("liabilities") if isinstance(output.get("liabilities"), list) else []
    liab_html = (
        f'<p class="note">{len(liabs)} sequence-liability flag(s) recorded — '
        "see the raw output below.</p>"
        if liabs
        else ""
    )

    scope_note = _selected_positions_note(output.get("annotation_scope"), output.get("warnings"))
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
    raw_muts = output.get("mutations") if isinstance(output.get("mutations"), list) else []
    parsed = [
        _seqedit_parse_mutation(m.get("change") if isinstance(m, dict) else m) for m in raw_muts
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
        f"{_h(_SEQEDIT_OP_LABELS.get(operation, operation.replace('_', ' ').title()))}"
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
                chips.append(f'<span class="seqedit-mut-chip">{_h(p["raw"])}</span>')
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
        alignment_html = '<p class="note">No sequence data was emitted for this edit.</p>'

    # ---- warnings -----------------------------------------------------
    warnings = output.get("warnings") if isinstance(output.get("warnings"), list) else []
    warn_html = "".join(f'<div class="seqedit-warn">{_h(str(w))}</div>' for w in warnings)

    recon_note = (
        " The wild-type row is reconstructed from the mutant sequence and the substitution list."
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


def _objective_rank_widget(output: dict[str, Any]) -> str:
    candidates = output.get("ranked_candidates")
    if not isinstance(candidates, list) or not candidates:
        return (
            '<div class="widget widget-rank">'
            '<h4 class="widget-title">Objective ranking</h4>'
            '<p class="empty">No ranked candidates emitted.</p></div>'
        )
    # Auto-detect interesting columns from the first non-empty row.
    candidate_keys: list[str] = []
    for c in candidates:
        if isinstance(c, dict):
            for k in c.keys():
                if k not in candidate_keys and k not in {"structure_path", "sequence"}:
                    candidate_keys.append(k)
    display_keys = [
        k
        for k in (
            "rank",
            "candidate_id",
            "objective_score",
            "score",
            "passed_gates",
            "failed_gates",
        )
        if k in candidate_keys
    ][:8]
    header = "".join(f"<th>{_h(_pretty_metric_name(k))}</th>" for k in display_keys)

    # Sort by the tool's own rank ascending so #1 (best) leads the table and the
    # ordering agrees with the board badges (rows are often emitted in archive
    # order, not rank order). Rows without a numeric rank sink to the bottom.
    def _rank_of(c: dict[str, Any]) -> float:
        r = _num(c.get("rank")) if isinstance(c, dict) else None
        return r if r is not None else float("inf")

    candidates = sorted((c for c in candidates if isinstance(c, dict)), key=_rank_of)
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
        f"<strong>objective_rank</strong> for {_h(output.get('objective_name') or 'objective')}. "
        f"Score: {_h(output.get('formula_text') or 'recorded formula')}; "
        f"{_h(output.get('score_direction') or 'higher')} is better."
        "</p>"
    )
    return (
        '<div class="widget widget-rank">'
        '<h4 class="widget-title">Objective ranking</h4>'
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
        cards.append(
            _score_card(
                "Predicted affinity (pIC50)",
                _fmt(num),
                "Boltz-2 predicted pIC50. Higher = tighter. Calibrated only when the target/ligand pair is within training distribution.",
                "neutral",
                gauge={"value": num, "lo": 0.0, "hi": 12.0},
            )
        )
    if aff_prob is not None:
        num = _num(aff_prob) or 0.0
        cards.append(
            _score_card(
                "Binary binding probability",
                f"{num * 100:.0f}%" if num <= 1.0 else _fmt(num),
                "Probability the predicted complex binds at experimentally relevant concentrations.",
                "good" if num >= 0.7 else ("ok" if num >= 0.4 else "poor"),
                gauge={"value": num, "lo": 0.0, "hi": 1.0},
            )
        )
    if conf is not None:
        num = _num(conf) or 0.0
        cards.append(
            _score_card(
                "Model confidence",
                _fmt(num),
                "Aggregate Boltz-2 confidence (akin to AlphaFold's confidence head).",
                "good" if num >= 0.7 else ("ok" if num >= 0.4 else "poor"),
                gauge={"value": num, "lo": 0.0, "hi": 1.0},
            )
        )
    cards_html = _card_grid(cards)
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
    "A": "C",
    "C": "C",
    "CA": "C",
    "CG0": "C",
    "CG1": "C",
    "G0": "C",
    "G1": "C",
    "N": "N",
    "NA": "N",
    "NS": "N",
    "NX": "N",
    "O": "O",
    "OA": "O",
    "OS": "O",
    "OX": "O",
    "S": "S",
    "SA": "S",
    "H": "H",
    "HD": "H",
    "HS": "H",
    "P": "P",
    "F": "F",
    "CL": "Cl",
    "BR": "Br",
    "I": "I",
    "SI": "Si",
    "MG": "Mg",
    "MN": "Mn",
    "ZN": "Zn",
    "FE": "Fe",
}
_PDBQT_ACCEPTOR_TYPES = {"NA", "NS", "OA", "OS", "SA"}
_PDBQT_DONOR_H_TYPES = {"HD", "HS"}

# Element swatch colours for the ligand-composition bar — brand-aligned and
# mutually distinct rather than literal CPK colouring.
_VINA_ELEMENT_COLOR = {
    "C": "#4b4565",
    "N": "#4cc4e4",
    "O": "#cc0980",
    "H": "#c9bbed",
    "S": "#f39e3a",
    "P": "#6b1ea8",
    "F": "#2bb59a",
    "Cl": "#2bb59a",
    "Br": "#a3553a",
    "I": "#7c2d8a",
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
            "affinity": None,
            "rmsd_lb": None,
            "rmsd_ub": None,
            "inter": None,
            "intra": None,
            "inter_intra": None,
            "unbound": None,
            "torsions": None,
            "torsdof": None,
            "atoms": [],
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
        "elements": elements,
        "donors": donors,
        "acceptors": acceptors,
        "aromatic": aromatic,
        "heavy": heavy,
        "total": len(atoms),
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
    return "".join(_h(el) + (f"<sub>{heavy[el]}</sub>" if heavy[el] != 1 else "") for el in order)


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
            f"RMSD <b>{lb:.2f}</b> / <b>{ub:.2f}</b> Å"
            if lb is not None and ub is not None
            else "—"
        )
        aff_txt = f"{aff:+.2f}" if aff is not None else "—"
        rows.append(
            '<div class="vina-pose">'
            f'<span class="vina-pose-rank">#{_h(str(rank))}</span>'
            '<div class="vina-pose-bar-track">'
            f'<i class="vina-pose-bar" style="width:{width:.1f}%;'
            f'background:{colour}"></i></div>'
            f'<span class="vina-pose-aff" style="color:{colour}">{aff_txt} '
            "<small>kcal/mol</small></span>"
            f'<span class="vina-pose-rmsd">{rmsd}</span></div>'
        )
    head = (
        '<div class="vina-pose vina-pose-head">'
        '<span class="vina-pose-rank">Pose</span>'
        "<span>Binding affinity</span>"
        '<span class="vina-pose-aff">score</span>'
        '<span class="vina-pose-rmsd">RMSD l.b. / u.b.</span></div>'
    )
    return f'<div class="vina-poses">{head}{"".join(rows)}</div>'


def _vina_energy_html(pose: dict[str, Any]) -> str:
    """Vina score-term breakdown — diverging bars about a zero axis."""
    terms = [
        (
            "Intermolecular",
            pose.get("inter"),
            "Receptor-ligand interaction energy - the actual binding contribution.",
        ),
        (
            "Intramolecular",
            pose.get("intra"),
            "Ligand internal (strain) energy in the bound conformation.",
        ),
        (
            "Unbound reference",
            pose.get("unbound"),
            "Ligand energy in the free state, subtracted as the scoring reference.",
        ),
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
            "<small>kcal/mol</small></span></div>"
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
            f"{_h(el)} <b>{count}</b></span>"
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
                pq_poses = _parse_pdbqt(p.read_text(encoding="utf-8", errors="replace"))
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
        _pdbqt_ligand_profile(best_pose["atoms"]) if best_pose and best_pose.get("atoms") else None
    )
    rotatable = None
    if best_pose:
        rotatable = best_pose.get("torsdof")
        if rotatable is None:
            rotatable = best_pose.get("torsions")

    # ---- summary cards ----
    cards: list[str] = []
    if best_aff is not None:
        cards.append(
            _score_card(
                "Best affinity (kcal/mol)",
                _fmt(best_aff),
                "Predicted free energy of binding for the top-ranked pose. More "
                "negative is tighter; weak / non-binders sit near 0.",
                _vina_affinity_quality(best_aff),
                gauge={"value": best_aff, "frame": True},
            )
        )
    if profile and best_aff is not None and profile["heavy"]:
        le = -best_aff / profile["heavy"]
        leq = "good" if le >= 0.35 else ("ok" if le >= 0.25 else "poor")
        cards.append(
            _score_card(
                "Ligand efficiency",
                f"{le:.3f}",
                "Binding free energy per heavy atom (-dG / N) - a medicinal-"
                "chemistry yardstick; ≈0.3 kcal/mol per atom marks a useful hit.",
                leq,
                gauge={"value": le, "lo": 0.0, "hi": 0.5},
            )
        )
    if modes:
        cards.append(
            _score_card(
                "Binding poses",
                str(len(modes)),
                "Distinct docked conformations Vina retained and scored.",
                "neutral",
                gauge={"value": len(modes), "frame": True},
            )
        )
    cards_html = _card_grid(cards) if cards else ""

    runtime = _num(output.get("runtime_ms"))
    meta_bits: list[str] = []
    if modes:
        meta_bits.append(f"{len(modes)} pose{'s' if len(modes) != 1 else ''}")
    if runtime is not None:
        meta_bits.append(f"docked in {_duration(int(runtime))}")
    meta = f'<p class="note">{_h(" · ".join(meta_bits))}</p>' if meta_bits else ""

    poses_html = (
        '<h5 class="widget-subhead">Binding poses</h5>' + _vina_poses_html(modes) if modes else ""
    )
    energy_html = ""
    if best_pose:
        ed = _vina_energy_html(best_pose)
        if ed:
            energy_html = '<h5 class="widget-subhead">Top-pose score breakdown</h5>' + ed
    ligand_html = (
        '<h5 class="widget-subhead">Docked ligand profile</h5>'
        + _vina_ligand_html(profile, rotatable)
        if profile
        else ""
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
        (f"{tool} designs", f"<strong>{_h(tool)}</strong> emitted the protein designs below."),
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
        chips.append(
            _dsn_chip(
                "seed" + ("s" if len(seeds) > 1 else ""),
                ", ".join(str(s) for s in seeds),
            )
        )
    rt = _num(output.get("runtime_ms"))
    if rt is not None:
        chips.append(f'<span class="dsn-chip">{_h(_duration(int(rt)))}</span>')
    chips_html = f'<div class="dsn-chips">{"".join(chips)}</div>'

    # ---- per-design table ----
    if not designs:
        return (
            '<div class="widget widget-design">'
            f'<h4 class="widget-title">{_h(title)}</h4>'
            f"{explainer}{chips_html}"
            '<p class="empty">No designs were emitted.</p></div>'
        )
    present = [
        (key, label)
        for key, label in _DESIGN_COLUMNS
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
            if length is not None
            else "<td>—</td>",
        ]
        for key, _ in present:
            num = _num(_plddt_on_100(key, d.get(key)))
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
            cells.append(f'<td><span class="dsn-file">{_h(Path(struct).name)}</span></td>')
        else:
            cells.append("<td>—</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_html = (
        '<div class="table-wrap"><table><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    return (
        '<div class="widget widget-design">'
        f'<h4 class="widget-title">{_h(title)}</h4>'
        f"{explainer}{chips_html}{table_html}</div>"
    )


# ProteinMPNN sequence recovery against a random sequence sits near 1/20
# (~5%) for the 20 canonical amino acids. A recovery at or below this floor
# signals there was no meaningful native reference (a de novo backbone), so
# the number is treated as "not a real recovery" rather than a poor design.
_SEQ_RECOVERY_MEANINGFUL_FLOOR = 0.15


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
        (f"{tool} designs", f"<strong>{_h(tool)}</strong> emitted the sequence designs below."),
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
            f"{explainer}{chips_html}"
            '<p class="empty">No designs were emitted.</p></div>'
        )

    # ProteinMPNN always *can* emit a seq_recovery number, but it only means
    # something when the backbone carried a real native sequence to score
    # against. For a de novo backbone (e.g. from rfdiffusion) the "native"
    # residues are placeholders, so recovery collapses to roughly the 1/20
    # (~5%) random baseline and the number is noise rather than a quality
    # signal. We therefore distinguish:
    #   * has_recovery        — at least one non-null seq_recovery present
    #   * meaningful_recovery — at least one recovery clears the random floor
    # and only draw the recovery column when it's actually meaningful; below
    # the floor we show a score-only table plus a one-line caveat so the run
    # stays auditable without implying a real 2% recovery.
    recoveries = [r for r in (_num(d.get("seq_recovery")) for d in designs) if r is not None]
    has_recovery = bool(recoveries)
    meaningful_recovery = any(r > _SEQ_RECOVERY_MEANINGFUL_FLOOR for r in recoveries)

    table_html = ""
    if meaningful_recovery:
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
                    f"<span>{pct:.0f}%</span></div></td>"
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
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
    elif has_recovery and any(_num(d.get("score")) is not None for d in designs):
        # Recovery was reported but is at the random baseline (no native
        # reference). Drop the noisy recovery column and just show the
        # still-informative score; the per-design score therefore lives in
        # this table rather than riding along in the sequence labels below.
        rows = [
            f"<tr><td>{i}</td>"
            f"<td>{_h(_fmt(_num(d.get('score')))) if _num(d.get('score')) is not None else '—'}</td></tr>"
            for i, d in enumerate(designs, 1)
        ]
        table_html = (
            '<h5 class="widget-subhead">Score</h5>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>#</th><th>Score</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
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
                    f"Highlighted cells are positions where all {len(present)} "
                    "designs agree on the same residue "
                    f"(<strong>{fully_conserved}/{max_len}</strong> &middot; "
                    f"{pct:.0f}% fully conserved). Dimmed cells are positions "
                    "where the model picked different residues across designs."
                    "</p>"
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
            f"{conservation_caption}"
            f'<div class="tmpl-seq-scroll">{"".join(seq_rows)}</div>'
        )
    return (
        f'<div class="widget widget-mpnn">'
        f'<h4 class="widget-title">{_h(title)}</h4>'
        f"{explainer}{chips_html}{table_html}{seq_html}</div>"
    )


def _mpnn_seq_cells(sequence: str, max_len: int, conservation: list[float]) -> str:
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
                    f"{_h(aa)}</span>"
                )
            else:
                cells.append(
                    f'<span class="mpnn-cell-variable" '
                    f'title="Position {idx + 1}: varies across designs">'
                    f"{_h(aa)}</span>"
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
            _lerp_hex("#ece8fb", "#cc0980", t) if t >= 0 else _lerp_hex("#ece8fb", "#1f9bd0", -t)
        )
        rects.append(
            f'<rect x="{i * bw:.4f}" y="0" width="{bw:.4f}" height="{height}" fill="{colour}"/>'
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
            # pLDDT-family → [0,100]; pTM and other heads stay on [0,1].
            value = _plddt_on_100(key, value)
            num = _num(value) or 0.0
            gauge_hi = 1.0 if num <= 1.5 else 100.0
            cards.append(
                _score_card(
                    label,
                    _fmt(value),
                    tip,
                    _quality_class(key.replace("mean_", ""), value),
                    gauge={"value": num, "lo": 0.0, "hi": gauge_hi},
                )
            )
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
                "negative</span>"
                '<span><span class="esm-emb-pip" style="background:#ece8fb"></span>'
                "~zero</span>"
                '<span><span class="esm-emb-pip" style="background:#cc0980"></span>'
                "positive</span></div>"
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
            cap = f'<p class="note">Filled positions: {_h(_format_position_list(filled))}</p>'
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
        value = _plddt_on_100(key, value)
        num = _num(value) or 0.0
        gauge_hi = 1.0 if (key == "plddt" and num <= 1.5) else hi
        cards.append(
            _score_card(
                label,
                _fmt(value),
                tip,
                _quality_class(key, value),
                gauge={"value": num, "lo": lo, "hi": gauge_hi},
            )
        )
    cards_html = _card_grid(cards)

    # ---- summary chips ----
    chips: list[str] = []
    seeds = output.get("seeds_used")
    if isinstance(seeds, list) and seeds:
        chips.append(
            _dsn_chip(
                "seed" + ("s" if len(seeds) > 1 else ""),
                ", ".join(str(s) for s in seeds),
            )
        )
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
                if p.get("seed") is not None
                else "<td>—</td>",
                f"<td>{_h(_fmt(p.get('rank')))}</td>"
                if p.get("rank") is not None
                else "<td>—</td>",
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
                if isinstance(cif, str)
                else "<td>—</td>"
            )
            rows.append("<tr>" + "".join(cells) + "</tr>")
        table_html = (
            '<h5 class="widget-subhead">Per-seed predictions</h5>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Seed</th><th>Rank</th><th>pLDDT</th><th>ipTM</th><th>pTM</th>"
            "<th>Ranking score</th><th>Structure</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
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
    shape_html = _output_shape_widget(output) if has_complex_output and include_shape else ""

    cards_html = _card_grid(cards)
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
        0.0,
        1.0,
    ),
    "bb_lddt": (
        "Backbone lDDT",
        "lDDT restricted to backbone atoms (0-1).",
        0.0,
        1.0,
    ),
    "dockq": (
        "DockQ",
        "Interface quality (0-1). >0.23 acceptable, >0.49 medium, >0.80 high.",
        0.0,
        1.0,
    ),
    "fnat": (
        "fnat",
        "Fraction of native interface contacts recovered (0-1).",
        0.0,
        1.0,
    ),
    "f1": (
        "F1",
        "F1 score over interface contacts (0-1).",
        0.0,
        1.0,
    ),
    "fnonnat": (
        "fnonnat",
        "Fraction of model interface contacts absent from the reference (0-1). Lower = better.",
        0.0,
        1.0,
    ),
    "irmsd": (
        "iRMSD (Å)",
        "Interface RMSD after superposition. Lower = better; <2 A is high quality.",
        None,
        None,
    ),
    "lrmsd": (
        "LRMSD (Å)",
        "Ligand RMSD after receptor superposition. Lower = better.",
        None,
        None,
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
    gauge = {"value": num, "frame": True} if lo is None else {"value": num, "lo": lo, "hi": hi}
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
            bits.append(f"{info['class1']} · {info['len1']} aa")
        if info.get("class2") and info.get("len2") is not None:
            bits.append(f"{info['class2']} · {info['len2']} aa")
        nat_c, nat_t = info.get("nat_correct"), info.get("nat_total")
        if nat_c is not None and nat_t:
            bits.append(f"{nat_c}/{nat_t} native contacts recovered")
        if bits:
            context = f'<p class="px-context">{_h(" · ".join(str(b) for b in bits))}</p>'
    sub_html = f'<span class="px-section-sub">{_h(subtitle)}</span>' if subtitle else ""
    return (
        '<section class="px-section">'
        f'<div class="px-section-head"><h5 class="px-section-title">{_h(title)}</h5>'
        f"{sub_html}</div>{_card_grid(cards)}{context}"
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
            _card_grid([card])
            if card
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
    ref_info = (
        report.get("ref_chain_info") if isinstance(report.get("ref_chain_info"), dict) else {}
    )
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
                sections.append(_px_section(f"Interface {iid}", "chain-chain interface", imetrics))
    entry = report.get("entry_id")
    header = (
        f'<p class="note">Scored against reference entry <code>{_h(str(entry))}</code>.</p>'
        if entry
        else ""
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
    common record shape used by the literature-search cards.

    Accepts both shapes: the raw Europe PMC keys (``authorString``,
    ``journalTitle``, ``isOpenAccess`` …) and the snake_case names the
    adapter writes into ``top_records`` (``authors``, ``journal``,
    ``is_open_access`` …).
    """
    open_access = r.get("is_open_access")
    if not isinstance(open_access, bool):
        open_access = str(r.get("isOpenAccess") or "").upper() == "Y"
    return {
        "title": str(r.get("title") or "").strip().rstrip("."),
        "authors": str(r.get("authorString") or r.get("authors") or "").strip(),
        "journal": str(r.get("journalTitle") or r.get("journal") or "").strip(),
        "volume": str(r.get("journalVolume") or r.get("journal_volume") or "").strip(),
        "issue": str(r.get("issue") or "").strip(),
        "page": str(r.get("pageInfo") or r.get("page") or "").strip(),
        "year": str(r.get("pubYear") or r.get("year") or "").strip(),
        "doi": (str(r["doi"]).strip() or None) if r.get("doi") else None,
        "pmid": (str(r["pmid"]).strip() or None) if r.get("pmid") else None,
        "pmcid": (str(r["pmcid"]).strip() or None) if r.get("pmcid") else None,
        "source": (str(r.get("source")).strip() or None) if r.get("source") else None,
        "open_access": open_access,
        "cited_by": r.get("citedByCount") or r.get("cited_by_count"),
        "date": str(r.get("firstPublicationDate") or r.get("first_publication_date") or ""),
        "snippet": str(r.get("abstract_snippet") or r.get("abstractText") or "").strip(),
    }


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
        if link
        else f'<span class="lit-title">{title}</span>'
    )
    oa = '<span class="lit-badge lit-oa">Open access</span>' if rec.get("open_access") else ""
    cite: list[str] = []
    if rec.get("journal"):
        cite.append(f'<em class="lit-journal">{_h(rec["journal"])}</em>')
    vip = rec.get("volume") or ""
    if rec.get("issue"):
        vip += f"({rec['issue']})"
    if rec.get("page"):
        vip += f":{rec['page']}"
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
            f"PMID {_h(rec['pmid'])}</a>"
        )
    if rec.get("pmcid"):
        chips.append(
            f'<a class="lit-id" href="https://europepmc.org/article/PMC/'
            f'{_attr(rec["pmcid"])}" target="_blank" rel="noopener noreferrer">'
            f"{_h(rec['pmcid'])}</a>"
        )
    snippet = f'<p class="lit-snippet">{_h(rec["snippet"])}</p>' if rec.get("snippet") else ""
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
    card per retrieved record (title + link + citation + snippet).

    Render is self-contained: every field consumed here lives in the tool's
    JSON output, so moving / archiving the workspace doesn't blank the
    widget. (Previously this read ``literature_raw_pages.json`` from disk.)
    """
    if not output:
        return ""
    query = output.get("normalized_query") or output.get("query")
    backend = output.get("backend")
    runtime = _num(output.get("runtime_ms"))
    result_count = _num(output.get("result_count"))
    total_hits = _num(output.get("total_hits"))
    status = output.get("status")
    date_filter = output.get("date_filter") if isinstance(output.get("date_filter"), dict) else {}

    top = output.get("top_records") if isinstance(output.get("top_records"), list) else []
    records = [_lit_normalize(r) for r in top if isinstance(r, dict)]

    stats: list[str] = []
    if backend:
        stats.append(f'<span class="lit-stat">backend <b>{_h(str(backend))}</b></span>')
    if total_hits is not None:
        stats.append(
            f'<span class="lit-stat"><b>{len(records)}</b> shown · '
            f"<b>{int(total_hits)}</b> total hits</span>"
        )
    elif result_count is not None:
        stats.append(f'<span class="lit-stat"><b>{int(result_count)}</b> results</span>')
    if status and status != "ok":
        stats.append(f'<span class="lit-stat lit-status">status <b>{_h(str(status))}</b></span>')
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
        if query
        else ""
    )
    summary_html = (
        f'<div class="lit-summary">{query_html}<div class="lit-stats">{"".join(stats)}</div></div>'
    )
    if records:
        cards = "".join(_lit_card(rec, i + 1) for i, rec in enumerate(records))
        body = f'<div class="lit-list">{cards}</div>'
    elif status == "no_hits":
        body = '<p class="note">Europe PMC returned no matches for this query.</p>'
    elif status == "filtered_out":
        body = (
            '<p class="note">Europe PMC matched records, but all were dropped '
            "by the local date filter.</p>"
        )
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
    "sequence",
    "designed_sequence",
    "wild_type_sequence",
    "mutant_sequence",
    "query_sequence",
    "edited_sequence",
    "fasta",
    "a3m",
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
    "good": "#4cc4e4",  # matches --good
    "ok": "#f39e3a",  # matches --ok
    "poor": "#cc0980",  # matches --poor
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
    return f'<div class="qc-card-grid" style="--cols:{cols}">{"".join(cards)}</div>'


def _score_card(
    label: str,
    value: str,
    explanation: str,
    quality: str,
    *,
    gauge: dict[str, Any] | None = None,
    source: str = "",
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
    src_html = f'<span class="qc-card-src">{_h(source)}</span>' if source else ""
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
            f'<div class="qc-card-name">{_h(label)}{src_html}</div>'
            f'<p class="qc-card-explain">{_h(explanation)}</p>'
            "</div></article>"
        )
    # Fallback: no gauge data — keep the simple value-and-label header.
    return (
        f'<article class="qc-card metric-{quality}">'
        "<header>"
        f"<div class='qc-card-figure'><strong>{_h(value)}</strong>"
        f'<span class="qc-card-name">{_h(label)}{src_html}</span></div>'
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
    """Product-grade reproducibility block backed by recorded audit fields."""
    repro = model.reproducibility
    if repro is None or not repro.replay_commands:
        return "<p class='empty'>No replay command available.</p>"
    commands = "".join(
        f'<div class="cmd repro-command"><pre>{_h(cmd)}</pre>'
        f'<button class="copy-btn" data-copy="{_attr(cmd)}">Copy</button></div>'
        for cmd in repro.replay_commands
    )
    ids = ", ".join(repro.ids[:3]) + ("..." if len(repro.ids) > 3 else "")
    id_line = (
        f'<p class="repro-id-line"><span>Recorded IDs</span><code>{_h(ids)}</code></p>'
        if repro.ids
        else ""
    )
    support_cards = (
        '<div class="repro-support-grid">'
        '<a class="repro-support-card" href="#details"><span>Technical provenance</span><strong>Per-tool results</strong></a>'
        "</div>"
    )
    return (
        '<div class="repro-panel">'
        '<div class="repro-copy-block">'
        '<p class="note">Run from the same workspace to regenerate this report '
        "from stored job results. No design tools are re-executed.</p>"
        f"{commands}"
        "</div>"
        f"{support_cards}"
        f"{id_line}"
        "</div>"
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
    "total_score",
    "ddg",
    "ddg_reu",
    "mean_ddg",
    "delta_delta_g",
    "polybasic_hits",
    "aromatic_fraction",
)

# Context-only metrics that should never appear as score bars (they belong
# in the candidate header / footer instead).
_SCORE_BAR_SKIP: frozenset[str] = frozenset(
    {
        "num_residues",
        "length",
        "runtime_ms",
        "n_residues",
        "elapsed_ms",
        "elapsed_s",
        "seed",
    }
)


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
    # The candidate may arrive with archive-nested scores (``{tool: {metric:
    # val}}``) when it comes straight off a review bundle, or already flat from
    # the flattened candidate view. ``_flatten_scores`` is idempotent on flat
    # maps, so flattening here gives the ``tool:metric`` keys this renderer's
    # alias-collapse / dedup logic relies on, regardless of which path fed us.
    combined.update(_flatten_scores(candidate.scores))

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
    # pLDDT family is presented on [0,100] everywhere; rescale here so the
    # bar fill, the printed value, and the quality colour all agree.
    value = _plddt_on_100(key, value)
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
    if k in {
        "plddt",
        "mean_plddt",
        "ptm",
        "iptm",
        "dockq",
        "ipsae",
        "seq_recovery",
        "recovery",
        "lddt",
        "bb_lddt",
    }:
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
    if k in {"ddg", "ddg_reu", "mean_ddg", "delta_delta_g"}:
        if num <= -1:
            return "good"
        if num <= 1:
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


# pLDDT and its named derivatives are logged by some adapters on the [0,1]
# unit scale (e.g. esm3 ``mean_plddt`` ≈ 0.98, protein_qc ``plddt`` ≈ 0.98)
# and by others already on AlphaFold's [0,100] scale. The report always
# presents pLDDT on [0,100], so a value that arrives as a unit fraction
# (<= 1.5) is scaled up by 100 before it drives any label, gauge fill, or
# quality colour. Keyed on the canonical (provenance-prefix-stripped) base
# metric name so ``esm3:mean_plddt`` and bare ``plddt`` both match.
_PLDDT_FAMILY_KEYS = frozenset(
    {
        "plddt",
        "mean_plddt",
        "scaffold_plddt",
        "binder_plddt",
        "iplddt",
    }
)


def _plddt_on_100(key: str, value: Any) -> Any:
    """Rescale a pLDDT-family value to the [0,100] scale when needed.

    Only touches pLDDT-family metrics whose value looks like a unit
    fraction (<= 1.5). Non-pLDDT keys, already-[0,100] values, and
    non-numeric inputs pass through unchanged.
    """
    base = str(key).split(":", 1)[-1].lower()
    if base not in _PLDDT_FAMILY_KEYS:
        return value
    num = _num(value)
    if num is None or not (0.0 <= num <= 1.5):
        return value
    return num * 100.0


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
  --sidebar-w: 280px;
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
.widget-title { margin:0 0 9px; font-size:13px; font-weight:650; color:var(--ink); }
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
/* ── Executive summary — compact, typography-driven (no metric cards) ──────── */
.exec-summary { margin:10px 0 4px; }
/* Status dots — the only colour in the summary, reserved for state signalling. */
.exec-stat-dot, .exec-cm-dot, .exec-node-dot {
  display:inline-block; width:7px; height:7px; border-radius:50%; flex:0 0 auto;
  background:var(--line-strong);
}
.dot-good, .dot-completed, .dot-cached { background:var(--good); }
.dot-ok, .dot-queued, .dot-pending, .dot-running { background:var(--ok); }
.dot-poor, .dot-failed, .dot-cancelled { background:var(--poor); }
.dot-neutral, .dot-skipped { background:var(--subtle); }
/* Grouped stat strip — values lead, labels sit quietly beneath; thin dividers. */
.exec-stats {
  display:flex; flex-wrap:wrap; gap:0;
  border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  padding:14px 0; margin-bottom:18px;
}
.exec-stat { padding:0 22px; border-left:1px solid var(--line); }
.exec-stat:first-child { padding-left:2px; border-left:none; }
.exec-stat-val {
  display:flex; align-items:center; gap:7px;
  font-size:19px; font-weight:600; color:var(--ink); letter-spacing:-.01em;
  font-variant-numeric:tabular-nums;
}
.exec-stat-lbl {
  font-size:11px; color:var(--muted); margin-top:3px;
  letter-spacing:.02em;
}
/* Two-column signal band — confidence | pipeline, divider not boxes. */
.exec-band {
  display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:32px;
}
.exec-band .exec-col + .exec-col,
.exec-band > .exec-col:nth-child(2) { }
.exec-col { min-width:0; }
.exec-col-head {
  font-size:11px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); padding-bottom:9px; margin-bottom:10px;
  border-bottom:1px solid var(--line);
}
.exec-col-sub { font-weight:500; text-transform:none; letter-spacing:0; color:var(--subtle); margin-left:6px; }
.exec-cm-list { display:flex; flex-direction:column; gap:11px; }
.exec-cm { display:flex; align-items:baseline; gap:9px; }
.exec-cm-dot { align-self:center; }
.exec-cm-val {
  font-size:17px; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums;
  min-width:46px;
}
.exec-cm-lbl { font-size:13px; color:var(--ink); }
.exec-cm-src { color:var(--muted); font-weight:500; }
.exec-cm-empty .exec-cm-lbl { color:var(--muted); font-size:12.5px; }
.exec-cm-empty code { font-family:ui-monospace,monospace; font-size:11.5px; }
.exec-node-list { display:flex; flex-direction:column; gap:8px; }
.exec-node {
  display:flex; align-items:center; gap:8px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; color:var(--ink);
}
@media (max-width: 900px) { .exec-band { grid-template-columns:1fr; gap:18px; } }

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

/* Candidate groups — visually separate the submitted "Final picks" from
   the "Explored candidates" that were evaluated but not submitted. The two
   groups previously shared identical chrome, so they read as one flat list.
   Final picks get an accent-tinted frame + glow; explored gets a muted,
   recessed frame so the eye lands on the answer first. */
.candidate-group {
  border-radius:16px; padding:8px 20px 18px; margin:20px 0;
  border:1px solid var(--line);
}
.candidate-group-final {
  border:1px solid color-mix(in srgb, var(--accent) 45%, var(--line));
  background:color-mix(in srgb, var(--accent) 6%, var(--panel));
  box-shadow:0 0 0 1px color-mix(in srgb, var(--accent) 18%, transparent),
             var(--shadow-card);
}
.candidate-group-explored {
  border:1px dashed var(--line);
  background:color-mix(in srgb, var(--ink) 2.5%, var(--panel));
  opacity:0.97;
}
.candidate-group-head {
  display:flex; justify-content:space-between; align-items:center;
  gap:16px; flex-wrap:wrap;
  padding:14px 0 12px; margin-bottom:6px;
  border-bottom:1px solid var(--line);
}
.candidate-group-title { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
.candidate-group-badge {
  display:inline-flex; align-items:center;
  font-size:13px; font-weight:700; letter-spacing:-0.005em;
  padding:5px 13px; border-radius:999px; white-space:nowrap;
}
.candidate-group-badge-final {
  color:#fff;
  background:var(--accent);
  box-shadow:0 1px 4px color-mix(in srgb, var(--accent) 40%, transparent);
}
.candidate-group-badge-explored {
  color:var(--muted);
  background:color-mix(in srgb, var(--ink) 7%, transparent);
  border:1px solid var(--line);
}
.candidate-group-sub {
  color:var(--subtle); font-size:12.5px; font-weight:500; letter-spacing:0;
}
/* Explored cards sit one level down the hierarchy — flatten their shadow so
   they don't compete with the final picks. */
.candidate-group-explored .candidate-card {
  box-shadow:none; background:var(--panel);
}

/* Candidate card */
.candidate-card {
  border:1px solid var(--line); border-radius:12px; padding:22px 24px; margin:16px 0;
  background:var(--panel);
  box-shadow:var(--shadow-tile);
}
/* Small uppercase eyebrow ("Candidate sequence · N residues", ...).
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
/* Caption explaining a backbone-only or partially-modelled structure. Sits as
   a slim banner between the controls and the sequence track. */
.structure-coverage-note {
  margin:6px 0 0; padding:6px 10px;
  font-size:11.5px; line-height:1.4; color:var(--muted);
  background:var(--accent-soft); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:4px;
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
/* Backbone-only models have no sidechain atoms, so the toggle is disabled.
   Mute it and switch the cursor to the not-allowed affordance; the label's
   title attribute carries the explanation on hover. */
.structure-control-disabled {
  opacity:0.45; cursor:not-allowed;
}
.structure-control-disabled:hover {
  background:var(--panel); border-color:var(--line); color:var(--ink);
}
.structure-control-disabled input[type="checkbox"] { cursor:not-allowed; }
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
/* The legend reads "Very low → Very high", i.e. a confidence scale — it is only
   meaningful in pLDDT mode. N→C spectrum encodes chain direction (not
   confidence) and Chain encodes identity, so the legend is hidden for both. */
.structure-legend[data-legend="spectrum"],
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
/* Per-tool node = a divider-separated section, not a nested card. */
.node-block {
  background:transparent; border:none; border-radius:0;
  padding:18px 0 4px; margin:0; box-shadow:none;
  border-top:1px solid var(--line);
}
.node-block:first-of-type { border-top:none; padding-top:6px; }
.node-block-head {
  display:flex; justify-content:space-between; align-items:center;
  gap:12px; margin-bottom:14px;
}
.node-block-title {
  display:block; font-weight:650; font-size:14.5px; color:var(--ink);
}
.node-block-meta {
  display:block; margin-top:3px; font-size:11.5px; color:var(--muted);
}
.node-block-meta code {
  font-family:ui-monospace, monospace; font-size:11px; color:var(--muted);
}
/* State = a quiet text label with a status dot; no coloured pill. */
.node-block-state {
  font-size:11.5px; font-weight:600; color:var(--muted);
  display:inline-flex; align-items:center; gap:6px;
}
.node-block-state::before {
  content:""; width:7px; height:7px; border-radius:50%; background:var(--subtle);
}
.node-block-state.state-completed::before, .node-block-state.state-cached::before { background:var(--good); }
.node-block-state.state-failed { color:var(--poor); }
.node-block-state.state-failed::before { background:var(--poor); }
.node-block-headright { display:flex; align-items:center; gap:12px; flex-shrink:0; }
.node-block-cohort {
  font-size:11px; font-weight:500; color:var(--muted);
}
.node-block-cohort::before { content:""; }

/* ── Two bands: per-design vs all-designs — neutral section headers, no tints,
   distinguished by label + sub-text (and the cohort "shows all" note). */
.nodeband { margin:8px 0 0; padding:0; border:none; background:none; }
.nodeband + .nodeband { margin-top:30px; padding-top:22px; border-top:1px solid var(--line); }
.nodeband-head {
  display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; padding:0 0 4px;
}
.nodeband-title {
  font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:var(--ink);
}
.nodeband-sub { font-size:12px; color:var(--muted); }
.nodeband-head .nb-glyph { display:none; }

/* ── Candidate browser: the per-design sample selector ──────────────────────
   A polished horizontal rail of candidate cards. Each card has a prominent
   rank token, a clear name, and a headline metric; the selected card is
   unmistakable (accent ring + tint + check), and the Final pick is special. */
.node-cand-toolbar {
  position:sticky; top:0; z-index:5;
  display:flex; flex-direction:column; gap:12px;
  padding:15px 16px; margin:0 0 18px;
  background:var(--panel);
  border:1px solid var(--line); border-radius:14px;
  box-shadow:var(--shadow-tile);
}
.cbrowse-head { display:flex; align-items:baseline; gap:10px; }
.cbrowse-title {
  font-size:11px; font-weight:750; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted);
}
.cbrowse-count {
  font-size:11px; font-weight:600; color:var(--subtle);
  background:var(--bg); border:1px solid var(--line); border-radius:var(--radius-pill);
  padding:2px 9px;
}
/* Cards scroll horizontally so a 30+ design run stays one compact rail with
   the final / ranked picks (ordered first) visible up front. */
.node-cand-btnrow {
  flex:1 1 auto; min-width:0;
  display:flex; flex-wrap:nowrap; align-items:stretch; gap:10px;
  overflow-x:auto; padding:2px 2px 8px; scrollbar-width:thin;
}
.node-cand-btnrow::-webkit-scrollbar { height:8px; }
.node-cand-btnrow::-webkit-scrollbar-thumb { background:var(--line-strong); border-radius:4px; }
.node-cand-btnrow::-webkit-scrollbar-track { background:transparent; }
.cbrowse-card {
  position:relative; flex:0 0 auto; min-width:176px; max-width:220px;
  display:flex; align-items:center; gap:11px; text-align:left;
  background:var(--panel); border:1.5px solid var(--line);
  padding:10px 12px; border-radius:12px; cursor:pointer;
  transition:border-color .14s, box-shadow .14s, transform .14s, background .14s;
}
.cbrowse-card:hover { border-color:var(--line-strong); box-shadow:var(--shadow-card-hover); }
.cbrowse-card.is-active {
  border-color:var(--accent); background:#faf8ff;
  box-shadow:0 0 0 3px rgba(107,30,168,.14), 0 6px 18px rgba(45,4,110,.12);
  transform:translateY(-1px);
}
/* Rank token — big numeral chip on the left; medal star for the final pick. */
.cbc-token {
  flex:0 0 auto; width:38px; height:38px; border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  background:var(--bg); border:1px solid var(--line);
}
.cbc-token-num { font-size:15px; font-weight:650; color:var(--muted); letter-spacing:-.01em; font-variant-numeric:tabular-nums; }
.cbc-token-num::before { content:"#"; font-size:10px; font-weight:600; opacity:.5; margin-right:1px; }
.cbc-token-dash::before { content:""; }
.cbc-token-final { font-size:15px; font-weight:700; color:var(--ink); }
.cbc-body { display:flex; flex-direction:column; gap:1px; min-width:0; }
.cbc-eyebrow {
  font-size:9px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:var(--subtle);
}
.cbc-name {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; font-weight:550;
  color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.cbc-metric { font-size:11px; font-weight:550; color:var(--muted); margin-top:2px; }
.cbc-check {
  position:absolute; top:7px; right:8px; width:15px; height:15px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  background:var(--accent); color:#fff; font-size:9px; font-weight:700;
  opacity:0; transform:scale(.6); transition:opacity .14s, transform .14s;
}
.cbrowse-card.is-active .cbc-check { opacity:1; transform:scale(1); }
/* Status: the final pick carries a neutral, authoritative treatment (ink ✓ +
   ink eyebrow) — no medal, no colour. Selection (is-active) owns the accent. */
.cbrowse-card.status-failed .cbc-token-num { color:var(--poor); }
.cbrowse-card.is-final .cbc-eyebrow { color:var(--ink); }

/* One variant per sample inside each node block — only the selected one
   shows; the shared sentinel (retrieval/input nodes) is always visible. */
.node-cand-variant { display:none; }
.node-cand-variant.is-active,
.node-cand-variant[data-cand="__shared__"] { display:block; }
.node-cand-na-note {
  margin:6px 0; padding:12px 14px; font-size:13px; color:var(--muted);
  background:var(--panel); border:1px dashed var(--line); border-radius:8px;
}

/* Widget = an open section inside a node block (no nested card chrome); its
   leaf tiles (QC cards, tables) carry whatever structure they need. */
.widget {
  background:transparent; border:none; border-radius:0;
  padding:0; margin:0 0 4px; box-shadow:none;
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
/* Candidate-availability + merge-audit event rows. These live in
   ``.artifact-list`` and share a node-title + metadata-pill + tool-chip
   shape. Without these rules the pills had no chip styling (so labels and
   values ran together — "binder_length_maxsource…") and the trailing
   <code> tool floated into a ragged staircase. Scope the layout to
   ``.artifact-list`` so the download-artifact rows (``.artifact-section``)
   are untouched. */
.artifact-list { display:flex; flex-direction:column; gap:10px; margin-top:4px; }
.artifact-list .artifact-row {
  align-items:flex-start; gap:12px;
  padding:12px 14px; border:1px solid var(--line); border-radius:10px;
  background:var(--panel); box-shadow:var(--shadow-tile);
}
.artifact-list .artifact-row > div:first-child { flex:1 1 auto; min-width:0; }
.artifact-list .artifact-row > div:first-child > strong {
  display:block; margin-bottom:7px; font-size:13.5px; color:var(--ink);
}
.artifact-list .artifact-row > code {
  flex:0 0 auto; align-self:flex-start;
  font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:12px; padding:3px 9px; border-radius:6px;
  background:var(--accent-soft); color:var(--accent); white-space:nowrap;
}
.artifact-list .artifact-meta { display:flex; flex-wrap:wrap; gap:6px; }
.artifact-pill {
  display:inline-flex; align-items:baseline; gap:5px;
  padding:3px 9px; border-radius:999px;
  background:color-mix(in srgb, var(--ink) 4%, var(--panel));
  border:1px solid var(--line);
  font-size:12px; color:var(--muted); line-height:1.55;
}
.artifact-pill strong { color:var(--ink); font-weight:600; }
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

/* Per-residue strips (structure_analysis widgets). One white card
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
/* Resolved-term relevance rows */
.seqret-terms { display:flex; flex-direction:column; gap:9px; margin-bottom:14px; }
.seqret-term {
  background:#fbfafe; border:1px solid var(--line); border-radius:9px;
  padding:9px 12px;
}
.seqret-term-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:7px; }
.seqret-term-src {
  font-size:10px; font-weight:700; letter-spacing:0.04em; color:#fff;
  background:var(--accent); border-radius:4px; padding:1px 6px;
}
.seqret-term-id {
  font-family:ui-monospace, monospace; font-size:12px; font-weight:700; color:var(--ink);
}
.seqret-term-label { font-size:12px; color:var(--muted); }
.seqret-relrow { display:flex; align-items:center; gap:10px; margin-top:7px; }
.seqret-relbar {
  position:relative; flex:1; height:9px; min-width:80px;
  background:#e7e3f3; border-radius:999px; overflow:hidden;
}
.seqret-relbar-fill {
  position:absolute; left:0; top:0; bottom:0;
  background:linear-gradient(90deg, #8574c0, #5a45a8); border-radius:999px;
}
.seqret-relval {
  font-variant-numeric:tabular-nums; font-size:12px; font-weight:700;
  color:var(--ink); min-width:34px; text-align:right;
}
.seqret-term-reason { font-size:11px; color:var(--subtle); }
/* Per-source yield bars */
.seqret-yields { display:flex; flex-direction:column; gap:7px; margin-bottom:14px; }
.seqret-yield { display:flex; align-items:center; gap:10px; }
.seqret-yield-label {
  flex-shrink:0; width:64px; font-size:11.5px; font-weight:700; color:var(--ink);
}
.seqret-yield-track {
  position:relative; flex:1; height:18px; min-width:80px;
  background:#f0eef8; border-radius:5px; overflow:hidden;
}
.seqret-yield-fill {
  position:absolute; left:0; top:0; bottom:0;
  background:linear-gradient(90deg, #65CBF3, #0053D6); border-radius:5px;
}
.seqret-yield.is-empty .seqret-yield-fill { background:#d6d0e6; }
.seqret-yield-val {
  flex-shrink:0; font-variant-numeric:tabular-nums; font-size:12px;
  font-weight:700; color:var(--ink);
}
.seqret-yield-val .muted { font-weight:400; font-size:11px; color:var(--muted); }
.seqret-unresolved {
  list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:5px;
}
.seqret-unresolved li {
  font-size:11.5px; color:var(--muted);
  background:#fbfafe; border:1px solid var(--line); border-radius:7px;
  padding:6px 10px;
}
.seqret-unresolved b { color:var(--ink); }
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
/* Source-tool tag on a metric card (e.g. confidence reported by Protein QC). */
.qc-card-src {
  display:inline; margin-left:5px; vertical-align:baseline;
  font-size:11px; font-weight:550; color:var(--muted);
}
.qc-card-src::before { content:"\\00b7 "; color:var(--subtle); }
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
  .console { grid-template-columns:1fr; }
  .console-board { max-height:none; }
}

/* ---- Candidate review console ---- */
.side-nav-divider {
  margin:14px 0 4px; padding:0 12px;
  font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color:var(--subtle);
}
.appendix-divider {
  margin:38px 0 8px; padding:16px 4px 0; border-top:1px dashed var(--line-strong);
  opacity:.82;
}
.appendix-divider-label {
  display:inline-block; font-size:12px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted);
}
.appendix-divider-note { margin:4px 0 0; color:var(--subtle); font-size:12.5px; max-width:64ch; }

.console-banner {
  margin:0 0 16px; padding:14px 16px; border-radius:12px;
  background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12;
}
.console-banner p { margin:4px 0 0; color:#9a3412; font-size:13px; }
.display-limit-notice {
  margin:0 0 16px; padding:12px 14px; border-radius:10px;
  background:#f8fafc; border:1px solid var(--line); color:var(--muted);
  font-size:13px; font-weight:600;
}

.console {
  display:grid; grid-template-columns:minmax(240px, 320px) 1fr; gap:18px; align-items:start;
}
.console-board {
  max-height:760px; overflow-y:auto; padding-right:2px;
}
/* ── Candidate result browser — a dense table/list hybrid ────────────────────
   Shared 3-column grid (rank · candidate · score). No per-row cards, no stars,
   no success-green: structure and a single hairline divider carry the records.
   Accent appears only on the *selected* row (a left bar + faint tint). */
.cand-list { --crow-grid: 34px minmax(0,1fr) auto; }
.cand-list-head {
  display:grid; grid-template-columns:var(--crow-grid); align-items:center;
  gap:10px; padding:0 12px 7px; border-bottom:1px solid var(--line);
  font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--subtle);
}
.cand-list-head .clh-rank { text-align:center; }
.cand-list-head .clh-score { text-align:right; }
.board-group { margin-top:14px; }
.board-group:first-of-type { margin-top:10px; }
.board-group-head {
  display:flex; align-items:center; gap:8px; padding:4px 12px;
}
.board-group-title { font-size:10.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); }
.board-group-count { font-size:10.5px; font-weight:600; color:var(--subtle); }
.board-group-count::before { content:"·"; margin-right:5px; color:var(--line-strong); }
.board-group-cards { display:flex; flex-direction:column; }
/* Collapsed groups (incomplete / failed / rejected) — expand on demand. */
.board-group-collapsible > summary.board-group-head {
  cursor:pointer; list-style:none;
}
.board-group-collapsible > summary.board-group-head::-webkit-details-marker { display:none; }
.board-group-collapsible > summary.board-group-head::before {
  content:"\25B8"; margin-right:2px; color:var(--subtle); font-size:9px;
  transition:transform .12s ease; display:inline-block;
}
.board-group-collapsible[open] > summary.board-group-head::before { transform:rotate(90deg); }

.crow {
  display:grid; grid-template-columns:var(--crow-grid); align-items:center; gap:10px;
  width:100%; text-align:left; cursor:pointer;
  padding:8px 12px; background:transparent; border:none;
  border-left:2px solid transparent; border-bottom:1px solid var(--line);
  transition:background .12s;
}
.crow:hover { background:var(--bg); }
.crow[aria-selected="true"] {
  background:rgba(107,30,168,.055); border-left-color:var(--accent);
}
.crow-rank { text-align:center; }
.crow-rank-num { font-size:13px; font-weight:600; color:var(--subtle); font-variant-numeric:tabular-nums; }
.crow-rank-dash { color:var(--line-strong); }
.crow-rank-final { font-size:14px; font-weight:700; color:var(--ink); }
/* Main column: id line + metric/source meta line, both truncating in the
   flexible 1fr column so a long "metric · source" can never overrun the row. */
.crow-main { display:flex; flex-direction:column; gap:1px; min-width:0; }
.crow-id-line { display:flex; align-items:center; gap:7px; min-width:0; }
.crow-id {
  min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; font-weight:550; color:var(--ink);
}
.crow[aria-selected="true"] .crow-id { font-weight:650; }
.crow-meta {
  font-size:10.5px; color:var(--subtle);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
/* Final = a neutral, authoritative label (NOT a celebratory badge). */
.crow-tag {
  flex:0 0 auto;
  font-size:9px; font-weight:700; letter-spacing:.05em; text-transform:uppercase;
  color:var(--ink); background:var(--bg);
  border:1px solid var(--line-strong); border-radius:4px; padding:1px 5px;
}
.crow-score {
  justify-self:end; text-align:right; padding-left:8px;
  font-size:13.5px; font-weight:650; color:var(--ink); font-variant-numeric:tabular-nums;
}
.crow-score-empty { color:var(--line-strong); font-weight:400; }

.console-detail {
  min-height:420px; border:1px solid var(--line); border-radius:14px;
  background:var(--panel); padding:18px; overflow:hidden;
}
.console-empty { color:var(--subtle); font-size:14px; padding:24px 4px; }
.cand-detail-head { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
/* Status label — a restrained outline chip; the candidate id (h3) carries the
   emphasis. Final is authoritative (solid ink), not celebratory. */
.cand-detail-eyebrow {
  font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  padding:2px 8px; border-radius:5px; background:var(--bg);
  border:1px solid var(--line-strong); color:var(--muted);
}
.cand-detail-head.status-final .cand-detail-eyebrow {
  background:var(--ink); border-color:var(--ink); color:#fff;
}
.cand-detail-head.status-failed .cand-detail-eyebrow {
  background:var(--poor-soft); border-color:var(--poor-line); color:#991b1b;
}
.cand-detail-id h3 { margin:6px 0 0; font-size:19px; font-weight:650; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--ink); }
.cand-detail-ranknote {
  margin:7px 0 0; font-size:12px; color:var(--muted);
  border-left:2px solid var(--line-strong); padding:2px 0 2px 10px;
}
.cand-detail-ranknote strong { color:var(--ink); font-weight:650; }
.cand-tabs {
  display:flex; flex-wrap:wrap; gap:4px; border-bottom:1px solid var(--line); margin-bottom:14px;
}
.cand-tab {
  padding:8px 13px; border:none; background:none; cursor:pointer;
  font-size:13px; font-weight:600; color:var(--muted);
  border-bottom:2px solid transparent; margin-bottom:-1px;
}
.cand-tab:hover { color:var(--ink); }
.cand-tab.is-active { color:var(--accent); border-bottom-color:var(--accent); }
.cand-tabpanel { display:none; }
.cand-tabpanel.is-active { display:block; animation:fadeIn .15s ease; }
@keyframes fadeIn { from { opacity:0; } to { opacity:1; } }

/* Quality analysis tab: a compact full-width metrics strip (score bars flow
   into as many columns as fit, so there is no ragged empty column) above the
   full-width sequence track. The 3D render lives in its own Structure tab. */
.cand-quality-scores { margin-bottom:18px; }
.cand-quality-scores .score-bars {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));
  gap:2px 36px; margin-bottom:0;
}
.cand-quality-scores .score-row { margin:5px 0; }

.detail-block { margin-bottom:16px; }
.detail-block-title { margin:0 0 7px; font-size:11px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); }
.detail-note { margin:0 0 8px; font-size:12px; color:var(--subtle); }

.prov {
  font-size:9.5px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
  padding:1px 6px; border-radius:999px; white-space:nowrap;
}
.prov-exact { background:#dcfce7; color:#166534; }
.prov-derived { background:var(--accent-soft); color:var(--accent); }
.prov-inferred { background:#fef3c7; color:#92400e; }
.prov-not_recorded { background:#f1f1f5; color:#9ca3af; }

.estep-row { display:flex; flex-wrap:wrap; gap:8px; }
.estep {
  display:inline-flex; align-items:center; gap:7px;
  padding:6px 10px; border-radius:9px; background:#faf9fe; border:1px solid var(--line);
}
.estep-tool { font-size:12px; font-weight:700; color:var(--ink); }
.estep-summary { font-size:11px; color:var(--muted); }
.estep-generator { border-left:3px solid #16a34a; }
.estep-evaluator { border-left:3px solid #6b1ea8; }
.estep-ranker { border-left:3px solid #d97706; }
.estep-submission { border-left:3px solid #2563eb; }
.pipe-row { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.pipe-step {
  font-size:11px; font-weight:600; color:var(--muted);
  padding:3px 9px; border-radius:999px; background:#f3f1f9; border:1px solid var(--line);
}
.pipe-generator { color:#166534; background:#dcfce7; }
.pipe-evaluator { color:var(--accent); background:var(--accent-soft); }

.score-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.score-family { border:1px solid var(--line); border-radius:11px; padding:12px 14px; background:var(--panel); }
.score-family-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:8px; }
.score-family-name { font-size:13px; font-weight:700; color:var(--ink); }
.score-family-body.is-scrollable {
  max-height:174px;
  overflow-y:auto;
  overflow-x:hidden;
  padding-right:6px;
  scrollbar-gutter:stable;
}
.score-family-body.is-scrollable::-webkit-scrollbar { width:8px; }
.score-family-body.is-scrollable::-webkit-scrollbar-track { background:#eef1f7; border-radius:999px; }
.score-family-body.is-scrollable::-webkit-scrollbar-thumb { background:#c7cedd; border-radius:999px; }
.metric-row { display:flex; align-items:baseline; justify-content:space-between; gap:10px; padding:4px 0; border-top:1px solid var(--line); }
.metric-row:first-child { border-top:none; }
.metric-label { font-size:12px; color:var(--ink); display:flex; align-items:baseline; gap:5px; flex-wrap:wrap; }
/* Provenance — restrained inline "· Tool" text naming the source model. */
.metric-src { font-size:11px; font-weight:550; color:var(--muted); }
.metric-src::before { content:"\\00b7 "; color:var(--subtle); }
.metric-value { font-size:13px; font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums; }
.tool-evidence { margin-bottom:8px; border:1px solid var(--line); border-radius:9px; padding:8px 12px; }
.tool-evidence summary { font-size:13px; font-weight:600; cursor:pointer; color:var(--ink); }
.tool-evidence-json { margin:8px 0 0; font-size:11px; max-height:320px; overflow:auto; background:#faf9fe; padding:10px; border-radius:7px; }

/* SaaS redesign scaffold */
:root {
  --saas-bg:#f7f8fb;
  --saas-violet:#5b35e6;
  --saas-blue:#2276d9;
  --saas-teal:#0f9f9a;
  --saas-green:#49a51d;
  --saas-amber:#b7791f;
  --saas-red:#c2410c;
}
body { background:var(--saas-bg); }
.sidebar {
  background:linear-gradient(180deg,#ffffff 0%,#f7f6ff 100%);
  color:var(--ink);
  border-right:1px solid var(--line);
  box-shadow:1px 0 16px rgba(15,23,42,.04);
}
.sidebar-brand { color:var(--ink); align-items:flex-start; }
.brand-text { display:flex; flex-direction:column; gap:1px; line-height:1.15; }
.brand-text strong { font-size:15px; }
.brand-text small { font-size:10.5px; color:var(--muted); font-weight:500; }
.sidebar-meta { background:#fff; border-color:var(--line); }
.sidebar-meta-label { color:var(--subtle); }
.sidebar-meta-value { color:var(--ink); }
.side-nav-item { color:#2d2a4a; }
.side-nav-item:hover { background:var(--accent-soft); color:var(--saas-violet); }
.side-nav-glyph { color:var(--saas-violet); }
.sidebar-foot { color:var(--muted); border-top-color:var(--line); }
.top-bar { min-height:var(--topbar-h); }
.top-bar-left, .top-bar-right { gap:12px; }
.top-bar-right { display:flex; align-items:center; flex-wrap:wrap; }
.top-bar-date { font-size:13px; color:var(--muted); }
.top-bar-action {
  display:inline-flex; align-items:center; justify-content:center;
  min-height:34px; padding:0 13px; border-radius:7px;
  border:1px solid var(--line-strong); background:#fff;
  color:var(--ink); text-decoration:none; font-size:12.5px; font-weight:600;
  font-family:inherit; cursor:pointer;
}
.top-bar-action-primary {
  background:var(--saas-violet); border-color:var(--saas-violet); color:#fff;
  box-shadow:0 8px 18px rgba(91,53,230,.18);
}
.content { max-width:1680px; padding:24px 28px 56px; }
.saas-section { border-radius:8px; padding:18px 20px; }
.saas-section-subtitle { margin:4px 0 0; color:var(--muted); font-size:13.5px; }
.saas-overview { background:transparent; border:none; box-shadow:none; padding:0; }
.saas-overview:hover { box-shadow:none; }
.saas-overview-grid {
  display:grid; grid-template-columns:1.05fr 1.1fr 2.35fr 1.1fr;
  gap:12px; align-items:stretch;
}
.saas-card {
  background:#fff; border:1px solid var(--line-strong); border-radius:8px;
  padding:14px 15px; box-shadow:0 1px 2px rgba(15,23,42,.035);
  min-width:0;
}
.overview-summary-card .overview-summary-body {
  max-height:220px;
  overflow-y:auto;
  overscroll-behavior:contain;
  padding-right:6px;
  white-space:normal;
  overflow-wrap:anywhere;
  word-break:normal;
}
.overview-summary-card .overview-summary-body p {
  margin:0;
}
.overview-objective-body {
  display:flex;
  flex-direction:column;
  gap:9px;
}
.overview-objective-block {
  padding:9px 10px;
  border:1px solid #ece8ff;
  border-radius:8px;
  background:#fbfaff;
}
.overview-objective-block b {
  display:block;
  margin:0 0 4px;
  color:#5b35e6;
  font-size:10.5px;
  font-weight:850;
  letter-spacing:.04em;
  text-transform:uppercase;
}
.overview-objective-block p {
  margin:0;
  color:var(--ink);
  font-size:13px;
  line-height:1.55;
}
.overview-summary-list {
  margin:0;
  padding:0 0 0 17px;
  display:flex;
  flex-direction:column;
  gap:7px;
}
.overview-summary-list li {
  margin:0;
  font-size:13.5px;
  line-height:1.55;
  color:var(--ink);
}
.saas-card p { margin:0; font-size:13.5px; line-height:1.55; color:var(--ink); }
.saas-icon-card-head {
  display:flex;
  align-items:center;
  gap:9px;
  margin:0 0 9px;
}
.saas-icon-card-head .saas-card-kicker {
  margin:0 !important;
}
.overview-card-icon {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:26px;
  height:26px;
  border-radius:8px;
  color:var(--saas-violet);
  background:#f3efff;
  border:1px solid #e4dcff;
  flex:0 0 auto;
}
.overview-card-icon svg {
  width:16px;
  height:16px;
  fill:none;
  stroke:currentColor;
  stroke-width:2;
  stroke-linecap:round;
  stroke-linejoin:round;
}
.saas-count-warning {
  margin:8px 0 0; color:var(--muted); font-size:12px;
}
.saas-card-kicker {
  margin:0 0 8px !important; color:var(--saas-violet) !important;
  font-size:11.5px !important; font-weight:700; letter-spacing:.01em;
}
.saas-stats { display:grid; grid-template-columns:repeat(4,1fr); gap:0; padding:0; overflow:hidden; }
.saas-stat { padding:15px 14px; border-left:1px solid var(--line); min-width:0; }
.saas-stat:first-child { border-left:none; }
.saas-stat-value {
  display:block; color:var(--saas-violet); font-size:28px; line-height:1; font-weight:760;
  font-variant-numeric:tabular-nums;
}
.saas-stat-label { display:block; margin-top:6px; font-size:12px; color:var(--ink); font-weight:700; }
.saas-stat-detail { display:block; margin-top:3px; font-size:11.5px; color:var(--muted); }
.saas-stat .prov { margin-top:7px; display:inline-block; }
.saas-caveat { border-color:#f1d7a7; background:#fffaf0; }
.saas-caveat p {
  max-width:100%;
  overflow-wrap:anywhere;
  word-break:break-word;
}
.saas-caveat .saas-card-kicker { color:var(--saas-amber) !important; }
.saas-caveat-ok { border-color:var(--line-strong); background:#fff; }
.saas-rationale-grid {
  display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px;
}
.agent-rationale-card {
  display:flex;
  flex-direction:column;
  min-height:0;
}
.agent-rationale-card-body {
  max-height:220px;
  overflow-y:auto;
  overscroll-behavior:contain;
  padding-right:6px;
  min-width:0;
}
.agent-rationale-section .agent-rationale-card-body p,
.agent-rationale-section .agent-rationale-card-body li,
.agent-rationale-section .agent-rationale-card-body span {
  white-space:normal;
  overflow-wrap:anywhere;
  word-break:normal;
  overflow:visible;
  text-overflow:clip;
  -webkit-line-clamp:unset;
  -webkit-box-orient:initial;
}
.saas-stage-chips { display:flex; flex-wrap:wrap; gap:6px; }
.saas-stage-chips span, .saas-chip {
  display:inline-flex; align-items:center; gap:4px;
  min-height:24px; padding:3px 8px; border-radius:6px;
  background:#f6f4ff; border:1px solid #e2ddff; color:#31216f;
  font-size:11.5px; font-weight:650; white-space:nowrap;
}
.saas-chip b { font-weight:750; }
.saas-chip-muted { color:var(--muted); background:#f8f8fb; border-color:var(--line); }
.saas-decisions ul { margin:0; padding:0; list-style:none; display:flex; flex-direction:column; gap:10px; }
.saas-decisions li { border-top:1px solid var(--line); padding-top:10px; }
.saas-decisions li:first-child { border-top:none; padding-top:0; }
.saas-decisions .decision-label {
  display:block;
  margin-bottom:6px;
  font-size:12px;
  font-weight:800;
  color:var(--ink);
}
.saas-decisions .decision-text {
  margin:0;
  color:#344054;
  font-size:13px;
  line-height:1.6;
}
.decision-text-structured {
  display:flex;
  flex-direction:column;
  gap:8px;
}
.decision-section {
  padding:8px 10px;
  border:1px solid #ece8ff;
  border-radius:8px;
  background:#fbfaff;
}
.decision-section b {
  display:block;
  margin:0 0 3px;
  color:#5b35e6;
  font-size:10.5px;
  font-weight:850;
  letter-spacing:.04em;
  text-transform:uppercase;
}
.decision-section p {
  margin:0;
  color:#344054;
  font-size:12.5px;
  line-height:1.55;
}
.decision-justification-card {
  padding:9px 11px;
  border:1px solid #c9e8ef;
  border-radius:8px;
  background:linear-gradient(180deg,#f7fdff 0%,#eef9fb 100%);
  box-shadow:0 5px 14px rgba(15,118,128,.06);
}
.decision-rationale-card {
  padding:9px 11px;
  border:1px solid #ece8ff;
  border-radius:8px;
  background:linear-gradient(180deg,#fbfaff 0%,#f5f1ff 100%);
  box-shadow:0 5px 14px rgba(91,53,230,.06);
}
.decision-justification-card p,
.decision-rationale-card p {
  margin:0;
  color:#29485a;
  font-size:12.7px;
  line-height:1.58;
}
.decision-rationale-card p {
  color:#3b3268;
}
.saas-table-wrap { overflow:auto; }
.saas-table { width:100%; border-collapse:separate; border-spacing:0; font-size:13px; }
.saas-table th {
  text-align:left; padding:9px 10px; color:var(--muted); font-size:11px;
  text-transform:uppercase; letter-spacing:.06em; border-bottom:1px solid var(--line);
}
.saas-table td { padding:10px; border-bottom:1px solid var(--line); vertical-align:top; }
.saas-table code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
.comparison-summary-row {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:12px;
}
.comparison-summary-card {
  border:1px solid var(--line); border-radius:8px; background:#fff; padding:11px 12px;
  display:flex; justify-content:space-between; align-items:center; gap:10px;
}
.comparison-summary-card span { color:var(--muted); font-size:12px; font-weight:700; }
.comparison-summary-card strong { color:var(--saas-violet); font-size:20px; line-height:1; }
.comparison-matrix-wrap {
  border:1px solid var(--line); border-radius:9px; background:#fff;
  max-height:clamp(360px, 55vh, 680px);
  overflow:auto;
}
.candidate-comparison-matrix thead th {
  position:sticky; top:0; z-index:2; background:#fff;
}
.candidate-comparison-matrix th,
.candidate-comparison-matrix td { white-space:nowrap; }
.candidate-comparison-matrix .comparison-verdict {
  min-width:220px; max-width:320px; white-space:normal; color:#3b3268; font-weight:550;
}
.candidate-comparison-matrix tr:first-child td {
  background:#fcfbff;
}
.comparison-rank {
  display:inline-flex; min-width:26px; min-height:26px; align-items:center; justify-content:center;
  border-radius:7px; background:#eef2ff; color:#31216f; font-weight:850;
}
.matrix-chip {
  display:inline-flex; align-items:center; min-height:24px; padding:2px 8px;
  border-radius:6px; background:#f6f4ff; border:1px solid #e2ddff;
  color:#20163f; font-weight:750; font-variant-numeric:tabular-nums;
}
.metric-missing {
  display:inline-flex; min-height:24px; align-items:center; padding:2px 8px;
  border-radius:6px; border:1px dashed var(--line-strong); color:var(--muted);
  background:#fafafa; font-size:11.5px; font-weight:650;
}
.comparison-note { margin:10px 0 0; color:var(--muted); font-size:12.5px; }
.saas-dossier-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
.saas-dossier-card {
  border:1px solid var(--line); border-radius:8px; padding:14px; background:#fff;
}
.saas-dossier-card h3 {
  margin:0 0 8px; font-size:15px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
}
.saas-dossier-card p { margin:0 0 10px; font-size:13px; color:var(--muted); }
.saas-chip-row { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:10px; }
.saas-dossier-meta { display:flex; flex-wrap:wrap; gap:6px; }
.saas-dossier-meta span {
  font-size:11px; color:var(--muted); background:#fafafa; border:1px solid var(--line);
  border-radius:5px; padding:2px 6px;
}
.saas-pattern-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
.saas-pattern-card { border:1px solid var(--line); border-radius:8px; background:#fff; padding:13px; min-height:132px; }
.saas-pattern-card h3 { margin:0 0 10px; font-size:13px; }
.saas-pattern-card p { margin:8px 0 0; color:var(--muted); font-size:12px; }
.saas-bars { height:72px; display:flex; align-items:flex-end; gap:4px; border-bottom:1px solid var(--line); }
.saas-bars > span {
  flex:1; min-width:5px; border-radius:4px 4px 0 0;
  background:linear-gradient(180deg,#7c5cff,#2aa7a0);
}
.saas-count-row { display:flex; justify-content:space-between; gap:8px; padding:5px 0; border-top:1px solid var(--line); }
.saas-count-row:first-of-type { border-top:none; }
.saas-count-row span { color:var(--muted); font-size:12px; }
.saas-count-row b { font-size:12px; color:var(--ink); }
.metric-range-card .range-line {
  position:relative; height:8px; margin:28px 4px 18px; border-radius:999px;
  background:linear-gradient(90deg,#ede9fe,#5eead4);
}
.metric-range-card .range-line span,
.metric-range-card .range-line b {
  position:absolute; top:50%; transform:translate(-50%,-50%);
  width:12px; height:12px; border-radius:50%; background:#fff;
  border:2px solid var(--saas-violet);
}
.metric-range-card .range-line b {
  width:16px; height:16px; background:var(--saas-violet); border-color:#fff;
  box-shadow:0 0 0 2px rgba(91,53,230,.25);
}
.saas-insight-note {
  margin-top:12px; border:1px solid #ded8ff; border-radius:8px; background:#fbfaff;
  padding:10px 12px; color:#3b3268; font-size:12.5px; line-height:1.45;
}
.structure-comparison-panel { overflow:hidden; }
.structure-comparison-panel:fullscreen {
  display:flex; flex-direction:column; width:100vw; height:100vh;
  max-width:none; border-radius:0; border:none; padding:18px; background:#fff;
}
.structure-comparison-panel:fullscreen .structure-comparison-grid {
  flex:1 1 auto; max-height:none; min-height:0;
  align-content:start; grid-auto-rows:minmax(1040px, auto); overflow-y:auto;
}
.structure-comparison-panel:fullscreen .structure-compare-card {
  display:flex; flex-direction:column; min-height:1040px;
}
.structure-comparison-panel:fullscreen .structure-compare-card .structure-section {
  flex:1 1 auto; min-height:0;
}
.structure-comparison-panel:fullscreen .structure-section-compact .structure-viewer {
  flex:1 1 auto; min-height:820px; height:auto;
}
.structure-comparison-head {
  display:flex; align-items:center; justify-content:space-between; gap:14px;
  margin-bottom:12px;
}
.structure-comparison-actions {
  display:flex; align-items:center; gap:10px; flex:0 0 auto;
}
.structure-comparison-actions > span {
  color:#667085; font-size:12px; font-style:normal; font-weight:760; white-space:nowrap;
}
.structure-comparison-subtitle-row {
  display:flex; align-items:center; flex-wrap:wrap; gap:12px; margin-top:4px;
}
.structure-comparison-subtitle-row .saas-section-subtitle { margin:0; }
.structure-comparison-fullscreen-btn {
  display:inline-flex; align-items:center; justify-content:center; gap:6px;
  min-height:36px; padding:0 13px; border:1px solid #d9d2ff;
  border-radius:9px; background:#f7f4ff; color:var(--primary-700);
  font-size:12.5px; font-weight:850; cursor:pointer;
  box-shadow:0 6px 14px rgba(91,53,230,.10);
}
.structure-comparison-fullscreen-btn:hover {
  border-color:var(--primary); background:var(--primary); color:#fff;
  box-shadow:0 10px 20px rgba(91,53,230,.20);
}
.structure-comparison-intro {
  display:flex; align-items:flex-start; justify-content:space-between; gap:16px;
  margin:14px 0 16px; padding:12px 14px; border:1px solid var(--border);
  border-radius:var(--radius-md); background:#fbfcff;
}
.structure-comparison-intro p {
  margin:0; color:#344054; font-size:13px; line-height:1.45; max-width:72ch;
}
.structure-comparison-intro span {
  color:#667085; font-size:12px; font-weight:760; text-align:right;
}
.structure-comparison-grid {
  display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px;
  max-height:clamp(500px, 68vh, 620px);
  overflow-y:auto; overscroll-behavior:contain; padding-right:6px;
}
.structure-compare-card {
  min-width:0; border:1px solid var(--border); border-radius:var(--radius-lg);
  background:#fff; overflow:hidden; cursor:grab;
}
.structure-compare-card:active { cursor:grabbing; }
.structure-compare-card.is-dragging { opacity:.62; border-color:var(--primary); box-shadow:var(--shadow-md); }
.structure-compare-head,
.structure-compare-foot {
  display:flex; align-items:center; justify-content:space-between; gap:10px; padding:10px 12px;
}
.structure-compare-head { border-bottom:1px solid var(--border); }
.structure-compare-head h3 {
  margin:2px 0 0; color:#172033; font-size:14px; line-height:1.2; word-break:break-word;
}
.structure-compare-rank {
  color:#667085; font-size:11px; font-weight:820; text-transform:uppercase;
}
.structure-compare-metrics {
  display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:8px; padding:10px 12px 0;
}
.structure-compare-metric {
  min-width:0; padding:7px 8px; border:1px solid var(--border); border-radius:8px; background:#f8fafc;
}
.structure-compare-metric b,
.structure-compare-metric em {
  display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.structure-compare-metric b {
  color:#667085; font-size:10.5px; text-transform:uppercase;
}
.structure-compare-metric strong {
  display:block; margin-top:2px; color:#172033; font-size:13px;
}
.structure-compare-metric em {
  color:#667085; font-size:10px; font-style:normal;
}
.structure-compare-metric.is-missing { color:#667085; font-size:12px; }
.structure-compare-card .structure-section { margin:10px 12px 12px; }
.structure-section-compact .structure-header { gap:8px; }
.structure-section-compact .structure-hint,
.structure-section-compact .structure-coverage-note { display:none; }
.structure-section-compact .structure-viewer { min-height:320px; }
.structure-section-compact .structure-legend { margin-top:8px; }
.structure-compare-foot { border-top:1px solid var(--border); }
.structure-compare-foot a {
  color:var(--primary-700); text-decoration:none; font-size:12px; font-weight:820;
}
.saas-trace {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px;
  list-style:none; padding:0; margin:0;
}
.saas-trace-stage {
  position:relative; border:1px solid var(--line); background:#fff; border-radius:8px;
  padding:14px 13px 12px;
}
.saas-trace-dot {
  display:inline-block; width:28px; height:28px; border-radius:50%;
  background:linear-gradient(135deg,var(--saas-violet),var(--saas-teal));
  margin-bottom:8px;
}
.saas-trace-stage strong { display:block; font-size:13px; }
.saas-trace-stage p { margin:5px 0; color:var(--muted); font-size:12px; }
.saas-trace-stage small { color:var(--subtle); font-size:11px; }
.candidate-review-board-layout {
  grid-template-columns:minmax(620px, .9fr) minmax(840px, 1.25fr);
  gap:20px;
}
.card-console {
  border:none; background:transparent; box-shadow:none; padding:0;
}
.card-console:hover { box-shadow:none; }
.card-console > .card-header,
.card-console > .widget-explainer,
.card-console > .console-banner {
  margin-left:0; margin-right:0;
}
.card-console > .card-header {
  padding:0 0 12px; border-bottom:none;
}
.candidate-ranking-panel {
  display:flex;
  flex-direction:column;
  gap:7px;
  max-width:1120px;
  margin-top:10px;
  padding:10px 12px;
  border:1px solid var(--line-strong);
  border-radius:8px;
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
  box-shadow:0 1px 2px rgba(15,23,42,.035);
}
.candidate-ranking-head {
  display:flex;
  align-items:center;
  gap:10px;
  min-width:0;
}
.candidate-ranking-label {
  flex:0 0 auto;
  color:#6b5aa9;
  font-size:10px;
  font-weight:850;
  letter-spacing:.055em;
  text-transform:uppercase;
}
.candidate-ranking-meta {
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:6px;
  min-width:0;
  color:var(--muted);
  font-size:12px;
}
.candidate-ranking-meta strong {
  color:var(--ink);
  font-weight:840;
}
.candidate-ranking-meta span {
  min-width:0;
}
.candidate-ranking-line {
  display:grid;
  grid-template-columns:48px minmax(0,1fr);
  gap:8px;
  margin:0;
  color:var(--text-soft);
  font-size:12px;
  line-height:1.45;
}
.candidate-ranking-line b {
  color:var(--muted);
  font-weight:800;
}
.candidate-ranking-line span,
.candidate-ranking-line code {
  min-width:0;
  white-space:normal;
  overflow-wrap:anywhere;
  word-break:normal;
}
.candidate-ranking-line code {
  padding:0;
  border-radius:0;
  background:transparent;
  color:#1f2937;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.8px;
}
.candidate-review-board {
  max-height:clamp(760px, 84vh, 1120px);
  overflow-y:auto;
  overscroll-behavior:contain;
  padding:14px;
  border:1px solid var(--line-strong); border-radius:10px; background:#fff;
  box-shadow:0 8px 28px rgba(15,23,42,.04);
}
.candidate-board .board-group { margin-top:12px; }
.candidate-board .board-group-head {
  padding:8px 2px 10px; border-bottom:1px solid var(--line);
}
.candidate-board .board-group-title {
  font-size:11px; color:#3b3268; letter-spacing:.08em; font-weight:850;
}
.candidate-board .board-group-count {
  min-width:22px; height:22px; display:inline-flex; align-items:center; justify-content:center;
  border-radius:999px; background:#f3f0ff; color:var(--saas-violet);
  font-weight:800; font-size:11px;
}
.candidate-board .board-group-count::before { content:none; }
.candidate-board .board-group-cards { gap:10px; padding:10px 0 12px; }
.candidate-board-row {
  display:grid; grid-template-columns:1fr; align-items:start; gap:10px; width:100%;
  padding:14px; border:1px solid var(--line); border-radius:10px;
  background:#fff; text-align:left; cursor:pointer;
  box-shadow:0 1px 2px rgba(15,23,42,.035);
  transition:border-color .14s ease, box-shadow .14s ease, transform .14s ease;
}
.candidate-board-row:hover {
  border-color:#cfc8ff; box-shadow:0 12px 28px rgba(91,53,230,.09);
  transform:translateY(-1px);
}
.candidate-board-row[aria-selected="true"] {
  border-color:var(--saas-violet); box-shadow:0 0 0 1px rgba(91,53,230,.18), 0 14px 32px rgba(91,53,230,.10);
  background:#fcfbff;
}
.candidate-board-row.is-final {
  border-color:#8c75ff; background:linear-gradient(180deg,#ffffff 0%,#fbf9ff 100%);
}
.candidate-card-top {
  display:flex; align-items:flex-start; justify-content:space-between; gap:12px;
}
.candidate-card-title { min-width:0; display:flex; flex-direction:column; gap:5px; align-items:flex-end; }
.candidate-board-row .crow-rank { display:flex; align-items:center; justify-content:center; }
.candidate-board-row .crow-rank-num,
.candidate-board-row .crow-rank-final {
  min-width:32px; min-height:32px; display:inline-flex; align-items:center; justify-content:center;
  border-radius:8px; background:#eef2ff; color:#31216f; font-size:13px; font-weight:850;
}
.candidate-board-row.is-final .crow-rank-final {
  background:var(--saas-violet); color:#fff;
}
.board-row-id {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:14px; font-weight:850;
  color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.status-badge {
  display:inline-flex; align-items:center; min-height:22px; padding:2px 8px;
  border-radius:999px; background:#f4f1ff; border:1px solid #ded8ff;
  color:#37206f; font-size:10.5px; font-weight:800;
}
.status-badge-final { background:#ecfdf3; border-color:#bbf7d0; color:#067647; }
.status-badge-ranked { background:#eff6ff; border-color:#bfdbfe; color:#1d4ed8; }
.status-badge-evaluated { background:#eef2ff; border-color:#c7d2fe; color:#3730a3; }
.status-badge-generated { background:#fff7ed; border-color:#fed7aa; color:#9a3412; }
.status-badge-reference { background:#f8fafc; border-color:#cbd5e1; color:#475569; }
.status-badge-incomplete { background:#fff7ed; border-color:#fed7aa; color:#9a3412; }
.status-badge-failed { background:#fef2f2; border-color:#fecaca; color:#991b1b; }
.board-row-verdict { color:#302a52; font-size:13px; line-height:1.42; font-weight:600; }
.board-row-metrics { display:flex; flex-wrap:wrap; gap:6px; }
.board-row-footer {
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  padding-top:8px; border-top:1px solid var(--line);
}
.candidate-assets, .comparison-assets { display:flex; flex-wrap:wrap; gap:5px; }
.asset-pill {
  display:inline-flex; align-items:center; min-height:22px; padding:2px 7px;
  border-radius:6px; font-size:10.5px; font-weight:750;
}
.asset-pill.is-on { background:#f0fdfa; border:1px solid #99f6e4; color:#0f766e; }
.asset-pill.is-missing { background:#f8fafc; border:1px solid var(--line); color:var(--muted); }
.open-dossier-action {
  color:var(--saas-violet); font-size:12px; padding:6px 10px; border:1px solid #ded8ff;
  border-radius:7px; background:#f6f4ff; white-space:nowrap;
}
.console-detail {
  min-height:720px; border-color:var(--line-strong); border-radius:10px; background:#fff;
  box-shadow:0 8px 28px rgba(15,23,42,.04);
}
.candidate-dossier-primary {
  display:grid; grid-template-columns:1fr;
  gap:14px; align-items:start; margin-top:14px;
}
.dossier-summary-panel {
  display:grid; grid-template-columns:minmax(0,1.1fr) minmax(260px,.9fr) auto;
  gap:14px; align-items:start; margin:0 0 14px;
  border:1px solid #ded8ff; background:linear-gradient(180deg,#ffffff 0%,#fbfaff 100%);
  border-radius:10px; padding:14px;
}
.dossier-verdict { margin:0; font-size:14px; line-height:1.5; color:#20163f; font-weight:650; }
.dossier-decision-points { margin:0; padding-left:18px; color:#3b3268; font-size:12.5px; line-height:1.45; }
.dossier-decision-points li + li { margin-top:4px; }
.dossier-structure-primary {
  border:1px solid var(--line); border-radius:8px; background:#fff; overflow:hidden;
}
.dossier-structure-primary .structure-section {
  border:none; border-radius:0; min-height:720px;
}
.dossier-structure-primary .structure-viewer {
  min-height:580px; height:clamp(580px, 64vh, 840px);
}
.dossier-evidence-primary {
  border:1px solid var(--line); border-radius:8px; background:#fff; padding:14px;
}
.dossier-evidence-primary h4,
.dossier-alert h4 {
  margin:0 0 10px; font-size:12px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted);
}
.dossier-evidence-primary .score-grid { grid-template-columns:1fr; }
.candidate-dossier-primary .dossier-evidence-primary .score-grid {
  grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
}
.dossier-alert-grid {
  display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:14px 0;
}
.dossier-alert {
  border:1px solid var(--line); border-radius:8px; background:#fff; padding:12px 14px;
}
.dossier-risk { background:#fffaf0; border-color:#f1d7a7; }
.dossier-missing { background:#f8fbff; border-color:#cfe0ff; }
.dossier-alert ul { margin:0; padding-left:18px; color:var(--ink); font-size:12.5px; }
.dossier-alert li + li { margin-top:4px; }
.dossier-no-structure { padding:14px; }
.dossier-advanced {
  margin-top:14px; border:1px solid var(--line); border-radius:8px; background:#fff;
}
.dossier-advanced > summary {
  cursor:pointer; padding:10px 13px; font-size:12.5px; font-weight:800;
  color:var(--saas-violet); list-style:none;
}
.dossier-advanced > summary::-webkit-details-marker { display:none; }
.dossier-advanced[open] > summary { border-bottom:1px solid var(--line); }
.dossier-advanced .cand-tabs,
.dossier-advanced .cand-tabpanels { margin-left:12px; margin-right:12px; }
.structure-health {
  margin:10px 14px; padding:10px 12px; border:1px solid #f1d7a7; border-radius:7px;
  background:#fffaf0; color:#7c2d12; font-size:12px;
}
.structure-health strong { display:block; margin-bottom:2px; color:#7c2d12; }
.structure-viewer {
  min-height:420px;
  contain:layout paint;
}
.structure-viewer-context-lost {
  outline:2px solid #f1d7a7;
  outline-offset:-2px;
}
.appendix-panel {
  padding:0; overflow:hidden; background:#fbfbfd; border-style:dashed;
  box-shadow:none; opacity:.92;
}
.appendix-panel:hover { box-shadow:0 4px 18px rgba(15,23,42,.04); opacity:1; }
.appendix-panel > summary {
  cursor:pointer; list-style:none; display:flex; justify-content:space-between; align-items:center;
  gap:16px; padding:13px 18px; color:#3b3268; font-weight:800;
}
.appendix-panel > summary::-webkit-details-marker { display:none; }
.appendix-panel > summary::before {
  content:"+"; width:20px; height:20px; display:inline-flex; align-items:center; justify-content:center;
  border:1px solid var(--line-strong); border-radius:6px; color:var(--saas-violet); margin-right:6px;
  background:#fff;
}
.appendix-panel[open] > summary::before { content:"-"; }
.appendix-panel > summary span { flex:1; }
.appendix-panel > summary small { color:var(--muted); font-weight:600; }
.appendix-panel > :not(summary) { margin-left:20px; margin-right:20px; }
.appendix-panel > :last-child { margin-bottom:20px; }
@media (max-width: 1200px) {
  .saas-overview-grid { grid-template-columns:1fr 1fr; }
  .saas-stats { grid-column:1 / -1; }
  .saas-rationale-grid, .saas-pattern-grid { grid-template-columns:1fr 1fr; }
  .candidate-review-board-layout, .candidate-dossier-primary { grid-template-columns:1fr; }
  .candidate-review-board { max-height:420px; }
  .dossier-summary-panel { grid-template-columns:1fr; }
}
@media (max-width: 720px) {
  .saas-overview-grid, .saas-rationale-grid, .saas-pattern-grid { grid-template-columns:1fr; }
  .saas-stats { grid-template-columns:1fr 1fr; }
  .top-bar { align-items:flex-start; flex-direction:column; height:auto; padding:10px 16px; }
  .top-bar-right { width:100%; }
  .dossier-alert-grid { grid-template-columns:1fr; }
  .board-row-footer, .candidate-card-top { align-items:flex-start; flex-direction:column; }
}

/* SaaS visual polish pass.
   This final layer only changes presentation. It intentionally keeps the
   existing report hierarchy, view-model data flow, and 3Dmol viewer logic. */
:root {
  --app-bg:#f8fafc;
  --app-bg-soft:#fbfcff;
  --surface:#ffffff;
  --surface-soft:#f7f8fc;
  --surface-tint:#fbfaff;
  --surface-violet:#f5f2ff;
  --surface-teal:#effdfb;
  --border:#e6e8f0;
  --border-strong:#d9ddec;
  --text:#111827;
  --text-soft:#374151;
  --text-muted:#667085;
  --text-subtle:#98a2b3;
  --primary:#5b35e6;
  --primary-700:#4320c8;
  --primary-soft:#f1edff;
  --blue:#2563eb;
  --teal:#0f9f9a;
  --success:#16a34a;
  --success-soft:#ecfdf3;
  --warning:#b7791f;
  --warning-soft:#fffbeb;
  --danger:#dc2626;
  --danger-soft:#fef2f2;
  --shadow-xs:0 1px 2px rgba(15,23,42,.04);
  --shadow-sm:0 6px 18px rgba(15,23,42,.055);
  --shadow-md:0 14px 36px rgba(15,23,42,.075);
  --radius-xs:5px;
  --radius-sm:7px;
  --radius-md:8px;
  --radius-lg:10px;
}

html { scroll-behavior:smooth; }
body {
  background:
    radial-gradient(circle at 45% -10%, rgba(91,53,230,.07), transparent 32%),
    linear-gradient(180deg,#fbfcff 0%,var(--app-bg) 42%,#f6f8fc 100%);
  color:var(--text);
  -webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
  overflow-x:hidden;
  transition:grid-template-columns .18s ease;
}
body.sidebar-collapsed {
  --sidebar-w:74px;
}

a, button, summary { -webkit-tap-highlight-color:transparent; }
a:focus-visible,
button:focus-visible,
summary:focus-visible,
.board-card:focus-visible,
.candidate-board-row:focus-visible,
.top-bar-action:focus-visible {
  outline:3px solid rgba(91,53,230,.24);
  outline-offset:2px;
}

.sidebar {
  background:rgba(255,255,255,.92);
  backdrop-filter:blur(18px);
  border-right:1px solid var(--border);
  box-shadow:1px 0 18px rgba(16,24,40,.045);
  overflow:visible;
  z-index:30;
}
.sidebar-brand {
  position:relative;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:20px 22px 20px;
  gap:10px;
  margin:0 -14px 14px;
  border-bottom:1px solid var(--border);
  overflow:visible;
}
.sidebar-brand-main {
  display:flex;
  align-items:center;
  gap:11px;
  min-width:0;
  padding-right:30px;
}
.brand-lockup {
  display:block;
  position:relative;
  width:238px;
  height:80px;
  overflow:hidden;
  flex:0 0 238px;
}
.brand-lockup img {
  position:absolute;
  width:330px;
  height:auto;
  left:50%;
  top:53%;
  transform:translate(-50%, -50%);
  display:block;
  max-width:none;
  pointer-events:none;
}
.brand-fallback {
  color:var(--text);
  font-size:17px;
  font-weight:820;
  line-height:1.15;
  white-space:nowrap;
}
.brand-mark {
  width:40px;
  height:40px;
  border-radius:0;
  display:block;
  flex:0 0 auto;
  box-shadow:none;
}
.brand-text strong {
  color:var(--text);
  font-size:17px;
  font-weight:820;
  line-height:1.15;
  white-space:nowrap;
}
.brand-text small {
  color:var(--text-muted);
  font-size:12px;
  white-space:nowrap;
}
.sidebar-toggle {
  position:absolute;
  right:-24px;
  top:16px;
  z-index:70;
  width:40px;
  height:40px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border-radius:12px;
  border:1px solid #d8dded;
  background:#fff;
  color:var(--primary);
  box-shadow:0 12px 26px rgba(16,24,40,.13);
  cursor:pointer;
  transition:background .14s ease, border-color .14s ease, box-shadow .14s ease, transform .14s ease;
}
.sidebar-toggle:hover {
  background:#f7f4ff;
  border-color:#cfc4ff;
  box-shadow:0 14px 30px rgba(91,53,230,.16);
  transform:translateX(-1px);
}
.sidebar-toggle svg {
  width:19px;
  height:19px;
  fill:none;
  stroke:currentColor;
  stroke-width:2.4;
  stroke-linecap:round;
  stroke-linejoin:round;
  transition:transform .18s ease;
}
body.sidebar-collapsed .sidebar-toggle svg {
  transform:rotate(180deg);
}
body.sidebar-collapsed .brand-text,
body.sidebar-collapsed .sidebar-meta,
body.sidebar-collapsed .side-nav-label,
body.sidebar-collapsed .side-nav-divider {
  display:none;
}
body.sidebar-collapsed .sidebar {
  padding-left:10px;
  padding-right:10px;
}
body.sidebar-collapsed .sidebar-brand {
  justify-content:center;
  padding-left:10px;
  padding-right:10px;
}
body.sidebar-collapsed .sidebar-brand-main {
  justify-content:center;
  padding-right:0;
}
body.sidebar-collapsed .brand-lockup {
  width:44px;
  height:44px;
  flex-basis:44px;
  border-radius:12px;
}
body.sidebar-collapsed .brand-lockup img {
  width:116px;
  left:22%;
}
body.sidebar-collapsed .side-nav {
  padding-left:4px;
  padding-right:4px;
}
body.sidebar-collapsed .side-nav-item {
  justify-content:center;
  padding:0;
}
body.sidebar-collapsed .side-nav-glyph {
  margin:0;
}
.side-nav { padding:12px 12px; gap:5px; }
.side-nav-item {
  min-height:38px;
  padding:0 11px;
  border-radius:var(--radius-md);
  color:#27304a;
  font-size:13px;
  font-weight:650;
  transition:background .14s ease, color .14s ease, box-shadow .14s ease, transform .14s ease;
}
.side-nav-item:hover {
  background:var(--primary-soft);
  color:var(--primary-700);
  transform:translateX(1px);
}
.side-nav-item.active,
.side-nav-item[aria-current="page"] {
  background:linear-gradient(180deg,#f6f3ff 0%,#eee9ff 100%);
  color:var(--primary-700);
  box-shadow:inset 0 0 0 1px #ded6ff;
}
.side-nav-glyph {
  width:18px;
  color:var(--primary);
  opacity:.92;
}
.side-nav-glyph svg {
  width:17px;
  height:17px;
  display:block;
  fill:none;
  stroke:currentColor;
  stroke-width:2;
  stroke-linecap:round;
  stroke-linejoin:round;
}
.sidebar-meta {
  margin:10px 13px;
  padding:11px 12px;
  border-radius:var(--radius-md);
  background:var(--surface);
  border:1px solid var(--border);
  box-shadow:var(--shadow-xs);
}
.sidebar-meta-value {
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size:13px;
  font-weight:650;
  letter-spacing:0;
  line-height:1.35;
  color:#27304a;
}
.sidebar-foot {
  margin:auto 12px 0;
  padding:12px 4px 4px;
  border:none;
  background:transparent;
  box-shadow:none;
  color:#667085;
  font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size:12px;
  font-weight:600;
  letter-spacing:.01em;
}
body.sidebar-collapsed .sidebar-foot {
  justify-content:center;
  margin-left:0;
  margin-right:0;
}

.top-bar {
  background:rgba(255,255,255,.86);
  backdrop-filter:blur(18px);
  border-bottom:1px solid var(--border);
  box-shadow:0 1px 2px rgba(16,24,40,.03);
  padding:0 24px;
  gap:16px;
}
.top-bar-left { min-width:0; flex:1 1 auto; }
.top-bar-right { flex:0 1 auto; justify-content:flex-end; min-width:0; }
.top-bar-status {
  border:1px solid #b8efcf;
  background:var(--success-soft);
  color:#067647;
  box-shadow:none;
}
.top-bar-workflow {
  min-width:0;
  color:#344054;
  font-size:13px;
}
.top-bar-date {
  color:var(--text-muted);
  white-space:nowrap;
}
.top-bar-action {
  min-height:34px;
  border-radius:var(--radius-sm);
  border-color:var(--border-strong);
  background:var(--surface);
  color:#293047;
  box-shadow:var(--shadow-xs);
  transition:background .14s ease, border-color .14s ease, box-shadow .14s ease, transform .14s ease;
}
.top-bar-action:hover {
  background:#fbfbff;
  border-color:#cfd4e4;
  box-shadow:var(--shadow-sm);
  transform:translateY(-1px);
}
.top-bar-action-primary {
  border-color:var(--primary);
  background:linear-gradient(180deg,#6849ee 0%,var(--primary) 100%);
  color:#fff;
  box-shadow:0 10px 22px rgba(91,53,230,.22);
}

.content {
  max-width:1720px;
  padding:26px 30px 64px;
  min-width:0;
}
.main,
.content,
.saas-section,
.card,
.saas-card,
.console-detail,
.candidate-board-row {
  min-width:0;
}
.section,
.saas-section,
.card {
  border-radius:var(--radius-lg);
  border-color:var(--border);
  background:rgba(255,255,255,.96);
  box-shadow:var(--shadow-xs);
}
.section:hover,
.saas-section:hover,
.card:hover {
  box-shadow:var(--shadow-sm);
}
.saas-overview,
.card-console {
  background:transparent;
  border:none;
  box-shadow:none;
}
.saas-overview:hover,
.card-console:hover {
  box-shadow:none;
}
.card-header {
  border-bottom-color:var(--border);
  gap:12px;
}
.card-title,
.section-title,
.saas-section h2 {
  color:var(--text);
  font-weight:820;
}
.card-eyebrow,
.kicker,
.saas-card-kicker,
.dossier-evidence-primary h4,
.dossier-alert h4 {
  color:#6b5aa9 !important;
  letter-spacing:.055em;
  font-weight:820;
}
.muted,
.widget-explainer,
.saas-section-subtitle,
.comparison-note {
  color:var(--text-muted);
}
.card-header .saas-section-subtitle {
  margin-top:5px;
  max-width:900px;
}
.card-console > .card-header {
  align-items:flex-start;
  padding-bottom:14px;
}
.card-console > .card-header + .console,
.card-console > .card-header + .console-banner,
.card-console > .card-header + .display-limit-notice,
.card-console > .card-header + .display-limit-notice + .console,
.card-console > .card-header + .display-limit-notice + .console-banner,
.card-console > .card-header + .display-limit-notice + .console-banner + .console,
.card-console > .card-header + .console-banner + .console {
  margin-top:0;
}

.saas-overview-grid {
  grid-template-columns:minmax(190px,1fr) minmax(220px,1.1fr) minmax(350px,2.05fr) minmax(190px,.95fr);
  gap:14px;
}
body {
  background:#ffffff;
}
.sidebar {
  background:linear-gradient(180deg,#fbfbff 0%,#f6f7fb 100%);
  border-right-color:#edf0f6;
}
.content {
  background:#ffffff;
}
.saas-card {
  border-color:var(--border);
  border-radius:var(--radius-lg);
  box-shadow:var(--shadow-xs);
}
.saas-card:hover {
  border-color:#d9dcf0;
  box-shadow:var(--shadow-sm);
}
.saas-card p {
  color:var(--text-soft);
}
.saas-stats {
  border-radius:var(--radius-lg);
  background:var(--surface);
}
.saas-stat {
  padding:16px 10px 14px;
  border-left-color:var(--border);
}
.saas-stat-value {
  font-size:30px;
  color:var(--primary);
}
.saas-stat-label {
  font-size:11.5px;
  overflow-wrap:normal;
}
.saas-stat-detail {
  font-size:11px;
}
.saas-stat:nth-child(2) .saas-stat-value { color:var(--blue); }
.saas-stat:nth-child(3) .saas-stat-value { color:var(--teal); }
.saas-stat:nth-child(4) .saas-stat-value { color:var(--success); }
.saas-caveat {
  background:linear-gradient(180deg,#fffdf5 0%,#fffaf0 100%);
  border-color:#f3d7a6;
}
.saas-caveat-ok {
  background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);
  border-color:var(--border);
}
.prov,
.status-badge,
.asset-pill,
.saas-chip,
.matrix-chip,
.metric-missing {
  letter-spacing:0;
}

.candidate-review-board-layout {
  grid-template-columns:minmax(640px,.92fr) minmax(860px,1.24fr);
  gap:24px;
  align-items:start;
}
.candidate-review-board {
  padding:16px;
  max-height:clamp(760px, 84vh, 1120px);
  overflow-y:auto;
  overscroll-behavior:contain;
  border-radius:var(--radius-lg);
  border-color:var(--border);
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
  box-shadow:var(--shadow-sm);
}
.candidate-board .board-group {
  border-top:1px solid var(--border);
  margin-top:14px;
  padding-top:2px;
}
.candidate-board .board-group:first-child {
  border-top:none;
  margin-top:0;
  padding-top:0;
}
.candidate-board .board-group-head {
  border-bottom:none;
  padding:7px 2px 8px;
}
.candidate-board .board-group-title {
  color:#344054;
  font-size:11px;
  letter-spacing:.07em;
}
.candidate-board .board-group-count {
  background:#f2f4f7;
  color:#475467;
}
.candidate-board-row {
  border-radius:var(--radius-lg);
  border-color:var(--border);
  background:var(--surface);
  box-shadow:0 1px 2px rgba(16,24,40,.035);
}
.candidate-board-row:hover {
  border-color:#cfc6ff;
  box-shadow:0 14px 34px rgba(91,53,230,.095);
}
.candidate-board-row[aria-selected="true"] {
  border-color:var(--primary);
  background:linear-gradient(180deg,#ffffff 0%,#fbfaff 100%);
  box-shadow:0 0 0 1px rgba(91,53,230,.18), 0 16px 36px rgba(91,53,230,.11);
}
.candidate-board-row.is-final {
  border-color:#9b88ff;
}
.candidate-board-row .crow-rank-num,
.candidate-board-row .crow-rank-final,
.comparison-rank {
  border-radius:var(--radius-sm);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.28);
}
.candidate-board-row.is-final .crow-rank-final {
  background:linear-gradient(180deg,#6e50f2 0%,var(--primary) 100%);
}
.board-row-id {
  color:#101828;
}
.board-row-verdict {
  color:#344054;
  font-weight:620;
}
.board-row-footer {
  border-top-color:var(--border);
}
.open-dossier-action {
  color:var(--primary-700);
  border-color:#d9d2ff;
  background:#f7f4ff;
  font-weight:760;
}
.candidate-board-row:hover .open-dossier-action {
  background:var(--primary);
  color:#fff;
  border-color:var(--primary);
}
.status-badge-final {
  background:var(--success-soft);
  border-color:#b7efca;
  color:#067647;
}
.status-badge-ranked {
  background:#eff6ff;
  border-color:#c7d7fe;
  color:#1d4ed8;
}
.status-badge-evaluated {
  background:#eef2ff;
  border-color:#c7d2fe;
  color:#3730a3;
}
.status-badge-generated {
  background:var(--warning-soft);
  border-color:#f4d29b;
  color:#92400e;
}
.status-badge-reference {
  background:#f8fafc;
  border-color:#cbd5e1;
  color:#475569;
}
.status-badge-incomplete {
  background:var(--warning-soft);
  border-color:#f4d29b;
  color:#92400e;
}
.status-badge-failed {
  background:var(--danger-soft);
  border-color:#fecaca;
  color:#991b1b;
}
.saas-chip,
.matrix-chip {
  border-color:#e2ddff;
  background:#f7f5ff;
  color:#282056;
}
.saas-chip-muted,
.metric-missing {
  border-color:var(--border);
  background:#f8fafc;
  color:var(--text-muted);
}
.asset-pill.is-on {
  background:var(--surface-teal);
  border-color:#a7f3e8;
  color:#0f766e;
}
.asset-pill.is-missing {
  background:#f8fafc;
  border-color:var(--border);
}

.console-detail {
  border-radius:var(--radius-lg);
  border-color:var(--border);
  box-shadow:var(--shadow-sm);
  position:sticky;
  top:78px;
}
.cand-detail-head {
  border-bottom-color:var(--border);
  background:linear-gradient(180deg,#ffffff 0%,#fbfbff 100%);
}
.dossier-summary-panel {
  border-radius:var(--radius-lg);
  border-color:#ddd6ff;
  background:linear-gradient(180deg,#ffffff 0%,#fbfaff 100%);
  box-shadow:0 8px 24px rgba(91,53,230,.055);
}
.dossier-verdict {
  color:#21164a;
}
.dossier-decision-points {
  color:#3b3268;
}
.dossier-structure-primary {
  border-radius:var(--radius-lg);
  border-color:var(--border);
  box-shadow:var(--shadow-xs);
}
.dossier-structure-primary .structure-section {
  background:#ffffff;
  min-height:720px;
}
.dossier-structure-primary .structure-viewer {
  border-top-color:var(--border);
  min-height:580px;
  height:clamp(580px, 64vh, 840px);
}
.dossier-structure-primary .downloads,
.cand-tabpanel[data-tab="structure"] > .downloads {
  margin:14px 14px 14px;
  padding:12px 14px;
  border:1px dashed var(--border);
  border-radius:var(--radius-md);
  background:#fbfcff;
}
.dossier-structure-primary .downloads-label,
.cand-tabpanel[data-tab="structure"] > .downloads .downloads-label {
  margin-right:2px;
}
.dossier-evidence-primary,
.dossier-alert,
.score-family,
.saas-dossier-card,
.saas-pattern-card,
.comparison-summary-card,
.comparison-matrix-wrap,
.saas-trace-stage {
  border-radius:var(--radius-lg);
  border-color:var(--border);
  background:var(--surface);
  box-shadow:var(--shadow-xs);
}
.score-family:hover,
.saas-dossier-card:hover,
.saas-pattern-card:hover,
.comparison-summary-card:hover,
.saas-trace-stage:hover {
  border-color:#d7dbea;
  box-shadow:var(--shadow-sm);
}
.metric-row {
  border-top-color:var(--border);
}
.metric-bar-row {
  --family-accent:#4f46e5;
  --family-soft:#f5f3ff;
  padding:8px 0;
  border-top:1px solid var(--border);
}
.metric-bar-row:first-child {
  border-top:none;
  padding-top:2px;
}
.metric-bar-top {
  display:flex;
  align-items:baseline;
  justify-content:space-between;
  gap:12px;
  min-width:0;
}
.metric-bar-track {
  display:block;
  height:7px;
  margin-top:7px;
  border-radius:999px;
  background:#eef1f7;
  overflow:hidden;
}
.metric-bar-track i {
  display:block;
  height:100%;
  border-radius:inherit;
  background:var(--family-accent);
}
.metric-bar-row.is-missing .metric-bar-track i {
  background:#cbd5e1;
}
.metric-value-pill {
  display:inline-flex;
  align-items:center;
  min-height:22px;
  padding:2px 8px;
  border-radius:999px;
  color:var(--family-accent);
  background:var(--family-soft);
  border:1px solid color-mix(in srgb, var(--family-accent) 22%, #ffffff);
  font-size:12px;
  text-transform:capitalize;
}
.metric-bar-row.is-categorical .metric-bar-top {
  align-items:flex-start;
  flex-wrap:wrap;
}
.metric-bar-row.is-categorical .metric-label {
  flex:0 1 150px;
  min-width:96px;
}
.metric-token-list {
  display:flex;
  flex:1 1 220px;
  min-width:min(100%, 160px);
  max-width:100%;
  flex-wrap:wrap;
  justify-content:flex-end;
  gap:6px;
}
.metric-token {
  display:inline-flex;
  align-items:center;
  max-width:100%;
  min-height:24px;
  padding:3px 8px;
  border-radius:6px;
  border:1px solid color-mix(in srgb, var(--family-accent) 18%, #d7dbea);
  background:color-mix(in srgb, var(--family-soft) 68%, #ffffff);
  color:#182230;
  font-size:12px;
  font-weight:650;
  line-height:1.35;
  white-space:normal;
  overflow-wrap:anywhere;
  word-break:normal;
}
.metric-token-more {
  color:var(--text-muted);
  background:#f8fafc;
  border-color:var(--border);
}
.score-family {
  --family-accent:#4f46e5;
  --family-soft:#f5f3ff;
  position:relative;
}
.score-family::before {
  content:"";
  position:absolute;
  left:0;
  top:12px;
  bottom:12px;
  width:3px;
  border-radius:999px;
  background:var(--family-accent);
  opacity:.72;
}
.score-family-interface-binding,
.metric-bar-row.score-family-interface-binding {
  --family-accent:#0f766e;
  --family-soft:#ecfdf5;
}
.score-family-structure-confidence,
.metric-bar-row.score-family-structure-confidence {
  --family-accent:#2563eb;
  --family-soft:#eff6ff;
}
.score-family-sequence-biophysics,
.metric-bar-row.score-family-sequence-biophysics {
  --family-accent:#7c3aed;
  --family-soft:#f5f3ff;
}
.score-family-novelty-similarity,
.metric-bar-row.score-family-novelty-similarity {
  --family-accent:#c2410c;
  --family-soft:#fff7ed;
}
.score-family-flags-liabilities,
.metric-bar-row.score-family-flags-liabilities {
  --family-accent:#16a34a;
  --family-soft:#ecfdf3;
}
.metric-value {
  color:#101828;
  font-weight:780;
}
@media (max-width: 760px) {
  .metric-token-list {
    justify-content:flex-start;
    flex-basis:100%;
  }
}
.dossier-alert-grid {
  gap:12px;
}
.dossier-risk {
  background:linear-gradient(180deg,#fffdf7 0%,#fffbeb 100%);
  border-color:#f2d195;
}
.dossier-missing {
  background:linear-gradient(180deg,#ffffff 0%,#f5f9ff 100%);
  border-color:#cfe0ff;
}
.dossier-advanced {
  border-color:var(--border);
  background:#fbfcff;
}
.dossier-advanced > summary {
  color:#4b35b4;
}

.comparison-summary-row {
  gap:12px;
}
.comparison-summary-card {
  padding:13px 14px;
}
.comparison-summary-card strong {
  color:var(--primary);
}
.comparison-matrix-wrap {
  max-height:clamp(360px, 55vh, 680px);
  overflow:auto;
  overscroll-behavior:contain;
}
.candidate-comparison-matrix {
  min-width:1080px;
}
.candidate-comparison-matrix thead th {
  position:sticky;
  top:0;
  z-index:2;
  background:#fff;
}
.saas-table th {
  background:#fbfcff;
  color:#667085;
  border-bottom-color:var(--border);
}
.saas-table td {
  border-bottom-color:var(--border);
}
.candidate-comparison-matrix tr:first-child td {
  background:#fbfaff;
}
.candidate-comparison-matrix .status-final td {
  background:#fcfbff;
}
.candidate-comparison-matrix .status-final .comparison-rank {
  background:linear-gradient(180deg,#6e50f2 0%,var(--primary) 100%);
  color:#fff;
}
.candidate-comparison-matrix tbody tr:hover td {
  background:#fbfcff;
}
.comparison-candidate-id {
  display:block;
  color:#111827;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12px;
  font-weight:850;
}
.comparison-candidate-sub {
  display:block;
  margin-top:3px;
  color:#98a2b3;
  font-size:10.5px;
  font-weight:700;
}
.candidate-comparison-matrix .status-badge {
  min-height:22px;
  padding:2px 8px;
  border-radius:999px;
  font-size:11px;
}
.metric-cell-missing {
  text-align:center;
}
.metric-missing {
  min-width:22px;
  min-height:22px;
  justify-content:center;
  padding:0 6px;
  border-style:solid;
  color:#a8a0bd;
  background:#fbfcff;
  font-size:13px;
  font-weight:800;
}
.matrix-chip {
  min-width:44px;
  justify-content:center;
}

.saas-pattern-grid {
  gap:14px;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
}
.saas-pattern-card {
  min-height:168px;
  padding:15px;
  border-radius:var(--radius-md);
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
}
.saas-pattern-card h3 {
  color:#1f2937;
  font-weight:800;
  margin-bottom:12px;
  letter-spacing:0;
}
.saas-pattern-card-empty {
  background:#fbfcff;
}
.saas-pattern-card-empty .empty {
  min-height:74px;
  display:flex;
  align-items:center;
  justify-content:center;
  border:1px dashed var(--border);
  border-radius:var(--radius-sm);
  color:#98a2b3;
}
.saas-missing-metrics-card {
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
}
.missing-metric-list {
  display:flex;
  flex-wrap:wrap;
  gap:6px;
  margin:2px 0 10px;
}
.missing-metric-token {
  display:inline-flex;
  align-items:center;
  min-height:24px;
  padding:3px 8px;
  border:1px dashed var(--border);
  border-radius:999px;
  background:#f8fafc;
  color:#667085;
  font-size:11.5px;
  font-weight:720;
}
.saas-bars {
  border-bottom-color:var(--border);
  height:76px;
  gap:5px;
  padding:0 2px;
}
.saas-bars > span {
  background:linear-gradient(180deg,#7c5cff 0%,#2fb7ad 100%);
  box-shadow:inset 0 -1px 0 rgba(15,23,42,.08);
  border-radius:5px 5px 0 0;
}
.saas-insight-note {
  background:linear-gradient(180deg,#ffffff 0%,#f7f4ff 100%);
  border-color:#ddd6ff;
}
.metric-range-card .range-line {
  margin:14px 4px 12px;
  height:7px;
  background:linear-gradient(90deg,#ede9fe 0%,#d6f5ee 52%,#5eead4 100%);
}
.metric-range-card p {
  font-size:11.5px;
  font-weight:650;
}
.range-filter-control {
  position:relative;
  height:34px;
  margin:12px 2px 6px;
}
.range-filter-track {
  position:absolute;
  left:0;
  right:0;
  top:14px;
  height:8px;
  border-radius:999px;
  background:linear-gradient(90deg,#ede9fe 0%,#d6f5ee 52%,#5eead4 100%);
  overflow:hidden;
}
.range-filter-window {
  position:absolute;
  top:0;
  bottom:0;
  left:0;
  right:0;
  border-radius:inherit;
  background:rgba(91,53,230,.18);
  box-shadow:inset 0 0 0 1px rgba(91,53,230,.18);
}
.range-filter-control input[type="range"] {
  position:absolute;
  inset:0;
  width:100%;
  height:34px;
  margin:0;
  appearance:none;
  -webkit-appearance:none;
  background:transparent;
  pointer-events:none;
}
.range-filter-control input[type="range"]::-webkit-slider-thumb {
  appearance:none;
  -webkit-appearance:none;
  width:20px;
  height:20px;
  border-radius:999px;
  border:4px solid #ffffff;
  background:var(--primary);
  box-shadow:0 0 0 2px rgba(91,53,230,.35), 0 2px 7px rgba(16,24,40,.18);
  pointer-events:auto;
  cursor:pointer;
}
.range-filter-control input[type="range"]::-moz-range-thumb {
  width:14px;
  height:14px;
  border-radius:999px;
  border:4px solid #ffffff;
  background:var(--primary);
  box-shadow:0 0 0 2px rgba(91,53,230,.35), 0 2px 7px rgba(16,24,40,.18);
  pointer-events:auto;
  cursor:pointer;
}
.candidate-range-filter {
  margin-top:14px;
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  background:#fff;
  padding:14px;
}
.candidate-range-filter-head {
  display:flex;
  justify-content:space-between;
  gap:12px;
  align-items:flex-start;
  margin-bottom:12px;
}
.candidate-range-filter-head h3 {
  margin:0 0 4px;
  font-size:14px;
}
.candidate-range-filter-head p,
.candidate-filter-empty {
  margin:0;
  color:#667085;
  font-size:12.5px;
  line-height:1.45;
}
.candidate-filter-count {
  white-space:nowrap;
  border:1px solid #d6bbfb;
  border-radius:999px;
  background:#f4ebff;
  color:#6941c6;
  padding:5px 9px;
  font-size:11.5px;
  font-weight:820;
}
.candidate-filter-results {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:10px;
}
.candidate-filter-result {
  display:grid;
  gap:8px;
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  background:#fbfcff;
  padding:11px;
}
.candidate-filter-result[hidden] {
  display:none;
}
.candidate-filter-result > div:first-child {
  display:flex;
  justify-content:space-between;
  gap:10px;
  align-items:center;
}
.candidate-filter-result strong {
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:12.5px;
  overflow:hidden;
  text-overflow:ellipsis;
}
.candidate-filter-result > div:first-child span {
  color:#667085;
  font-size:11px;
  font-weight:760;
}
.candidate-filter-result-metrics {
  display:flex;
  flex-wrap:wrap;
  gap:5px;
}
.candidate-range-filter .candidate-filter-result-metrics > span {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  flex:0 0 auto;
  width:auto;
  min-width:0;
  height:24px;
  min-height:24px;
  max-height:24px;
  border:1px solid #e4defb;
  border-radius:999px;
  background:#fbf9ff;
  color:#5b35e6;
  padding:0 8px;
  font-size:10.5px;
  line-height:1;
  font-weight:760;
  white-space:nowrap;
  box-sizing:border-box;
  align-self:flex-start;
}
.candidate-range-filter .candidate-filter-result-metrics > .candidate-filter-primary-metric {
  border-color:#7c5cff;
  background:#f0ebff;
  color:#3b237e;
  font-weight:900;
}
.saas-count-card {
  display:flex;
  flex-direction:column;
}
.saas-count-row {
  align-items:center;
  gap:10px;
}
.saas-count-row span {
  min-width:92px;
  color:#475467;
  font-weight:650;
}
.saas-count-row em {
  flex:1;
  height:7px;
  overflow:hidden;
  border-radius:999px;
  background:#eef2ff;
}
.saas-count-row i {
  display:block;
  height:100%;
  border-radius:inherit;
  background:linear-gradient(90deg,#7c5cff,#2fb7ad);
}
.saas-count-row b {
  min-width:18px;
  text-align:right;
  color:#1f2937;
  font-weight:850;
}

.evidence-snapshot-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr));
  gap:14px;
}
.evidence-snapshot-card {
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
  box-shadow:var(--shadow-xs);
  padding:15px;
}
.evidence-snapshot-head,
.evidence-snapshot-foot {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
}
.evidence-snapshot-head h3 {
  margin:3px 0 0;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:15px;
  color:#111827;
}
.evidence-snapshot-body {
  display:grid;
  grid-template-columns:96px minmax(0,1fr);
  gap:16px;
  align-items:center;
  margin:14px 0;
}
.snapshot-score {
  width:92px;
  height:92px;
  display:grid;
  grid-template-rows:auto auto;
  place-content:center;
  justify-items:center;
  text-align:center;
  border-radius:50%;
  background:
    radial-gradient(circle at center,#ffffff 0 54%,transparent 55%),
    conic-gradient(var(--primary) calc(var(--score) * 1%), #eef2ff 0);
  box-shadow:inset 0 0 0 1px #e8e4ff;
}
.snapshot-score strong {
  color:#241653;
  font-size:19px;
  line-height:1;
  font-weight:880;
  display:block;
  width:82px;
  max-width:calc(100% - 10px);
  text-align:center;
  white-space:nowrap;
  letter-spacing:0;
  font-variant-numeric:tabular-nums;
}
.snapshot-score.is-compact strong {
  font-size:16px;
}
.snapshot-score span {
  margin-top:4px;
  color:#706898;
  font-size:10px;
  font-weight:760;
  display:block;
  max-width:78px;
  text-align:center;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.snapshot-bars {
  display:flex;
  flex-direction:column;
  gap:7px;
}
.snapshot-bar {
  display:grid;
  grid-template-columns:72px minmax(56px,1fr) minmax(42px,auto);
  gap:8px;
  align-items:center;
}
.snapshot-bar span,
.snapshot-bar b {
  font-size:11px;
  font-weight:780;
  color:#3b3268;
}
.snapshot-bar b {
  text-align:right;
  font-variant-numeric:tabular-nums;
}
.snapshot-bar em {
  height:7px;
  overflow:hidden;
  border-radius:999px;
  background:#eef2ff;
}
.snapshot-bar i {
  display:block;
  height:100%;
  border-radius:inherit;
  background:linear-gradient(90deg,#7c5cff,#2fb7ad);
}
.snapshot-bar.is-missing span,
.snapshot-bar.is-missing b {
  color:#98a2b3;
}
.snapshot-bar.is-missing em {
  background:#f4f6fb;
}
.selection-summary-card {
  position:relative;
  margin:12px 0 14px;
  padding:16px 16px 15px;
  border:1px solid #d9def8;
  border-radius:14px;
  background:
    linear-gradient(180deg,rgba(255,255,255,.96) 0%,rgba(247,249,255,.98) 100%);
  box-shadow:0 12px 28px rgba(76,93,174,.10);
  overflow:hidden;
}
.selection-summary-card::before {
  content:"";
  position:absolute;
  inset:0 auto 0 0;
  width:4px;
  background:linear-gradient(180deg,#7c5cff,#2fb7ad);
}
.selection-summary-kicker {
  margin:0 0 10px;
  color:#6b54d6;
  font-size:10px;
  font-weight:850;
  letter-spacing:.08em;
  text-transform:uppercase;
}
.selection-summary-card h4 {
  margin:0;
  color:#172033;
  font-size:14.5px;
  line-height:1.38;
  font-weight:850;
}
.selection-summary-card p {
  margin:9px 0 0;
  color:#384152;
  font-size:12.8px;
  line-height:1.55;
}
.selection-summary-card ul {
  margin:11px 0 0;
  padding:0;
  display:flex;
  flex-direction:column;
  gap:6px;
  list-style:none;
  color:#475467;
  font-size:12.2px;
  line-height:1.45;
}
.selection-summary-card li {
  position:relative;
  padding-left:16px;
}
.selection-summary-card li::before {
  content:"";
  position:absolute;
  left:0;
  top:.58em;
  width:6px;
  height:6px;
  border-radius:999px;
  background:#2fb7ad;
}
.selection-summary-card li + li {
  margin-top:0;
}
.evidence-snapshot-foot {
  padding-top:10px;
  border-top:1px solid var(--border);
  color:#667085;
  font-size:11.5px;
  font-weight:700;
}
.evidence-snapshot-foot a {
  color:var(--primary-700);
  text-decoration:none;
  font-weight:820;
}
.top-candidate-msa-block {
  margin:0 0 18px;
  padding:14px;
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  background:#fbfcff;
}
.top-candidate-msa-block h3 {
  margin:0 0 10px;
  color:#172033;
  font-size:15px;
  line-height:1.25;
}
.evidence-snapshot-msa {
  margin:0 0 10px;
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  background:#fff;
  overflow:hidden;
}
.evidence-snapshot-msa > summary {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding:8px 10px;
  cursor:pointer;
  list-style:none;
  color:#3b3268;
  font-size:12px;
  font-weight:820;
}
.evidence-snapshot-msa > summary::-webkit-details-marker {
  display:none;
}
.evidence-snapshot-msa > summary em {
  color:#667085;
  font-size:11px;
  font-style:normal;
  font-weight:700;
}
.evidence-snapshot-msa .msa-section {
  margin:0;
  border-top:1px solid var(--border);
  padding:10px;
}
.evidence-snapshot-msa .msa-grid-wrap {
  max-height:190px;
}

.workflow-audit-section {
  background:#ffffff;
}
.workflow-audit-header {
  display:flex;
  align-items:flex-start;
  justify-content:space-between;
  gap:18px;
  margin-bottom:16px;
}
.workflow-audit-meta {
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
  justify-content:flex-end;
  color:#667085;
  font-size:11.5px;
  font-weight:720;
}
.workflow-audit-meta > span:not(.workflow-status-pill) {
  padding:6px 9px;
  border:1px solid var(--border);
  border-radius:999px;
  background:#fbfcff;
}
.workflow-status-pill,
.audit-stage-status {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  gap:5px;
  min-height:24px;
  padding:4px 10px;
  border-radius:999px;
  border:1px solid #bbf7d0;
  background:#dcfae6;
  color:#067647;
  font-size:11px;
  font-weight:820;
  line-height:1;
}
.workflow-status-pill::before,
.audit-stage-status::before {
  content:"";
  width:6px;
  height:6px;
  border-radius:999px;
  background:#12b76a;
}
.workflow-status-pill.is-recorded {
  border-color:#d6bbfb;
  background:#f4ebff;
  color:#6941c6;
}
.workflow-status-pill.is-recorded::before {
  background:#7f56d9;
}
.workflow-graph-panel {
  padding:18px;
  border:1px solid #e4e7ec;
  border-radius:18px;
  background:
    linear-gradient(180deg,rgba(255,255,255,.94),rgba(248,250,252,.94)),
    linear-gradient(90deg,rgba(99,102,241,.055) 1px,transparent 1px),
    linear-gradient(180deg,rgba(99,102,241,.045) 1px,transparent 1px);
  background-size:auto,28px 28px,28px 28px;
  box-shadow:var(--shadow-xs);
}
.audit-stage-strip {
  display:flex;
  align-items:stretch;
  justify-content:flex-start;
  gap:18px;
  list-style:none;
  margin:0;
  padding:0 2px 12px;
  overflow-x:auto;
  scrollbar-width:thin;
  scrollbar-color:#cfd5e8 transparent;
}
.audit-stage-strip::before,
.audit-stage-strip::after {
  content:"";
  flex:1 0 0;
  min-width:0;
}
.audit-stage-strip > .audit-stage {
  flex:0 0 145px;
}
.audit-stage-strip::-webkit-scrollbar {
  height:8px;
}
.audit-stage-strip::-webkit-scrollbar-thumb {
  background:#cfd5e8;
  border-radius:999px;
}
.audit-stage {
  position:relative;
  flex:0 0 145px;
  min-height:150px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:10px;
  padding:16px 14px;
  border:1px solid #dfe4ee;
  border-radius:14px;
  background:rgba(255,255,255,.94);
  box-shadow:0 10px 28px rgba(16,24,40,.045);
}
.audit-stage:not(:last-child)::after {
  content:"";
  position:absolute;
  top:50%;
  right:-19px;
  width:19px;
  height:1.5px;
  background:#aeb8cf;
  z-index:1;
}
.audit-stage:not(:last-child)::before {
  content:"";
  position:absolute;
  top:calc(50% - 4px);
  right:-19px;
  width:8px;
  height:8px;
  border-top:1.5px solid #aeb8cf;
  border-right:1.5px solid #aeb8cf;
  transform:rotate(45deg);
  z-index:2;
}
.audit-stage-band {
  position:relative;
  flex:0 0 auto;
  min-width:180px;
  padding:14px;
  border:1px solid #e1e6f0;
  border-radius:18px;
  background:linear-gradient(180deg,rgba(255,255,255,.92),rgba(248,250,252,.92));
  box-shadow:0 12px 30px rgba(16,24,40,.04);
}
.audit-stage-band:not(:last-child)::after {
  content:"";
  position:absolute;
  top:50%;
  right:-19px;
  width:19px;
  height:1.5px;
  background:#aeb8cf;
  z-index:1;
}
.audit-stage-band:not(:last-child)::before {
  content:"";
  position:absolute;
  top:calc(50% - 4px);
  right:-19px;
  width:8px;
  height:8px;
  border-top:1.5px solid #aeb8cf;
  border-right:1.5px solid #aeb8cf;
  transform:rotate(45deg);
  z-index:2;
}
.audit-stage-head {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  margin-bottom:12px;
}
.audit-stage-head span {
  color:#3b3268;
  font-size:12px;
  font-weight:880;
  letter-spacing:.05em;
  text-transform:uppercase;
}
.audit-stage-head em {
  padding:4px 8px;
  border:1px solid #e4e7ec;
  border-radius:999px;
  background:#ffffff;
  color:#667085;
  font-size:10.5px;
  font-style:normal;
  font-weight:760;
}
.audit-stage-group {
  display:flex;
  align-items:stretch;
  gap:18px;
  list-style:none;
  margin:0;
  padding:0;
}
.audit-stage-group .audit-stage {
  min-height:138px;
}
.audit-stage-group .audit-stage-icon {
  width:48px;
  height:48px;
}
.audit-stage-group .audit-stage-icon svg {
  width:22px;
  height:22px;
}
.audit-stage-number {
  position:absolute;
  top:12px;
  left:12px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:22px;
  height:22px;
  border-radius:999px;
  background:#f2f4f7;
  color:#667085;
  font-size:11px;
  font-weight:840;
}
.audit-stage-icon {
  width:54px;
  height:54px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border-radius:999px;
  color:#5b35e6;
  background:#eee9ff;
  box-shadow:inset 0 0 0 1px rgba(91,53,230,.06);
}
.audit-stage-icon svg {
  width:25px;
  height:25px;
  fill:none;
  stroke:currentColor;
  stroke-width:2.15;
  stroke-linecap:round;
  stroke-linejoin:round;
}
.audit-stage[data-stage="preprocessing"] .audit-stage-icon {
  color:#1570ef;
  background:#eaf3ff;
}
.audit-stage[data-stage="engineering"] .audit-stage-icon {
  color:#0e7490;
  background:#e6f7fb;
}
.audit-stage[data-stage="backbone_generation"] .audit-stage-icon {
  color:#1570ef;
  background:#eaf3ff;
}
.audit-stage[data-stage="binder_generation"] .audit-stage-icon {
  color:#7f56d9;
  background:#f4ebff;
}
.audit-stage[data-stage="generation"] .audit-stage-icon {
  color:#7f56d9;
  background:#f4ebff;
}
.audit-stage[data-stage="model_call"] .audit-stage-icon {
  color:#7f56d9;
  background:#f4ebff;
}
.audit-stage[data-stage="structure_comparison"] .audit-stage-icon {
  color:#1570ef;
  background:#eaf3ff;
}
.audit-stage[data-stage="energy"] .audit-stage-icon {
  color:#b54708;
  background:#fff4df;
}
.audit-stage[data-stage="validation"] .audit-stage-icon,
.audit-stage[data-stage="completed"] .audit-stage-icon {
  color:#039855;
  background:#e8f8ef;
}
.audit-stage[data-stage="report_generation"] .audit-stage-icon {
  color:#dc6803;
  background:#fff4df;
}
.audit-stage strong {
  color:#111827;
  font-size:14px;
  line-height:1.2;
  text-align:center;
  font-weight:820;
}
.audit-stage-status {
  min-height:22px;
  padding:4px 9px;
  font-size:10.8px;
}
.audit-stage.has-failures .audit-stage-status {
  border-color:#fecaca;
  background:#fef2f2;
  color:#b42318;
}
.audit-stage.has-failures .audit-stage-status::before {
  background:#f04438;
}
.audit-tool-row {
  display:flex;
  align-items:center;
  justify-content:center;
  flex-wrap:wrap;
  gap:5px;
  min-height:24px;
  margin-top:2px;
}
.audit-tool-chip {
  display:inline-flex;
  align-items:center;
  max-width:116px;
  min-height:22px;
  padding:3px 8px;
  border:1px solid #e4defb;
  border-radius:999px;
  background:#fbf9ff;
  color:#5b35e6;
  font-size:10.6px;
  font-weight:800;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.audit-tool-chip.is-muted {
  border-color:#e4e7ec;
  background:#f8fafc;
  color:#98a2b3;
}
.audit-stage.is-skipped {
  opacity:.58;
  border-style:dashed;
  background:rgba(255,255,255,.72);
}
.audit-stage.is-skipped .audit-stage-icon {
  color:#98a2b3;
  background:#f2f4f7;
  box-shadow:none;
}
.workflow-graph-legend {
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:12px;
  padding:10px 2px 0;
  border-top:1px solid rgba(228,231,236,.72);
  color:#667085;
  font-size:11.5px;
  font-weight:720;
}
.workflow-tool-call-list {
  margin:12px 0;
  border:1px solid rgba(228,231,236,.9);
  border-radius:12px;
  background:rgba(255,255,255,.78);
  overflow:hidden;
}
.workflow-tool-call-list > summary {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  padding:10px 12px;
  cursor:pointer;
  list-style:none;
}
.workflow-tool-call-list > summary::-webkit-details-marker {
  display:none;
}
.workflow-tool-call-list > summary strong {
  font-size:12.5px;
  color:#111827;
}
.workflow-tool-call-list > summary em {
  color:#667085;
  font-size:11.5px;
  font-style:normal;
}
.workflow-tool-call-list ol {
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:10px;
  margin:0;
  padding:0 12px 12px;
  list-style:none;
  max-height:clamp(240px, 34vh, 520px);
  overflow-y:auto;
  overscroll-behavior:contain;
}
.workflow-tool-call-list li {
  display:grid;
  grid-template-columns:24px minmax(0,.8fr) minmax(0,1fr);
  gap:7px;
  align-items:center;
  border:1px solid #edf0f5;
  border-radius:8px;
  background:#fff;
  padding:8px;
}
.workflow-tool-call-list li.has-failure {
  border-color:#fecaca;
  background:#fff7f7;
  box-shadow:0 0 0 1px rgba(220,38,38,.06) inset;
}
.workflow-tool-call-list li.tool-call-group {
  grid-template-columns:24px minmax(0,1fr) auto;
  align-items:start;
  gap:8px 10px;
  min-height:112px;
  align-content:start;
}
.tool-call-group-main {
  display:grid;
  grid-template-columns:minmax(0,1fr) auto;
  align-items:start;
  gap:5px 8px;
  min-width:0;
}
.workflow-tool-call-list li > span {
  display:inline-flex;
  justify-content:center;
  color:#667085;
  font-size:11px;
  font-weight:820;
}
.workflow-tool-call-list li strong {
  font-size:12px;
  color:#2f1c72;
  min-width:0;
  overflow:visible;
  text-overflow:clip;
  white-space:normal;
  overflow-wrap:anywhere;
  line-height:1.25;
}
.workflow-tool-call-list li code {
  color:#667085;
  font-size:11px;
  overflow:hidden;
  text-overflow:ellipsis;
}
.workflow-tool-call-list li em {
  grid-column:2 / -1;
  color:#667085;
  font-size:10.5px;
  font-style:normal;
}
.workflow-tool-call-list li em.state-failed {
  color:#b42318;
  font-weight:820;
}
.tool-call-count {
  align-self:start;
  border:1px solid #d0d5dd;
  border-radius:999px;
  background:#f8fafc;
  color:#344054 !important;
  padding:3px 8px;
  white-space:nowrap;
}
.tool-call-outcome {
  grid-column:2 / -1;
  margin:0;
  color:#475467;
  font-size:12px;
  line-height:1.45;
  min-width:0;
  overflow-wrap:anywhere;
}
.tool-call-error {
  grid-column:1 / -1;
  display:block;
  margin-top:2px;
  padding:6px 8px;
  border-radius:7px;
  background:#fee4e2;
  color:#912018;
  font-size:10.8px;
  line-height:1.35;
  overflow-wrap:anywhere;
}
.workflow-graph-legend span {
  display:inline-flex;
  align-items:center;
  gap:6px;
}
.legend-dot {
  width:8px;
  height:8px;
  border-radius:999px;
  background:#12b76a;
}
.legend-dot.is-skipped {
  background:#cfd5e8;
}
.legend-chip {
  width:22px;
  height:12px;
  border:1px solid #e4defb;
  border-radius:999px;
  background:#fbf9ff;
}
@media (max-width: 900px) {
  .workflow-audit-header {
    flex-direction:column;
    align-items:stretch;
  }
  .workflow-audit-meta {
    justify-content:flex-start;
  }
  .audit-stage-strip {
    padding-bottom:14px;
  }
  .audit-stage-group .audit-stage,
  .audit-stage-strip > .audit-stage {
    flex-basis:150px;
  }
}
.workflow-audit-details > summary {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:14px;
  cursor:pointer;
  list-style:none;
}
.workflow-audit-details > summary::-webkit-details-marker {
  display:none;
}
.workflow-audit-details > summary strong {
  display:block;
  color:#111827;
  font-size:20px;
  line-height:1.2;
}
.workflow-audit-details > summary em {
  color:#706898;
  font-size:12px;
  font-style:normal;
  font-weight:760;
}
.workflow-audit-details > summary::after {
  content:"+";
  width:26px;
  height:26px;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border:1px solid var(--border);
  border-radius:8px;
  color:var(--primary);
  background:#fff;
  font-weight:850;
}
.workflow-audit-details[open] > summary::after {
  content:"-";
}
.workflow-audit-details > .workflow {
  margin-top:16px;
}
.workflow-audit-details .dag-wrap {
  margin-top:16px;
  overflow-x:auto;
  padding:10px 0 4px;
}
.workflow-audit-details .dag {
  max-width:none;
}
.workflow-node-dag-details {
  margin-top:14px;
  border:1px dashed var(--border);
  border-radius:var(--radius-lg);
  background:#fff;
}
.workflow-node-dag-details > summary {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  padding:12px 14px;
  cursor:pointer;
  list-style:none;
}
.workflow-node-dag-details > summary::-webkit-details-marker {
  display:none;
}
.workflow-node-dag-details > summary span {
  color:#33236f;
  font-weight:850;
}
.workflow-node-dag-details > summary small {
  color:#98a2b3;
  font-weight:700;
}
.workflow-node-dag-details > :not(summary) {
  margin-left:14px;
  margin-right:14px;
}

.saas-trace-stage {
  padding:15px 14px;
}
.saas-trace-dot {
  box-shadow:0 8px 18px rgba(91,53,230,.16);
}
.saas-trace-stage strong {
  color:#1f2937;
  font-weight:800;
}

.workflow-stage-strip {
  --stage-line:#d9ddec;
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(128px,1fr));
  gap:12px;
  list-style:none;
  padding:0;
  margin:0;
  counter-reset:workflow-stage;
}
.workflow-stage {
  position:relative;
  min-height:176px;
  display:flex;
  flex-direction:column;
  gap:10px;
  padding:14px 12px 13px;
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  background:linear-gradient(180deg,#ffffff 0%,#fbfcff 100%);
  box-shadow:var(--shadow-xs);
}
.workflow-stage:not(:last-child)::after {
  content:"";
  position:absolute;
  top:25px;
  right:-12px;
  width:14px;
  height:14px;
  border-top:2px solid var(--stage-line);
  border-right:2px solid var(--stage-line);
  transform:rotate(45deg);
  opacity:.9;
  z-index:1;
}
.workflow-stage:not(:last-child)::before {
  content:"";
  position:absolute;
  top:31px;
  right:-12px;
  width:18px;
  height:2px;
  background:var(--stage-line);
  opacity:.9;
  z-index:1;
}
.workflow-stage-marker {
  display:flex;
  align-items:center;
  gap:8px;
}
.workflow-stage-index,
.workflow-stage-icon {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  border-radius:999px;
  font-weight:850;
}
.workflow-stage-index {
  width:25px;
  height:25px;
  color:#fff;
  background:linear-gradient(135deg,var(--primary),var(--teal));
  font-size:12px;
  box-shadow:0 8px 18px rgba(91,53,230,.16);
}
.workflow-stage-icon {
  min-width:26px;
  min-height:26px;
  max-width:96px;
  padding:5px 7px;
  color:#33236f;
  background:#f3f0ff;
  border:1px solid #ddd6ff;
  font-size:10px;
  line-height:1.1;
  white-space:nowrap;
  overflow-wrap:anywhere;
  text-align:center;
}
.workflow-stage-body {
  display:flex;
  flex-direction:column;
  gap:6px;
}
.workflow-stage-body h3 {
  margin:0;
  color:#111827;
  font-size:13px;
  font-weight:850;
}
.workflow-stage-body p {
  margin:0;
  color:#475467;
  font-size:11.5px;
  line-height:1.38;
}
.workflow-stage-body small {
  color:#706898;
  font-size:10.5px;
  line-height:1.35;
}
.workflow-stage-body small strong {
  color:#3b3268;
}
.workflow-stage.is-recorded .workflow-stage-index {
  background:linear-gradient(135deg,#6f4cff,#8b5cf6);
}
.workflow-stage.is-complete .workflow-stage-index {
  background:linear-gradient(135deg,var(--primary),var(--teal));
}
.workflow-stage.has-failures {
  border-color:#fecaca;
  background:linear-gradient(180deg,#ffffff 0%,#fff7f7 100%);
}
.workflow-stage.has-failures .workflow-stage-index {
  background:linear-gradient(135deg,#dc2626,#f97316);
}
.workflow-stage.is-skipped {
  background:#fbfcff;
  border-style:dashed;
  opacity:.68;
}
.workflow-stage.is-skipped .workflow-stage-index,
.workflow-stage.is-skipped .workflow-stage-icon {
  background:#f2f4f7;
  color:#98a2b3;
  border-color:#e4e7ec;
  box-shadow:none;
}
.workflow-stage.is-skipped::after {
  opacity:.42;
}
.workflow-stage.is-skipped::before {
  opacity:.42;
}

@media (min-width: 1180px) {
  .workflow-stage-strip {
    grid-template-columns:repeat(7,minmax(0,1fr));
  }
  .workflow-stage-count-4 {
    grid-template-columns:repeat(4,minmax(0,1fr));
  }
}

.comparison-assets {
  align-items:flex-start;
}
.comparison-assets a.asset-pill,
.asset-download-menu summary.asset-pill {
  text-decoration:none;
  cursor:pointer;
}
.comparison-assets a.asset-pill:hover,
.asset-download-menu summary.asset-pill:hover {
  border-color:var(--primary);
  background:#f7f4ff;
  color:var(--primary-700);
}
.asset-download-link {
  flex-direction:row;
  gap:4px;
}
.asset-download-link .artifact-name {
  font-weight:800;
}
.asset-download-link .artifact-meta {
  color:inherit;
  font-size:10.5px;
  opacity:.72;
}
.asset-download-menu {
  position:relative;
}
.asset-download-menu > summary {
  list-style:none;
}
.asset-download-menu > summary::-webkit-details-marker {
  display:none;
}
.asset-download-menu-body {
  position:absolute;
  right:0;
  z-index:4;
  min-width:170px;
  display:flex;
  flex-direction:column;
  gap:6px;
  margin-top:6px;
  padding:8px;
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  background:#fff;
  box-shadow:var(--shadow-md);
}

#reproducibility .card,
#reproducibility .saas-card,
.repro {
  border-color:var(--border);
  background:var(--surface);
}
.repro-panel {
  display:grid;
  grid-template-columns:minmax(0,1.65fr) minmax(260px,.75fr);
  gap:16px;
  align-items:start;
}
.repro-copy-block {
  min-width:0;
}
.repro-command {
  position:relative;
  padding:16px 88px 16px 18px;
  border-radius:var(--radius-md);
  background:#18114a;
  border:1px solid #2a2069;
  box-shadow:var(--shadow-sm);
}
.repro-command pre {
  margin:0;
  padding:0;
  border:none;
  box-shadow:none;
  background:transparent;
  white-space:pre-wrap;
  overflow-wrap:anywhere;
}
.repro-command .copy-btn {
  position:absolute;
  top:12px;
  right:12px;
}
.repro-support-grid {
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:10px;
  align-self:end;
}
.repro-support-card {
  text-decoration:none;
  border:1px solid var(--border);
  border-radius:var(--radius-md);
  background:#fbfcff;
  padding:12px;
}
.repro-support-card span {
  display:block;
  color:#667085;
  font-size:11px;
  font-weight:780;
}
.repro-support-card strong {
  display:block;
  margin-top:7px;
  color:#33236f;
  font-size:14px;
  line-height:1;
}
.repro-support-card:hover {
  border-color:#d9d2ff;
  background:#f7f4ff;
}
.repro-id-line {
  grid-column:1 / -1;
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  align-items:center;
  margin:0;
  color:#667085;
  font-size:12px;
}
.repro-id-line code {
  border:1px solid var(--border);
  border-radius:var(--radius-sm);
  background:#f7f4ff;
  color:#33236f;
  padding:3px 7px;
}
.appendix-panel {
  background:#fbfcff;
  border-color:#dbe0ed;
  opacity:.86;
}
.appendix-panel:hover,
.appendix-panel[open] {
  opacity:1;
}
.appendix-panel > summary {
  color:#3b3268;
}
.appendix-panel > summary small {
  color:var(--text-muted);
}
pre,
.tool-evidence-json {
  background:#0f172a;
  color:#e5e7eb;
  border:1px solid #1e293b;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04);
}

@media (max-width: 1320px) {
  .content { padding:22px 22px 54px; }
  .saas-overview-grid { grid-template-columns:1fr 1fr; }
  .saas-stats { grid-column:1 / -1; }
  .candidate-review-board-layout { grid-template-columns:1fr; }
  .console-detail { position:static; }
}
@media (max-width: 980px) {
  body { display:block; }
  .sidebar-toggle { display:none; }
  body.sidebar-collapsed .brand-text { display:flex; }
  body.sidebar-collapsed .sidebar-meta { display:flex; }
  body.sidebar-collapsed .side-nav-label,
  body.sidebar-collapsed .side-nav-divider { display:inline; }
  body.sidebar-collapsed .side-nav-item { justify-content:flex-start; padding:0 11px; }
  .sidebar {
    position:relative;
    width:auto;
    min-height:auto;
    border-right:none;
    border-bottom:1px solid var(--border);
  }
  .side-nav {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
  }
  .top-bar {
    position:relative;
    left:auto;
    height:auto;
    flex-wrap:wrap;
    gap:10px;
    padding:12px 18px;
  }
  .main {
    margin-left:0;
  }
  .content { padding:20px 16px 48px; }
  .saas-rationale-grid,
  .saas-pattern-grid {
    grid-template-columns:1fr 1fr;
  }
  .repro-panel {
    grid-template-columns:1fr;
  }
  .console-detail { position:static; }
}
@media (max-width: 720px) {
  .saas-overview-grid,
  .saas-rationale-grid,
  .saas-pattern-grid,
  .saas-trace {
    grid-template-columns:1fr;
  }
  .saas-stats { grid-template-columns:1fr 1fr; }
  .saas-stat { border-left:none; border-top:1px solid var(--border); }
  .saas-stat:nth-child(-n+2) { border-top:none; }
  .repro-support-grid { grid-template-columns:1fr 1fr; }
  .dossier-structure-primary .structure-viewer {
    min-height:360px;
    height:420px;
  }
}
"""


# ----------------------------------------------------------------- JS


def _js() -> str:
    # Sentinel ``VPA_OFFLINE_VIEWER`` marks this build as fully self-contained:
    # no CDN, no LLM, no network requests. Test contracts assert its presence
    # and the absence of external viewer CDN imports. The report's narrative and visuals
    # are produced deterministically from JobStore data — re-running
    # ``vpa report <id>`` yields byte-equivalent output for the same inputs.
    return (
        """
// VPA_OFFLINE_VIEWER · deterministic offline 3D viewer · no external CDN, no LLM calls
(function () {
  // ----- Sidebar collapse -----
  const sidebarToggle = document.querySelector('[data-sidebar-toggle]');
  if (sidebarToggle) {
    const storageKey = 'vpa-report-sidebar-collapsed';
    const applySidebarState = (collapsed) => {
      document.body.classList.toggle('sidebar-collapsed', collapsed);
      sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
      sidebarToggle.setAttribute('aria-label', collapsed ? 'Expand sidebar' : 'Collapse sidebar');
    };
    try {
      applySidebarState(window.localStorage.getItem(storageKey) === '1');
    } catch (e) {
      applySidebarState(false);
    }
    sidebarToggle.addEventListener('click', () => {
      const collapsed = !document.body.classList.contains('sidebar-collapsed');
      applySidebarState(collapsed);
      try { window.localStorage.setItem(storageKey, collapsed ? '1' : '0'); } catch (e) {}
    });
  }

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

  // ----- Top-bar report actions -----
  function reportFileStem() {
    const title = (document.title || 'vpa-report').trim() || 'vpa-report';
    return title.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'vpa-report';
  }
  document.querySelectorAll('[data-export-report]').forEach((btn) => {
    btn.addEventListener('click', () => {
      try {
        const html = '<!doctype html>\\n' + document.documentElement.outerHTML;
        const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = reportFileStem() + '.html';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1500);
      } catch (e) {
        console.warn('Report export failed:', e);
        window.print();
      }
    });
  });
  // ----- Reproducibility support cards -----
  document.querySelectorAll('.repro-support-card').forEach((card) => {
    card.addEventListener('click', (event) => {
      const href = card.getAttribute('href') || '';
      if (!href.startsWith('#')) return;
      const target = document.querySelector(href);
      if (!target) return;
      event.preventDefault();
      if (target.tagName && target.tagName.toLowerCase() === 'details') {
        target.open = true;
      }
      if (href === '#appendix') {
        const firstPanel = document.querySelector('.appendix-panel');
        if (firstPanel) firstPanel.open = true;
      }
      const nested = target.querySelector && target.querySelector('details');
      if (nested) nested.open = true;
      target.scrollIntoView({ block: 'start', behavior: 'smooth' });
      history.replaceState(null, '', href);
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

  // ----- Cross-candidate metric range filters -----
  document.querySelectorAll('[data-candidate-range-filter]').forEach((panel) => {
    const section = panel.closest('section') || document;
    const cards = Array.from(section.querySelectorAll('[data-filter-card]'));
    const results = Array.from(panel.querySelectorAll('[data-filter-result]'));
    const countEl = panel.querySelector('[data-filter-count]');
    const emptyEl = panel.querySelector('[data-filter-empty]');

    function num(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    function nearlyEqual(a, b) {
      return Math.abs(a - b) < 1e-9;
    }
    function fmtRangeValue(value) {
      if (!Number.isFinite(value)) return '';
      if (Math.abs(value) >= 10) return value.toFixed(1).replace(/\\.0$/,'');
      return value.toFixed(3).replace(/\\.?0+$/,'');
    }
    function updateCard(card) {
      const minInput = card.querySelector('.metric-filter-min');
      const maxInput = card.querySelector('.metric-filter-max');
      if (!minInput || !maxInput) return null;
      let lo = num(minInput.value);
      let hi = num(maxInput.value);
      const absMin = num(minInput.min);
      const absMax = num(maxInput.max);
      if (lo === null || hi === null || absMin === null || absMax === null) return null;
      if (lo > hi) {
        if (document.activeElement === minInput) {
          hi = lo;
          maxInput.value = String(hi);
        } else {
          lo = hi;
          minInput.value = String(lo);
        }
      }
      const span = Math.max(absMax - absMin, 1e-9);
      const left = Math.max(0, Math.min(100, (lo - absMin) / span * 100));
      const right = Math.max(0, Math.min(100, (absMax - hi) / span * 100));
      const windowEl = card.querySelector('.range-filter-window');
      if (windowEl) {
        windowEl.style.left = left + '%';
        windowEl.style.right = right + '%';
      }
      const label = card.querySelector('.range-filter-label');
      if (label) {
        label.innerHTML = '<b>' + fmtRangeValue(lo) + '</b> to <b>'
                        + fmtRangeValue(hi) + '</b>';
      }
      const defaultMin = num(card.dataset.resolvedDefaultMin || card.dataset.defaultMin);
      const defaultMax = num(card.dataset.resolvedDefaultMax || card.dataset.defaultMax);
      const wasTouched = card.dataset.filterTouched === 'true';
      const active = wasTouched && defaultMin !== null && defaultMax !== null
        && (!nearlyEqual(lo, defaultMin) || !nearlyEqual(hi, defaultMax));
      return { metric: card.dataset.metric || '', lo, hi, active };
    }
    function readFilters() {
      return cards.map(updateCard).filter(Boolean).filter((f) => f.active);
    }
    function metricsFor(result) {
      try {
        const parsed = JSON.parse(result.dataset.metrics || '{}');
        return parsed && typeof parsed === 'object' ? parsed : {};
      } catch (e) {
        return {};
      }
    }
    function applyFilters() {
      const active = readFilters();
      let shown = 0;
      for (const result of results) {
        const metrics = metricsFor(result);
        const keep = active.every((filter) => {
          const value = num(metrics[filter.metric]);
          return value !== null && value >= filter.lo && value <= filter.hi;
        });
        result.hidden = !keep;
        if (keep) shown += 1;
      }
      if (countEl) {
        countEl.textContent = shown + ' matching structure' + (shown === 1 ? '' : 's');
      }
      if (emptyEl) {
        emptyEl.hidden = shown !== 0;
      }
    }
    for (const card of cards) {
      const inputs = Array.from(card.querySelectorAll('input[type="range"]'));
      const minInput = card.querySelector('.metric-filter-min');
      const maxInput = card.querySelector('.metric-filter-max');
      if (minInput && maxInput) {
        card.dataset.resolvedDefaultMin = minInput.value;
        card.dataset.resolvedDefaultMax = maxInput.value;
        card.dataset.filterTouched = 'false';
      }
      inputs.forEach((input) => {
        input.addEventListener('input', () => {
          card.dataset.filterTouched = 'true';
          applyFilters();
        });
      });
      updateCard(card);
    }
    applyFilters();
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

  // Expose an id→payload map so the candidate console can paint a viewer the
  // moment its detail template is cloned into the live DOM. Every viewer lives
  // inside an inert <template> until its board card is selected, so eager
  // getElementById can't reach them — the console drives initialisation lazily.
  const payloadMap = {};
  for (const payload of structurePayloads) {
    const ref = payload.content_ref ? payloadMap[payload.content_ref] : null;
    payloadMap[payload.id] = ref
      ? Object.assign({}, ref, payload, {
          content: ref.content,
          format: ref.format,
          has_sidechains: ref.has_sidechains,
          has_plddt: ref.has_plddt
        })
      : payload;
  }
  window.__vpaPayloadMap = payloadMap;

  if (typeof $3Dmol === 'undefined') {
    // Vendor not present — degrade gracefully per spec §1 guard. Surface the
    // same notice for lazily-activated viewers so the console stays honest.
    window.__vpaSetupViewer = function (payload) {
      const el = payload && document.getElementById(payload.id);
      if (el) {
        el.innerHTML = '<div class="structure-status">3D viewer unavailable (3Dmol.js asset missing)</div>';
        showViewerFailure(el, '3Dmol.js asset missing.');
      }
    };
    for (const payload of structurePayloads) {
      const el = document.getElementById(payload.id);
      if (el) {
        el.innerHTML = '<div class="structure-status">3D viewer unavailable (3Dmol.js asset missing)</div>';
        showViewerFailure(el, '3Dmol.js asset missing.');
      }
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

  // Each viewer is self-contained: its container, its selection map, its style
  // applier. Expose the initialiser so the candidate console controller can
  // paint a viewer lazily when a detail panel is activated, then init only the
  // viewers already present in the live DOM (templates are inert and skipped).
  window.__vpaSetupViewer = setupViewer;
  for (const payload of structurePayloads) {
    const el = document.getElementById(payload.id);
    if (el && !el.closest('[data-structure-compare-grid]')) {
      setupViewer(payload);
    }
  }
  document.querySelectorAll('[data-structure-compare-grid]').forEach((grid) => {
    const viewers = Array.from(grid.querySelectorAll('.structure-viewer'));
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const payload = payloadMap[entry.target.id];
          if (payload) setupViewer(payload);
          observer.unobserve(entry.target);
        });
      }, { root: grid, rootMargin: '160px 0px', threshold: 0.01 });
      viewers.forEach((el) => observer.observe(el));
    } else {
      viewers.slice(0, 6).forEach((el) => {
        const payload = payloadMap[el.id];
        if (payload) setupViewer(payload);
      });
    }
    setupStructureCompareGrid(grid);
  });

  function setupStructureCompareGrid(grid) {
    let dragged = null;
    const panel = grid.closest('.structure-comparison-panel');
    const panelFullscreenBtn = panel ? panel.querySelector('[data-structure-comparison-fullscreen]') : null;
    if (panel && panelFullscreenBtn) {
      panelFullscreenBtn.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        toggleStructureComparisonFullscreen(panel);
      });
      panel.addEventListener('fullscreenchange', () => {
        grid.querySelectorAll('.structure-viewer').forEach((el) => {
          const payload = payloadMap[el.id];
          if (payload) setupViewer(payload);
        });
        window.setTimeout(() => {
          try { window.dispatchEvent(new Event('resize')); } catch (e) {}
        }, 80);
      });
    }
    grid.querySelectorAll('[data-structure-compare-card]').forEach((card) => {
      card.addEventListener('dragstart', () => {
        dragged = card;
        card.classList.add('is-dragging');
      });
      card.addEventListener('dragend', () => {
        card.classList.remove('is-dragging');
        dragged = null;
      });
    });
    grid.addEventListener('dragover', (event) => {
      if (!dragged) return;
      event.preventDefault();
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('[data-structure-compare-card]');
      if (!target || target === dragged || target.parentElement !== grid) return;
      const rect = target.getBoundingClientRect();
      const before = event.clientY < rect.top + rect.height / 2;
      grid.insertBefore(dragged, before ? target : target.nextSibling);
    });
  }

  function toggleStructureComparisonFullscreen(panel) {
    if (document.fullscreenElement === panel) {
      document.exitFullscreen();
    } else if (panel.requestFullscreen) {
      panel.requestFullscreen().catch((e) => console.warn('structure comparison fullscreen denied', e));
    }
  }

  function setupViewer(payload) {
    const container = document.getElementById(payload.id);
    if (!container) return;
    if (container.dataset.viewerModelLoaded === 'true') return;
    container.innerHTML = '<div class="structure-status">Loading structure...</div>';
    const fmtRaw = (payload.format || 'pdb').toLowerCase();
    const fmt = (fmtRaw === 'mmcif' || fmtRaw === 'cif') ? 'cif'
              : (fmtRaw === 'ent' ? 'pdb' : fmtRaw);

    let viewer;
    try {
      container.innerHTML = '';
      viewer = $3Dmol.createViewer(container, {
        backgroundColor: 'white',
        antialias: true,
        cartoonQuality: 8
      });
      viewer.addModel(payload.content, fmt);
    } catch (err) {
      console.error('3Dmol load error', payload.id, err);
      container.innerHTML = '<div class="structure-status">Failed to load structure</div>';
      showViewerFailure(container, '3Dmol could not load the structure model.');
      return;
    }

    // Detect whether B-factor (== pLDDT for AF/Protenix/Boltz outputs) is
    // populated. Some PDBs leave it at 0 — fall back to spectrum colouring.
    // Prefer the authoritative flag computed in Python (scans the raw atom
    // records) so the runtime gate agrees with the disabled pLDDT <option>;
    // fall back to a live 3Dmol scan only when the flag is absent.
    let hasPlddt;
    if (typeof payload.has_plddt === 'boolean') {
      hasPlddt = payload.has_plddt;
    } else {
      hasPlddt = false;
      try {
        const atoms = viewer.selectedAtoms({ atom: 'CA' });
        for (const a of atoms) {
          if (typeof a.b === 'number' && a.b > 0.01) { hasPlddt = true; break; }
        }
      } catch (e) { /* selectedAtoms may not be available in some versions */ }
    }

    // Detect whether the model carries any sidechain atoms at all. Backbone-
    // only outputs (rfdiffusion scaffolds, esm3 backbone predictions, ...)
    // contain just N/CA/C/O per residue, so the "Sidechains" toggle has
    // literally nothing to draw. We disable + explain it rather than letting
    // the user click a dead control. Any non-hydrogen protein atom outside
    // the backbone set proves sidechains exist.
    // Prefer the authoritative flag computed in Python (scans the raw atom
    // records) — the browser-side 3Dmol scan below misses sidechains for some
    // mmCIF inputs. Fall back to the runtime scan only when the flag is absent.
    let hasSidechains;
    if (typeof payload.has_sidechains === 'boolean') {
      hasSidechains = payload.has_sidechains;
    } else {
      hasSidechains = false;
      try {
        const BACKBONE = { N: 1, CA: 1, C: 1, O: 1, OXT: 1 };
        const allAtoms = viewer.selectedAtoms({});
        for (const a of allAtoms) {
          if (a.hetflag) continue;                       // ligands / waters
          const el = (a.elem || '').toUpperCase();
          if (el === 'H' || el === 'D') continue;        // sticks show heavy atoms
          const name = (a.atom || '').toUpperCase().trim();
          if (!BACKBONE[name]) { hasSidechains = true; break; }
        }
      } catch (e) { /* selectedAtoms may not be available in some versions */ }
    }

    // Honour the per-section default colour set by the Python renderer
    // (pLDDT when the artifact carries per-residue confidence, else N→C
    // spectrum — for monomers and complexes alike; Chain stays selectable but
    // is never the default). Falls back to the pLDDT-detection heuristic when
    // no default is set. pLDDT is still gated on real B-factor data so a
    // placeholder zero column stays muted.
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
      structureExtent: null,
    };
    state.structureExtent = structureExtent(viewer);

    applyStructureStyles(state);
    // Click behaviour intentionally minimal — the report's residue
    // inspection happens through Pick (hover) mode, so clicking only
    // rotates / zooms the model rather than dropping label pins.
    zoomViewerToStructure(state, true);
    markViewerHealthy(container);
    installViewerHealthMonitor(container);
    scheduleViewerResize(state, container);

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
      if (!hasSidechains) {
        // Backbone-only model: nothing for the toggle to draw. Disable it and
        // explain why so the dead control doesn't read as a bug.
        sidechainsBox.checked = false;
        sidechainsBox.disabled = true;
        state.sidechains = false;
        const lbl = sidechainsBox.closest('label');
        if (lbl) {
          lbl.classList.add('structure-control-disabled');
          lbl.title = 'Backbone-only structure — no sidechain atoms were '
                    + 'generated for this model, so there is nothing to show.';
        }
      } else {
        sidechainsBox.addEventListener('change', () => {
          state.sidechains = sidechainsBox.checked;
          applyStructureStyles(state);
        });
      }
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
    // For a binder-design complex the FASTA strip is the binder chain only.
    // Empty string => monomer (strip applies to the single chain).
    state.binderChain = section.getAttribute('data-binder-chain') || '';
    if (pickerBtn && infoPanel) {
      pickerBtn.addEventListener('click', () => togglePicker(state, section));
    }

    section.addEventListener('fullscreenchange', () => {
      // Resize after the layout settles so the canvas matches the new container.
      setTimeout(() => {
        try {
          viewer.resize();
          zoomViewerToStructure(state, false);
        } catch (e) {}
      }, 60);
    });
    window.addEventListener('resize', () => {
      try { viewer.resize(); viewer.render(); } catch (e) {}
    });
  }

  function installViewerHealthMonitor(container) {
    const canvas = container.querySelector('canvas');
    if (!canvas) return;
    canvas.addEventListener('webglcontextlost', (event) => {
      try { event.preventDefault(); } catch (e) {}
      // A context-loss event alone is not a user-facing failure. Headless
      // browsers and screenshot capture can emit CONTEXT_LOST_WEBGL after a
      // successful visible render. Keep it as developer telemetry unless it is
      // paired with an actual init/load/render failure elsewhere.
      container.dataset.viewerContextLost = 'true';
      console.warn('VPA structure viewer WebGL context lost', container.id);
    }, false);
    canvas.addEventListener('webglcontextrestored', () => {
      markViewerHealthy(container);
      container.classList.remove('structure-viewer-context-lost');
      console.info('VPA structure viewer WebGL context restored', container.id);
    }, false);
  }

  function markViewerHealthy(container) {
    const section = container.closest('.structure-section');
    const health = section ? section.querySelector('.structure-health') : null;
    container.dataset.viewerModelLoaded = 'true';
    container.dataset.viewerRendered = 'true';
    container.dataset.viewerHealth = 'healthy';
    container.classList.remove('structure-viewer-context-lost');
    if (health) health.hidden = true;
  }

  function showViewerFailure(container, reason) {
    const section = container.closest('.structure-section');
    const health = section ? section.querySelector('.structure-health') : null;
    container.dataset.viewerHealth = 'failed';
    container.dataset.viewerFailureReason = reason || 'viewer failed';
    container.classList.add('structure-viewer-context-lost');
    if (health) {
      const msg = health.querySelector('span');
      if (msg && reason) {
        msg.textContent = reason + ' Reload the page, close other GPU-heavy tabs, or reopen this report in a browser with WebGL enabled. Structure downloads remain available below.';
      }
      health.hidden = false;
    }
  }

  function scheduleViewerResize(state, container, attempt) {
    attempt = attempt || 0;
    const run = () => {
      const rect = container.getBoundingClientRect();
      if ((rect.width < 2 || rect.height < 2) && attempt < 6) {
        scheduleViewerResize(state, container, attempt + 1);
        return;
      }
      try {
        state.viewer.resize();
        zoomViewerToStructure(state, false);
        markViewerHealthy(container);
      } catch (e) {
        console.warn('VPA structure viewer resize failed', container.id, e);
        if (attempt >= 6 && !viewerHasUsableCanvas(container)) {
          showViewerFailure(container, 'The viewer could not size or render the structure canvas.');
        }
      }
      if (attempt < 2) {
        window.setTimeout(() => scheduleViewerResize(state, container, attempt + 1), 120);
      }
    };
    if (window.requestAnimationFrame) {
      window.requestAnimationFrame(run);
    } else {
      window.setTimeout(run, 60);
    }
  }

  function viewerHasUsableCanvas(container) {
    const canvas = container.querySelector('canvas');
    if (!canvas) return false;
    const rect = canvas.getBoundingClientRect();
    return rect.width > 2 && rect.height > 2 && canvas.width > 2 && canvas.height > 2;
  }

  function structureExtent(viewer) {
    // Lightweight bounding box used only to choose a conservative post-fit zoom.
    // It does not alter selections or styles and deliberately ignores waters.
    let atoms = [];
    try {
      atoms = viewer.selectedAtoms({ hetflag: false }) || [];
    } catch (e) {}
    if (!atoms.length) {
      try { atoms = viewer.selectedAtoms({}) || []; } catch (e) { atoms = []; }
    }
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let count = 0;
    for (const atom of atoms) {
      const x = atom.x, y = atom.y, z = atom.z;
      if (typeof x !== 'number' || typeof y !== 'number' || typeof z !== 'number') continue;
      minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
      count += 1;
    }
    if (!count) return null;
    const spans = [maxX - minX, maxY - minY, maxZ - minZ].sort((a, b) => b - a);
    return { atomCount: count, longAxis: spans[0], midAxis: spans[1], shortAxis: spans[2] };
  }

  function structureFillZoom(extent) {
    if (!extent) return 1.12;
    const longAxis = Math.max(extent.longAxis || 0, 0.001);
    const midAxis = Math.max(extent.midAxis || 0, 0.001);
    const shortAxis = Math.max(extent.shortAxis || 0, 0.001);
    const elongation = longAxis / Math.max((midAxis + shortAxis) / 2, 0.001);
    // Small or elongated binders otherwise look sparse after zoomTo(). Keep the
    // multiplier modest so compact proteins and complexes are not clipped.
    if (elongation >= 4.0) return 1.22;
    if (longAxis < 45) return 1.18;
    if (longAxis < 75) return 1.14;
    return 1.08;
  }

  function zoomViewerToStructure(state, animated) {
    const v = state.viewer;
    try { v.zoomTo(); } catch (e) {}
    try { v.zoom(structureFillZoom(state.structureExtent), animated ? 500 : 0); } catch (e) {}
    try { v.render(); } catch (e) {}
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
    // The FASTA strip belongs to the binder chain only on a complex; a
    // residue on the target (or any non-binder chain) has no cell to map to,
    // so we must NOT translate its resi into a binder sequence position.
    const onSeqChain = !state.binderChain
      || (atom.chain || '') === state.binderChain;
    if (seqEl) {
      // Align with the candidate's FASTA — atom.resi is 1-indexed in the
      // PDB. When the design's sequence is shorter than the structure
      // (e.g. an unmodelled tail) we still show the position but fall
      // back to "seq N/A" when out of range.
      const idx = (atom.resi | 0) - 1;
      if (!onSeqChain) {
        seqEl.textContent = 'target chain — no binder seq';
        seqEl.classList.add('sri-placeholder');
      } else if (state.sequence && idx >= 0 && idx < state.sequence.length) {
        const ch = state.sequence.charAt(idx);
        seqEl.textContent = 'seq ' + (idx + 1) + ' (' + ch + ')';
        seqEl.classList.remove('sri-placeholder');
      } else {
        seqEl.textContent = 'seq ' + atom.resi;
        seqEl.classList.remove('sri-placeholder');
      }
    }
    highlightSequenceCell(state, onSeqChain ? atom.resi : null);
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

    // Protein backbone cartoon.
    const cartoonStyle = { cartoon: cartoonVisualSpec(state.colourMode) };
    v.setStyle({ hetflag: false }, cartoonStyle);

    // Hetero atoms (ligands etc.) as ball-and-stick
    v.setStyle({ hetflag: true }, { stick: { radius: 0.22 }, sphere: { scale: 0.32 } });
    // Hide water
    v.setStyle({ resn: 'HOH' }, {});
    v.setStyle({ resn: 'WAT' }, {});

    if (state.sidechains) {
      // Keep CA in the selection so each side chain stick stays bonded to the
      // backbone (excluding CA leaves the sticks floating); exclude only the
      // peptide N/C/O. radius 0.13 was effectively invisible over the cartoon.
      v.addStyle(
        { hetflag: false, not: { atom: ['N', 'C', 'O'] } },
        { stick: { radius: 0.2 } }
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

  function cartoonVisualSpec(mode) {
    // AFDB-like preset: retain all existing colour modes, but make the cartoon
    // ribbon slightly thicker/smoother so small binders read as protein
    // backbones rather than thin wire fragments.
    const colour = cartoonColourSpec(mode);
    return Object.assign({
      thickness: 0.58,
      width: 1.25,
      arrows: true,
    }, colour);
  }

  function toggleFullscreen(section) {
    if (document.fullscreenElement === section) {
      document.exitFullscreen();
    } else if (section.requestFullscreen) {
      section.requestFullscreen().catch((e) => console.warn('fullscreen denied', e));
    }
  }
})();

// ----- Candidate review console (board → detail, lazy <template> panels) -----
// Runs as its own IIFE *after* the viewer IIFE so window.__vpaSetupViewer and
// window.__vpaPayloadMap are already populated. Selecting a board card clones
// the matching detail <template> into the live host, wires its tab strip, and
// paints any structure viewers it carries on demand — keeping the page light
// (one cloned detail at a time) and the 3D viewers genuinely lazy.
(function () {
  const host = document.getElementById('cand-detail-host');
  if (!host) return;
  const cards = Array.from(document.querySelectorAll('.board-card'));
  if (!cards.length) return;

  const templates = {};
  document.querySelectorAll('template.cand-detail-tpl').forEach((tpl) => {
    templates[tpl.dataset.cand] = tpl;
  });

  function wireDetailTabs(root) {
    const tabs = Array.from(root.querySelectorAll('.cand-tab'));
    const panels = Array.from(root.querySelectorAll('.cand-tabpanel'));
    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        tabs.forEach((t) => {
          const on = t === tab;
          t.classList.toggle('is-active', on);
          t.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        panels.forEach((p) => p.classList.toggle('is-active', p.dataset.tab === target));
      });
    });
  }

  function paintViewers(root) {
    const setup = window.__vpaSetupViewer;
    if (typeof setup !== 'function') return;
    const map = window.__vpaPayloadMap || {};
    root.querySelectorAll('.structure-viewer').forEach((el) => {
      const payload = map[el.id];
      if (payload) setup(payload);
    });
  }

  function activate(card) {
    cards.forEach((c) => c.setAttribute('aria-selected', c === card ? 'true' : 'false'));
    // If the selected card lives inside a collapsed status group, open that
    // group so the highlighted selection is actually visible in the board
    // column — otherwise the detail pane shows a card the reader can't see.
    const group = card.closest('details.board-group-collapsible');
    if (group && !group.open) group.open = true;
    host.innerHTML = '';
    const tpl = templates[card.dataset.cand];
    if (!tpl) {
      host.innerHTML = '<div class="console-empty">No detail recorded for this candidate.</div>';
      return;
    }
    host.appendChild(tpl.content.cloneNode(true));
    wireDetailTabs(host);
    paintViewers(host);
  }

  cards.forEach((card) => {
    card.addEventListener('click', () => activate(card));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(card); }
    });
  });

  document.querySelectorAll('[data-open-cand]').forEach((link) => {
    link.addEventListener('click', (e) => {
      const target = link.getAttribute('data-open-cand');
      const card = cards.find((c) => c.dataset.cand === target);
      if (!card) return;
      e.preventDefault();
      activate(card);
      const detail = document.getElementById('cand-detail-host');
      if (detail && detail.scrollIntoView) {
        detail.scrollIntoView({ block: 'start', behavior: 'smooth' });
      }
    });
  });

  // Auto-open the first card so the detail pane is never empty on load.
  // Prefer a card that's already visible (in an expanded group); only fall
  // back to a collapsed card — which ``activate`` then reveals — when every
  // candidate is collapsed (e.g. an all-incomplete retrieval run).
  const firstVisible = cards.find(
    (c) => !c.closest('details.board-group-collapsible:not([open])'));
  activate(firstVisible || cards[0]);
})();

// Per-tool section sample selector: filters every node block to one
// candidate's result. Runs independently of the candidate console above so
// the appendix stays self-contained (its own selector, own default).
(function () {
  const section = document.getElementById('details');
  if (!section) return;
  const toolbar = section.querySelector('.node-cand-toolbar');
  if (!toolbar) return;  // single / no candidate sample — nothing to switch
  const buttons = Array.prototype.slice.call(
    toolbar.querySelectorAll('.node-cand-btn'));
  const variants = Array.prototype.slice.call(
    section.querySelectorAll('.node-cand-variant'));
  function select(cid) {
    buttons.forEach((b) => {
      const on = b.dataset.ptCand === cid;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    variants.forEach((v) => {
      const c = v.dataset.cand;
      v.classList.toggle('is-active', c === cid || c === '__shared__');
    });
  }
  buttons.forEach((b) => {
    b.addEventListener('click', () => select(b.dataset.ptCand));
  });
})();
"""
        + PLDDT_HOVER_JS
    )


__all__ = ["render_report_html", "resolve_report_output_path", "write_report"]

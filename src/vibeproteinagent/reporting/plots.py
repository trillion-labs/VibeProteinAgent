from __future__ import annotations

import html
import math
from pathlib import Path

from .metrics import (
    pretty_tool_name,
)
from .models import PlotSpec, ReportModel


def build_plots(model: ReportModel) -> list[PlotSpec]:
    """Build Appendix plots only for MSA-generation evidence.

    Candidate-level analytics now live in the review and comparison sections.
    The Appendix Plots panel is reserved for alignment context, so it does not
    emit pLDDT traces, score bars, or cross-metric scatter charts.
    """
    plots: list[PlotSpec] = []
    for block in model.msa_blocks:
        identities = [value for value in block.identities if _num(value) is not None]
        if len(identities) < 3:
            continue
        source = pretty_tool_name(block.source_tool or "") if block.source_tool else "MSA"
        plots.append(
            PlotSpec(
                f"{source} identity distribution",
                "msa_identity",
                score_distribution_svg(
                    identities,
                    tool_label=source,
                    output_kind="MSA sequence identity",
                    score_definition="Fractional identity to the query over aligned, non-gap query columns.",
                ),
            )
        )
    return plots


def _extract_plddt_per_residue(path: Path) -> list[float]:
    """Parse PDB B-factor column for CA atoms — that's pLDDT in AF/ESM3 output.

    Returns one value per residue (CA) along the chain, or empty list when
    the file is missing/unreadable, when the suffix is not PDB, or when the
    B-factor column carries no real confidence signal (every value equal —
    typically all 0.00 from generators like Proteina-Complexa that don't
    fold/score, or a uniform placeholder). A flat trace is worse than no
    trace: it implies "we measured pLDDT and it's terrible" when the truth
    is "no pLDDT was ever computed."
    """
    if not path.is_file() or path.suffix.lower() not in {".pdb", ".ent"}:
        return []
    values: list[float] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        if not (line.startswith("ATOM") and len(line) >= 66):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            values.append(float(line[60:66].strip()))
        except ValueError:
            continue
    # Reject placeholder vectors: an all-equal B-factor column (most often
    # all-zero) carries no information and should not be plotted.
    if not values or max(values) - min(values) < 1e-6:
        return []
    return values


def _nice_residue_ticks(n: int) -> list[int]:
    """Pick ~8 evenly spaced 0-based residue indices for the x-axis.

    Interior ticks land on round 1-based residue numbers (50, 100, …); the
    first (1) and last (n) residues are always included.
    """
    if n <= 2:
        return list(range(n))
    nice = (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000)
    raw = (n - 1) / 8
    step = next((s for s in nice if s >= raw), nice[-1])
    idxs = [0]
    label = step
    while label < n - step * 0.35:
        idxs.append(label - 1)  # residue number `label` -> 0-based index
        label += step
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return idxs


def plddt_trace_svg(
    values: list[float], *, width: int = 860, height: int = 264, idx: int = 0
) -> str:
    """Per-residue pLDDT line plot — product-analytics styling.

    A white plot field with subtle horizontal gridlines, a gradient area
    fill under the trace, a crisp accent-coloured line, a multi-tick residue
    axis, and a dashed reference at the pLDDT 70 "confident" threshold.

    The per-residue values and plot geometry are emitted as ``data-*``
    attributes so :data:`PLDDT_HOVER_JS` can map a cursor position back to a
    residue and show a marker + tooltip on hover.
    """
    if not values:
        return ""
    # Auto-detect 0-1 vs 0-100 scale and normalize to 0-100.
    max_v = max(values)
    in_unit = max_v <= 1.0
    norm = [v * 100 if in_unit else v for v in values]
    n = len(norm)

    pad_l, pad_r, pad_t, pad_b = 48, 20, 36, 42
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    base_y = pad_t + plot_h
    step = plot_w / max(1, n - 1)

    def x_of(i: float) -> float:
        return pad_l + i * step

    def y_of(v: float) -> float:
        return pad_t + plot_h * (1 - v / 100.0)

    pts = [(x_of(i), y_of(v)) for i, v in enumerate(norm)]
    line_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area_d = (
        f"M{pts[0][0]:.1f},{base_y:.1f} "
        + " ".join(f"L{x:.1f},{y:.1f}" for x, y in pts)
        + f" L{pts[-1][0]:.1f},{base_y:.1f} Z"
    )

    # Horizontal gridlines (25/50/75) + y-axis tick labels (0..100).
    grid = ""
    yticks = ""
    for tick in (0, 25, 50, 75, 100):
        y = y_of(tick)
        if tick not in (0, 100):
            grid += (
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" '
                f'y2="{y:.1f}" stroke="#eceaf4" stroke-width="1"/>'
            )
        yticks += (
            f'<text x="{pad_l - 9}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'class="plddt-axis">{tick}</text>'
        )

    # X-axis: round residue ticks + marks.
    xticks = ""
    for ri in _nice_residue_ticks(n):
        x = x_of(ri)
        xticks += (
            f'<line x1="{x:.1f}" y1="{base_y:.1f}" x2="{x:.1f}" '
            f'y2="{base_y + 4:.1f}" stroke="#cfc8e0" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{base_y + 16:.1f}" text-anchor="middle" '
            f'class="plddt-axis">{ri + 1}</text>'
        )

    # Dashed reference at the pLDDT 70 "confident" threshold.
    y70 = y_of(70)
    ref = (
        f'<line x1="{pad_l}" y1="{y70:.1f}" x2="{pad_l + plot_w}" y2="{y70:.1f}" '
        f'stroke="#c2b6e6" stroke-width="1" stroke-dasharray="4 4"/>'
        f'<text x="{pad_l + plot_w - 5:.1f}" y="{y70 - 5:.1f}" text-anchor="end" '
        f'class="plddt-ref-label">confident ≥ 70</text>'
    )

    mean_v = sum(norm) / n
    subtitle = (
        f'<text x="{pad_l}" y="20" class="plddt-subtitle">'
        f'mean pLDDT {mean_v:.1f} · n = {n} residues</text>'
    )
    xlabel = (
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 6}" text-anchor="middle" '
        f'class="plddt-axis-title">residue index</text>'
    )
    data_values = ",".join(f"{v:.1f}" for v in norm)
    grad = f"plddtArea{idx}"

    return (
        f'<svg class="plot-svg plddt-plot" viewBox="0 0 {width} {height}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'data-values="{data_values}" data-pad-l="{pad_l}" data-pad-t="{pad_t}" '
        f'data-plot-w="{plot_w}" data-plot-h="{plot_h}" data-n="{n}">'
        f'<defs><linearGradient id="{grad}" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#6b1ea8" stop-opacity="0.26"/>'
        '<stop offset="1" stop-color="#6b1ea8" stop-opacity="0.02"/>'
        '</linearGradient></defs>'
        f'<rect class="plddt-field" x="{pad_l}" y="{pad_t}" width="{plot_w}" '
        f'height="{plot_h}" rx="6"/>'
        f'{grid}{ref}'
        f'<path d="{area_d}" fill="url(#{grad})"/>'
        f'<polyline class="plddt-line" points="{line_pts}"/>'
        f'<line class="plddt-axis-line" x1="{pad_l}" y1="{pad_t}" '
        f'x2="{pad_l}" y2="{base_y}"/>'
        f'<line class="plddt-axis-line" x1="{pad_l}" y1="{base_y}" '
        f'x2="{pad_l + plot_w}" y2="{base_y}"/>'
        f'{yticks}{xticks}{subtitle}{xlabel}'
        '<g class="plddt-hover">'
        f'<line class="plddt-crosshair" y1="{pad_t}" y2="{base_y}"/>'
        '<circle class="plddt-cursor" r="4.5"/>'
        '<g class="plddt-tip">'
        '<rect class="plddt-tip-bg" width="104" height="36" rx="6"/>'
        '<text class="plddt-tip-res" x="52" y="14" text-anchor="middle"></text>'
        '<text class="plddt-tip-val" x="52" y="28" text-anchor="middle"></text>'
        '</g></g>'
        '</svg>'
    )


def _nice_ceiling(v: float) -> float:
    """Smallest 'nice' number (1/2/2.5/5 x 10^k) that is >= ``v``."""
    if v <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(v))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if m * mag >= v:
            return m * mag
    return 10.0 * mag


def score_distribution_svg(
    values: list[float],
    *,
    width: int = 860,
    height: int = 248,
    tool_label: str = "",
    output_kind: str = "",
    score_definition: str = "",
) -> str:
    """Per-candidate score bar chart — product-analytics styling.

    One gradient-filled bar per candidate, sorted ascending so the chart
    reads as a clean distribution profile. White plot field with horizontal
    gridlines, a y-axis scaled to a 'nice' ceiling, a value label above each
    bar, and a dashed mean reference. Bar width is capped so a handful of
    candidates never balloon into solid blocks, while the bars still span
    the full plot width so the panel never looks empty.

    ``tool_label`` / ``output_kind`` / ``score_definition`` populate the
    plot subtitle so the chart is self-describing — a reader sees which
    tool produced these scores, what kind of artifact each candidate is
    (sequence design, structure prediction, ...), and what "score" means
    in context (with direction). All three are optional; when omitted the
    subtitle collapses back to the generic ``n = N · range MIN-MAX`` line.
    """
    if not values:
        return ""
    vals = sorted(values)
    n = len(vals)
    vmin, vmax = vals[0], vals[-1]
    mean_v = sum(vals) / n
    y_top = 1.0 if vmax <= 1.0 else _nice_ceiling(vmax)

    # Reserve extra vertical padding when we have a 2-line subtitle to keep
    # the gridlines from collapsing onto the caption text.
    has_context_line = bool(tool_label or output_kind or score_definition)
    pad_l, pad_r, pad_t, pad_b = 46, 18, (58 if has_context_line else 42), 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    base_y = pad_t + plot_h

    def y_of(v: float) -> float:
        return base_y - plot_h * (v / y_top)

    def fmt(v: float) -> str:
        return f"{v:.3g}"

    # Horizontal gridlines + y-axis tick labels.
    grid = ""
    yticks = ""
    n_ticks = 4
    for k in range(n_ticks + 1):
        tv = y_top * k / n_ticks
        y = y_of(tv)
        if 0 < k < n_ticks:
            grid += (
                f'<line class="sd-grid" x1="{pad_l}" y1="{y:.1f}" '
                f'x2="{pad_l + plot_w}" y2="{y:.1f}"/>'
            )
        yticks += (
            f'<text class="sd-axis" x="{pad_l - 9}" y="{y + 3.5:.1f}" '
            f'text-anchor="end">{fmt(tv)}</text>'
        )

    # Bars — capped width, centred in evenly spaced slots.
    slot = plot_w / n
    bar_w = max(3.0, min(96.0, slot * 0.64))
    show_idx = n <= 20
    bars = ""
    labels = ""
    xidx = ""
    for i, v in enumerate(vals):
        cx = pad_l + (i + 0.5) * slot
        by = y_of(v)
        bars += (
            f'<rect class="sd-bar" x="{cx - bar_w / 2:.1f}" y="{by:.1f}" '
            f'width="{bar_w:.1f}" height="{base_y - by:.1f}" rx="4"/>'
        )
        labels += (
            f'<text class="sd-bar-label" x="{cx:.1f}" y="{by - 6:.1f}" '
            f'text-anchor="middle">{fmt(v)}</text>'
        )
        if show_idx:
            xidx += (
                f'<text class="sd-axis" x="{cx:.1f}" y="{base_y + 15:.1f}" '
                f'text-anchor="middle">{i + 1}</text>'
            )

    # Dashed mean reference.
    ymean = y_of(mean_v)
    mean_ref = (
        f'<line class="sd-mean" x1="{pad_l}" y1="{ymean:.1f}" '
        f'x2="{pad_l + plot_w}" y2="{ymean:.1f}"/>'
        f'<text class="sd-mean-label" x="{pad_l + plot_w - 5:.1f}" '
        f'y="{ymean - 5:.1f}" text-anchor="end">mean {fmt(mean_v)}</text>'
    )

    # Subtitle: when caller provided context, line 1 carries the
    # tool + output kind + score definition; line 2 carries the
    # numeric summary. Without context we keep the one-line generic.
    if has_context_line:
        # Line 1: "<Tool> <output kind> · score = <definition>". Each
        # part is optional; only include separators when neighbouring
        # parts are present so we don't emit dangling "  · "s.
        line1_parts: list[str] = []
        head = " ".join(p for p in (tool_label, output_kind) if p).strip()
        if head:
            line1_parts.append(head)
        if score_definition:
            line1_parts.append(f"score = {score_definition}")
        line1 = " · ".join(line1_parts)
        subtitle = (
            f'<text class="sd-subtitle" x="{pad_l}" y="20">{html.escape(line1)}</text>'
            f'<text class="sd-subtitle sd-subtitle-meta" x="{pad_l}" y="36">'
            f'n = {n} candidates · range {fmt(vmin)}-{fmt(vmax)}</text>'
        )
    else:
        subtitle = (
            f'<text class="sd-subtitle" x="{pad_l}" y="20">'
            f'n = {n} candidates · score {fmt(vmin)}-{fmt(vmax)}</text>'
        )
    xtitle = (
        f'<text class="sd-axis-title" x="{pad_l + plot_w / 2:.1f}" y="{height - 5}" '
        f'text-anchor="middle">candidate · sorted by score</text>'
    )

    return (
        f'<svg class="plot-svg sd-plot" viewBox="0 0 {width} {height}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="sdBar" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#8a3fc8"/>'
        '<stop offset="1" stop-color="#6b1ea8"/>'
        '</linearGradient></defs>'
        f'<rect class="sd-field" x="{pad_l}" y="{pad_t}" width="{plot_w}" '
        f'height="{plot_h}" rx="6"/>'
        f'{grid}{bars}{mean_ref}'
        f'<line class="sd-axis-line" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{base_y}"/>'
        f'<line class="sd-axis-line" x1="{pad_l}" y1="{base_y}" '
        f'x2="{pad_l + plot_w}" y2="{base_y}"/>'
        f'{yticks}{xidx}{labels}{subtitle}{xtitle}'
        '</svg>'
    )


def scatter_svg(
    pairs: list[tuple[object, object, str]],
    *,
    width: int = 520,
    height: int = 220,
) -> str:
    xs = [_num(x) for x, _, _ in pairs]
    ys = [_num(y) for _, y, _ in pairs]
    nums = [
        (x, y, label)
        for x, y, (_, _, label) in zip(xs, ys, pairs, strict=False)
        if x is not None and y is not None
    ]
    if not nums:
        return ""
    min_x, max_x = min(x for x, _, _ in nums), max(x for x, _, _ in nums)
    min_y, max_y = min(y for _, y, _ in nums), max(y for _, y, _ in nums)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    pad = 34
    dots = []
    for x_val, y_val, label in nums:
        x = pad + (x_val - min_x) / span_x * (width - pad * 2)
        y = height - pad - (y_val - min_y) / span_y * (height - pad * 2)
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5"><title>'
            f'{html.escape(label)}: {x_val:.2f}, {y_val:.2f}</title></circle>'
        )
    return _svg(width, height, "".join(dots), f"{min_x:.1f}", f"{max_x:.1f}")


def bar_svg(items: dict[str, int], *, width: int = 520, height: int = 220) -> str:
    if not items:
        return ""
    ordered = sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    max_count = max(v for _, v in ordered) or 1
    pad = 28
    row_h = (height - pad * 2) / max(1, len(ordered))
    rows = []
    for idx, (label, count) in enumerate(ordered):
        y = pad + idx * row_h
        w = (width - 190) * count / max_count
        rows.append(
            f'<text x="8" y="{y + row_h * 0.65:.1f}">{html.escape(label[:24])}</text>'
            f'<rect x="180" y="{y + 4:.1f}" width="{w:.1f}" '
            f'height="{max(4, row_h - 8):.1f}" rx="3" />'
            f'<text x="{185 + w:.1f}" y="{y + row_h * 0.65:.1f}">{count}</text>'
        )
    return _svg(width, height, "".join(rows), "", "")


def _svg(width: int, height: int, body: str, left_label: str, right_label: str) -> str:
    labels = ""
    if left_label or right_label:
        labels = (
            f'<text x="28" y="{height - 6}">{html.escape(left_label)}</text>'
            f'<text x="{width - 52}" y="{height - 6}">{html.escape(right_label)}</text>'
        )
    return (
        f'<svg class="plot-svg" viewBox="0 0 {width} {height}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">{body}{labels}</svg>'
    )


def _num(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None


# Hover interactivity for the per-residue pLDDT trace. Self-contained IIFE —
# included inline in the report's <script> by render._js(). On mousemove over
# a ``.plddt-plot`` SVG it snaps to the nearest residue, shows a marker dot +
# crosshair, and a tooltip with the residue number and pLDDT value; the whole
# hover layer is hidden again on mouseleave.
PLDDT_HOVER_JS = """
(function () {
  var TIP_W = 104, TIP_H = 36;
  document.querySelectorAll('svg.plddt-plot').forEach(function (svg) {
    var values = (svg.dataset.values || '').split(',').map(parseFloat);
    var n = values.length;
    if (n < 2) return;
    var padL = parseFloat(svg.dataset.padL);
    var padT = parseFloat(svg.dataset.padT);
    var plotW = parseFloat(svg.dataset.plotW);
    var plotH = parseFloat(svg.dataset.plotH);
    var step = plotW / (n - 1);
    var hover = svg.querySelector('.plddt-hover');
    var cross = svg.querySelector('.plddt-crosshair');
    var dot = svg.querySelector('.plddt-cursor');
    var tip = svg.querySelector('.plddt-tip');
    var tipRes = svg.querySelector('.plddt-tip-res');
    var tipVal = svg.querySelector('.plddt-tip-val');
    if (!hover || !cross || !dot || !tip) return;
    function hide() { hover.style.opacity = '0'; }
    function move(ev) {
      var ctm = svg.getScreenCTM();
      if (!ctm) return;
      var pt = svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      var loc = pt.matrixTransform(ctm.inverse());
      if (loc.x < padL - 4 || loc.x > padL + plotW + 4 ||
          loc.y < padT - 12 || loc.y > padT + plotH + 12) { hide(); return; }
      var i = Math.round((loc.x - padL) / step);
      if (i < 0) i = 0;
      if (i > n - 1) i = n - 1;
      var v = values[i];
      var cx = padL + i * step;
      var cy = padT + plotH * (1 - v / 100);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx);
      dot.setAttribute('cx', cx); dot.setAttribute('cy', cy);
      tipRes.textContent = 'Residue ' + (i + 1);
      tipVal.textContent = 'pLDDT ' + v.toFixed(1);
      var tx = cx - TIP_W / 2;
      if (tx < padL) tx = padL;
      if (tx + TIP_W > padL + plotW) tx = padL + plotW - TIP_W;
      var ty = cy - TIP_H - 13;
      if (ty < padT) ty = cy + 15;
      tip.setAttribute('transform', 'translate(' + tx.toFixed(1) + ',' + ty.toFixed(1) + ')');
      hover.style.opacity = '1';
    }
    svg.addEventListener('mousemove', move);
    svg.addEventListener('mouseleave', hide);
  });
})();
"""


__all__ = [
    "PLDDT_HOVER_JS",
    "bar_svg",
    "build_plots",
    "plddt_trace_svg",
    "scatter_svg",
    "score_distribution_svg",
]

# SVG template strings are intentionally kept readable as literal snippets.
# ruff: noqa: E501

from __future__ import annotations

import html
from pathlib import Path

from .models import PlotSpec, ReportModel


def build_plots(model: ReportModel) -> list[PlotSpec]:
    """Build only meaningful plots — skip tautological single-value bar charts.

    Multi-candidate runs get distributions; single-candidate runs get the
    per-residue pLDDT trace if their structure carries B-factor confidence.
    """
    plots: list[PlotSpec] = []

    # Per-residue pLDDT (B-factor) plot — most informative for structure
    # prediction outputs. Skip when no candidate has a structure file.
    for candidate in model.candidates[:5]:
        if not candidate.structure_path:
            continue
        plddt_values = _extract_plddt_per_residue(Path(candidate.structure_path))
        if not plddt_values:
            continue
        plots.append(
            PlotSpec(
                f"Per-residue pLDDT · {candidate.candidate_id}",
                "plddt_trace",
                plddt_trace_svg(plddt_values),
            )
        )
        # Only one per-residue plot is shown unless this is a multi-candidate
        # run — keep the report from getting noisy.
        if len(model.candidates) == 1:
            break

    scores = [c.score for c in model.candidates if c.score is not None]
    if len(scores) >= 3:
        plots.append(PlotSpec("Candidate score distribution", "histogram", histogram_svg(scores)))

    pairs = [
        (c.scores.get("plddt"), c.scores.get("iptm"), c.candidate_id)
        for c in model.candidates
        if _num(c.scores.get("plddt")) is not None and _num(c.scores.get("iptm")) is not None
    ]
    if len(pairs) >= 3:
        plots.append(PlotSpec("pLDDT vs ipTM", "scatter", scatter_svg(pairs)))

    # QC flag counts and workflow node-state breakdowns used to be plotted
    # here, but both add visual noise without information value once the
    # per-tool widgets render the underlying data directly (QC widget shows
    # every metric + flag, and the workflow DAG already shows node states).
    return plots


def _extract_plddt_per_residue(path: Path) -> list[float]:
    """Parse PDB B-factor column for CA atoms — that's pLDDT in AF/ESM3 output.

    Returns one value per residue (CA) along the chain, or empty list when
    the file is missing/unreadable. CIF/mmCIF is skipped for now; the
    standard ESM3/AlphaFold output is PDB.
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
    return values


def plddt_trace_svg(values: list[float], *, width: int = 720, height: int = 200) -> str:
    """Line plot of pLDDT vs residue index with AlphaFold confidence bands."""
    if not values:
        return ""
    # Auto-detect 0-1 vs 0-100 scale and normalize for color bands.
    max_v = max(values)
    in_unit = max_v <= 1.0
    norm = [v * 100 if in_unit else v for v in values]
    pad_l = 42
    pad_r = 18
    pad_t = 14
    pad_b = 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    n = len(norm)
    step = plot_w / max(1, n - 1)
    # Build polyline
    points = " ".join(
        f"{pad_l + idx * step:.1f},{pad_t + plot_h * (1 - v / 100.0):.1f}"
        for idx, v in enumerate(norm)
    )
    # AF/ESM confidence bands (90/70/50)
    bands_html = ""
    for low, high, color in [
        (90, 100, "#4ec3e0"),    # very high
        (70, 90, "#76d8a0"),     # confident
        (50, 70, "#f4d35e"),     # low
        (0, 50, "#e07a5f"),      # very low
    ]:
        y_top = pad_t + plot_h * (1 - high / 100.0)
        y_bot = pad_t + plot_h * (1 - low / 100.0)
        bands_html += (
            f'<rect x="{pad_l}" y="{y_top:.1f}" width="{plot_w}" '
            f'height="{y_bot - y_top:.1f}" fill="{color}" opacity="0.13"/>'
        )
    # Y-axis ticks
    ticks_html = ""
    for tick in (0, 25, 50, 75, 100):
        y = pad_t + plot_h * (1 - tick / 100.0)
        ticks_html += (
            f'<line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l}" y2="{y:.1f}" stroke="#999" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{y + 3:.1f}" text-anchor="end" font-size="10" fill="#666">{tick}</text>'
        )
    # X axis label
    mean_v = sum(norm) / len(norm)
    title = f'<text x="{pad_l}" y="10" font-size="11" fill="#555">mean pLDDT {mean_v:.1f} · n={n} residues</text>'
    return (
        f'<svg class="plot-svg" viewBox="0 0 {width} {height}" role="img" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{bands_html}{ticks_html}{title}'
        f'<polyline points="{points}" fill="none" stroke="#1f5a4a" stroke-width="1.8"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h:.1f}" stroke="#aaa"/>'
        f'<line x1="{pad_l}" y1="{pad_t + plot_h:.1f}" x2="{width - pad_r}" y2="{pad_t + plot_h:.1f}" stroke="#aaa"/>'
        f'<text x="{pad_l}" y="{height - 8}" font-size="10" fill="#666">1</text>'
        f'<text x="{width - pad_r - 12}" y="{height - 8}" font-size="10" fill="#666" text-anchor="end">{n}</text>'
        f'<text x="{(pad_l + width - pad_r) / 2:.1f}" y="{height - 8}" font-size="10" '
        f'fill="#666" text-anchor="middle">residue index</text>'
        '</svg>'
    )


def histogram_svg(values: list[float], *, width: int = 520, height: int = 180) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    bins = min(10, max(1, len(values)))
    span = hi - lo or 1.0
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / span * bins))
        counts[idx] += 1
    max_count = max(counts) or 1
    pad = 28
    bar_w = (width - pad * 2) / bins
    bars = []
    for idx, count in enumerate(counts):
        h = (height - pad * 2) * count / max_count
        x = pad + idx * bar_w
        y = height - pad - h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(2, bar_w - 4):.1f}" '
            f'height="{h:.1f}" rx="3"><title>{count}</title></rect>'
        )
    return _svg(width, height, "".join(bars), f"{lo:.2f}", f"{hi:.2f}")


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


__all__ = ["bar_svg", "build_plots", "histogram_svg", "plddt_trace_svg", "scatter_svg"]

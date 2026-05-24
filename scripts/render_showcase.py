# ruff: noqa: E402, I001
"""Generate a self-contained HTML showcase of the Phase 6 pretty-mode renderers.

Runs each ``render_*`` function in :mod:`vibeproteinagent.cli.render` against a
deterministic synthetic payload, captures the pretty-mode output as SVG via
Rich's record mode, captures the plain-mode counterpart as text, and emits a
single self-contained ``docs/render-showcase.html`` page.

Why this exists: the renderers only produce ANSI in a TTY, so neither CI logs
nor scrollback give a faithful picture of what users actually see. The HTML
snapshot is the source-controlled "what does ``vpa`` look like" reference.

Usage:
    # from the project root, inside the ``vpa`` conda env
    python scripts/render_showcase.py

Outputs:
    docs/render-showcase.html   (overwritten)

The HTML is intentionally gitignored — re-run this script when ``render.py`` or
the descriptor surface changes.
"""

from __future__ import annotations

import html
import sys
from io import StringIO
from pathlib import Path

# Make the in-tree package importable without requiring an editable install.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.panel import Panel

from vibeproteinagent.cli import render
from vibeproteinagent.core.log import LogStyle


OUT_DIR = ROOT / "docs"
OUT_HTML = OUT_DIR / "render-showcase.html"


# ---------------------------------------------------------------- helpers


def _console(width: int, *, pretty: bool) -> Console:
    return Console(
        record=True,
        force_terminal=pretty,
        color_system="truecolor" if pretty else None,
        no_color=not pretty,
        width=width,
        file=StringIO(),
        highlight=False,
    )


def _shoot(fn, payload, *, width: int = 100) -> tuple[str, str]:
    """Return ``(svg, plain_text)`` for the given renderer + payload."""
    pretty = _console(width, pretty=True)
    fn(payload, style=LogStyle.PRETTY, console=pretty)
    svg = pretty.export_svg(title=fn.__name__, clear=True)

    plain = _console(width, pretty=False)
    fn(payload, style=LogStyle.PLAIN, console=plain)
    text = plain.export_text(clear=True)
    return svg, text


def _shoot_error(payload, *, width: int = 100) -> tuple[str, str]:
    """Mirror the visual produced by :func:`render.render_error`.

    The framework's ``render_error`` writes to its own internal stderr console
    (it ignores the ``console`` arg), so we can't capture its SVG directly. We
    reconstruct the same Panel shape here so the showcase reflects what a user
    would see when an error envelope fires.
    """
    err = payload["error"]
    detail_str = ", ".join(f"{k}={v}" for k, v in (err.get("details") or {}).items())
    body = (
        f"[bold red]Error:[/] [bold]{err['code']}[/]\n"
        f"  message: {err['message']}\n"
        f"  details: {detail_str}"
    )

    pretty = _console(width, pretty=True)
    pretty.print(Panel(body, border_style="red", title="vpa run", title_align="left"))
    svg = pretty.export_svg(title="render_error", clear=True)

    plain = _console(width, pretty=False)
    plain.print(Panel(body, border_style="red", title="vpa run", title_align="left"))
    text = plain.export_text(clear=True)
    return svg, text


# ---------------------------------------------------------------- synthetic payloads


VERSION = {
    "version": "0.0.1", "python": "3.12.13",
    "torch": "2.7.1+cu128", "cuda": "12.8",
}

TOOLS = {"tools": [
    {"name": "noop", "version": "1.0.0", "category": "misc",
     "description": "Returns its input unchanged. Used for IPC latency benchmarking.",
     "gpu_required": False, "vram_min_gib": None},
    {"name": "proteinmpnn", "version": "1.0.1", "category": "inverse_folding",
     "description": "Sequence design over a fixed backbone (Dauparas et al., 2022).",
     "gpu_required": True, "vram_min_gib": 2},
    {"name": "mpnn_to_protenix_input", "version": "1.0.1", "category": "misc",
     "description": (
         "Glue adapter: convert top-K ProteinMPNN designs (FASTA) into "
         "Protenix-v2 input JSON files."
     ),
     "gpu_required": False, "vram_min_gib": None},
    {"name": "protenix_v2", "version": "1.0.0", "category": "structure_prediction",
     "description": (
         "Protein/complex 3D structure prediction (Protenix v1, "
         "AlphaFold3-class). ByteDance, B200-patched."
     ),
     "gpu_required": True, "vram_min_gib": 16},
    {"name": "pxdesign", "version": "1.0.0", "category": "protein_design",
     "description": "PXDesign target-conditioned binder generation (ByteDance DiT, B200-patched).",
     "gpu_required": True, "vram_min_gib": 24},
]}

DOCTOR = {
    "python_version": "3.12.13",
    "executable": "/opt/miniconda3/envs/vpa/bin/python3.12",
    "torch_version": "2.7.1+cu128", "cuda_version": "12.8",
    "cuda_available": True, "num_gpus": 8,
    "active_workspace": "demo",
    "workspace_root": "/tmp/vpa_morning_demo/home/workspaces/demo",
    "model_cache_root": "/opt/vpa-data/models",
    "daemon_running": True,
    "registry_tools": ["mpnn_to_protenix_input", "noop", "proteinmpnn",
                       "protenix_v2", "pxdesign"],
    "gpus": [{"index": i, "name": "NVIDIA B200", "compute_capability": "10.0",
              "total_memory_gib": 178, "multi_processor_count": 148}
             for i in range(8)],
}

WORKSPACE_STATUS = {"active": "demo", "via": "file",
                    "root": "/tmp/vpa_morning_demo/home/workspaces/demo"}

DAEMON_STATUS = {
    "running": True, "pid": 1697357,
    "socket": "/tmp/vpa_morning_demo/run/supervisor.sock",
    "log": "/tmp/vpa_morning_demo/run/supervisor.log",
    "workers": [{
        "gpu_id": 0, "pid": 1697905, "alive": True, "busy": False,
        "current_job_id": None,
        "free_vram_mib": 177878, "total_vram_mib": 182641,
        "loaded_adapters": [
            {"name": "proteinmpnn", "version": "1.0.1", "vram_used_mib": 14,
             "last_used_ts": 0.0, "pin_warm": True, "run_count": 3},
            {"name": "mpnn_to_protenix_input", "version": "1.0.1",
             "vram_used_mib": 0, "last_used_ts": 0.0,
             "pin_warm": True, "run_count": 1},
            {"name": "protenix_v2", "version": "1.0.0", "vram_used_mib": 2910,
             "last_used_ts": 0.0, "pin_warm": True, "run_count": 1},
        ],
    }],
}

WARM = {
    "tool": "protenix_v2", "version": "*",
    "warmed": [{"gpu_id": 0, "success": True, "vram_used_mib": 2910,
                "free_vram_mib": 177878, "pin_warm": True,
                "worker_state_before": "idle",
                "waited_for_busy_worker_ms": 0}],
}

JOB = {"job": {
    "job_id": "job_31c72d72ee42719b4671acdebe58e972",
    "call_id": None, "state": "completed",
    "tool": "proteinmpnn", "version": "1.0.1",
    "params_hash": "ab7e0a1b54b2a3681d8c4bfdeaecaf24cc47b8456927540ebf20b2d1d50b6f78",
    "workspace": "demo", "priority": "normal",
    "submitted_at": 1747400000123456, "queued_at": 1747400000125000,
    "started_at": 1747400000130000, "finished_at": 1747400000750000,
    "worker_id": 0, "gpu_ids": [0],
    "error_code": None, "error_message": None, "error_details": None,
    "output_json": '{"designs":[/* 4 sequences */],"designs_fasta":"outputs/designs.fa"}',
    "retry_count": 0, "max_retries": 0, "cache_hit": 0,
}}


def _workflow(state_map: dict) -> dict:
    return {
        "workflow_id": "a8cb8f1e6e3ca062",
        "name": "user_design_workflow", "version": "1.0.0",
        "state": "completed",
        "nodes": [
            {"node_id": "design", "tool": "proteinmpnn",
             "tool_version": ">=1.0.0,<2",
             "state": state_map["design"],
             "job_id": "job_31c72d72ee42719b4671acdebe58e972"},
            {"node_id": "convert", "tool": "mpnn_to_protenix_input",
             "tool_version": ">=1.0.0,<2",
             "state": state_map["convert"],
             "job_id": "job_8e2dbf3a4f9c1d20a17f0e3e4ab6c4d5"},
            {"node_id": "refold", "tool": "protenix_v2",
             "tool_version": ">=1.0.0,<2",
             "state": "skipped", "job_id": None},
            {"node_id": "refold[0]", "tool": "protenix_v2",
             "tool_version": ">=1.0.0,<2",
             "state": state_map["refold[0]"],
             "job_id": "job_a8b3c0d9e1f4a07b2c3d4e5f6a7b8c9d"},
        ],
    }


WORKFLOW_FIRST = _workflow(
    {"design": "completed", "convert": "completed", "refold[0]": "completed"}
)
WORKFLOW_CACHED = _workflow(
    {"design": "cached", "convert": "cached", "refold[0]": "cached"}
)

ERROR = {"ok": False, "error": {
    "code": "E_CONFLICT",
    "message": "call_id 'k1' already bound to job job_31c72d… with a different submission",
    "details": {"call_id": "k1",
                "existing_job_id": "job_31c72d72ee42719b4671acdebe58e972",
                "diff": {"params_hash": "differs"}},
}}


# ---------------------------------------------------------------- tiles


TILES = [
    ("vpa version",               "Three-line version block.",
     render.render_version,         VERSION, 100),
    ("vpa tools",
     "Registered tools table — category column colored, GPU/VRAM badges.",
     render.render_tool_list,       TOOLS, 110),
    ("vpa doctor",                 "Environment summary + per-GPU table.",
     render.render_doctor,          DOCTOR, 92),
    ("vpa workspace status",       "Active workspace + resolution source.",
     render.render_workspace_status, WORKSPACE_STATUS, 100),
    ("vpa daemon status",          "Worker table with VRAM + loaded adapters.",
     render.render_daemon_status,   DAEMON_STATUS, 120),
    ("vpa daemon warm protenix_v2", "Per-worker warm summary.",
     render.render_warm_result,     WARM, 100),
    ("vpa status <job_id>",        "Single job record.",
     render.render_job,             JOB, 110),
    ("vpa workflow run user_design_workflow.yaml (first run)",
     "DAG node states — 3 ran, 1 skipped (foreach parent).",
     render.render_workflow_status, WORKFLOW_FIRST, 110),
    ("vpa workflow run user_design_workflow.yaml (re-run)",
     "Same workflow, all 3 nodes hit the cache.",
     render.render_workflow_status, WORKFLOW_CACHED, 110),
    ("vpa run --call-id collision (E_CONFLICT)",
     "Error envelope rendered as a red-bordered panel.",
     None,                          ERROR, 100),
]


# ---------------------------------------------------------------- main


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(TILES)} tiles → {OUT_HTML}")

    sections: list[tuple[str, str, str, str]] = []
    for idx, (cmd, desc, fn, payload, width) in enumerate(TILES, 1):
        if fn is None:
            svg, text = _shoot_error(payload, width=width)
        else:
            svg, text = _shoot(fn, payload, width=width)
        print(f"  {idx:>2}/{len(TILES)}  {cmd}")
        sections.append((cmd, desc, svg, text))

    parts: list[str] = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>vibeproteinagent — pretty-mode showcase</title>',
        '<style>',
        'body{font-family:system-ui,sans-serif;max-width:1300px;margin:2em auto;'
        'padding:0 2em;color:#222}',
        'h1{border-bottom:2px solid #333;padding-bottom:.4em}',
        'h2{margin-top:2.5em;color:#0a3d62}',
        '.desc{color:#555;margin:.5em 0 1em}',
        'svg{max-width:100%;height:auto;border:1px solid #ddd;border-radius:6px;background:#000}',
        'pre{background:#1e1e1e;color:#ddd;padding:1em;border-radius:6px;overflow-x:auto;font-size:13px}',
        'details{margin-top:.5em}',
        '</style></head><body>',
        '<h1>vibeproteinagent — pretty-mode renderers</h1>',
        '<p>Each tile shows the <em>pretty-mode</em> SVG (what a real terminal renders) '
        'and the matching <em>plain-mode</em> output below it (collapsed; what scrapers / '
        'pipes / <code>NO_COLOR=1</code> see). Regenerated by '
        '<code>python scripts/render_showcase.py</code>.</p>',
    ]
    for cmd, desc, svg, text in sections:
        parts.append(f'<h2>{html.escape(cmd)}</h2>')
        parts.append(f'<p class="desc">{html.escape(desc)}</p>')
        parts.append(svg)
        parts.append(
            f'<details><summary>plain mode</summary><pre>{html.escape(text)}</pre></details>'
        )
    parts.append('</body></html>')

    OUT_HTML.write_text('\n'.join(parts))
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

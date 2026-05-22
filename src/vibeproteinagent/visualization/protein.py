# HTML / CSS / JS templates are intentionally kept readable as literal
# multi-line snippets (the report's render.py uses the same convention).
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError, NotFoundError
from vibeproteinagent.workspace import ArtifactStore, JobStore, Workspace

STRUCTURE_SUFFIXES = frozenset({".pdb", ".ent", ".cif", ".mmcif"})


@dataclass(frozen=True)
class StructureEntry:
    label: str
    path: Path
    format: str
    content: str
    metadata: dict[str, Any]
    source: dict[str, Any]


def structure_entry_from_path(
    path: str | Path,
    *,
    label: str | None = None,
    source: dict[str, Any] | None = None,
) -> StructureEntry:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise NotFoundError(f"structure file not found: {p}", details={"path": str(p)})
    if p.suffix.lower() not in STRUCTURE_SUFFIXES:
        raise InvalidParamsError(
            f"unsupported structure format for {p.name!r}",
            details={
                "path": str(p),
                "suffix": p.suffix,
                "allowed_suffixes": sorted(STRUCTURE_SUFFIXES),
            },
        )
    return StructureEntry(
        label=label or p.name,
        path=p,
        format=_structure_format_token(p),
        content=p.read_text(errors="replace"),
        metadata=_structure_metadata(p),
        source=source or {"kind": "path"},
    )


def resolve_job_structures(
    workspace: Workspace,
    job_id: str,
    *,
    field: str | None = None,
) -> tuple[list[StructureEntry], dict[str, Any]]:
    store = JobStore(workspace)
    rec = store.get(job_id)
    artifacts = ArtifactStore(workspace).for_job(rec.job_id)
    entries: list[StructureEntry] = []
    seen: set[Path] = set()

    for artifact_field, artifact in sorted(artifacts.items()):
        if field and artifact_field != field:
            continue
        path = Path(artifact.pool_path)
        if not _looks_like_structure(path, artifact.kind):
            continue
        entry = structure_entry_from_path(
            path,
            label=artifact_field,
            source={
                "kind": "artifact",
                "field": artifact_field,
                "artifact_id": artifact.artifact_id,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
            },
        )
        entries.append(entry)
        seen.add(entry.path)

    if not entries and rec.output:
        for output_field, path in _iter_structure_paths(rec.output):
            if field and output_field != field:
                continue
            resolved = path.expanduser().resolve()
            if resolved in seen:
                continue
            entry = structure_entry_from_path(
                resolved,
                label=output_field or resolved.name,
                source={"kind": "job_output", "field": output_field},
            )
            entries.append(entry)
            seen.add(entry.path)

    if field and not entries:
        raise NotFoundError(
            f"job {rec.job_id} has no structure artifact/output field {field!r}",
            details={"job_id": rec.job_id, "field": field},
        )
    if not entries:
        raise NotFoundError(
            f"job {rec.job_id} has no PDB/CIF structure artifact",
            details={
                "job_id": rec.job_id,
                "artifact_fields": sorted(artifacts),
                "hint": "run `vpa artifacts <job_id> --json` to inspect available artifacts",
            },
        )

    summary = {
        "job_id": rec.job_id,
        "state": rec.state.value,
        "tool": rec.tool_name,
        "version": rec.tool_version,
        "workflow_id": rec.workflow_id,
        "workflow_node_id": rec.workflow_node_id,
        "output": _compact(rec.output),
    }
    return entries, summary


def build_protein_viewer(
    structures: list[StructureEntry],
    out_path: str | Path,
    *,
    title: str = "VPA protein viewer",
    summary: dict[str, Any] | None = None,
    viewer_mode: str = "online",
) -> Path:
    if not structures:
        raise InvalidParamsError("at least one structure is required for visualization")
    if viewer_mode not in {"online", "offline"}:
        raise InvalidParamsError(
            f"unknown viewer mode {viewer_mode!r}",
            details={"viewer_mode": viewer_mode, "allowed": ["online", "offline"]},
        )
    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_protein_viewer_html(
            structures,
            title=title,
            summary=summary,
            viewer_mode=viewer_mode,
        ),
        encoding="utf-8",
    )
    return out


def render_protein_viewer_html(
    structures: list[StructureEntry],
    *,
    title: str = "VPA protein viewer",
    summary: dict[str, Any] | None = None,
    viewer_mode: str = "online",
) -> str:
    if not structures:
        raise InvalidParamsError("at least one structure is required for visualization")
    if viewer_mode not in {"online", "offline"}:
        raise InvalidParamsError(
            f"unknown viewer mode {viewer_mode!r}",
            details={"viewer_mode": viewer_mode, "allowed": ["online", "offline"]},
        )
    payload = {
        "title": title,
        "viewer_mode": viewer_mode,
        "summary": summary or {},
        "structures": [
            {
                "label": item.label,
                "format": item.format,
                "content": item.content,
                "metadata": item.metadata,
                "source": item.source,
                "path": str(item.path),
            }
            for item in structures
        ],
    }
    return _render_html(payload, viewer_mode=viewer_mode)


def _looks_like_structure(path: Path, kind: str | None = None) -> bool:
    suffix_ok = path.suffix.lower() in STRUCTURE_SUFFIXES
    kind_ok = (kind or "").lower() in {"pdb", "cif", "mmcif", "structure"}
    return path.is_file() and (suffix_ok or kind_ok)


def _iter_structure_paths(value: Any, prefix: str = "") -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_iter_structure_paths(item, child_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value[:100]):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            out.extend(_iter_structure_paths(item, child_prefix))
    elif isinstance(value, str):
        path = Path(value)
        if path.suffix.lower() in STRUCTURE_SUFFIXES and path.expanduser().is_file():
            out.append((prefix, path))
    return out


def _structure_format_token(path: Path) -> str:
    """Map a structure file extension to the format string 3Dmol expects.

    Used both in the JSON payload (consumed by the inline viewer JS) and
    in :func:`_structure_metadata` for the human-readable file info.
    """
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    return "pdb"


def _structure_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "file_name": path.name,
        "format": _structure_format_token(path),
        "size_bytes": path.stat().st_size,
    }
    try:
        from Bio.PDB import MMCIFParser, PDBParser

        parser = (
            MMCIFParser(QUIET=True)
            if path.suffix.lower() in {".cif", ".mmcif"}
            else PDBParser(QUIET=True)
        )
        structure = parser.get_structure("structure", str(path))
        chains: set[str] = set()
        residue_count = 0
        hetero_count = 0
        atom_count = 0
        for model in structure:
            for chain in model:
                chains.add(str(chain.id))
                for residue in chain:
                    atom_count += len(residue)
                    if residue.id[0] == " ":
                        residue_count += 1
                    else:
                        hetero_count += 1
            break
        metadata.update(
            {
                "chains": sorted(chains),
                "chain_count": len(chains),
                "residue_count": residue_count,
                "hetero_residue_count": hetero_count,
                "atom_count": atom_count,
            }
        )
    except Exception as exc:
        metadata["parse_warning"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _compact(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 30:
                out["..."] = f"{len(value) - 30} more keys"
                break
            out[str(key)] = _compact(item, depth=depth + 1)
        return out
    return str(value)


def _render_html(payload: dict[str, Any], *, viewer_mode: str) -> str:
    if viewer_mode == "offline":
        return _render_offline_html(payload)
    return _render_3dmol_html(payload, viewer_mode=viewer_mode)


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _threedmol_source() -> str:
    asset = Path(__file__).resolve().parent / "assets" / "3Dmol-min.js"
    fallback = Path(__file__).resolve().parents[1] / "reporting" / "assets" / "3Dmol-min.js"
    try:
        return asset.read_text(encoding="utf-8")
    except OSError:
        try:
            return fallback.read_text(encoding="utf-8")
        except OSError:
            return ""


def _render_3dmol_html(payload: dict[str, Any], *, viewer_mode: str) -> str:
    title = str(payload.get("title") or "VPA protein viewer")
    payload_json = _json_for_script(payload)
    escaped_title = html.escape(title)
    threedmol = _threedmol_source()
    if viewer_mode == "online":
        viewer_script = (
            '<script src="https://3Dmol.org/build/3Dmol-min.js"></script>\n'
            f"<script>if (typeof $3Dmol === 'undefined') {{{threedmol}}}</script>"
        )
    else:
        viewer_script = f"<script>{threedmol}</script>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  {viewer_script}
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f8f8;
      --panel: #ffffff;
      --ink: #101828;
      --muted: #667085;
      --line: #d7e1e8;
      --accent: #157a6e;
      --accent-soft: #e8f6f3;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(300px, 390px) minmax(0, 1fr);
      min-height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    main {{
      position: relative;
      min-height: 100vh;
      background: linear-gradient(180deg, #ffffff, #fbfdfd);
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 21px;
      line-height: 1.18;
      font-weight: 760;
    }}
    .subtitle {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }}
    .mode {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      margin: 0 0 16px;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid #b7ddd6;
      background: var(--accent-soft);
      color: #14554c;
      font-size: 12px;
      font-weight: 700;
    }}
    .section {{
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 12px;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 800;
    }}
    .structure-list, .controls {{
      display: grid;
      gap: 8px;
    }}
    .structure-button {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: #fbfcfc;
      color: var(--ink);
      border-radius: 8px;
      padding: 10px;
      cursor: pointer;
      font: inherit;
      min-height: 48px;
    }}
    .structure-button[aria-current="true"] {{
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
      background: var(--accent-soft);
    }}
    .label {{
      display: block;
      font-weight: 750;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .meta {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }}
    label {{
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 24px;
      color: var(--ink);
      font-size: 13px;
    }}
    input[type="checkbox"], select {{
      accent-color: var(--accent);
    }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 8px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    dl {{
      display: grid;
      grid-template-columns: minmax(108px, auto) minmax(0, 1fr);
      gap: 6px 12px;
      margin: 0;
      font-size: 13px;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
      font-weight: 560;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 260px;
      overflow: auto;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfc;
      color: #263238;
      font-size: 12px;
      line-height: 1.45;
    }}
    #viewport {{
      position: absolute;
      inset: 0;
    }}
    #viewport canvas {{
      outline: none;
      display: block;
    }}
    .toolbar {{
      position: absolute;
      top: 14px;
      right: 14px;
      display: flex;
      gap: 8px;
      z-index: 10;
    }}
    .toolbar button {{
      border: 1px solid rgba(16, 24, 40, 0.16);
      background: rgba(255, 255, 255, 0.94);
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 11px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
      box-shadow: 0 4px 14px rgba(15, 23, 42, .07);
    }}
    .status {{
      position: absolute;
      left: 16px;
      bottom: 14px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(16, 24, 40, 0.13);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 13px;
      color: var(--muted);
      max-width: min(700px, calc(100vw - 32px));
      box-shadow: 0 4px 14px rgba(15, 23, 42, .07);
    }}
    .legend {{
      position: absolute;
      right: 16px;
      bottom: 14px;
      z-index: 9;
      width: 220px;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid rgba(16, 24, 40, .13);
      background: rgba(255,255,255,.92);
      box-shadow: 0 4px 14px rgba(15, 23, 42, .07);
    }}
    .legend-title {{
      font-size: 12px;
      font-weight: 800;
      color: var(--muted);
      margin-bottom: 7px;
    }}
    .legend-bar {{
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #ff7d45, #ffdb13, #65cbf3, #0053d6);
    }}
    .legend-labels {{
      display: flex;
      justify-content: space-between;
      margin-top: 5px;
      font-size: 11px;
      color: var(--muted);
    }}
    .empty {{
      display: grid;
      place-items: center;
      height: 100%;
      color: var(--muted);
      padding: 20px;
      text-align: center;
    }}
    @media (max-width: 880px) {{
      .shell {{
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(520px, 70vh);
      }}
      aside {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
        max-height: 48vh;
      }}
      main {{ min-height: 520px; }}
      .legend {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1 id="title"></h1>
      <p class="subtitle" id="path"></p>
      <div class="mode" id="mode"></div>
      <div class="section">
        <h2>Structures</h2>
        <div class="structure-list" id="structures"></div>
      </div>
      <div class="section">
        <h2>Display</h2>
        <div class="controls">
          <label for="color-mode">Colour mode</label>
          <select id="color-mode">
            <option value="plddt" selected>pLDDT / B-factor</option>
            <option value="spectrum">N to C spectrum</option>
            <option value="chain">Chain</option>
          </select>
          <label><input type="checkbox" id="sidechains"> Sidechains</label>
          <label><input type="checkbox" id="ligand" checked> Ligand / hetero atoms</label>
          <label><input type="checkbox" id="surface"> Transparent surface</label>
          <label><input type="checkbox" id="water"> Water</label>
        </div>
      </div>
      <div class="section">
        <h2>Current Structure</h2>
        <dl id="metadata"></dl>
      </div>
    </aside>
    <main>
      <div class="toolbar">
        <button type="button" id="center">Center</button>
        <button type="button" id="spin">Spin</button>
        <button type="button" id="snapshot">Snapshot</button>
      </div>
      <div id="viewport"></div>
      <div class="legend" id="legend">
        <div class="legend-title">pLDDT / B-factor colours</div>
        <div class="legend-bar"></div>
        <div class="legend-labels"><span>low</span><span>high</span></div>
      </div>
      <div class="status" id="status">Loading viewer...</div>
    </main>
  </div>
  <script>
    const payload = {payload_json};
    const titleEl = document.getElementById("title");
    const pathEl = document.getElementById("path");
    const modeEl = document.getElementById("mode");
    const listEl = document.getElementById("structures");
    const metaEl = document.getElementById("metadata");
    const statusEl = document.getElementById("status");
    const viewport = document.getElementById("viewport");
    const controls = {{
      color: document.getElementById("color-mode"),
      sidechains: document.getElementById("sidechains"),
      ligand: document.getElementById("ligand"),
      surface: document.getElementById("surface"),
      water: document.getElementById("water"),
    }};
    let viewer = null;
    let currentItem = null;
    let spinning = false;

    titleEl.textContent = payload.title || "VPA protein viewer";
    modeEl.textContent = (payload.viewer_mode || "offline") + " 3Dmol viewer";

    function fmt(value) {{
      if (Array.isArray(value)) return value.join(", ");
      if (value && typeof value === "object") return JSON.stringify(value);
      return value === undefined || value === null || value === "" ? "n/a" : String(value);
    }}

    function showError(error) {{
      const message = error && error.message ? error.message : String(error);
      statusEl.textContent = "Viewer error: " + message;
      viewport.innerHTML = '<div class="empty">Unable to render this structure.</div>';
      console.error(error);
    }}

    function renderMetadata(item) {{
      metaEl.innerHTML = "";
      const metadata = item.metadata || {{}};
      const keys = [
        "file_name", "format", "chain_count", "chains", "residue_count",
        "hetero_residue_count", "atom_count", "size_bytes", "parse_warning"
      ];
      for (const key of keys) {{
        if (!(key in metadata)) continue;
        const dt = document.createElement("dt");
        dt.textContent = key.replaceAll("_", " ");
        const dd = document.createElement("dd");
        dd.textContent = fmt(metadata[key]);
        metaEl.append(dt, dd);
      }}
    }}

    function plddtColor(b) {{
      const v = b > 1.5 ? b : b * 100;
      if (v >= 90) return "#0053d6";
      if (v >= 70) return "#65cbf3";
      if (v >= 50) return "#ffdb13";
      return "#ff7d45";
    }}

    function cartoonStyle() {{
      if (controls.color.value === "plddt") {{
        return {{ cartoon: {{ colorfunc: atom => plddtColor(atom.b || 0) }} }};
      }}
      if (controls.color.value === "chain") {{
        return {{ cartoon: {{ colorscheme: "chain" }} }};
      }}
      return {{ cartoon: {{ color: "spectrum" }} }};
    }}

    function applyStyles() {{
      if (!viewer) return;
      viewer.setStyle({{}}, {{}});
      viewer.setStyle({{ hetflag: false }}, cartoonStyle());
      if (controls.sidechains.checked) {{
        viewer.addStyle(
          {{ hetflag: false, not: {{ atom: ["N", "CA", "C", "O"] }} }},
          {{ stick: {{ radius: 0.13 }} }}
        );
      }}
      if (controls.ligand.checked) {{
        viewer.setStyle(
          {{ hetflag: true, not: {{ resn: ["HOH", "WAT"] }} }},
          {{ stick: {{ radius: 0.22 }}, sphere: {{ scale: 0.32 }} }}
        );
      }}
      if (controls.surface.checked) {{
        viewer.addSurface($3Dmol.SurfaceType.VDW, {{ opacity: 0.18, color: "white" }}, {{ hetflag: false }});
      }}
      if (controls.water.checked) {{
        viewer.setStyle({{ resn: ["HOH", "WAT"] }}, {{ sphere: {{ radius: 0.35, color: "#9ca3af" }} }});
      }}
      viewer.render();
    }}

    function loadStructure(index) {{
      if (typeof $3Dmol === "undefined") {{
        throw new Error("3Dmol.js is unavailable");
      }}
      const item = payload.structures[index];
      currentItem = item;
      for (const button of listEl.querySelectorAll("button")) {{
        button.setAttribute("aria-current", button.dataset.index === String(index));
      }}
      pathEl.textContent = item.path || "";
      renderMetadata(item);
      viewport.innerHTML = "";
      viewer = $3Dmol.createViewer(viewport, {{ backgroundColor: "white" }});
      const fmtRaw = (item.format || "pdb").toLowerCase();
      const fmt = fmtRaw === "mmcif" ? "cif" : (fmtRaw === "ent" ? "pdb" : fmtRaw);
      viewer.addModel(item.content, fmt);
      applyStyles();
      viewer.zoomTo();
      viewer.render();
      try {{ viewer.zoom(1.08, 300); }} catch (e) {{}}
      statusEl.textContent = `${{item.label || "structure"}} · drag to rotate · scroll to zoom`;
    }}

    function buildStructureList() {{
      listEl.innerHTML = "";
      payload.structures.forEach((item, index) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.className = "structure-button";
        button.dataset.index = String(index);
        const label = document.createElement("span");
        label.className = "label";
        label.textContent = item.label || `Structure ${{index + 1}}`;
        const meta = document.createElement("span");
        meta.className = "meta";
        const metadata = item.metadata || {{}};
        meta.textContent = `${{metadata.format || item.format || "structure"}} · ${{metadata.residue_count ?? "?"}} residues`;
        button.append(label, meta);
        button.addEventListener("click", () => {{
          try {{ loadStructure(index); }} catch (error) {{ showError(error); }}
        }});
        listEl.appendChild(button);
      }});
    }}

    for (const control of Object.values(controls)) {{
      control.addEventListener("change", applyStyles);
    }}
    document.getElementById("center").addEventListener("click", () => {{
      if (!viewer) return;
      viewer.zoomTo();
      viewer.render();
    }});
    document.getElementById("spin").addEventListener("click", () => {{
      if (!viewer) return;
      spinning = !spinning;
      viewer.spin(spinning);
      document.getElementById("spin").textContent = spinning ? "Stop" : "Spin";
    }});
    document.getElementById("snapshot").addEventListener("click", () => {{
      if (!viewer) return;
      viewer.pngURI(uri => {{
        const link = document.createElement("a");
        link.href = uri;
        link.download = "vpa-structure.png";
        link.click();
      }});
    }});
    window.addEventListener("resize", () => {{
      if (!viewer) return;
      try {{ viewer.resize(); viewer.render(); }} catch (e) {{}}
    }});

    buildStructureList();
    try {{ loadStructure(0); }} catch (error) {{ showError(error); }}
  </script>
</body>
</html>
"""




def _render_offline_html(payload: dict[str, Any]) -> str:
    title = str(payload.get("title") or "VPA protein viewer")
    payload_json = _json_for_script(payload)
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #626863;
      --line: #d9ddd4;
      --panel: #ffffff;
      --accent: #246b5a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      letter-spacing: 0;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      min-height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 18px;
      overflow: auto;
    }}
    main {{
      position: relative;
      min-height: 100vh;
      background: #fdfdfa;
    }}
    canvas {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
    }}
    h1 {{
      margin: 0 0 4px;
      font-size: 20px;
      line-height: 1.2;
    }}
    .subtitle {{
      margin: 0 0 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }}
    .section {{
      border-top: 1px solid var(--line);
      padding-top: 14px;
      margin-top: 14px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .structure-list, .controls {{
      display: grid;
      gap: 8px;
    }}
    .structure-button {{
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: #fbfbf8;
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 10px;
      cursor: pointer;
      font: inherit;
      min-height: 42px;
    }}
    .structure-button[aria-current="true"] {{
      border-color: var(--accent);
      box-shadow: inset 3px 0 0 var(--accent);
      background: #eef7f3;
    }}
    .label {{
      display: block;
      font-weight: 650;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .meta {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }}
    label {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--ink);
      font-size: 13px;
    }}
    input[type="checkbox"] {{ accent-color: var(--accent); }}
    dl {{
      display: grid;
      grid-template-columns: minmax(95px, auto) minmax(0, 1fr);
      gap: 6px 10px;
      margin: 0;
      font-size: 13px;
    }}
    dt {{ color: var(--muted); }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      max-height: 280px;
      overflow: auto;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfbf8;
      font-size: 12px;
      line-height: 1.45;
    }}
    .toolbar {{
      position: absolute;
      top: 14px;
      right: 14px;
      display: flex;
      gap: 8px;
      z-index: 10;
    }}
    .toolbar button {{
      border: 1px solid rgba(32, 33, 36, 0.22);
      background: rgba(255, 255, 255, 0.92);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
      font-size: 13px;
    }}
    .status {{
      position: absolute;
      left: 16px;
      bottom: 14px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(32, 33, 36, 0.18);
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 13px;
      color: var(--muted);
      max-width: min(620px, calc(100vw - 32px));
    }}
    @media (max-width: 840px) {{
      .shell {{
        grid-template-columns: 1fr;
        grid-template-rows: auto minmax(520px, 70vh);
      }}
      aside {{
        border-right: 0;
        border-bottom: 1px solid var(--line);
        max-height: 46vh;
      }}
      main {{ min-height: 520px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1 id="title"></h1>
      <p class="subtitle" id="path"></p>
      <div class="section">
        <h2>Structures</h2>
        <div class="structure-list" id="structures"></div>
      </div>
      <div class="section">
        <h2>Display</h2>
        <div class="controls">
          <label><input type="checkbox" id="backbone" checked> Backbone trace</label>
          <label><input type="checkbox" id="atoms" checked> Atoms</label>
          <label><input type="checkbox" id="ligand" checked> Ligand / hetero atoms</label>
          <label><input type="checkbox" id="water"> Water</label>
        </div>
      </div>
      <div class="section">
        <h2>Current Structure</h2>
        <dl id="metadata"></dl>
      </div>
    </aside>
    <main>
      <canvas id="viewport"></canvas>
      <div class="toolbar">
        <button type="button" id="center">Center</button>
        <button type="button" id="spin">Spin</button>
        <button type="button" id="snapshot">Snapshot</button>
      </div>
      <div class="status" id="status">Loading offline viewer...</div>
    </main>
  </div>
  <script>
    const VPA_OFFLINE_VIEWER = true;
    const payload = {payload_json};
    const canvas = document.getElementById("viewport");
    const ctx = canvas.getContext("2d");
    const titleEl = document.getElementById("title");
    const pathEl = document.getElementById("path");
    const listEl = document.getElementById("structures");
    const metaEl = document.getElementById("metadata");
    const statusEl = document.getElementById("status");
    let current = null;
    let rotation = {{ x: -0.45, y: 0.72 }};
    let zoom = 1.0;
    let dragging = false;
    let last = {{ x: 0, y: 0 }};
    let spinning = false;

    titleEl.textContent = payload.title || "VPA protein viewer";

    function tokenizeCif(line) {{
      const out = [];
      let token = "";
      let quote = null;
      for (let i = 0; i < line.length; i++) {{
        const ch = line[i];
        if (quote) {{
          if (ch === quote) {{
            quote = null;
          }} else {{
            token += ch;
          }}
        }} else if (ch === "'" || ch === '"') {{
          quote = ch;
        }} else if (/\\s/.test(ch)) {{
          if (token) {{
            out.push(token);
            token = "";
          }}
        }} else {{
          token += ch;
        }}
      }}
      if (token) out.push(token);
      return out;
    }}

    function parsePdb(text) {{
      const atoms = [];
      for (const line of text.split(/\\r?\\n/)) {{
        const rec = line.slice(0, 6).trim();
        if (rec !== "ATOM" && rec !== "HETATM") continue;
        const atom = line.slice(12, 16).trim();
        const res = line.slice(17, 20).trim();
        const chain = line.slice(21, 22).trim() || "_";
        const seq = parseInt(line.slice(22, 26).trim(), 10) || 0;
        const x = parseFloat(line.slice(30, 38));
        const y = parseFloat(line.slice(38, 46));
        const z = parseFloat(line.slice(46, 54));
        if (![x, y, z].every(Number.isFinite)) continue;
        const elem = (line.slice(76, 78).trim() || atom.replace(/[0-9]/g, "").slice(0, 1));
        atoms.push({{ rec, atom, res, chain, seq, x, y, z, elem: elem.toUpperCase() }});
      }}
      return atoms;
    }}

    function parseCif(text) {{
      const lines = text.split(/\\r?\\n/);
      const atoms = [];
      for (let i = 0; i < lines.length; i++) {{
        if (lines[i].trim() !== "loop_") continue;
        const headers = [];
        let j = i + 1;
        while (j < lines.length && lines[j].trim().startsWith("_atom_site.")) {{
          headers.push(lines[j].trim().replace("_atom_site.", ""));
          j++;
        }}
        if (!headers.includes("Cartn_x")) continue;
        const idx = Object.fromEntries(headers.map((h, n) => [h, n]));
        while (j < lines.length) {{
          const line = lines[j].trim();
          if (!line || line === "#" || line.startsWith("_") || line === "loop_") break;
          const row = tokenizeCif(line);
          const x = parseFloat(row[idx.Cartn_x]);
          const y = parseFloat(row[idx.Cartn_y]);
          const z = parseFloat(row[idx.Cartn_z]);
          if ([x, y, z].every(Number.isFinite)) {{
            atoms.push({{
              rec: row[idx.group_PDB] || "ATOM",
              atom: row[idx.label_atom_id] || row[idx.auth_atom_id] || "",
              res: row[idx.auth_comp_id] || row[idx.label_comp_id] || "",
              chain: row[idx.auth_asym_id] || row[idx.label_asym_id] || "_",
              seq: parseInt(row[idx.auth_seq_id] || row[idx.label_seq_id] || "0", 10) || 0,
              x, y, z,
              elem: (row[idx.type_symbol] || "").toUpperCase()
            }});
          }}
          j++;
        }}
      }}
      return atoms;
    }}

    function parseStructure(item) {{
      const atoms = item.format === "cif" ? parseCif(item.content) : parsePdb(item.content);
      if (!atoms.length) throw new Error("No atoms parsed from " + item.label);
      const center = atoms.reduce((acc, a) => {{
        acc.x += a.x; acc.y += a.y; acc.z += a.z; return acc;
      }}, {{ x: 0, y: 0, z: 0 }});
      center.x /= atoms.length; center.y /= atoms.length; center.z /= atoms.length;
      for (const atom of atoms) {{
        atom.x -= center.x; atom.y -= center.y; atom.z -= center.z;
      }}
      const backbone = atoms.filter(a => a.atom === "CA" || a.atom === "P");
      return {{ item, atoms, backbone }};
    }}

    function color(atom) {{
      const elem = atom.elem || "";
      if (atom.rec === "HETATM" && atom.res !== "HOH") return "#9a5a12";
      if (elem === "O") return "#d5432f";
      if (elem === "N") return "#2f62c8";
      if (elem === "S") return "#caa21b";
      if (elem === "P") return "#8a55c2";
      if (elem === "H") return "#c8c8c8";
      return "#3d8a72";
    }}

    function projected(atom) {{
      const cy = Math.cos(rotation.y), sy = Math.sin(rotation.y);
      const cx = Math.cos(rotation.x), sx = Math.sin(rotation.x);
      let x = atom.x * cy - atom.z * sy;
      let z = atom.x * sy + atom.z * cy;
      let y = atom.y * cx - z * sx;
      z = atom.y * sx + z * cx;
      const scale = Math.min(canvas.width, canvas.height) / 95 * zoom;
      return {{
        x: canvas.width / 2 + x * scale,
        y: canvas.height / 2 - y * scale,
        z,
      }};
    }}

    function shouldDrawAtom(atom) {{
      const hetero = atom.rec === "HETATM";
      const water = atom.res === "HOH" || atom.res === "WAT";
      if (water) return document.getElementById("water").checked;
      if (hetero) return document.getElementById("ligand").checked;
      return document.getElementById("atoms").checked;
    }}

    function draw() {{
      if (!current) return;
      resize();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fdfdfa";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (document.getElementById("backbone").checked) {{
        const byChain = new Map();
        for (const atom of current.backbone) {{
          if (!byChain.has(atom.chain)) byChain.set(atom.chain, []);
          byChain.get(atom.chain).push(atom);
        }}
        ctx.lineWidth = 2.4;
        ctx.strokeStyle = "#246b5a";
        for (const chainAtoms of byChain.values()) {{
          chainAtoms.sort((a, b) => a.seq - b.seq);
          ctx.beginPath();
          chainAtoms.forEach((atom, idx) => {{
            const p = projected(atom);
            if (idx === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
          }});
          ctx.stroke();
        }}
      }}
      const visible = current.atoms
        .filter(shouldDrawAtom)
        .map(atom => ({{ atom, p: projected(atom) }}))
        .sort((a, b) => a.p.z - b.p.z);
      for (const item of visible) {{
        const r = item.atom.rec === "HETATM" ? 4.2 : 2.6;
        ctx.beginPath();
        ctx.arc(item.p.x, item.p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = color(item.atom);
        ctx.fill();
      }}
      statusEl.textContent = `${{current.item.label}} · ${{current.atoms.length}} atoms`;
    }}

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.floor(rect.width * ratio));
      const h = Math.max(1, Math.floor(rect.height * ratio));
      if (canvas.width !== w || canvas.height !== h) {{
        canvas.width = w;
        canvas.height = h;
      }}
    }}

    function renderMetadata(item) {{
      pathEl.textContent = item.path || "";
      metaEl.innerHTML = "";
      const metadata = item.metadata || {{}};
      const keys = [
        "file_name", "format", "chain_count", "chains", "residue_count",
        "hetero_residue_count", "atom_count", "size_bytes", "parse_warning"
      ];
      for (const key of keys) {{
        if (!(key in metadata)) continue;
        const dt = document.createElement("dt");
        dt.textContent = key.replaceAll("_", " ");
        const dd = document.createElement("dd");
        const value = metadata[key];
        dd.textContent = Array.isArray(value) ? value.join(", ") : String(value);
        metaEl.append(dt, dd);
      }}
    }}

    function loadStructure(index) {{
      const item = payload.structures[index];
      for (const button of listEl.querySelectorAll("button")) {{
        button.setAttribute("aria-current", button.dataset.index === String(index));
      }}
      renderMetadata(item);
      current = parseStructure(item);
      zoom = 1.0;
      draw();
    }}

    payload.structures.forEach((item, index) => {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "structure-button";
      button.dataset.index = String(index);
      button.innerHTML = `<span class="label"></span><span class="meta"></span>`;
      button.querySelector(".label").textContent = item.label || `Structure ${{index + 1}}`;
      const metadata = item.metadata || {{}};
      const fmt = metadata.format || item.format || "structure";
      const residues = metadata.residue_count ?? "?";
      button.querySelector(".meta").textContent = `${{fmt}} · ${{residues}} residues`;
      button.addEventListener("click", () => {{
        try {{ loadStructure(index); }} catch (error) {{ showError(error); }}
      }});
      listEl.appendChild(button);
    }});

    function showError(error) {{
      statusEl.textContent = "Viewer error: " + (error && error.message ? error.message : error);
      console.error(error);
    }}

    for (const id of ["backbone", "atoms", "ligand", "water"]) {{
      document.getElementById(id).addEventListener("change", draw);
    }}
    canvas.addEventListener("mousedown", event => {{
      dragging = true;
      last = {{ x: event.clientX, y: event.clientY }};
    }});
    window.addEventListener("mouseup", () => {{ dragging = false; }});
    window.addEventListener("mousemove", event => {{
      if (!dragging) return;
      rotation.y += (event.clientX - last.x) * 0.01;
      rotation.x += (event.clientY - last.y) * 0.01;
      last = {{ x: event.clientX, y: event.clientY }};
      draw();
    }});
    canvas.addEventListener("wheel", event => {{
      event.preventDefault();
      zoom *= event.deltaY > 0 ? 0.9 : 1.1;
      zoom = Math.max(0.2, Math.min(8, zoom));
      draw();
    }}, {{ passive: false }});
    document.getElementById("center").addEventListener("click", () => {{
      rotation = {{ x: -0.45, y: 0.72 }};
      zoom = 1.0;
      draw();
    }});
    document.getElementById("spin").addEventListener("click", () => {{
      spinning = !spinning;
      document.getElementById("spin").textContent = spinning ? "Stop" : "Spin";
    }});
    document.getElementById("snapshot").addEventListener("click", () => {{
      const link = document.createElement("a");
      link.href = canvas.toDataURL("image/png");
      link.download = "vpa-structure.png";
      link.click();
    }});
    function tick() {{
      if (spinning) {{
        rotation.y += 0.012;
        draw();
      }}
      requestAnimationFrame(tick);
    }}
    window.addEventListener("resize", draw);
    try {{ loadStructure(0); }} catch (error) {{ showError(error); }}
    tick();
  </script>
</body>
</html>
"""

from __future__ import annotations

import csv
import difflib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters._remote.fasta import FastaRecord, parse_fasta, write_fasta
from vibeproteinagent.adapters._remote.temporal import before_cutoff
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext, null_context
from vibeproteinagent.adapters.foldseek.adapter import FoldseekAdapter, FoldseekInput
from vibeproteinagent.adapters.mmseqs2_gpu.adapter import MMseqs2GpuAdapter, MMseqs2GpuInput


def _read_first_fasta(path: str) -> FastaRecord:
    records = parse_fasta(Path(path).expanduser().read_text())
    if not records:
        raise ValueError(f"no FASTA records found in {path}")
    return records[0]


def _clean_sequence(seq: str) -> str:
    return "".join(seq.split()).upper()


def _similarity(query: str, target: str) -> tuple[float, float]:
    """Return local-alignment identity and query coverage.

    The local FASTA fallback should tolerate shifted domains and terminal
    insertions. For normal template sizes this uses a compact Smith-Waterman
    dynamic program that tracks matches, aligned query residues, and alignment
    length. Very large comparisons fall back to difflib matching blocks to keep
    a local metadata search bounded and deterministic.
    """

    if not query or not target:
        return 0.0, 0.0
    max_cells = 2_000_000
    if len(query) * len(target) > max_cells:
        matcher = difflib.SequenceMatcher(a=query, b=target, autojunk=False)
        matches = sum(block.size for block in matcher.get_matching_blocks())
        coverage = matches / len(query)
        span = max(matches, min(len(query), len(target)))
        identity = matches / span if span else 0.0
        return identity, coverage

    # State tuple: score, matches, aligned query residues, aligned columns.
    prev: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * (len(target) + 1)
    best = (0, 0, 0, 0)
    for q in query:
        cur: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)]
        for j, t in enumerate(target, start=1):
            diag = prev[j - 1]
            diag_state = (
                diag[0] + (2 if q == t else -1),
                diag[1] + (1 if q == t else 0),
                diag[2] + 1,
                diag[3] + 1,
            )
            up = prev[j]
            up_state = (up[0] - 1, up[1], up[2] + 1, up[3] + 1)
            left = cur[j - 1]
            left_state = (left[0] - 1, left[1], left[2], left[3] + 1)
            state = max(
                (diag_state, up_state, left_state, (0, 0, 0, 0)),
                key=lambda item: (item[0], item[1], -item[3]),
            )
            cur.append(state)
            if (state[0], state[1], -state[3]) > (best[0], best[1], -best[3]):
                best = state
        prev = cur
    if best[3] == 0:
        return 0.0, 0.0
    identity = best[1] / best[3]
    coverage = best[2] / len(query)
    return identity, coverage


def _identity(value: str | float) -> float:
    ident = float(value)
    return ident / 100.0 if ident > 1.0 else ident


def _metadata(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    p = Path(path).expanduser()
    out: dict[str, dict[str, Any]] = {}
    if p.suffix.lower() == ".jsonl":
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tid = str(row.get("template_id") or row.get("id") or row.get("accession") or "")
            if tid:
                out[tid] = row
    else:
        with p.open() as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                tid = str(row.get("template_id") or row.get("id") or row.get("accession") or "")
                if tid:
                    out[tid] = dict(row)
    return out


def _release_date(row: dict[str, Any]) -> str | None:
    for key in ("release_date", "first_public_date", "date", "created"):
        if row.get(key):
            return str(row[key])
    return None


def _template_id(raw: str) -> str:
    return raw.split("|")[1] if "|" in raw else raw


def _write_query_fasta(out_dir: Path, params: TemplateSelectInput) -> tuple[Path, str, str]:
    if params.query_sequence:
        query_seq = _clean_sequence(params.query_sequence)
        query_kind = "sequence"
        query_path = out_dir / "query.fasta"
        write_fasta([FastaRecord("query", query_seq, "query")], query_path)
        return query_path, query_seq, query_kind
    if params.query_fasta:
        query_record = _read_first_fasta(params.query_fasta)
        return (
            Path(params.query_fasta).expanduser(),
            _clean_sequence(query_record.sequence),
            "fasta",
        )
    raise ValueError("query_sequence or query_fasta is required")


def _parse_mmseqs_hits(path: Path, query_len: int) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 3:
            continue
        alnlen = int(float(cols[3])) if len(cols) > 3 and cols[3] else query_len
        evalue = float(cols[10]) if len(cols) > 10 and cols[10] else 1.0
        hits.append({
            "template_id": _template_id(cols[1]),
            "identity": round(_identity(cols[2]), 6),
            "coverage": round(min(1.0, alnlen / max(1, query_len)), 6),
            "evalue": evalue,
            "source": "mmseqs",
            "description": cols[1],
            "length": None,
        })
    return hits


def _structure_residue_count(path: str | None) -> int | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.is_file():
        return None
    suffix = p.suffix.lower()
    residues: set[tuple[str, str, str]] = set()
    if suffix in {".cif", ".mmcif"}:
        # A lightweight count is enough for coverage. Structure analysis owns
        # full mmCIF parsing; this only needs stable residue cardinality.
        text = p.read_text(errors="ignore")
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 9 and parts[0] in {"ATOM", "HETATM"}:
                chain = parts[6] if len(parts) > 6 else ""
                seq = parts[8] if len(parts) > 8 else ""
                comp = parts[5] if len(parts) > 5 else ""
                residues.add((chain, seq, comp))
    else:
        for line in p.read_text(errors="ignore").splitlines():
            if line.startswith(("ATOM  ", "HETATM")) and line[12:16].strip() == "CA":
                residues.add((line[21].strip(), line[22:26].strip(), line[17:20].strip()))
    return len(residues) or None


def _parse_foldseek_hits(path: Path, query_len: int | None) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        alnlen = int(float(cols[3])) if cols[3] else 0
        hits.append({
            "template_id": _template_id(cols[1]),
            "identity": round(_identity(cols[2]), 6),
            "coverage": round(min(1.0, alnlen / query_len), 6) if query_len else 1.0,
            "evalue": float(cols[4]),
            "source": "foldseek",
            "description": cols[1],
            "length": alnlen or None,
        })
    return hits


def _template_records_for_ids(
    records: list[FastaRecord],
    selected_ids: set[str],
) -> list[FastaRecord]:
    return [r for r in records if _template_id(r.identifier) in selected_ids]


class TemplateSelectInput(BaseModel):
    query_sequence: str | None = None
    query_fasta: str | None = Field(None, json_schema_extra={"vpa:digest": "sha256-if-path"})
    query_structure: str | None = Field(None, json_schema_extra={"vpa:digest": "sha256-if-path"})
    template_db: str = Field(min_length=1, json_schema_extra={"vpa:digest": "sha256-if-path"})
    template_db_kind: str = Field("fasta", pattern=r"^(fasta|mmseqs|foldseek)$")
    max_templates: int = Field(50, ge=1, le=1000)
    min_identity: float = Field(0.2, ge=0, le=1)
    min_coverage: float = Field(0.5, ge=0, le=1)
    max_evalue: float = Field(0.001, gt=0)
    temporal_cutoff: str | None = None
    metadata_table: str | None = Field(None, json_schema_extra={"vpa:digest": "sha256-if-path"})

    @model_validator(mode="after")
    def _has_query(self) -> TemplateSelectInput:
        if not self.query_sequence and not self.query_fasta and not self.query_structure:
            raise ValueError("one of query_sequence, query_fasta, or query_structure is required")
        if self.template_db_kind in {"fasta", "mmseqs"} and self.query_structure:
            raise ValueError(
                "fasta/mmseqs template selection requires query_sequence or query_fasta"
            )
        if self.template_db_kind == "foldseek" and not self.query_structure:
            raise ValueError("foldseek template selection requires query_structure")
        return self


class TemplateSelectOutput(BaseModel):
    query_kind: str
    engine: str
    template_count: int
    templates_json: str
    templates_tsv: str
    top_templates: list[dict[str, Any]]
    artifacts: dict[str, str | None]
    runtime_ms: int


@register_adapter("descriptor.yaml")
class TemplateSelectAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: TemplateSelectInput, ctx: AdapterRunContext) -> TemplateSelectOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = _metadata(params.metadata_table)
        template_db_path = Path(params.template_db).expanduser()
        records: list[FastaRecord] = []
        if (
            params.template_db_kind in {"fasta", "mmseqs"}
            and template_db_path.is_file()
        ):
            records = parse_fasta(template_db_path.read_text())

        if params.template_db_kind == "fasta":
            query_path, query_seq, query_kind = _write_query_fasta(out_dir, params)
            del query_path
            engine = "local_fasta"
            raw_hits: list[dict[str, Any]] = []
            for record in records:
                identity, coverage = _similarity(query_seq, _clean_sequence(record.sequence))
                evalue = max(1e-300, (1.0 - identity + 1e-6) / max(1, len(query_seq)) ** 2)
                raw_hits.append({
                    "template_id": _template_id(record.identifier),
                    "identity": round(identity, 6),
                    "coverage": round(coverage, 6),
                    "evalue": evalue,
                    "source": "local_fasta",
                    "description": record.description,
                    "length": len(record.sequence),
                })
        elif params.template_db_kind == "mmseqs":
            query_path, query_seq, query_kind = _write_query_fasta(out_dir, params)
            engine = "mmseqs2_gpu"
            search_limit = min(50000, max(params.max_templates * 20, params.max_templates + 200))
            mmseqs_ctx = null_context(ctx.job_dir)
            mmseqs_out = MMseqs2GpuAdapter({}, Path(), self.device).run(
                MMseqs2GpuInput(
                    query_fasta=str(query_path),
                    target_db=params.template_db,
                    min_seq_id=params.min_identity,
                    coverage=params.min_coverage,
                    max_seqs=search_limit,
                ),
                mmseqs_ctx,
            )
            raw_hits = _parse_mmseqs_hits(Path(mmseqs_out.hits_tsv), len(query_seq))
        else:
            query_kind = "structure"
            engine = "foldseek"
            search_limit = min(50000, max(params.max_templates * 20, params.max_templates + 200))
            foldseek_ctx = null_context(ctx.job_dir)
            foldseek_out = FoldseekAdapter({}, Path(), self.device).run(
                FoldseekInput(
                    query_pdb=str(params.query_structure),
                    target_db=params.template_db,
                    e_value=params.max_evalue,
                    max_seqs=search_limit,
                ),
                foldseek_ctx,
            )
            raw_hits = _parse_foldseek_hits(
                Path(foldseek_out.hits_tsv),
                _structure_residue_count(params.query_structure),
            )

        ranked = []
        dropped_temporal = 0
        for hit in raw_hits:
            tid = str(hit["template_id"])
            m = meta.get(tid, {})
            release = _release_date(m)
            if not before_cutoff(
                release,
                params.temporal_cutoff,
                include_undated=params.temporal_cutoff is None,
            ):
                dropped_temporal += 1
                continue
            if hit["identity"] < params.min_identity or hit["coverage"] < params.min_coverage:
                continue
            if float(hit["evalue"]) > params.max_evalue:
                continue
            row = dict(hit)
            row["release_date"] = release
            row["source"] = m.get("source") or row.get("source") or params.template_db_kind
            ranked.append(row)
        ranked.sort(
            key=lambda row: (
                -row["identity"],
                -row["coverage"],
                row["evalue"],
                row["template_id"],
            )
        )
        selected = ranked[: params.max_templates]
        json_path = out_dir / "templates.json"
        tsv_path = out_dir / "templates.tsv"
        fasta_path = out_dir / "template_sequences.fasta"
        summary_path = out_dir / "template_select_summary.json"
        json_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        with tsv_path.open("w") as fh:
            fh.write("template_id\tidentity\tcoverage\tevalue\trelease_date\tsource\tlength\n")
            for row in selected:
                fh.write(
                    f"{row['template_id']}\t{row['identity']}\t{row['coverage']}\t"
                    f"{row['evalue']}\t{row.get('release_date') or ''}\t"
                    f"{row['source']}\t{row['length']}\n"
                )
        selected_ids = {row["template_id"] for row in selected}
        selected_records = _template_records_for_ids(records, selected_ids)
        template_sequences_fasta: str | None = None
        if selected_records:
            write_fasta(selected_records, fasta_path)
            template_sequences_fasta = str(fasta_path)
        summary_path.write_text(
            json.dumps(
                {
                    "dropped_by_temporal_filter": dropped_temporal,
                    "scanned_templates": len(records),
                    "raw_hit_count": len(raw_hits),
                    "template_sequences_fasta_available": template_sequences_fasta is not None,
                    "engine": engine,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        ctx.emit_progress(1.0)
        return TemplateSelectOutput(
            query_kind=query_kind,
            engine=engine,
            template_count=len(selected),
            templates_json=str(json_path),
            templates_tsv=str(tsv_path),
            top_templates=selected[:10],
            artifacts={
                "template_sequences_fasta": template_sequences_fasta,
                "summary_json": str(summary_path),
            },
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: TemplateSelectInput) -> int:
        return 0


__all__ = ["TemplateSelectAdapter", "TemplateSelectInput", "TemplateSelectOutput"]

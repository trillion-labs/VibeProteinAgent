from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


class ArtifactSummarizer:
    """Compact local artifact summaries suitable for LLM result-boundary review."""

    max_text_bytes = 16_384

    def summarize_path(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        if not p.exists():
            return {"path": str(p), "exists": False}
        if p.is_dir():
            return self._summarize_dir(p)
        suffix = p.suffix.lower()
        if suffix in {".json"}:
            return self._summarize_json(p)
        if suffix in {".csv", ".tsv"}:
            return self._summarize_table(p)
        if suffix in {".fa", ".fasta", ".faa"}:
            return self._summarize_fasta(p)
        if suffix in {".pdb", ".cif"}:
            return self._summarize_structure(p)
        return {"path": str(p), "kind": "file", "size_bytes": p.stat().st_size}

    def _summarize_dir(self, path: Path) -> dict[str, Any]:
        files = [p for p in path.rglob("*") if p.is_file()]
        return {
            "path": str(path),
            "kind": "directory",
            "file_count": len(files),
            "size_bytes": sum(p.stat().st_size for p in files),
            "examples": [str(p.relative_to(path)) for p in files[:10]],
        }

    def _summarize_json(self, path: Path) -> dict[str, Any]:
        try:
            obj = json.loads(path.read_text()[: self.max_text_bytes])
        except Exception as exc:
            return {"path": str(path), "kind": "json", "error": str(exc)}
        if isinstance(obj, dict):
            selected = {
                k: v for k, v in obj.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            return {
                "path": str(path),
                "kind": "json",
                "keys": sorted(obj)[:50],
                "selected_fields": selected,
            }
        if isinstance(obj, list):
            return {
                "path": str(path),
                "kind": "json",
                "items": len(obj),
                "first_items": obj[:5],
            }
        return {"path": str(path), "kind": "json", "value_type": type(obj).__name__}

    def _summarize_table(self, path: Path) -> dict[str, Any]:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        lines = path.read_text(errors="replace").splitlines()
        if not lines:
            return {"path": str(path), "kind": "table", "rows": 0}
        headers = lines[0].split(delimiter)
        rows = [line.split(delimiter) for line in lines[1:101]]
        numeric: dict[str, list[float]] = {}
        for row in rows:
            for idx, value in enumerate(row[: len(headers)]):
                try:
                    numeric.setdefault(headers[idx], []).append(float(value))
                except ValueError:
                    pass
        distributions = {
            key: {"min": min(vals), "max": max(vals), "mean": mean(vals)}
            for key, vals in numeric.items()
            if vals
        }
        return {
            "path": str(path),
            "kind": "table",
            "columns": headers,
            "rows_sampled": len(rows),
            "top_rows": [dict(zip(headers, row, strict=False)) for row in rows[:10]],
            "numeric": distributions,
        }

    def _summarize_fasta(self, path: Path) -> dict[str, Any]:
        count = 0
        lengths: list[int] = []
        current = 0
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(">"):
                if count:
                    lengths.append(current)
                count += 1
                current = 0
            else:
                current += len(line.strip())
        if count:
            lengths.append(current)
        return {
            "path": str(path),
            "kind": "fasta",
            "sequence_count": count,
            "length_min": min(lengths) if lengths else 0,
            "length_max": max(lengths) if lengths else 0,
            "length_mean": mean(lengths) if lengths else 0,
        }

    def _summarize_structure(self, path: Path) -> dict[str, Any]:
        chains: set[str] = set()
        residues: set[tuple[str, str]] = set()
        atoms = 0
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith(("ATOM", "HETATM")):
                atoms += 1
                chain = line[21:22].strip() or "_"
                res = line[22:27].strip()
                chains.add(chain)
                residues.add((chain, res))
        return {
            "path": str(path),
            "kind": path.suffix.lower().lstrip("."),
            "atom_count": atoms,
            "chain_count": len(chains),
            "chains": sorted(chains),
            "residue_count": len(residues),
        }

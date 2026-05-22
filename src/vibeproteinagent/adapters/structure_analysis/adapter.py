from __future__ import annotations

import json
import math
import shlex
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.dssp.adapter import _parse_dssp_ss
from vibeproteinagent.core.binary import find_binary

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
HYDROPHOBIC = frozenset("AILMFWVY")
BURIED_POLAR = frozenset("DENQKRHSTYC")
_DSSP_BINARY_CANDIDATES = ("mkdssp", "dssp")
_HELIX_LABELS = frozenset("HGI")
_SHEET_LABELS = frozenset("EB")


@dataclass
class Residue:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    aa: str
    atoms: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    het: bool = False

    @property
    def key(self) -> tuple[str, int, str]:
        return self.chain_id, self.resseq, self.icode

    @property
    def ca(self) -> tuple[float, float, float] | None:
        return self.atoms.get("CA") or next(iter(self.atoms.values()), None)


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _parse_pdb(path: Path) -> tuple[list[Residue], list[Residue]]:
    residues: dict[tuple[str, int, str, str], Residue] = {}
    ligands: dict[tuple[str, int, str, str], Residue] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
            continue
        atom = line[12:16].strip()
        resname = line[17:20].strip().upper()
        chain = line[21:22].strip() or "_"
        try:
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        icode = line[26:27].strip()
        is_ligand = line.startswith("HETATM") and resname not in {"HOH", "WAT", "DOD"}
        aa = AA3.get(resname, "X")
        key = (chain, resseq, icode, resname)
        bucket = ligands if is_ligand or aa == "X" else residues
        residue = bucket.setdefault(
            key, Residue(chain, resseq, icode, resname, aa, het=is_ligand)
        )
        residue.atoms[atom] = (x, y, z)
    return list(residues.values()), list(ligands.values())


def _cif_value(parts: list[str], index: dict[str, int], key: str, default: str = "") -> str:
    pos = index.get(key)
    if pos is None or pos >= len(parts):
        return default
    value = parts[pos]
    return default if value in {".", "?"} else value


def _parse_mmcif(path: Path) -> tuple[list[Residue], list[Residue]]:
    residues: dict[tuple[str, int, str, str], Residue] = {}
    ligands: dict[tuple[str, int, str, str], Residue] = {}
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if not headers or not all(h.startswith("_atom_site.") for h in headers):
            while i < len(lines) and lines[i].strip() != "#":
                i += 1
            continue
        index = {header: n for n, header in enumerate(headers)}
        needed = [
            "_atom_site.group_PDB",
            "_atom_site.label_atom_id",
            "_atom_site.label_comp_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]
        if not all(k in index for k in needed):
            continue
        while i < len(lines):
            raw = lines[i].strip()
            if not raw or raw == "#" or raw == "loop_" or raw.startswith("_"):
                break
            parts = shlex.split(raw)
            if len(parts) < len(headers):
                i += 1
                continue
            group = _cif_value(parts, index, "_atom_site.group_PDB").upper()
            model = _cif_value(parts, index, "_atom_site.pdbx_PDB_model_num", "1")
            alt_id = _cif_value(parts, index, "_atom_site.label_alt_id", ".")
            if model not in {"1", ".", "?"} or alt_id not in {".", "?", "A", "1"}:
                i += 1
                continue
            atom = _cif_value(parts, index, "_atom_site.auth_atom_id") or _cif_value(
                parts, index, "_atom_site.label_atom_id"
            )
            resname = (
                _cif_value(parts, index, "_atom_site.auth_comp_id")
                or _cif_value(parts, index, "_atom_site.label_comp_id")
            ).upper()
            chain = (
                _cif_value(parts, index, "_atom_site.auth_asym_id")
                or _cif_value(parts, index, "_atom_site.label_asym_id")
                or "_"
            )
            seq = _cif_value(parts, index, "_atom_site.auth_seq_id") or _cif_value(
                parts, index, "_atom_site.label_seq_id"
            )
            try:
                resseq = int(seq)
                x = float(parts[index["_atom_site.Cartn_x"]])
                y = float(parts[index["_atom_site.Cartn_y"]])
                z = float(parts[index["_atom_site.Cartn_z"]])
            except (ValueError, KeyError):
                i += 1
                continue
            icode = _cif_value(parts, index, "_atom_site.pdbx_PDB_ins_code", "")
            aa = AA3.get(resname, "X")
            is_ligand = group == "HETATM" and resname not in {"HOH", "WAT", "DOD"}
            bucket = ligands if is_ligand or (group == "HETATM" and aa == "X") else residues
            key = (chain, resseq, icode, resname)
            residue = bucket.setdefault(
                key, Residue(chain, resseq, icode, resname, aa, het=is_ligand)
            )
            residue.atoms[atom] = (x, y, z)
            i += 1
    return list(residues.values()), list(ligands.values())


def _parse_structure(path: Path) -> tuple[list[Residue], list[Residue]]:
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return _parse_mmcif(path)
    return _parse_pdb(path)


def _burial_from_neighbors(
    n: int, *, buried_threshold: float = 0.10, surface_threshold: float = 0.25
) -> tuple[float, str]:
    rsa = max(0.0, min(1.0, 1.0 - n / 18.0))
    if rsa <= buried_threshold:
        return rsa, "buried"
    if rsa >= surface_threshold:
        return rsa, "exposed"
    return rsa, "intermediate"


def _secondary_structure(
    coords: list[tuple[Residue, tuple[float, float, float] | None]],
) -> dict[tuple[str, int, str], str]:
    ss = {residue.key: "loop" for residue, _ in coords}
    by_chain: dict[str, list[tuple[Residue, tuple[float, float, float]]]] = defaultdict(list)
    for residue, ca in coords:
        if ca is not None:
            by_chain[residue.chain_id].append((residue, ca))
    for chain_coords in by_chain.values():
        for i, (_residue, ca) in enumerate(chain_coords):
            if i + 3 < len(chain_coords) and _dist(ca, chain_coords[i + 3][1]) <= 6.4:
                for j in range(i, i + 4):
                    ss[chain_coords[j][0].key] = "helix"
        for i in range(1, len(chain_coords) - 1):
            residue, _ = chain_coords[i]
            if ss[residue.key] == "helix":
                continue
            if _dist(chain_coords[i - 1][1], chain_coords[i + 1][1]) >= 6.2:
                ss[residue.key] = "sheet"
    return ss


def _ss_label(code: str) -> str:
    if code in _HELIX_LABELS:
        return "helix"
    if code in _SHEET_LABELS:
        return "sheet"
    return "loop"


def _run_dssp(
    structure_path: Path,
    coords: list[tuple[Residue, tuple[float, float, float] | None]],
    out_dir: Path,
    output_prefix: str,
) -> tuple[dict[tuple[str, int, str], str] | None, Path | None, str | None]:
    try:
        binary = find_binary(_DSSP_BINARY_CANDIDATES, env_override="VPA_DSSP_BINARY")
    except FileNotFoundError as exc:
        return None, None, str(exc)

    classic_path = out_dir / f"{output_prefix}.dssp"
    try:
        subprocess.run(
            [
                str(binary),
                "--output-format",
                "dssp",
                str(structure_path),
                str(classic_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return None, None, f"DSSP execution failed: {exc}"

    ss_string = _parse_dssp_ss(classic_path.read_text())
    if not ss_string:
        return None, classic_path, "DSSP output did not contain residue annotations"
    mapped: dict[tuple[str, int, str], str] = {}
    for (residue, _ca), code in zip(coords, ss_string, strict=False):
        mapped[residue.key] = _ss_label(code)
    return mapped, classic_path, None


def _run_pydssp(
    residues: list[Residue],
) -> tuple[dict[tuple[str, int, str], str] | None, str | None]:
    try:
        import numpy as np
        import pydssp
    except Exception as exc:
        return None, f"pydssp unavailable: {exc}"

    keys: list[tuple[str, int, str]] = []
    coords: list[list[tuple[float, float, float]]] = []
    for residue in residues:
        if all(atom in residue.atoms for atom in ("N", "CA", "C", "O")):
            keys.append(residue.key)
            coords.append([residue.atoms[atom] for atom in ("N", "CA", "C", "O")])
    if not coords:
        return None, "pydssp requires residues with N, CA, C, and O atoms"

    try:
        ss_codes = pydssp.assign(np.array(coords, dtype=float), out_type="c3")
    except Exception as exc:
        return None, f"pydssp assignment failed: {exc}"
    return {key: _ss_label(str(code)) for key, code in zip(keys, ss_codes, strict=True)}, None


def _patches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    last_chain: str | None = None
    last_resseq: int | None = None
    for row in rows:
        is_patch = row["burial_class"] == "exposed" and row["aa"] in HYDROPHOBIC
        contiguous = (
            last_chain == row["chain_id"]
            and last_resseq is not None
            and row["resseq"] == last_resseq + 1
        )
        if is_patch and (not current or contiguous):
            current.append(row)
        else:
            if len(current) >= 2:
                patches.append({
                    "chain_id": current[0]["chain_id"],
                    "positions": [r["resseq"] for r in current],
                    "size": len(current),
                })
            current = [row] if is_patch else []
        last_chain, last_resseq = row["chain_id"], row["resseq"]
    if len(current) >= 2:
        patches.append({
            "chain_id": current[0]["chain_id"],
            "positions": [r["resseq"] for r in current],
            "size": len(current),
        })
    return patches


class StructureAnalysisInput(BaseModel):
    structure_path: str = Field(min_length=1, json_schema_extra={"vpa:digest": "sha256-if-path"})
    sequence: str | None = None
    chain_id: str | None = None
    interface_chain_ids: list[str] = Field(default_factory=list)
    ligand_resnames: list[str] = Field(default_factory=list)
    positions: list[int] = Field(default_factory=list)
    max_annotations: int = Field(200, ge=1, le=10000)
    contact_distance_angstrom: float = Field(5.0, gt=0, le=20)
    surface_rsa_threshold: float = Field(0.25, ge=0, le=1)
    buried_rsa_threshold: float = Field(0.10, ge=0, le=1)
    run_dssp: bool = True
    output_prefix: str = Field("structure_analysis", pattern=r"^[A-Za-z0-9_.-]+$")


class StructureAnalysisOutput(BaseModel):
    structure_path: str
    chains: list[str]
    residue_counts: dict[str, int]
    secondary_structure: dict[str, float]
    burial: dict[str, int]
    interface: dict[str, int]
    ligand_contacts: dict[str, int]
    liabilities: list[dict[str, Any]]
    annotation_scope: dict[str, Any]
    artifacts: dict[str, str]
    warnings: list[str]
    runtime_ms: int


@register_adapter("descriptor.yaml")
class StructureAnalysisAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: StructureAnalysisInput,
        ctx: AdapterRunContext,
    ) -> StructureAnalysisOutput:
        t0 = time.time()
        path = Path(params.structure_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"structure_analysis input not found: {path}")
        residues, ligands = _parse_structure(path)
        if params.chain_id:
            residues = [r for r in residues if r.chain_id == params.chain_id]
        if not residues:
            raise ValueError("structure_analysis found no protein residues")
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        ctx.emit_progress(0.25)
        coords = [(r, r.ca) for r in residues if r.ca is not None]
        dssp_source_path: Path | None = None
        ss_source = "ca_geometry"
        warnings: list[str] = []
        if params.run_dssp:
            ss_by_key, dssp_source_path, dssp_warning = _run_dssp(
                path,
                coords,
                out_dir,
                params.output_prefix,
            )
            if ss_by_key is None:
                pydssp_by_key, pydssp_warning = _run_pydssp(residues)
                if pydssp_by_key is not None:
                    ss_by_key = pydssp_by_key
                    ss_source = "pydssp"
                    if dssp_warning:
                        warnings.append(f"mkdssp unavailable; used pydssp: {dssp_warning}")
                else:
                    ss_by_key = _secondary_structure(coords)
                    if dssp_warning:
                        warnings.append(
                            f"DSSP unavailable; used CA-geometry heuristic: {dssp_warning}"
                        )
                    if pydssp_warning:
                        warnings.append(pydssp_warning)
            else:
                ss_source = "dssp"
                warnings.append("secondary_structure assigned from DSSP")
        else:
            ss_by_key = _secondary_structure(coords)
            warnings.append("run_dssp=false; secondary_structure uses CA-geometry heuristic")
        all_rows: list[dict[str, Any]] = []
        for residue, ca in coords:
            neighbors = 0
            if ca is not None:
                neighbors = sum(
                    1
                    for other, other_ca in coords
                    if other is not residue
                    and other_ca is not None
                    and _dist(ca, other_ca) <= 10.0
                )
            rsa, burial = _burial_from_neighbors(
                neighbors,
                buried_threshold=params.buried_rsa_threshold,
                surface_threshold=params.surface_rsa_threshold,
            )
            ss = ss_by_key.get(residue.key, "loop")
            row = {
                "chain_id": residue.chain_id,
                "resseq": residue.resseq,
                "icode": residue.icode,
                "aa": residue.aa,
                "resname": residue.resname,
                "rsa": round(rsa, 4),
                "burial_class": burial,
                "secondary_structure": ss,
                "neighbor_count": neighbors,
            }
            all_rows.append(row)
        wanted = set(params.positions) if params.positions else None
        rows = [
            row for row in all_rows
            if wanted is None or int(row["resseq"]) in wanted
        ]
        if wanted and len(rows) != len(wanted):
            observed = {int(row["resseq"]) for row in rows}
            missing_positions = sorted(wanted - observed)
            warnings.append(
                "requested positions missing from parsed structure: "
                + ",".join(str(pos) for pos in missing_positions)
            )
        if params.sequence:
            parsed_sequence = "".join(r.aa for r in residues if r.aa != "X")
            provided = "".join(params.sequence.split()).upper()
            if len(provided) != len(parsed_sequence):
                warnings.append(
                    "provided sequence length does not match parsed structure residues: "
                    f"sequence={len(provided)}, structure={len(parsed_sequence)}"
                )
            elif provided != parsed_sequence:
                warnings.append("provided sequence differs from parsed structure sequence")
        ctx.emit_progress(0.55)
        contacts: list[dict[str, Any]] = []
        interface_set: set[tuple[str, int, str]] = set()
        interface_chains = set(params.interface_chain_ids)
        for r, ca in coords:
            if ca is None:
                continue
            for other, other_ca in coords:
                if other is r or other_ca is None:
                    continue
                if (
                    interface_chains
                    and other.chain_id not in interface_chains
                    and r.chain_id not in interface_chains
                ):
                    continue
                if r.chain_id == other.chain_id:
                    continue
                d = _dist(ca, other_ca)
                if d <= params.contact_distance_angstrom:
                    contacts.append({
                        "chain_id": r.chain_id,
                        "resseq": r.resseq,
                        "other_chain_id": other.chain_id,
                        "other_resseq": other.resseq,
                        "distance": round(d, 3),
                    })
                    interface_set.add(r.key)
        ligand_names = {x.upper() for x in params.ligand_resnames}
        ligand_contacts = []
        for r, ca in coords:
            if ca is None:
                continue
            for lig in ligands:
                if ligand_names and lig.resname not in ligand_names:
                    continue
                points = list(lig.atoms.values())
                if (
                    points
                    and min(_dist(ca, p) for p in points) <= params.contact_distance_angstrom
                ):
                    ligand_contacts.append({
                        "chain_id": r.chain_id,
                        "resseq": r.resseq,
                        "ligand_resname": lig.resname,
                        "ligand_chain_id": lig.chain_id,
                        "ligand_resseq": lig.resseq,
                    })
        patches = _patches(all_rows)
        liabilities: list[dict[str, Any]] = []
        liabilities.extend(
            {
                "type": "exposed_hydrophobic_patch",
                "chain_id": p["chain_id"],
                "positions": p["positions"],
            }
            for p in patches
        )
        for row in all_rows:
            if row["burial_class"] == "buried" and row["aa"] in BURIED_POLAR:
                liabilities.append({
                    "type": "buried_polar_candidate",
                    "chain_id": row["chain_id"],
                    "positions": [row["resseq"]],
                })
        burial_counts = Counter(row["burial_class"] for row in all_rows)
        chains = sorted({r.chain_id for r in residues})
        ss_counts = Counter(row["secondary_structure"] for row in all_rows)
        n_rows = max(1, len(all_rows))
        annotation_scope = {
            "summary_scope": "all_residues_after_chain_filter",
            "annotation_scope": (
                "requested_positions" if wanted else "all_residues_after_chain_filter"
            ),
            "requested_positions": sorted(wanted) if wanted else [],
            "annotated_residue_count": len(rows),
            "summary_residue_count": len(all_rows),
        }
        summary = {
            "structure_path": str(path),
            "chains": chains,
            "residue_counts": dict(Counter(r.chain_id for r in residues)),
            "secondary_structure": {
                "helix": ss_counts["helix"],
                "sheet": ss_counts["sheet"],
                "loop": ss_counts["loop"],
                "helix_fraction": ss_counts["helix"] / n_rows,
                "sheet_fraction": ss_counts["sheet"] / n_rows,
                "loop_fraction": ss_counts["loop"] / n_rows,
            },
            "burial": dict(burial_counts),
            "interface": {
                "contact_count": len(contacts),
                "interface_residue_count": len(interface_set),
                "ligand_contact_count": len(ligand_contacts),
            },
            "ligand_contacts": {
                "contact_count": len(ligand_contacts),
                "contact_residue_count": len({
                    (c["chain_id"], c["resseq"]) for c in ligand_contacts
                }),
            },
            "liabilities": liabilities[: params.max_annotations],
            "secondary_structure_source": ss_source,
            "annotation_scope": annotation_scope,
        }
        summary_path = out_dir / f"{params.output_prefix}_summary.json"
        residue_path = out_dir / f"{params.output_prefix}_residue_structure_annotations.tsv"
        contacts_path = out_dir / f"{params.output_prefix}_interface_contacts.tsv"
        patches_path = out_dir / f"{params.output_prefix}_surface_patches.json"
        dssp_path = out_dir / f"{params.output_prefix}_dssp.tsv"
        summary["artifacts"] = {
            "summary_json": str(summary_path),
            "residue_annotations_tsv": str(residue_path),
            "interface_contacts_tsv": str(contacts_path),
            "surface_patches_json": str(patches_path),
            "dssp_tsv": str(dssp_path),
        }
        if dssp_source_path is not None and dssp_source_path != dssp_path:
            summary["artifacts"]["raw_dssp"] = str(dssp_source_path)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        with residue_path.open("w") as fh:
            fh.write("chain_id\tresseq\ticode\taa\tresname\trsa\tburial_class\tsecondary_structure\tneighbor_count\n")
            for row in rows:
                keys = (
                    "chain_id",
                    "resseq",
                    "icode",
                    "aa",
                    "resname",
                    "rsa",
                    "burial_class",
                    "secondary_structure",
                    "neighbor_count",
                )
                fh.write("\t".join(str(row[k]) for k in keys) + "\n")
        with contacts_path.open("w") as fh:
            fh.write("chain_id\tresseq\tother_chain_id\tother_resseq\tdistance\n")
            for c in contacts:
                fh.write(f"{c['chain_id']}\t{c['resseq']}\t{c['other_chain_id']}\t{c['other_resseq']}\t{c['distance']}\n")
        patches_path.write_text(json.dumps(patches, indent=2, sort_keys=True) + "\n")
        with dssp_path.open("w") as fh:
            fh.write("chain_id\tresseq\ticode\taa\tsecondary_structure\n")
            for row in rows:
                fh.write(
                    f"{row['chain_id']}\t{row['resseq']}\t{row['icode']}\t"
                    f"{row['aa']}\t{row['secondary_structure']}\n"
                )
        ctx.emit_progress(1.0)
        return StructureAnalysisOutput(
            structure_path=str(path),
            chains=chains,
            residue_counts=summary["residue_counts"],
            secondary_structure=summary["secondary_structure"],
            burial={
                "buried": burial_counts["buried"],
                "intermediate": burial_counts["intermediate"],
                "exposed": burial_counts["exposed"],
            },
            interface=summary["interface"],
            ligand_contacts=summary["ligand_contacts"],
            liabilities=liabilities[: params.max_annotations],
            annotation_scope=annotation_scope,
            artifacts=summary["artifacts"],
            warnings=warnings,
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: StructureAnalysisInput) -> int:
        return 0


__all__ = [
    "StructureAnalysisAdapter",
    "StructureAnalysisInput",
    "StructureAnalysisOutput",
    "_parse_structure",
]

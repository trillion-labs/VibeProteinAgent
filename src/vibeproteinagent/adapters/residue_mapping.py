from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

PositionBasis = Literal["auto", "sequence", "structure"]

AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class ResidueRef:
    chain_id: str
    resseq: int
    icode: str = ""
    aa: str = "X"

    @property
    def key(self) -> tuple[str, int, str]:
        return self.chain_id, self.resseq, self.icode

    @property
    def residue_id(self) -> str:
        return f"{self.chain_id}{self.resseq}{self.icode}"


@dataclass
class PositionResolution:
    residue_ids: list[str]
    mapped_positions: list[dict[str, Any]]
    missing_positions: list[int]
    position_index_basis: str
    warnings: list[str]

    @property
    def residue_ids_csv(self) -> str:
        return ",".join(self.residue_ids)

    def to_audit(self) -> dict[str, Any]:
        return {
            "position_index_basis": self.position_index_basis,
            "mapped_positions": self.mapped_positions,
            "missing_positions": self.missing_positions,
            "residue_ids": self.residue_ids,
        }


def clean_sequence(sequence: str) -> str:
    lines = []
    for line in sequence.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    return "".join(ch for ch in "".join(lines).upper() if ch.isalpha())


def read_fasta_sequence(path: str | None) -> str | None:
    if path is None:
        return None
    fasta_path = Path(path).expanduser()
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    return clean_sequence(fasta_path.read_text())


def resolve_reference_sequence(
    sequence: str | None = None,
    fasta_path: str | None = None,
) -> str | None:
    if sequence:
        return clean_sequence(sequence)
    return read_fasta_sequence(fasta_path)


def residue_refs_from_objects(residues: Sequence[Any]) -> list[ResidueRef]:
    refs: list[ResidueRef] = []
    for residue in residues:
        aa = str(getattr(residue, "aa", "X") or "X")
        if aa == "X":
            continue
        refs.append(
            ResidueRef(
                chain_id=str(getattr(residue, "chain_id")),
                resseq=int(getattr(residue, "resseq")),
                icode=str(getattr(residue, "icode", "") or ""),
                aa=aa,
            )
        )
    return refs


def parse_structure_residue_refs(path: Path, chain_id: str | None = None) -> list[ResidueRef]:
    if path.suffix.lower() in {".cif", ".mmcif"}:
        refs = _parse_mmcif_residue_refs(path)
    else:
        refs = _parse_pdb_residue_refs(path)
    if chain_id:
        refs = [ref for ref in refs if ref.chain_id == chain_id]
    return refs


def resolve_positions(
    *,
    residues: Sequence[ResidueRef],
    positions: Sequence[int],
    chain_id: str,
    sequence: str | None = None,
    fasta_path: str | None = None,
    position_basis: PositionBasis = "auto",
    explicit_residue_ids: Sequence[str] | None = None,
    label: str = "positions",
) -> PositionResolution:
    refs = list(residues)
    residue_by_key = {ref.key: ref for ref in refs}
    warnings: list[str] = []
    residue_ids: list[str] = []
    mapped_positions: list[dict[str, Any]] = []
    missing_positions: list[int] = []
    explicit_ids = [normalize_residue_id(item, default_chain=chain_id) for item in explicit_residue_ids or []]
    sequence_text = resolve_reference_sequence(sequence, fasta_path)
    basis: Literal["sequence", "structure"]

    if position_basis == "sequence":
        if not sequence_text:
            raise ValueError(f"{label} requested sequence basis but no sequence/fasta was provided")
        basis = "sequence"
    elif position_basis == "structure":
        basis = "structure"
    elif sequence_text:
        basis = "sequence"
    else:
        basis = "structure"
        if positions:
            warnings.append(
                f"{label} interpreted as structure residue numbers because no sequence/fasta was provided"
            )

    if basis == "sequence":
        assert sequence_text is not None
        seq_map, map_basis = sequence_position_map(sequence_text, refs)
        for pos in sorted(set(int(p) for p in positions)):
            residue = seq_map.get(pos)
            if residue is None or residue.key not in residue_by_key:
                missing_positions.append(pos)
                continue
            residue_ids.append(residue.residue_id)
            mapped_positions.append(_mapping_entry(pos, residue))
        position_index_basis = map_basis
        if missing_positions:
            warnings.append(
                f"{label} missing from parsed structure after sequence alignment: "
                + ",".join(str(pos) for pos in missing_positions)
            )
    else:
        by_resseq = {(ref.chain_id, ref.resseq): ref for ref in refs}
        for pos in sorted(set(int(p) for p in positions)):
            residue = by_resseq.get((chain_id, pos))
            if residue is None:
                missing_positions.append(pos)
                residue_ids.append(f"{chain_id}{pos}")
                continue
            residue_ids.append(residue.residue_id)
            mapped_positions.append(_mapping_entry(pos, residue))
        position_index_basis = "structure_resseq"
        if missing_positions:
            warnings.append(
                f"{label} missing from parsed structure: "
                + ",".join(str(pos) for pos in missing_positions)
            )

    residue_ids.extend(explicit_ids)
    residue_ids = _unique_preserve_order(residue_ids)
    return PositionResolution(
        residue_ids=residue_ids,
        mapped_positions=mapped_positions,
        missing_positions=missing_positions,
        position_index_basis=position_index_basis,
        warnings=warnings,
    )


def resolve_position_range(
    *,
    residues: Sequence[ResidueRef],
    range_text: str,
    chain_id: str,
    sequence: str | None = None,
    fasta_path: str | None = None,
    position_basis: PositionBasis = "auto",
    label: str = "target_residues",
) -> tuple[str, PositionResolution]:
    start_s, sep, end_s = str(range_text).partition("-")
    if sep != "-":
        raise ValueError(f"{label} must be an inclusive range like 3-117")
    start, end = int(start_s), int(end_s)
    if end < start:
        raise ValueError(f"{label} range end must be >= start")
    sequence_text = resolve_reference_sequence(sequence, fasta_path)
    if position_basis == "sequence" and not sequence_text:
        raise ValueError(f"{label} requested sequence basis but no sequence/fasta was provided")
    if position_basis == "structure" or (position_basis == "auto" and not sequence_text):
        warnings = []
        if position_basis == "auto":
            warnings.append(
                f"{label} interpreted as a structure residue range because no sequence/fasta was provided"
            )
        return f"{chain_id}{start}-{end}", PositionResolution(
            residue_ids=[],
            mapped_positions=[],
            missing_positions=[],
            position_index_basis="structure_resseq",
            warnings=warnings,
        )
    resolution = resolve_positions(
        residues=residues,
        positions=list(range(start, end + 1)),
        chain_id=chain_id,
        sequence=sequence_text,
        position_basis="sequence",
        label=label,
    )
    mapped = [
        ref
        for ref in (parse_residue_id(item, default_chain=chain_id) for item in resolution.residue_ids)
        if ref.chain_id == chain_id
    ]
    ranges = _contiguous_numeric_ranges(mapped)
    if not ranges:
        raise ValueError(f"{label} did not map to any structure residues")
    return "/".join(
        f"{chain_id}{lo}" if lo == hi else f"{chain_id}{lo}-{hi}"
        for lo, hi in ranges
    ), resolution


def sequence_position_map(
    provided_sequence: str,
    residues: Sequence[ResidueRef],
) -> tuple[dict[int, ResidueRef], str]:
    provided = clean_sequence(provided_sequence)
    structure_residues = [ref for ref in residues if ref.aa != "X"]
    structure_sequence = "".join(ref.aa for ref in structure_residues)
    if not provided or not structure_sequence:
        return {}, "sequence_alignment"

    offset = provided.find(structure_sequence)
    if offset >= 0:
        return {
            offset + i + 1: residue
            for i, residue in enumerate(structure_residues)
        }, "sequence_alignment_substring"

    match = 2
    mismatch = -1
    gap = -2
    n = len(provided)
    m = len(structure_sequence)
    prev = [j * gap for j in range(m + 1)]
    trace: list[bytearray] = [bytearray([2] * (m + 1)) for _ in range(n + 1)]
    trace[0][0] = 0
    for i in range(1, n + 1):
        curr = [i * gap] + [0] * m
        trace[i][0] = 1
        ai = provided[i - 1]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (match if ai == structure_sequence[j - 1] else mismatch)
            up = prev[j] + gap
            left = curr[j - 1] + gap
            if diag >= up and diag >= left:
                curr[j] = diag
                trace[i][j] = 0
            elif up >= left:
                curr[j] = up
                trace[i][j] = 1
            else:
                curr[j] = left
                trace[i][j] = 2
        prev = curr

    position_map: dict[int, ResidueRef] = {}
    i, j = n, m
    while i > 0 or j > 0:
        direction = trace[i][j]
        if i > 0 and j > 0 and direction == 0:
            position_map[i] = structure_residues[j - 1]
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or direction == 1):
            i -= 1
        else:
            j -= 1
    return position_map, "sequence_alignment"


def normalize_residue_id(value: str, *, default_chain: str) -> str:
    return parse_residue_id(value, default_chain=default_chain).residue_id


def parse_residue_id(value: str, *, default_chain: str) -> ResidueRef:
    text = str(value).strip()
    if not text:
        raise ValueError("empty residue id")
    chain = default_chain
    rest = text
    if text[0].isalpha() or text[0] == "_":
        chain = text[0]
        rest = text[1:]
    digits = []
    icode = []
    for ch in rest:
        if ch.isdigit() or (ch == "-" and not digits):
            digits.append(ch)
        else:
            icode.append(ch)
    if not digits:
        raise ValueError(f"residue id {value!r} does not contain a residue number")
    return ResidueRef(chain_id=chain, resseq=int("".join(digits)), icode="".join(icode))


def _mapping_entry(requested_position: int, residue: ResidueRef) -> dict[str, Any]:
    return {
        "requested_position": requested_position,
        "chain_id": residue.chain_id,
        "resseq": residue.resseq,
        "icode": residue.icode,
        "aa": residue.aa,
        "residue_id": residue.residue_id,
    }


def _unique_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _contiguous_numeric_ranges(residues: Sequence[ResidueRef]) -> list[tuple[int, int]]:
    numbers = sorted({ref.resseq for ref in residues if not ref.icode})
    if not numbers:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = previous = number
    ranges.append((start, previous))
    return ranges


def _parse_pdb_residue_refs(path: Path) -> list[ResidueRef]:
    refs: dict[tuple[str, int, str], ResidueRef] = {}
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        resname = line[17:20].strip().upper()
        aa = AA3.get(resname, "X")
        if aa == "X":
            continue
        chain = line[21:22].strip() or "_"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        icode = line[26:27].strip()
        refs.setdefault((chain, resseq, icode), ResidueRef(chain, resseq, icode, aa))
    return list(refs.values())


def _parse_mmcif_residue_refs(path: Path) -> list[ResidueRef]:
    refs: dict[tuple[str, int, str], ResidueRef] = {}
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
            "_atom_site.label_comp_id",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
        ]
        if not all(key in index for key in needed):
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
            if group != "ATOM":
                i += 1
                continue
            resname = (
                _cif_value(parts, index, "_atom_site.auth_comp_id")
                or _cif_value(parts, index, "_atom_site.label_comp_id")
            ).upper()
            aa = AA3.get(resname, "X")
            if aa == "X":
                i += 1
                continue
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
            except ValueError:
                i += 1
                continue
            icode = _cif_value(parts, index, "_atom_site.pdbx_PDB_ins_code", "")
            refs.setdefault((chain, resseq, icode), ResidueRef(chain, resseq, icode, aa))
            i += 1
    return list(refs.values())


def _cif_value(parts: list[str], index: dict[str, int], key: str, default: str = "") -> str:
    pos = index.get(key)
    if pos is None or pos >= len(parts):
        return default
    value = parts[pos]
    return default if value in {".", "?"} else value

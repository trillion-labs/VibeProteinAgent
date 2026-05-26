from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_VALID_AA = set("ABCDEFGHIKLMNPQRSTVWXYZUO-*")


@dataclass(frozen=True, slots=True)
class FastaRecord:
    identifier: str
    sequence: str
    description: str = ""


def clean_sequence(sequence: str) -> str:
    return "".join(ch for ch in sequence.upper() if ch in _VALID_AA)


def parse_fasta(text: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    seq: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                ident = header.split()[0]
                records.append(FastaRecord(ident, "".join(seq), header))
            header = line[1:]
            seq = []
        else:
            seq.append(line)
    if header is not None:
        ident = header.split()[0]
        records.append(FastaRecord(ident, "".join(seq), header))
    return records


def write_fasta(records: list[FastaRecord], path: str | Path) -> None:
    lines: list[str] = []
    for record in records:
        header = record.description or record.identifier
        lines.append(f">{header}")
        sequence = clean_sequence(record.sequence)
        lines.extend(sequence[i : i + 80] for i in range(0, len(sequence), 80))
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def deduplicate_fasta_records(records: list[FastaRecord]) -> tuple[list[FastaRecord], int]:
    seen: set[str] = set()
    kept: list[FastaRecord] = []
    dropped = 0
    for record in records:
        sequence = clean_sequence(record.sequence)
        if sequence in seen:
            dropped += 1
            continue
        seen.add(sequence)
        kept.append(FastaRecord(record.identifier, sequence, record.description))
    return kept, dropped

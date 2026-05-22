from __future__ import annotations

import urllib.parse
from dataclasses import dataclass
from typing import Any

from .fasta import FastaRecord, parse_fasta
from .http import HttpClient

UNIPROT = "https://rest.uniprot.org"


@dataclass(frozen=True)
class UniProtRecord:
    accession: str
    sequence: str
    first_public_date: str | None
    reviewed: bool | None
    length: int | None
    metadata: dict[str, Any]


def first_public_date(data: dict[str, Any]) -> str | None:
    try:
        return data["results"][0]["entryAudit"]["firstPublicDate"]
    except Exception:
        pass
    audit = data.get("entryAudit") if isinstance(data, dict) else None
    if isinstance(audit, dict):
        value = audit.get("firstPublicDate")
        return str(value) if value else None
    return None


def accession_from_entry(data: dict[str, Any]) -> str | None:
    primary = data.get("primaryAccession")
    if primary:
        return str(primary)
    results = data.get("results")
    if isinstance(results, list) and results:
        return accession_from_entry(results[0])
    return None


def sequence_from_entry(data: dict[str, Any]) -> str:
    seq = data.get("sequence")
    if isinstance(seq, dict) and seq.get("value"):
        return str(seq["value"])
    return ""


def reviewed_from_entry(data: dict[str, Any]) -> bool | None:
    value = data.get("entryType")
    if isinstance(value, str):
        return "reviewed" in value.lower() or "swiss-prot" in value.lower()
    return None


def fetch_accession(client: HttpClient, accession: str) -> UniProtRecord | None:
    data = client.get_json(f"{UNIPROT}/uniprotkb/{urllib.parse.quote(accession)}.json")
    accession_id = accession_from_entry(data) or accession
    seq = sequence_from_entry(data)
    if not seq:
        fasta_text = client.get(f"{UNIPROT}/uniprotkb/{urllib.parse.quote(accession)}.fasta").text
        records = parse_fasta(fasta_text)
        seq = records[0].sequence if records else ""
    return UniProtRecord(
        accession=accession_id,
        sequence=seq,
        first_public_date=first_public_date(data),
        reviewed=reviewed_from_entry(data),
        length=len(seq) if seq else None,
        metadata=data,
    )


def search_accessions(client: HttpClient, query: str, *, size: int = 100) -> list[str]:
    data = client.get_json(
        f"{UNIPROT}/uniprotkb/search",
        params={"query": query, "fields": "accession", "format": "json", "size": size},
    )
    accessions: list[str] = []
    for item in data.get("results", []) if isinstance(data, dict) else []:
        acc = item.get("primaryAccession")
        if acc:
            accessions.append(str(acc))
    return accessions


def to_fasta_record(record: UniProtRecord) -> FastaRecord:
    return FastaRecord(
        identifier=record.accession,
        sequence=record.sequence,
        description=f"{record.accession} first_public_date={record.first_public_date or 'undated'}",
    )

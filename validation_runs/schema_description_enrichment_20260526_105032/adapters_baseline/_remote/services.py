from __future__ import annotations

import re
from typing import Any

from .dates import normalized_date

_UNIPROT_RE = re.compile(r"\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9]\b|\b[A-Z0-9]{10}\b")
_IPR_RE = re.compile(r"\bIPR\d{6}\b")
_GO_RE = re.compile(r"\bGO:\d{7}\b")


def normalize_uniprot_record(record: dict[str, Any]) -> dict[str, Any]:
    accession = record.get("primaryAccession") or record.get("accession")
    audit = record.get("entryAudit") or {}
    sequence_obj = record.get("sequence") or {}
    protein = record.get("proteinDescription") or {}
    recommended = protein.get("recommendedName") or {}
    organism = record.get("organism") or {}
    sequence = sequence_obj.get("value") or record.get("sequence") or ""
    reviewed = "Swiss-Prot" in str(record.get("entryType", ""))
    if record.get("reviewed") is not None:
        reviewed = bool(record.get("reviewed"))
    return {
        "accession": accession,
        "id": record.get("uniProtkbId") or record.get("id"),
        "protein_name": (
            ((recommended.get("fullName") or {}).get("value"))
            or record.get("protein_name")
            or accession
        ),
        "organism": (
            organism.get("scientificName")
            or organism.get("commonName")
            or record.get("organism")
        ),
        "taxon_id": (organism.get("taxonId") if isinstance(organism, dict) else None),
        "sequence": sequence,
        "length": int(sequence_obj.get("length") or len(sequence or "")),
        "reviewed": reviewed,
        "first_public_date": normalized_date(
            audit.get("firstPublicDate") or record.get("first_public_date")
        ),
        "raw": record,
    }


def normalize_europepmc_record(record: dict[str, Any]) -> dict[str, Any]:
    first_date = (
        record.get("firstPublicationDate")
        or record.get("firstIndexDate")
        or record.get("journalInfo", {}).get("printPublicationDate")
        or record.get("pubYear")
    )
    abstract = record.get("abstractText") or ""
    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "title": record.get("title") or "",
        "year": int(record["pubYear"]) if str(record.get("pubYear", "")).isdigit() else None,
        "doi": record.get("doi"),
        "pmid": record.get("pmid") or (record.get("id") if record.get("source") == "MED" else None),
        "first_publication_date": normalized_date(first_date),
        "abstract": abstract,
        "abstract_snippet": abstract[:500],
        "journal": record.get("journalTitle"),
        "is_open_access": str(record.get("isOpenAccess", "")).upper() == "Y",
        "raw": record,
    }


def extract_interpro_accessions(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    accessions: list[str] = []
    for item in payload.get("results", []) or []:
        metadata = item.get("metadata") or {}
        accession = metadata.get("accession") or item.get("accession")
        if accession:
            accessions.append(str(accession))
    next_url = payload.get("next") or payload.get("next_url")
    return accessions, next_url


def extract_go_accessions(association: dict[str, Any]) -> set[str]:
    candidates: set[str] = set()
    for key in ("bioentity", "bioentity_internal_id", "bioentity_label", "subject"):
        value = association.get(key)
        if isinstance(value, dict):
            value = value.get("id") or value.get("label")
        if not value:
            continue
        text = str(value)
        if "UniProtKB:" in text:
            candidates.add(text.rsplit(":", 1)[-1])
        for match in _UNIPROT_RE.findall(text):
            candidates.add(match)
    return candidates


def extract_constraint_terms(text: str) -> dict[str, list[str]]:
    return {
        "interpro_ids": sorted(set(_IPR_RE.findall(text))),
        "go_ids": sorted(set(_GO_RE.findall(text))),
        "uniprot_accessions": sorted(set(_UNIPROT_RE.findall(text))),
        "pdb_ids": sorted(set(re.findall(r"\b[0-9][A-Za-z0-9]{3}\b", text))),
        "temperatures_c": sorted(set(re.findall(r"\b\d{2,3}\s*(?:deg\s*)?C\b", text))),
        "ph_values": sorted(set(re.findall(r"\bpH\s*[0-9](?:\.[0-9])?\b", text, re.I))),
        "mutations": sorted(set(re.findall(r"\b[A-Z][0-9]{1,5}[A-Z]\b", text))),
    }

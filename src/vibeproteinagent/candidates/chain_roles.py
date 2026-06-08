from __future__ import annotations

from pathlib import Path
from typing import Any

from vibeproteinagent.adapters.residue_mapping import parse_structure_residue_refs


def chain_sequences_from_structure(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    try:
        refs = parse_structure_residue_refs(Path(path).expanduser())
    except Exception:
        return {}
    chains: dict[str, list[str]] = {}
    for ref in refs:
        if not ref.aa or ref.aa == "X":
            continue
        chains.setdefault(ref.chain_id, []).append(ref.aa)
    return {chain_id: "".join(sequence) for chain_id, sequence in chains.items()}


def chain_sequence_from_structure(
    path: str | Path | None,
    chain_id: str | None,
) -> str | None:
    if not chain_id:
        return None
    return chain_sequences_from_structure(path).get(str(chain_id))


def observed_chain_for_sequence(
    structure_path: str | Path | None,
    sequence: str | None,
    *,
    preferred_chain_id: str | None = None,
) -> str | None:
    observed = chain_sequences_from_structure(structure_path)
    if not observed or not sequence:
        return None
    preferred = str(preferred_chain_id) if preferred_chain_id else None
    if preferred and observed.get(preferred) == sequence:
        return preferred
    matches = [
        chain_id
        for chain_id, chain_sequence in observed.items()
        if chain_sequence == sequence
    ]
    return matches[0] if len(matches) == 1 else None


def chain_role_entry(
    *,
    role: str,
    sequence: str | None = None,
    chain_id: str | None = None,
    input_order: int | None = None,
    structure_path: str | Path | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"role": role}
    if chain_id:
        row["chain_id"] = str(chain_id)
    if input_order is not None:
        row["input_order"] = int(input_order)
    if sequence:
        row["sequence_length"] = len(sequence)
    observed = _observed_chain_id(
        sequence=sequence,
        chain_id=chain_id,
        structure_path=structure_path,
    )
    if observed:
        row["observed_chain_id"] = observed
    return row


def complex_chain_metadata(
    *,
    target_sequence: str | None = None,
    binder_sequence: str | None = None,
    structure_path: str | Path | None = None,
    target_chain_id: str | None = None,
    binder_chain_id: str | None = None,
    target_input_order: int | None = 0,
    binder_input_order: int | None = 1,
    target_role: str = "target",
    binder_role: str = "binder",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if target_sequence or target_chain_id is not None:
        rows.append(
            chain_role_entry(
                role=target_role,
                sequence=target_sequence,
                chain_id=target_chain_id,
                input_order=target_input_order,
                structure_path=structure_path,
            )
        )
    if binder_sequence or binder_chain_id is not None:
        rows.append(
            chain_role_entry(
                role=binder_role,
                sequence=binder_sequence,
                chain_id=binder_chain_id,
                input_order=binder_input_order,
                structure_path=structure_path,
            )
        )
    return rows


def interface_chain_pair(
    chain_a: str | None,
    chain_b: str | None,
    *,
    source: str,
) -> dict[str, str] | None:
    if not chain_a or not chain_b:
        return None
    return {"chain_a": str(chain_a), "chain_b": str(chain_b), "source": source}


def add_interface_chain_pair(
    payload: dict[str, Any],
    chain_a: str | None,
    chain_b: str | None,
    *,
    source: str,
) -> dict[str, Any]:
    pair = interface_chain_pair(chain_a, chain_b, source=source)
    if pair is None:
        return payload
    pairs = payload.setdefault("interface_chain_pairs", [])
    if isinstance(pairs, list) and pair not in pairs:
        pairs.append(pair)
    return payload


def _observed_chain_id(
    *,
    sequence: str | None,
    chain_id: str | None,
    structure_path: str | Path | None,
) -> str | None:
    observed = chain_sequences_from_structure(structure_path)
    if not observed:
        return None
    if not sequence:
        return str(chain_id) if chain_id and str(chain_id) in observed else None
    return observed_chain_for_sequence(
        structure_path,
        sequence,
        preferred_chain_id=chain_id,
    )


__all__ = [
    "add_interface_chain_pair",
    "chain_role_entry",
    "chain_sequence_from_structure",
    "chain_sequences_from_structure",
    "complex_chain_metadata",
    "interface_chain_pair",
    "observed_chain_for_sequence",
]

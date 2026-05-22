"""Vendored ProteinMPNN inference subset.

Mirrors `dauparas/ProteinMPNN/protein_mpnn_utils.py` verbatim. See
``vendor/proteinmpnn/UPSTREAM.md`` for the exact commit hash, URL and license.
"""

from __future__ import annotations

from .protein_mpnn_utils import (
    ProteinMPNN,
    StructureDataset,
    StructureDatasetPDB,
    _S_to_seq,
    _scores,
    parse_PDB,
    parse_PDB_biounits,
    tied_featurize,
)

__all__ = [
    "ProteinMPNN",
    "StructureDataset",
    "StructureDatasetPDB",
    "_S_to_seq",
    "_scores",
    "parse_PDB",
    "parse_PDB_biounits",
    "tied_featurize",
]

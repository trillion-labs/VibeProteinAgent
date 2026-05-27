"""Named compatibility shims used by adapter runtimes.

Every compatibility behavior in this module has an explicit registry entry so
future cleanup can remove shims deliberately instead of leaving hidden patches
inside adapters.
"""
from __future__ import annotations

import contextlib
import functools
import os
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompatEntry:
    name: str
    affected_adapter: str
    why: str
    activation_condition: str
    removal_condition: str
    test: str


_ENTRIES: dict[str, CompatEntry] = {
    "rdkit_allchem_descriptors": CompatEntry(
        name="rdkit_allchem_descriptors",
        affected_adapter="boltz2_affinity",
        why="Some Boltz-2 code expects AllChem.Descriptors on newer RDKit builds.",
        activation_condition="RDKit imports but rdkit.Chem.AllChem lacks Descriptors.",
        removal_condition="Upstream Boltz-2 stops reading AllChem.Descriptors.",
        test="tests/test_adapter_boltz2_affinity_unit.py",
    ),
    "dgl_graphbolt_stub": CompatEntry(
        name="dgl_graphbolt_stub",
        affected_adapter="rfpeptides",
        why="Some DGL CUDA wheels import GraphBolt even when RFPeptides does not use it.",
        activation_condition="Importing dgl fails because graphbolt shared library is absent.",
        removal_condition="DGL wheel ships compatible GraphBolt for the target CUDA stack.",
        test="tests/test_adapter_rfpeptides_unit.py",
    ),
    "torch_full_load": CompatEntry(
        name="torch_full_load",
        affected_adapter="legacy checkpoint adapters",
        why="torch>=2.6 defaults to weights_only=True, rejecting trusted local legacy checkpoints.",
        activation_condition=(
            "Adapter enters allow_full_torch_load() around trusted local checkpoints."
        ),
        removal_condition="All affected checkpoints are migrated to weights-only-safe formats.",
        test="tests/test_adapter_pxdesign_unit.py",
    ),
    "esm3_shared_hf_cache": CompatEntry(
        name="esm3_shared_hf_cache",
        affected_adapter="esm3",
        why="ESM3 gated model access should use the shared VPA Hugging Face cache by default.",
        activation_condition="HF_HOME is unset before ESM3 import or model load.",
        removal_condition="Descriptor/config grows an explicit first-class model-cache field.",
        test="tests/test_adapter_esm3_real.py",
    ),
    "pxdesign_fair_esm_stub": CompatEntry(
        name="pxdesign_fair_esm_stub",
        affected_adapter="pxdesign",
        why="PXDesign imports a narrow fair-esm namespace during Protenix compatibility setup.",
        activation_condition="fair-esm is absent but PXDesign only needs the imported namespace.",
        removal_condition="PXDesign removes the obsolete fair-esm import path.",
        test="tests/test_adapter_pxdesign_pipeline_unit.py",
    ),
}


def compat_entries() -> tuple[CompatEntry, ...]:
    return tuple(_ENTRIES.values())


def compat_entry(name: str) -> CompatEntry:
    return _ENTRIES[name]


def ensure_rdkit_allchem_descriptors() -> None:
    """Expose ``Descriptors`` through ``AllChem`` when RDKit omits that alias."""

    from rdkit.Chem import AllChem, Descriptors

    if not hasattr(AllChem, "Descriptors"):
        AllChem.Descriptors = Descriptors  # type: ignore[attr-defined]


def ensure_esm3_shared_hf_cache(default_cache: str) -> None:
    os.environ.setdefault("HF_HOME", default_cache)


def install_dgl_graphbolt_stub() -> None:
    """Install a tiny graphbolt placeholder for DGL builds missing the library."""

    graphbolt = types.ModuleType("dgl.graphbolt")
    graphbolt.__dict__["__all__"] = []
    sys.modules.setdefault("dgl.graphbolt", graphbolt)


def _silence_dgl_backend_banner() -> None:
    """Ensure DGL doesn't print "Setting the default backend to 'pytorch'..."
    to STDOUT on first import.

    DGL routes that banner through ``print()`` (stdout, not stderr) when
    ``DGLBACKEND`` is unset, which pollutes any per-process stdout we
    capture downstream — most visibly the matrix runner's per-case
    ``<case>.json`` files (stderr is already isolated to .stderr.log).
    Using ``setdefault`` preserves an operator's explicit override.
    """

    os.environ.setdefault("DGLBACKEND", "pytorch")


def import_dgl_with_graphbolt_guard() -> Any:
    """Import DGL, retrying once with the documented GraphBolt guard."""

    _silence_dgl_backend_banner()
    try:
        import dgl  # type: ignore

        return dgl
    except Exception as exc:
        msg = str(exc)
        if "graphbolt" not in msg.lower():
            raise
        install_dgl_graphbolt_stub()
        import dgl  # type: ignore

        return dgl


@contextlib.contextmanager
def allow_full_torch_load() -> Iterator[None]:
    """Temporarily force ``torch.load`` to use ``weights_only=False``."""

    import torch

    original = torch.load

    @functools.wraps(original)
    def _patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original  # type: ignore[assignment]


__all__ = [
    "CompatEntry",
    "allow_full_torch_load",
    "compat_entries",
    "compat_entry",
    "ensure_esm3_shared_hf_cache",
    "ensure_rdkit_allchem_descriptors",
    "import_dgl_with_graphbolt_guard",
    "install_dgl_graphbolt_stub",
]

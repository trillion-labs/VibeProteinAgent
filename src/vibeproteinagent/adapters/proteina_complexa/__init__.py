from __future__ import annotations

from vibeproteinagent.adapters import _proteina_complexa_common as common

_CHECKPOINTS = {
    "protein_binder": ("complexa.ckpt", "complexa_ae.ckpt"),
    "ligand_binder": ("complexa_ligand.ckpt", "complexa_ligand_ae.ckpt"),
    "motif": ("complexa_ame.ckpt", "complexa_ame_ae.ckpt"),
}


def availability_skip_reason() -> str | None:
    skip = common.availability_skip_reason()
    if skip is not None:
        return skip
    missing = [
        mode
        for mode, (ckpt, ae_ckpt) in _CHECKPOINTS.items()
        if not common.ckpt_pair_present(ckpt, ae_ckpt)
    ]
    if missing:
        mode_list = ", ".join(missing)
        return (
            "Proteina-Complexa unified tool is missing checkpoints for "
            f"{mode_list} under {common.proteina_complexa_ckpts()}."
        )
    return None


from .adapter import (  # noqa: E402
    ProteinaComplexaAdapter,
    ProteinaComplexaDesign,
    ProteinaComplexaInput,
    ProteinaComplexaOutput,
)

__all__ = [
    "ProteinaComplexaAdapter",
    "ProteinaComplexaDesign",
    "ProteinaComplexaInput",
    "ProteinaComplexaOutput",
    "availability_skip_reason",
]

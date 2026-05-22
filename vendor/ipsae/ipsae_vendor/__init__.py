"""Vendored DunbrackLab/IPSAE re-export shim.

`ipsae.py` upstream is a command-line script — it has no library API. The
VibeProteinAgent adapter invokes it as a subprocess with this directory on
``sys.path`` so the file is resolvable as ``ipsae_vendor.ipsae``.
"""
from __future__ import annotations

from pathlib import Path

IPSAE_SCRIPT: Path = Path(__file__).resolve().parent / "ipsae.py"

__all__ = ["IPSAE_SCRIPT"]

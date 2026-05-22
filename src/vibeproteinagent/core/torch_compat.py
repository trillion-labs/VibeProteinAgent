"""Compatibility helpers for working with newer torch versions.

Adapters that load legacy Lightning / PyTorch checkpoints frequently need to
override the post-2.6 ``torch.load(weights_only=True)`` default. This module
hosts the shared contextmanager so adapter packages can import it from one
place instead of carrying private copies.
"""
from __future__ import annotations

from collections.abc import Iterator

from vibeproteinagent.core.compat import allow_full_torch_load as _allow_full_torch_load


def allow_full_torch_load() -> Iterator[None]:
    """Temporarily force ``torch.load`` to use ``weights_only=False``."""
    return _allow_full_torch_load()


__all__ = ["allow_full_torch_load"]

from __future__ import annotations

import os
from pathlib import Path

from .adapter import ESM3Adapter, ESM3Input, ESM3Output, _ensure_shared_hf_cache_env, _hf_credential


def _open_small_cache_present() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", "")).expanduser()
    if not str(hf_home):
        return False
    cache = hf_home / "hub" / "models--EvolutionaryScale--esm3-sm-open-v1"
    return (cache / "snapshots").is_dir() and any((cache / "snapshots").iterdir())


def availability_skip_reason() -> str | None:
    _ensure_shared_hf_cache_env()
    if _open_small_cache_present():
        return None
    if _hf_credential():
        return None
    try:
        from huggingface_hub import get_token
    except Exception:
        get_token = None
    if get_token is not None and get_token():
        return None
    return (
        "ESM3 open-small is Hugging Face gated on this host. Set HF_TOKEN "
        "(or VPA_HF_CREDENTIAL) and accept the EvolutionaryScale/biohub model terms "
        "before promoting to VERIFIED."
    )


__all__ = ["ESM3Adapter", "ESM3Input", "ESM3Output", "availability_skip_reason"]

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from pydantic import BaseModel

from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.descriptor import ToolDescriptor, load_descriptor

try:
    import torch
except ImportError:  # pragma: no cover - exercised by bootstrap envs.
    class _Device:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return self.value

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        ),
        device=_Device,
    )

# Discovery path #1 sink: every @register_adapter-decorated class that gets imported
# enrolls itself here. The registry walks `vibeproteinagent.adapters` subpackages,
# imports each, and reads this dict.
_BUILTIN_ADAPTERS: dict[str, tuple[ToolDescriptor, type[ToolAdapter]]] = {}


class ToolAdapter(ABC):
    """Base class every tool adapter inherits from.

    Subclasses must:
      * be decorated with `@register_adapter("descriptor.yaml")` (path relative to
        the file the decorator is invoked in), which sets `descriptor` as a ClassVar.
      * implement `__init__(self, config, weights_dir, device)` and `run(self, params, ctx)`.

    The supervisor / worker is responsible for: weight resolution into `weights_dir`,
    cancellation polling, log fan-out, artifact registration. Adapters MUST NOT touch
    SQLite directly; write files under `ctx.job_dir/outputs/` and return paths as strings
    in the output pydantic model.
    """

    descriptor: ClassVar[ToolDescriptor]

    @abstractmethod
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        ...

    @abstractmethod
    def run(self, params: BaseModel, ctx: AdapterRunContext) -> BaseModel:
        ...

    def healthcheck(self) -> bool:
        return True

    def estimate_vram(self, params: BaseModel) -> int:
        """Return the **TOTAL** VRAM in MiB this adapter needs to RUN ``params``,
        including the resident model weights (the cost of being loaded).

        Used by the worker's admission check to decide whether a (possibly
        already-loaded) worker can run the job. The check effectively asks::

            free_vram_at_admission_time
                + vram_already_used_by_THIS_adapter
                >= estimate_vram(params) + min_free_vram_mib

        so an implementer does **not** subtract the loaded base — return
        the total. When the adapter is not yet loaded, the worker treats
        the loaded residency as 0 and the formula collapses to "free ≥
        total + headroom" — what you'd expect on a cold load.

        Default: ``(descriptor.vram_min_gib or 4) * 1024`` — the descriptor's
        declared floor, treated as a total. Adapters whose true total varies
        with params (sequence length, batch size, etc.) should override.
        """
        gib = self.descriptor.vram_min_gib if self.descriptor.vram_min_gib is not None else 4
        return gib * 1024

    def supports_batching(self) -> bool:
        return self.descriptor.supports_batching

    def batch_run(
        self, params_list: list[BaseModel], ctx: AdapterRunContext
    ) -> list[BaseModel]:
        return [self.run(p, ctx) for p in params_list]

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def register_adapter(descriptor_path: str):
    """Class decorator: load `<class_dir>/<descriptor_path>` and attach it as `cls.descriptor`.

    Also enrols the (descriptor, class) pair into `_BUILTIN_ADAPTERS` keyed by descriptor name,
    so `ToolRegistry.discover()` can find it just by importing the package.
    """

    def _decorate(cls: type[ToolAdapter]) -> type[ToolAdapter]:
        try:
            cls_file = Path(inspect.getfile(cls)).resolve()
        except TypeError as exc:
            raise RuntimeError(
                f"register_adapter could not locate source file for {cls!r}"
            ) from exc
        full_path = cls_file.parent / descriptor_path
        descriptor = load_descriptor(full_path)
        cls.descriptor = descriptor
        _BUILTIN_ADAPTERS[descriptor.name] = (descriptor, cls)
        return cls

    return _decorate

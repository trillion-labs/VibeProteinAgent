from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel

from vibeproteinagent.core.errors import (
    DependencyError,
    InvalidParamsError,
    UnknownToolError,
)

from .base import _BUILTIN_ADAPTERS, ToolAdapter
from .descriptor import ToolDescriptor, load_descriptor
from .visibility import is_retired_registry_tool_name

_USER_TOOLS_ROOT = Path.home() / ".vibeproteinagent" / "tools"
_ENTRY_POINT_GROUP = "vibeproteinagent.tools"


class ToolRegistry:
    """Discovers tool adapters from three sources, merged later-wins per §6:

      1. In-tree built-ins under `vibeproteinagent.adapters.<tool>` (descriptor.yaml + class
         decorated with `@register_adapter`).
      2. User drop-in at `~/.vibeproteinagent/tools/<tool>/{descriptor.yaml,adapter.py}`.
      3. `importlib.metadata.entry_points(group="vibeproteinagent.tools")`.

    Lookup is by `name` + PEP-440 version specifier. Results are cached in-memory until
    `discover(refresh=True)` or `refresh()` is called.
    """

    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolDescriptor, type[ToolAdapter]]] = {}
        self._discovered = False

    # ------------------------------------------------------------------ discovery

    def discover(self, *, refresh: bool = False) -> None:
        if self._discovered and not refresh:
            return
        if refresh:
            self._tools.clear()
            # Note: do NOT clear `_BUILTIN_ADAPTERS`. The @register_adapter decorator
            # only fires on first import; clearing the dict and re-importing already-loaded
            # subpackages would leave it empty. Built-ins are stable Python source — the
            # only thing refresh genuinely needs to re-scan is filesystem drop-ins and
            # entry points.

        self._discover_builtins()
        self._tools.update(
            {
                name: entry
                for name, entry in _BUILTIN_ADAPTERS.items()
                if not is_retired_registry_tool_name(name)
            }
        )
        self._discover_user_dropins()
        self._discover_entry_points()
        self._discovered = True

    def refresh(self) -> None:
        self.discover(refresh=True)

    def _discover_builtins(self) -> None:
        adapters_pkg = importlib.import_module("vibeproteinagent.adapters")
        pkg_path = list(adapters_pkg.__path__)
        for mod_info in pkgutil.iter_modules(pkg_path):
            if not mod_info.ispkg:
                continue
            if mod_info.name.startswith("_"):
                continue
            if is_retired_registry_tool_name(mod_info.name):
                continue
            try:
                importlib.import_module(f"vibeproteinagent.adapters.{mod_info.name}")
            except Exception as exc:
                self._register_descriptor_only_builtin(pkg_path, mod_info.name, exc)

    def _register_descriptor_only_builtin(
        self,
        pkg_path: list[str],
        package_name: str,
        import_error: BaseException,
    ) -> None:
        """Keep the built-in catalog descriptor-complete even if imports fail.

        Tool listing and MCP discovery are descriptor operations; they must not
        disappear just because an adapter has optional runtime dependencies
        missing in the current environment. Execution still fails explicitly via
        the placeholder adapter.
        """
        for root in pkg_path:
            if is_retired_registry_tool_name(package_name):
                return
            descriptor_path = Path(root) / package_name / "descriptor.yaml"
            if not descriptor_path.is_file():
                continue
            try:
                descriptor = load_descriptor(descriptor_path)
            except Exception:
                return
            if descriptor.name in _BUILTIN_ADAPTERS:
                return
            self._tools[descriptor.name] = (
                descriptor,
                _unavailable_adapter_class(descriptor, import_error),
            )
            return

    def _discover_user_dropins(self) -> None:
        if not _USER_TOOLS_ROOT.is_dir():
            return
        for tool_dir in sorted(_USER_TOOLS_ROOT.iterdir()):
            if not tool_dir.is_dir():
                continue
            descriptor_path = self._find_descriptor(tool_dir)
            adapter_path = tool_dir / "adapter.py"
            if descriptor_path is None or not adapter_path.is_file():
                continue
            descriptor = load_descriptor(descriptor_path)
            adapter_cls = self._load_module_class(
                module_name=f"vibeproteinagent_user_tools.{tool_dir.name}",
                file_path=adapter_path,
                entry_point=descriptor.entry_point,
            )
            self._tools[descriptor.name] = (descriptor, adapter_cls)

    def _discover_entry_points(self) -> None:
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except TypeError:  # pragma: no cover - very old importlib.metadata
            eps = importlib.metadata.entry_points().get(_ENTRY_POINT_GROUP, [])
        for ep in eps:
            try:
                obj = ep.load()
            except Exception:  # pragma: no cover - skip broken entry points
                continue
            if not (isinstance(obj, type) and issubclass(obj, ToolAdapter)):
                continue
            descriptor = getattr(obj, "descriptor", None)
            if descriptor is None:
                continue
            self._tools[descriptor.name] = (descriptor, obj)

    @staticmethod
    def _find_descriptor(tool_dir: Path) -> Path | None:
        for candidate in ("descriptor.yaml", "descriptor.yml", "descriptor.json"):
            p = tool_dir / candidate
            if p.is_file():
                return p
        return None

    @staticmethod
    def _load_module_class(
        *, module_name: str, file_path: Path, entry_point: str | None
    ) -> type[ToolAdapter]:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise DependencyError(
                f"could not build import spec for adapter at {file_path}",
                details={"path": str(file_path)},
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        if entry_point and ":" in entry_point:
            cls_name = entry_point.split(":", 1)[1]
            cls = getattr(module, cls_name, None)
        else:
            cls = next(
                (
                    obj
                    for obj in vars(module).values()
                    if isinstance(obj, type)
                    and issubclass(obj, ToolAdapter)
                    and obj is not ToolAdapter
                ),
                None,
            )
        if cls is None:
            raise DependencyError(
                f"adapter file {file_path} did not export a ToolAdapter subclass",
                details={"path": str(file_path)},
            )
        return cls  # type: ignore[return-value]

    # ------------------------------------------------------------------ queries

    def get(
        self, name: str, version_spec: str = "*"
    ) -> tuple[ToolDescriptor, type[ToolAdapter]]:
        self.discover()
        entry = self._tools.get(name)
        if entry is None:
            raise UnknownToolError(
                f"no tool registered with name '{name}'",
                details={"name": name, "version_spec": version_spec},
            )
        descriptor, cls = entry
        if not _matches_spec(descriptor.version, version_spec):
            raise UnknownToolError(
                f"tool '{name}' version '{descriptor.version}' does not match "
                f"spec '{version_spec}'",
                details={
                    "name": name,
                    "available_version": descriptor.version,
                    "version_spec": version_spec,
                },
            )
        return descriptor, cls

    def list(self, category: str | None = None) -> list[ToolDescriptor]:
        self.discover()
        descriptors = [d for d, _ in self._tools.values()]
        if category is not None:
            # Accept either "protein-design" (hyphenated, user friendly) or
            # "protein_design" (descriptor enum value). Normalise to underscores.
            cat = category.strip().lower().replace("-", "_")
            descriptors = [d for d in descriptors if d.category.value == cat]
        return sorted(descriptors, key=lambda d: (d.category.value, d.name, d.version))

    def entries(
        self, category: str | None = None
    ) -> list[tuple[ToolDescriptor, type[ToolAdapter]]]:
        self.discover()
        entries = list(self._tools.values())
        if category is not None:
            cat = category.strip().lower().replace("-", "_")
            entries = [entry for entry in entries if entry[0].category.value == cat]
        return sorted(
            entries,
            key=lambda item: (item[0].category.value, item[0].name, item[0].version),
        )


def _matches_spec(version: str, version_spec: str) -> bool:
    spec = (version_spec or "*").strip()
    if spec in ("", "*"):
        return True
    try:
        v = Version(version)
    except InvalidVersion as exc:
        raise InvalidParamsError(
            f"tool descriptor version '{version}' is not PEP-440",
            details={"version": version, "error": str(exc)},
        ) from exc
    try:
        return v in SpecifierSet(spec)
    except InvalidSpecifier as exc:
        raise InvalidParamsError(
            f"version_spec '{spec}' is not a valid PEP-440 specifier",
            details={"version_spec": spec, "error": str(exc)},
        ) from exc


def _unavailable_adapter_class(
    descriptor: ToolDescriptor,
    import_error: BaseException,
) -> type[ToolAdapter]:
    message = (
        f"adapter {descriptor.name!r} could not be imported while building the "
        "tool catalog"
    )
    details = {
        "tool": descriptor.name,
        "version": descriptor.version,
        "import_error_type": type(import_error).__name__,
        "import_error": str(import_error),
    }

    class UnavailableAdapter(ToolAdapter):
        def __init__(self, config: dict, weights_dir: Path, device: Any) -> None:
            del config, weights_dir, device

        def run(self, params: BaseModel, ctx: Any) -> BaseModel:
            del params, ctx
            raise DependencyError(message, details=details)

        def healthcheck(self) -> bool:
            return False

    UnavailableAdapter.__name__ = f"{descriptor.name.title().replace('_', '')}UnavailableAdapter"
    UnavailableAdapter.descriptor = descriptor
    UnavailableAdapter._vpa_unavailable_details = details  # type: ignore[attr-defined]
    return UnavailableAdapter

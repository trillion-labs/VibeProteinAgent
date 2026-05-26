from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

from vibeproteinagent.adapters.bindcraft import adapter as bindcraft_adapter


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "src/vibeproteinagent/adapters/bindcraft/adapter.py"
INSTALLER = ROOT / "vendor/bindcraft/install_bindcraft.sh"
SRC_ROOT = ROOT / "src"
KNOWN_TRANSITIVE_SIDECAR_DEPS = {
    "vibeproteinagent.workspace.store": {"aiosqlite"},
}


def _bindcraft_smoke_import_code() -> str:
    tree = ast.parse(ADAPTER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_bindcraft_import_skip_reason":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = getattr(child.func, "id", None) or getattr(child.func, "attr", None)
            if name != "smoke_import":
                continue
            for keyword in child.keywords:
                if keyword.arg == "code" and isinstance(keyword.value, ast.Constant):
                    assert isinstance(keyword.value.value, str)
                    return keyword.value.value
    raise AssertionError("BindCraft smoke_import code string not found")


def _imported_modules(code: str) -> set[str]:
    normalized = "\n".join(part.strip() for part in code.split(";") if part.strip())
    imports: set[str] = set()
    for node in ast.parse(normalized).body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def _install_command_chunks(script: str) -> list[str]:
    chunks: list[str] = []
    lines = script.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if re.search(r"(^|\s)(?:pip3?|python -m pip|\$pkg_manager)\s+install\b", line):
            chunk = [line]
            while chunk[-1].rstrip().endswith("\\") and index + 1 < len(lines):
                index += 1
                chunk.append(lines[index])
            chunks.append("\n".join(chunk))
        index += 1
    return chunks


def _canonical(text: str) -> str:
    return re.sub(r"[-_.]+", "", text).lower()


def _installer_covers_package(chunks: list[str], package: str) -> bool:
    package_key = _canonical(package)
    return any(package_key in _canonical(chunk) for chunk in chunks)


def _vpa_module_exists(module: str) -> bool:
    relative = Path(*module.split("."))
    return (
        (SRC_ROOT / f"{relative}.py").is_file()
        or (SRC_ROOT / relative / "__init__.py").is_file()
    )


def _vpa_module_path(module: str) -> Path:
    relative = Path(*module.split("."))
    py_file = SRC_ROOT / f"{relative}.py"
    if py_file.is_file():
        return py_file
    init_file = SRC_ROOT / relative / "__init__.py"
    if init_file.is_file():
        return init_file
    raise AssertionError(f"{module} is not present under src/")


def _external_import_packages(module: str) -> set[str]:
    tree = ast.parse(_vpa_module_path(module).read_text())
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".", 1)[0]
                if top_level != "vibeproteinagent":
                    packages.add(top_level)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top_level = node.module.split(".", 1)[0]
            if top_level != "vibeproteinagent":
                packages.add(top_level)
    return packages


def test_bindcraft_smoke_imports_are_covered_by_installer() -> None:
    code = _bindcraft_smoke_import_code()
    modules = _imported_modules(code)
    installer_chunks = _install_command_chunks(INSTALLER.read_text())

    missing: list[str] = []
    for module in sorted(modules):
        if module.startswith("vibeproteinagent."):
            if not _vpa_module_exists(module):
                missing.append(f"{module} is not present under src/")
            continue
        package = module.split(".", 1)[0]
        if not _installer_covers_package(installer_chunks, package):
            missing.append(f"{package} is imported by smoke probe but not installed")

    assert not missing, "\n".join(missing)


def test_bindcraft_known_transitive_sidecar_deps_are_installed() -> None:
    installer_chunks = _install_command_chunks(INSTALLER.read_text())
    missing: list[str] = []

    for module, expected_packages in KNOWN_TRANSITIVE_SIDECAR_DEPS.items():
        imported_packages = _external_import_packages(module)
        missing_from_module = expected_packages - imported_packages
        assert not missing_from_module, (
            f"{module} no longer imports expected sidecar deps: "
            f"{sorted(missing_from_module)}"
        )
        for package in sorted(_external_import_packages(module)):
            if package in expected_packages and not _installer_covers_package(
                installer_chunks, package
            ):
                missing.append(f"{package} is imported by {module} but not installed")

    assert not missing, "\n".join(missing)


def test_bindcraft_smoke_import_helper_accepts_stub_python(tmp_path: Path) -> None:
    python = tmp_path / "python"
    python.write_text(
        f"""#!{sys.executable}
from __future__ import annotations

import sys

expected = {{
    "import msgpack",
    "import vibeproteinagent.workers.sidecar_child",
    "import pydantic",
    "import vibeproteinagent.adapters.bindcraft._runner",
    "import torch",
    "import jax",
    "import colabdesign",
    "import pyrosetta",
}}

code = sys.argv[sys.argv.index("-c") + 1]
missing = sorted(item for item in expected if item not in code)
if missing:
    print("missing imports: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
print("ok")
"""
    )
    python.chmod(0o755)

    reason = bindcraft_adapter._bindcraft_import_skip_reason(python, timeout_s=10)

    assert reason is None


@pytest.mark.slow
def test_bindcraft_real_sidecar_smoke_import_passes_when_configured() -> None:
    env_path = os.environ.get("VPA_BINDCRAFT_ENV_PATH")
    if not env_path:
        pytest.skip("$VPA_BINDCRAFT_ENV_PATH is unset")
    python = Path(env_path) / "bin/python"
    if not python.is_file():
        pytest.skip(f"BindCraft sidecar python not found at {python}")

    reason = bindcraft_adapter._bindcraft_import_skip_reason(python)

    assert reason is None

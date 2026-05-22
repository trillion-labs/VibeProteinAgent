from __future__ import annotations

from pathlib import Path


def test_vendored_3dmol_assets_ship_license_companions() -> None:
    root = Path(__file__).resolve().parents[1]
    asset_paths = sorted((root / "src" / "vibeproteinagent").glob("*/assets/3Dmol-min.js"))

    assert asset_paths
    for asset in asset_paths:
        license_file = asset.with_name("3Dmol-min.js.LICENSE.txt")
        assert license_file.is_file(), f"missing license companion for {asset}"
        text = license_file.read_text(encoding="utf-8")
        assert "BSD-3-Clause" in text
        assert "University of Pittsburgh" in text


def test_vendored_3dmol_license_companions_are_forced_into_wheel() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert (
        '"src/vibeproteinagent/reporting/assets/3Dmol-min.js.LICENSE.txt"'
        in pyproject
    )
    assert (
        '"src/vibeproteinagent/visualization/assets/3Dmol-min.js.LICENSE.txt"'
        in pyproject
    )
    assert '"THIRD_PARTY_NOTICES.md" = "vibeproteinagent/THIRD_PARTY_NOTICES.md"' in pyproject

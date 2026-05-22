from __future__ import annotations

import json

import yaml

from vibeproteinagent.adapters.msa_policy import (
    has_prepared_msa,
    json_file_has_prepared_msa,
    should_use_remote_msa,
    strip_prepared_msa,
    yaml_file_has_prepared_msa,
)


def test_should_use_remote_msa_resolver_contract() -> None:
    assert should_use_remote_msa(
        use_msa=False, msa_backend="auto", has_prepared_msa=False
    ) is False
    assert should_use_remote_msa(
        use_msa=True, msa_backend="none", has_prepared_msa=False
    ) is False
    assert should_use_remote_msa(
        use_msa=True, msa_backend="remote", has_prepared_msa=True
    ) is True
    assert should_use_remote_msa(
        use_msa=True, msa_backend="auto", has_prepared_msa=True
    ) is False
    assert should_use_remote_msa(
        use_msa=True, msa_backend="auto", has_prepared_msa=False
    ) is True


def test_has_prepared_msa_detects_nested_tool_native_fields() -> None:
    assert has_prepared_msa({"sequences": [{"proteinChain": {"msa_path": "x.a3m"}}]})
    assert has_prepared_msa({"sequences": [{"protein": {"msa": "x.a3m"}}]})
    assert not has_prepared_msa({"sequences": [{"protein": {"msa": ""}}]})
    assert not has_prepared_msa({"sequences": [{"protein": {"sequence": "AAAA"}}]})


def test_json_and_yaml_file_prepared_msa_detection(tmp_path) -> None:
    json_path = tmp_path / "input.json"
    json_path.write_text(json.dumps([{"proteinChain": {"paired_msa": "paired.a3m"}}]))
    assert json_file_has_prepared_msa(json_path)

    yaml_path = tmp_path / "input.yaml"
    yaml_path.write_text(yaml.safe_dump({"sequences": [{"protein": {"msa": "x.a3m"}}]}))
    assert yaml_file_has_prepared_msa(yaml_path)


def test_strip_prepared_msa_removes_nested_msa_fields() -> None:
    raw = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "ACDE", "msa": "x.a3m"}},
            {"proteinChain": {"msa_path": "y.a3m", "sequence": "FGHI"}},
        ],
        "paired_msa": "paired.a3m",
    }
    stripped = strip_prepared_msa(raw)
    assert not has_prepared_msa(stripped)
    assert stripped["sequences"][0]["protein"]["sequence"] == "ACDE"
    assert stripped["sequences"][1]["proteinChain"]["sequence"] == "FGHI"

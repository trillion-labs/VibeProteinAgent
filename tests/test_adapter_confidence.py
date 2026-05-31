from __future__ import annotations

import math

import pytest

from vibeproteinagent.adapters._confidence import normalize_plddt, plddt_unit_score


def test_normalize_plddt_accepts_unit_interval_and_percent_scales() -> None:
    assert normalize_plddt(0.981) == pytest.approx(98.1)
    assert normalize_plddt(98.1) == pytest.approx(98.1)
    assert plddt_unit_score(0.981) == pytest.approx(0.981)
    assert plddt_unit_score(98.1) == pytest.approx(0.981)


@pytest.mark.parametrize("value", [None, "not-a-number", math.nan, math.inf, -0.1])
def test_normalize_plddt_rejects_invalid_values(value) -> None:
    assert normalize_plddt(value) is None
    assert plddt_unit_score(value) is None


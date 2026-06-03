from __future__ import annotations

from vibeproteinagent.adapters.profile_guidance import (
    consensus_sequence,
    derive_profile_mask_positions,
    normalize_profile_guidance,
    proteinmpnn_bias_applicability,
    suggested_profile_length,
)


def _profile() -> dict:
    return {
        "schema_version": "family_profile_v1",
        "family_id": "family-a",
        "sequence_count": 3,
        "consensus_sequence": "ACDE",
        "length_stats": {"median": 4},
        "position_frequencies": [
            {
                "position": 1,
                "consensus": "A",
                "frequencies": {"A": 1.0},
            },
            {
                "position": 2,
                "consensus": "C",
                "frequencies": {"C": 0.8, "A": 0.2},
            },
        ],
        "conserved_positions": [
            {"position": 1, "consensus": "A", "conservation": 1.0},
            {"position": 2, "consensus": "C", "conservation": 0.8},
        ],
    }


def test_normalize_profile_guidance_builds_v1_preferences() -> None:
    guidance = normalize_profile_guidance(_profile())

    assert guidance is not None
    assert guidance["schema_version"] == "profile_guidance_v1"
    assert guidance["family_id"] == "family-a"
    assert guidance["position_preferences"][1]["frequencies"]["C"] == 0.8


def test_profile_guidance_derives_mask_positions_from_conserved_mismatches() -> None:
    assert derive_profile_mask_positions("AAAA", _profile()) == [2]
    assert derive_profile_mask_positions("ACAA", _profile()) == []


def test_profile_guidance_length_and_consensus_helpers() -> None:
    assert consensus_sequence(_profile()) == "ACDE"
    assert suggested_profile_length(_profile()) == 4
    assert proteinmpnn_bias_applicability(_profile(), num_residues=4)["applied"] is True
    assert proteinmpnn_bias_applicability(_profile(), num_residues=5)["applied"] is False

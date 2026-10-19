from pathlib import Path

import pytest

from scripts import audit_setspeed_observability_v01 as audit


def test_stock_speed_icon_matches_dll_categories():
    assert audit.stock_speed_icon(1.0, 1.04, playing=True) is None
    assert audit.stock_speed_icon(1.0, 1.5, playing=True) == "Rabbit"
    assert audit.stock_speed_icon(1.0, 8.0, playing=True) == "DoubleRabbit"
    assert audit.stock_speed_icon(1.0, 0.75, playing=True) == "Snail"
    assert audit.stock_speed_icon(1.0, 0.25, playing=True) == "DoubleSnail"
    assert audit.stock_speed_icon(1.0, 1.0, playing=False) == "SameSpeed"
    with pytest.raises(ValueError):
        audit.stock_speed_icon(0.0, 1.0, playing=True)


def test_train_bool_candidate_audit_abstains_without_real_capture():
    result = audit.run(Path(
        "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
    ))
    icon = result["candidates"]["stock_setspeed_speed_icon"]
    assert icon["renderer_semantics_original"] == "DoubleRabbit"
    assert icon["renderer_semantics_counterfactual"] is None
    assert icon["renderer_semantics_distinguish_pair"]
    assert icon["verdict"] == "UNVERIFIED"
    assert icon["first_observed_s"] is None

    for mode in ("normal", "jongyeol"):
        jrp = result["candidates"]["jrp_tbpm_cbpm_kps"][mode]
        assert jrp["on_show_equal"]
        assert jrp["first_on_hit_differs"]
        assert jrp["earliest_possible_update_s_posthoc"] > jrp["deadline_s_posthoc"]
        assert jrp["verdict"] == "INDISTINGUISHABLE"

    assert result["candidates"]["audio"]["audio_asset_present_in_train80"] is False
    assert result["overall"]["motor_connection_allowed"] is False

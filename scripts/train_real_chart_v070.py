from __future__ import annotations

"""v0.7.0: train the existing anchor-guard pipeline with player-visible HUD input.

The training/guard semantics remain v0.6.4 plus v0.6.5's exact proposal cache.
Only the observation interface changes: Tile BPM, Real BPM, and transient timing
feedback are appended to the previous 233D visible geometry/motor vector.
"""

import train_real_chart_v054 as v054
import train_real_chart_v060 as v060
import train_real_chart_v062 as v062
import train_real_chart_v063 as v063
import train_real_chart_v064 as v064
import train_real_chart_v065 as v065

from dmdod.real_chart_hud import DEFAULT_FEEDBACK_HOLD_S, DiagnosticHudRealChartMotorEnv
from dmdod.real_chart_hud_features import (
    HUD_FEATURE_VERSION,
    HUD_REAL_CHART_INPUT_DIM,
    encode_hud_real_chart_observation,
)


TRAINER_VERSION = "0.7.0-human-visible-hud"
CHECKPOINT_FORMAT_VERSION = 14
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v070_hud.pt"
HUD_OBSERVATION_VERSION = HUD_FEATURE_VERSION

_INSTALLED = False
_PARENT_RUN_SIGNATURE = None
_PARENT_CHECKPOINT_PAYLOAD = None
_PARENT_LOAD_PROGRESS = None


def _install_v070() -> None:
    global _INSTALLED, _PARENT_RUN_SIGNATURE, _PARENT_CHECKPOINT_PAYLOAD, _PARENT_LOAD_PROGRESS
    if _INSTALLED:
        return

    # Install v0.6.5 first so deterministic duplicate line searches remain
    # cached. Then replace only the observation/checkpoint identity surface.
    v065._install_v065()

    _PARENT_RUN_SIGNATURE = v064._run_signature
    _PARENT_CHECKPOINT_PAYLOAD = v064._checkpoint_payload
    _PARENT_LOAD_PROGRESS = v064._load_progress

    # v0.5.4 owns the shared evaluator. v0.6.0 owns expert/DAgger collection.
    # Both resolve these module globals at runtime, so swapping them here keeps
    # the mature training/guard pipeline while changing only what the policy sees.
    v054.DiagnosticRealChartMotorEnv = DiagnosticHudRealChartMotorEnv
    v054.encode_real_chart_observation = encode_hud_real_chart_observation
    v054.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM

    v060.encode_real_chart_observation = encode_hud_real_chart_observation
    v060.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM

    # These constants are used for model construction/checkpoint validation in
    # the multi-segment/resume/anchor-guard layers.
    v062.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM
    v063.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM
    v064.REAL_CHART_INPUT_DIM = HUD_REAL_CHART_INPUT_DIM

    v064.TRAINER_VERSION = TRAINER_VERSION
    v064.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v064.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    v064._run_signature = _hud_run_signature
    v064._checkpoint_payload = _hud_checkpoint_payload
    v064._load_progress = _hud_load_progress
    _INSTALLED = True


def _hud_run_signature(args, *, chart_path: str, train_pool, validation_window, sight_window) -> dict:
    assert _PARENT_RUN_SIGNATURE is not None
    signature = _PARENT_RUN_SIGNATURE(
        args,
        chart_path=chart_path,
        train_pool=train_pool,
        validation_window=validation_window,
        sight_window=sight_window,
    )
    signature["hud_observation"] = HUD_OBSERVATION_VERSION
    signature["hud_feedback_hold_s"] = DEFAULT_FEEDBACK_HOLD_S
    return signature


def _hud_checkpoint_payload(**kwargs) -> dict:
    assert _PARENT_CHECKPOINT_PAYLOAD is not None
    payload = _PARENT_CHECKPOINT_PAYLOAD(**kwargs)
    payload["trainer_version"] = TRAINER_VERSION
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION
    payload["input_dim"] = HUD_REAL_CHART_INPUT_DIM
    payload["hud_observation"] = HUD_OBSERVATION_VERSION
    payload["hud_feedback_hold_s"] = DEFAULT_FEEDBACK_HOLD_S
    payload["hud_semantics"] = {
        "always_visible": ["tile_bpm", "real_bpm"],
        "transient": ["last_judgement", "signed_timing_error_ms"],
        "excluded": [
            "target_timestamp",
            "time_to_next_target",
            "chart_time",
            "absolute_floor_index",
            "x_accuracy",
            "progress",
            "attempt_count",
            "kv",
            "overload_counter",
            "fatigue",
        ],
    }
    return payload


def _hud_load_progress(model, path, *, current_signature: dict, anchor_count: int, device):
    assert _PARENT_LOAD_PROGRESS is not None
    try:
        return _PARENT_LOAD_PROGRESS(
            model,
            path,
            current_signature=current_signature,
            anchor_count=anchor_count,
            device=device,
        )
    except SystemExit as exc:
        message = str(exc).replace("v0.6.5", "v0.7.0").replace("v0.6.4", "v0.7.0")
        raise SystemExit(message) from None


def main() -> None:
    _install_v070()
    print("=== DMDOD v0.7.0 Human-Visible HUD ===")
    print(
        f"observation={HUD_REAL_CHART_INPUT_DIM}D = old 233D + "
        "Tile BPM + Real BPM + transient judgement/error HUD"
    )
    print(
        f"feedback-hold={DEFAULT_FEEDBACK_HOLD_S:.2f}s | "
        "XAcc/progress/KV/attempt/internal timing truth remain hidden"
    )
    print("backend=v0.6.4 anchor guard + v0.6.5 exact proposal cache")
    v064.main()
    print(
        f"proposal-cache: hits={v065._PROPOSAL_CACHE.hits} misses={v065._PROPOSAL_CACHE.misses} "
        f"saved-line-searches={v065._PROPOSAL_CACHE.hits}"
    )


if __name__ == "__main__":
    main()

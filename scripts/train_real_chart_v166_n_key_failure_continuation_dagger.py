from __future__ import annotations

"""v1.6.6: trust DAgger with fixed-horizon collection after terminal failure.

This is a thin compatibility layer over the v1.6.4 trust-region trainer.  The
selection/evaluation path is unchanged.  Only student-state DAgger collection
is replaced: overload/fail-on-miss snapshots the failed evaluation result, then
continues body/policy/teacher dynamics to the normal episode horizon so failure
and recovery states are represented in the training data.
"""

import train_real_chart_v163_n_key_continuous_trust_dagger as base
from dmdod.n_key_dagger_continuation import (
    collect_n_key_dagger_sequence_with_continuation,
)


TRAINER_VERSION = "1.6.6-n-key-failure-continuation-dagger"
CHECKPOINT_FORMAT_VERSION = 24

_ORIGINAL_CHECKPOINT_PAYLOAD = base._checkpoint_payload


def _collect_with_failure_continuation(*args, **kwargs):
    kwargs["continue_after_failure"] = True
    return collect_n_key_dagger_sequence_with_continuation(*args, **kwargs)


def _checkpoint_payload(*args, **kwargs) -> dict:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(*args, **kwargs)
    payload.update(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "dagger_failure_continuation": "to-episode-end",
            "dagger_failure_continuation_preserves_terminal_stats": True,
            "dagger_failure_continuation_evaluation_unchanged": True,
        }
    )
    return payload


def main() -> None:
    base.TRAINER_VERSION = TRAINER_VERSION
    base.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    base.collect_n_key_dagger_sequence = _collect_with_failure_continuation
    base._checkpoint_payload = _checkpoint_payload
    print("=== DMDOD v1.6.6 N-Key Failure-Continuation Trust DAgger ===")
    print(
        "DAgger collection continues terminal failures to the normal episode horizon; "
        "Train/Validation evaluation still terminates normally."
    )
    base.main()


if __name__ == "__main__":
    main()

from __future__ import annotations

"""v1.7.2: connectome trust DAgger with reject early-stop.

This is a thin compatibility layer over v1.7.1.  If a proposal epoch has no
Train-safe improving trust candidate, v1.6.4 rolls model, optimizer, and DAgger
data back to the exact accepted state.  Re-running another epoch from that
unchanged state deterministically recreates the same proposal, so stop the epoch
loop after the first such rejection while preserving checkpoint and Validation
handling.
"""

import builtins

import train_real_chart_v171_n_key_connectome_failure_continuation_dagger as v171


TRAINER_VERSION = "1.7.2-n-key-connectome-reject-early-stop-dagger"
CHECKPOINT_FORMAT_VERSION = 28

_ORIGINAL_SELECT_IMPROVING_CANDIDATE = v171.v166.base._select_improving_candidate
_stop_before_next_epoch = False


def _select_improving_candidate(reference_results, candidates):
    global _stop_before_next_epoch
    chosen = _ORIGINAL_SELECT_IMPROVING_CANDIDATE(reference_results, candidates)
    _stop_before_next_epoch = chosen is None
    return chosen


def _epoch_range(*args):
    for epoch in builtins.range(*args):
        if _stop_before_next_epoch:
            print(
                "epoch-loop: EARLY STOP after rejected proposal; "
                "accepted model+optimizer+DAgger data are unchanged"
            )
            break
        yield epoch


def main() -> None:
    global _stop_before_next_epoch
    _stop_before_next_epoch = False

    v171.TRAINER_VERSION = TRAINER_VERSION
    v171.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v171.v166.base._select_improving_candidate = _select_improving_candidate
    # v1.6.4 has exactly one module-level range() call: the DAgger epoch loop.
    v171.v166.base.range = _epoch_range

    print("=== DMDOD v1.7.2 N-Key Connectome Reject-Early-Stop Trust DAgger ===")
    print(
        "A fully rejected proposal ends the epoch loop because rollback restores "
        "the identical model, optimizer, and DAgger dataset; checkpoint and "
        "Validation handling remain unchanged."
    )
    v171.main()


if __name__ == "__main__":
    main()

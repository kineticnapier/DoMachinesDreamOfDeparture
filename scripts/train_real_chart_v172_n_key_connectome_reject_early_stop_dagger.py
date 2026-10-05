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
_ORIGINAL_V166_CHECKPOINT_PAYLOAD = v171.v166._checkpoint_payload
_stop_before_next_epoch = False
_last_started_epoch = 0


def _select_improving_candidate(reference_results, candidates):
    global _stop_before_next_epoch
    chosen = _ORIGINAL_SELECT_IMPROVING_CANDIDATE(reference_results, candidates)
    _stop_before_next_epoch = chosen is None
    return chosen


def _epoch_range(*args):
    global _last_started_epoch
    for epoch in builtins.range(*args):
        if _stop_before_next_epoch:
            print(
                "epoch-loop: EARLY STOP after rejected proposal; "
                "accepted model+optimizer+DAgger data are unchanged"
            )
            break
        _last_started_epoch = int(epoch)
        yield epoch


def _checkpoint_payload(*args, **kwargs):
    requested_epochs = int(kwargs["requested_epochs"])
    kwargs["completed_epoch"] = int(_last_started_epoch)
    payload = _ORIGINAL_V166_CHECKPOINT_PAYLOAD(*args, **kwargs)
    payload.update(
        {
            "dagger_reject_early_stop": bool(
                _stop_before_next_epoch and _last_started_epoch < requested_epochs
            ),
            "dagger_reject_early_stop_epoch": (
                int(_last_started_epoch)
                if _stop_before_next_epoch and _last_started_epoch < requested_epochs
                else None
            ),
        }
    )
    return payload


def main() -> None:
    global _stop_before_next_epoch, _last_started_epoch
    _stop_before_next_epoch = False
    _last_started_epoch = 0

    v171.TRAINER_VERSION = TRAINER_VERSION
    v171.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v171.v166._checkpoint_payload = _checkpoint_payload
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

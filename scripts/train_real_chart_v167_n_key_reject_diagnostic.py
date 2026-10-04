from __future__ import annotations

"""v1.6.7: diagnose the best rejected trust candidate and skip duplicate epochs.

This is a thin layer over v1.6.6 failure-continuation trust DAgger. Train-only
selection semantics are unchanged. When an epoch produces no accepted trust
candidate, the best guard-rejected candidate by the existing Train selection
key is evaluated once on Validation for diagnosis only. Because a rejected
epoch rolls model, optimizer, and DAgger data back to the same accepted state,
later requested epochs would deterministically reproduce the same proposal;
those duplicate epochs are skipped.
"""

import argparse
from builtins import range as builtin_range
from pathlib import Path

import torch

import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
import train_real_chart_v163_n_key_continuous_trust_dagger as base
import train_real_chart_v166_n_key_failure_continuation_dagger as v166


TRAINER_VERSION = "1.6.7-n-key-reject-diagnostic"
CHECKPOINT_FORMAT_VERSION = 25


class _RunState:
    def __init__(self) -> None:
        self.stop_duplicate_epochs = False
        self.executed_epochs = 0
        self.diagnostics: list[dict] = []


class _ControlledRange:
    def __init__(self, values, state: _RunState) -> None:
        self._values = values
        self._state = state

    def __iter__(self):
        for value in self._values:
            if self._state.stop_duplicate_epochs:
                break
            yield value


def _parse_context():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--validation-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args, _ = parser.parse_known_args()
    return args


def _validation_segments(parent: dict, dataset_path: str, validation_limit: int | None):
    dataset = base.discover_multichart_dataset(dataset_path)
    validation_charts = base.v080._compile_role(dataset.validation)
    validation = base.v080._build_validation_segments(
        validation_charts,
        window_s=float(parent["validation_window"]),
    )
    parent_limit = parent.get("validation_limit")
    limit = validation_limit if validation_limit is not None else parent_limit
    if limit is not None:
        validation = validation[: int(limit)]
    return validation


def _diagnostic_record(epoch: int, candidate, results) -> dict:
    train_summary = v161._summarize(candidate.results)
    validation_summary = v161._summarize(results)
    return {
        "epoch": int(epoch),
        "alpha": float(candidate.alpha),
        "train_hits": int(train_summary.hits),
        "train_targets": int(train_summary.targets),
        "train_x_accuracy_percent": float(train_summary.x_accuracy_percent),
        "validation_hits": int(validation_summary.hits),
        "validation_targets": int(validation_summary.targets),
        "validation_x_accuracy_percent": float(validation_summary.x_accuracy_percent),
        "validation_early": int(validation_summary.early),
        "validation_overloaded": bool(validation_summary.overloaded),
        "validation_keydowns": int(validation_summary.keydowns),
        "used_for_selection": False,
    }


def main() -> None:
    context = _parse_context()
    parent = torch.load(Path(context.checkpoint), map_location="cpu", weights_only=False)
    device = v161._device_from_arg(context.device)
    validation = _validation_segments(parent, context.dataset, context.validation_limit)
    control_dt_s = float(parent["control_dt"])
    physics_dt_s = float(parent.get("physics_dt", 0.001))

    diagnostic_model = base._build_policy_from_checkpoint(parent, device=device)
    state = _RunState()

    original_select = base._select_improving_candidate
    original_payload = v166._ORIGINAL_CHECKPOINT_PAYLOAD

    def controlled_range(*args):
        return _ControlledRange(builtin_range(*args), state)

    def select_with_reject_diagnostic(base_results, candidates):
        chosen = original_select(base_results, candidates)
        state.executed_epochs += 1
        if chosen is not None:
            return chosen

        rejected = [candidate for candidate in candidates if not candidate.guard_accepted]
        if rejected:
            best = max(rejected, key=lambda candidate: v161._selection_key(candidate.results))
            diagnostic_model.load_state_dict(best.state)
            diagnostic_model.prepare_recurrent_runtime()
            print(
                "=== diagnostic-only best rejected candidate Validation "
                f"epoch {state.executed_epochs} alpha={best.alpha:g} ==="
            )
            print(
                "This Validation result is diagnostic only and is not used for "
                "trust selection or checkpoint selection."
            )
            results = v162._evaluate_role_continuous(
                diagnostic_model,
                validation,
                label=f"reject-diagnostic-e{state.executed_epochs:03d}-a{best.alpha:g}",
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
            )
            state.diagnostics.append(
                _diagnostic_record(state.executed_epochs, best, results)
            )
        else:
            print(
                "diagnostic-only: epoch had no accepted improvement and no "
                "guard-rejected candidate to validate"
            )

        state.stop_duplicate_epochs = True
        print(
            "epoch-skip: accepted model, optimizer, and DAgger data are unchanged; "
            "remaining requested epochs would reproduce the same deterministic proposal"
        )
        return None

    def payload_with_diagnostic(*args, **kwargs):
        kwargs["completed_epoch"] = int(state.executed_epochs)
        payload = original_payload(*args, **kwargs)
        payload.update(
            {
                "dagger_requested_epochs": int(kwargs["requested_epochs"]),
                "dagger_executed_epochs": int(state.executed_epochs),
                "dagger_duplicate_epoch_skip": bool(state.stop_duplicate_epochs),
                "dagger_duplicate_epoch_skip_reason": (
                    "rejected-epoch-restores-identical-model-optimizer-data"
                    if state.stop_duplicate_epochs
                    else None
                ),
                "dagger_rejected_validation_diagnostics": list(state.diagnostics),
                "dagger_rejected_validation_used_for_selection": False,
            }
        )
        return payload

    base.range = controlled_range
    base._select_improving_candidate = select_with_reject_diagnostic
    v166._ORIGINAL_CHECKPOINT_PAYLOAD = payload_with_diagnostic
    v166.TRAINER_VERSION = TRAINER_VERSION
    v166.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION

    print("=== DMDOD v1.6.7 N-Key Reject Diagnostic + Duplicate-Epoch Skip ===")
    print(
        "Best guard-rejected candidate gets one diagnostic-only Validation run; "
        "unchanged rejected continuation states skip duplicate later epochs."
    )
    v166.main()


if __name__ == "__main__":
    main()

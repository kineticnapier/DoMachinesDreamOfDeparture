from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Callable, Hashable

import torch

import train_real_chart_v064 as v064


TRAINER_VERSION = "0.6.5-exact-proposal-cache"
CHECKPOINT_FORMAT_VERSION = 13
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v065_proposal_cache.pt"
PROPOSAL_CACHE_VERSION = "exact-state-v1"


@dataclass(slots=True)
class LineSearchMemo:
    """Memoize deterministic trust-region line searches.

    v0.6.4 recreates a fresh optimizer from the same trusted model on every
    epoch.  With the same sequence order this produces the exact same proposal,
    so rejected odd/even epochs can repeat an expensive train+validation+anchor
    evaluation many times.  This cache only reuses a result when the base model,
    proposal model, evaluated segment, references, alphas, and evaluation
    settings are byte/metric identical.
    """

    entries: dict[Hashable, tuple] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get_or_run(self, key: Hashable, runner: Callable[[], tuple]) -> tuple[tuple, bool]:
        cached = self.entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached, True
        value = runner()
        self.entries[key] = value
        self.misses += 1
        return value, False


_PROPOSAL_CACHE = LineSearchMemo()
_ORIGINAL_ANCHOR_LINE_SEARCH = v064._evaluate_anchor_line_search
_ORIGINAL_RUN_SIGNATURE = v064._run_signature
_ORIGINAL_CHECKPOINT_PAYLOAD = v064._checkpoint_payload
_ORIGINAL_LOAD_PROGRESS = v064._load_progress
_INSTALLED = False


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.digest()


def _state_digest(state: dict[str, torch.Tensor]) -> bytes:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_tensor_digest(state[name]))
    return digest.digest()


def _eval_signature(result) -> tuple:
    stats = result.stats
    return (
        int(stats.hits),
        int(stats.misses),
        int(stats.targets),
        float(stats.x_accuracy_percent),
        float(stats.perfect_rate),
        int(stats.too_early_presses),
        bool(stats.overloaded),
        int(result.physical_keydowns),
    )


def _segment_signature(segment) -> tuple:
    # Exact target identity is evaluator-private data.  It is used only to make
    # cache reuse safe; it is never added to the student's observation.
    return tuple(
        (
            int(target.ordinal),
            int(target.floor_index),
            float(target.chart_time_s),
            float(target.episode_time_s),
        )
        for target in segment.targets
    )


def _proposal_cache_key(
    *,
    base_state: dict[str, torch.Tensor],
    proposal_state: dict[str, torch.Tensor],
    base_train_eval,
    validation_reference,
    anchor_references,
    alphas,
    train_segment,
    validation_segment,
    anchor_segments,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
) -> tuple:
    return (
        PROPOSAL_CACHE_VERSION,
        _state_digest(base_state),
        _state_digest(proposal_state),
        _eval_signature(base_train_eval),
        _eval_signature(validation_reference),
        tuple(_eval_signature(result) for result in anchor_references),
        tuple(float(alpha) for alpha in alphas),
        _segment_signature(train_segment),
        _segment_signature(validation_segment),
        tuple(_segment_signature(segment) for segment in anchor_segments),
        bool(same_hand),
        float(control_dt_s),
        str(device),
    )


def _cached_evaluate_anchor_line_search(model, **kwargs):
    key = _proposal_cache_key(
        base_state=kwargs["base_state"],
        proposal_state=kwargs["proposal_state"],
        base_train_eval=kwargs["base_train_eval"],
        validation_reference=kwargs["validation_reference"],
        anchor_references=kwargs["anchor_references"],
        alphas=kwargs["alphas"],
        train_segment=kwargs["train_segment"],
        validation_segment=kwargs["validation_segment"],
        anchor_segments=kwargs["anchor_segments"],
        same_hand=kwargs["same_hand"],
        control_dt_s=kwargs["control_dt_s"],
        device=kwargs["device"],
    )

    def run():
        return _ORIGINAL_ANCHOR_LINE_SEARCH(model, **kwargs)

    (choice, candidates, chosen_state), cache_hit = _PROPOSAL_CACHE.get_or_run(key, run)
    if cache_hit:
        label = kwargs["label_prefix"]
        if kwargs["verbose"]:
            print(
                f"{label}: CACHE-HIT exact base/proposal; "
                f"reused {len(candidates)} trust-region candidate evaluations"
            )
        else:
            print(
                f"{label}: CACHE "
                + " | ".join(v064._candidate_brief(candidate) for candidate in candidates)
            )
        if chosen_state is None:
            model.load_state_dict(kwargs["base_state"])
        else:
            model.load_state_dict(chosen_state)

    # v0.6.4 deep-copies an accepted chosen_state before retaining it. Return a
    # private copy on cache hits too so memoized tensors can never be mutated by
    # later training code.
    returned_state = copy.deepcopy(chosen_state) if chosen_state is not None else None
    return choice, candidates, returned_state


def _run_signature(args, *, chart_path: str, train_pool, validation_window, sight_window) -> dict:
    signature = _ORIGINAL_RUN_SIGNATURE(
        args,
        chart_path=chart_path,
        train_pool=train_pool,
        validation_window=validation_window,
        sight_window=sight_window,
    )
    signature["proposal_cache"] = PROPOSAL_CACHE_VERSION
    return signature


def _checkpoint_payload(**kwargs) -> dict:
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(**kwargs)
    payload["proposal_cache"] = PROPOSAL_CACHE_VERSION
    payload["proposal_cache_semantics"] = "evaluation-only; policy update semantics unchanged"
    return payload


def _load_progress(model, path, *, current_signature: dict, anchor_count: int, device: torch.device):
    try:
        return _ORIGINAL_LOAD_PROGRESS(
            model,
            path,
            current_signature=current_signature,
            anchor_count=anchor_count,
            device=device,
        )
    except SystemExit as exc:
        message = str(exc).replace("v0.6.4", "v0.6.5")
        message = message.replace("anchor-guard progress checkpoint", "proposal-cache progress checkpoint")
        raise SystemExit(message) from None


def _install_v065() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    # v0.6.4 resolves these globals at runtime, so the thin wrapper can preserve
    # its thoroughly tested trainer while changing only deterministic evaluation
    # reuse and checkpoint identity.
    v064.TRAINER_VERSION = TRAINER_VERSION
    v064.CHECKPOINT_FORMAT_VERSION = CHECKPOINT_FORMAT_VERSION
    v064.DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
    v064._evaluate_anchor_line_search = _cached_evaluate_anchor_line_search
    v064._run_signature = _run_signature
    v064._checkpoint_payload = _checkpoint_payload
    v064._load_progress = _load_progress
    _INSTALLED = True


def main() -> None:
    _install_v065()
    print(
        "=== DMDOD v0.6.5 Exact Proposal Cache ===\n"
        "deterministic duplicate trust-region evaluations are reused; "
        "policy/guard semantics stay v0.6.4-equivalent"
    )
    v064.main()
    print(
        f"proposal-cache: hits={_PROPOSAL_CACHE.hits} misses={_PROPOSAL_CACHE.misses} "
        f"saved-line-searches={_PROPOSAL_CACHE.hits}"
    )


if __name__ == "__main__":
    main()

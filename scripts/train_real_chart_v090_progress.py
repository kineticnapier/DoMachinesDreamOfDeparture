from __future__ import annotations

"""Execution-only fine-grained progress instrumentation for v0.9 training.

The functions below are byte-for-byte-semantic clones of the hot training/
rollout/evaluation loops with observational progress events inserted between
existing operations.  No loss, optimizer step, guard, cache key, state update,
or checkpoint field is changed.
"""

import sys

import torch

import train_real_chart_v054 as v054
import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
import train_real_chart_v060 as v060
import train_real_chart_v080 as v080
import train_real_chart_v080_fast as v080_fast
from dmdod.training_progress import emit_progress


PROGRESS_VERSION = "v090-live-internals-v1"
PROGRESS_EMIT_EVERY_STEPS = 25

_INSTALLED = False
_GUARD_DEPTH = 0
_ORIGINAL_LINE_SEARCH = None
_ORIGINAL_GROUPED_EVAL = None


def _progress_train_one_epoch(
    model,
    sequences,
    *,
    optimizer,
    chunk_steps: int,
    reverse_order: bool,
) -> float:
    """Exact v0.5.7 recurrent BC epoch with sequence/chunk progress events."""

    if not sequences:
        raise ValueError("at least one stable DAgger sequence is required")
    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    parameters = v057._policy_parameters(model)
    ordered = list(reversed(sequences)) if reverse_order else sequences
    total_chunks = sum(
        (int(stable.sequence.frames) + int(chunk_steps) - 1) // int(chunk_steps)
        for stable in ordered
    )
    emit_progress(
        "bc_start",
        sequences=len(ordered),
        chunks=total_chunks,
        reverse=bool(reverse_order),
    )

    loss_sum = 0.0
    weighted_elements = 0.0
    global_chunk = 0

    for sequence_index, stable in enumerate(ordered, 1):
        sequence = stable.sequence
        sequence_chunks = (int(sequence.frames) + int(chunk_steps) - 1) // int(chunk_steps)
        source = str(getattr(sequence, "source", f"sequence-{sequence_index}"))
        emit_progress(
            "bc_sequence_start",
            index=sequence_index,
            total=len(ordered),
            chunks=sequence_chunks,
            frames=int(sequence.frames),
            source=source,
            reverse=bool(reverse_order),
        )

        state = model.initial_state(sequence.observations.device)
        chunk_index = 0
        for start in range(0, sequence.frames, chunk_steps):
            end = min(sequence.frames, start + chunk_steps)
            state = state.detach()
            predictions: list[torch.Tensor] = []
            for x in sequence.observations[start:end]:
                mean, _, _, state = model.forward_step(x, state)
                predictions.append(torch.tanh(mean))

            predicted = torch.stack(predictions)
            target = sequence.teacher_actions[start:end]
            weights = stable.loss_weights[start:end]
            loss = v056._weighted_actuation_loss(predicted, target, weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()

            weight = max(float(weights.sum().item()), 1.0)
            loss_value = float(loss.detach().item())
            loss_sum += loss_value * weight
            weighted_elements += weight
            chunk_index += 1
            global_chunk += 1
            emit_progress(
                "bc_chunk",
                sequence=sequence_index,
                sequence_total=len(ordered),
                chunk=chunk_index,
                chunk_total=sequence_chunks,
                global_chunk=global_chunk,
                global_total=total_chunks,
                loss=loss_value,
                source=source,
                reverse=bool(reverse_order),
            )

        emit_progress(
            "bc_sequence_done",
            index=sequence_index,
            total=len(ordered),
            source=source,
        )

    final_loss = loss_sum / max(1.0, weighted_elements)
    emit_progress("bc_done", loss=final_loss, reverse=bool(reverse_order))
    return final_loss


def _progress_collect_expert_sequence(
    segment,
    *,
    lead_s: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
):
    """Exact finger-agnostic expert collection with live simulator progress."""

    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=v060.DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=v060.DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    observations: list[tuple[float, ...]] = []
    actions: list[tuple[float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    expected_steps = max(1, int(segment.duration_s / control_dt_s) + 1)
    emit_progress("rollout_start", mode="expert", total=expected_steps, source="round expert")
    actual_steps = 0
    for step_index in range(1, max_steps + 1):
        action = v060._teacher_action(env, observation, lead_s)
        observations.append(v060.encode_real_chart_observation(observation))
        actions.append((action.left, action.right))
        step = env.step(action)
        observation = step.observation
        actual_steps = step_index
        if step_index % PROGRESS_EMIT_EVERY_STEPS == 0 or step.done:
            emit_progress(
                "rollout_step",
                mode="expert",
                current=min(step_index, expected_steps),
                total=expected_steps,
                source="round expert",
            )
        if step.done:
            break
    else:
        raise RuntimeError("finger-agnostic teacher episode exceeded step budget")

    emit_progress("rollout_done", mode="expert", steps=actual_steps, source="round expert")
    return (
        torch.tensor(observations, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
        v054.StudentEvalResult(env.stats, env.physical_keydowns),
    )


def _progress_collect_mixture_rollout(
    model,
    segment,
    *,
    lead_s: float,
    teacher_fraction: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    source: str,
    seed: int,
    press_recovery_cap: int,
):
    """Exact DAgger mixture rollout with live simulator progress."""

    if not 0.0 <= teacher_fraction <= 1.0:
        raise ValueError("teacher_fraction must be in [0, 1]")

    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=v060.DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=v060.DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    rng = v060.random.Random(seed)
    observations: list[tuple[float, ...]] = []
    labels: list[tuple[float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    expected_steps = max(1, int(segment.duration_s / control_dt_s) + 1)
    emit_progress(
        "rollout_start",
        mode="mixture",
        total=expected_steps,
        source=str(source),
        teacher_fraction=float(teacher_fraction),
    )
    actual_steps = 0
    with torch.no_grad():
        for step_index in range(1, max_steps + 1):
            encoded = v060.encode_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            student, state = model.deterministic_action(x, state)
            teacher = v060._teacher_action(
                env,
                observation,
                lead_s,
                preferred_action=student,
            )

            observations.append(encoded)
            labels.append((teacher.left, teacher.right))
            applied = teacher if rng.random() < teacher_fraction else student
            step = env.step(applied)
            observation = step.observation
            actual_steps = step_index
            if step_index % PROGRESS_EMIT_EVERY_STEPS == 0 or step.done:
                emit_progress(
                    "rollout_step",
                    mode="mixture",
                    current=min(step_index, expected_steps),
                    total=expected_steps,
                    source=str(source),
                    teacher_fraction=float(teacher_fraction),
                )
            if step.done:
                break
        else:
            raise RuntimeError("finger-agnostic DAgger rollout exceeded step budget")

    sequence = v060.v055.DAggerSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        source=source,
    )
    stable = v056._make_stable_sequence(
        sequence,
        press_recovery_cap=press_recovery_cap,
        expert=False,
    )
    emit_progress("rollout_done", mode="mixture", steps=actual_steps, source=str(source))
    return v056.StableRollout(
        sequence=stable,
        evaluation=v054.StudentEvalResult(env.stats, env.physical_keydowns),
        teacher_fraction=teacher_fraction,
    )


def _eval_label(segments) -> str:
    roles = {str(getattr(named, "role", "eval")) for named in segments}
    if roles and all(role.startswith("anchor-") for role in roles):
        return "Anchor eval"
    if roles == {"validation"}:
        return "Validation eval"
    if roles == {"validation-full"}:
        return "Validation full"
    if roles == {"final-holdout"}:
        return "Final holdout"
    if roles == {"round-train"}:
        return "Train eval"
    return "/".join(sorted(roles))[:32] or "Eval"


def _progress_evaluate_states_on_segments(
    model,
    states,
    segments,
    *,
    same_hand: bool,
    control_dt_s: float,
):
    """Exact v0.8 flat evaluator with cache/task completion events."""

    if not segments:
        return {}

    result = {}
    pending = {}
    digests = {alpha: v080.v065._state_digest(state) for alpha, state in states.items()}

    for alpha, state in states.items():
        digest = digests[alpha]
        for named in segments:
            cache_key = v080._eval_cache_key(digest, named)
            cached = v080._EVAL_CACHE.get(cache_key)
            if cached is not None:
                result[(alpha, named.key)] = cached
                v080._EVAL_CACHE_HITS += 1
                continue
            pending[(alpha, named.key)] = (state, named, cache_key)

    total = len(states) * len(segments)
    cached_count = total - len(pending)
    visible = _GUARD_DEPTH == 0
    label = _eval_label(segments)
    if visible:
        emit_progress(
            "eval_start",
            label=label,
            current=cached_count,
            total=total,
            cached=cached_count,
            workers=v080._configured_workers(),
        )

    completed = cached_count
    if pending:
        if v080._configured_workers() <= 1:
            for key, (state, named, cache_key) in pending.items():
                raw = v080.evaluate_hud_state_on_segment(
                    state,
                    int(model.hidden_dim),
                    named.segment,
                    bool(same_hand),
                    float(control_dt_s),
                )
                evaluated = v054.StudentEvalResult(raw[0], raw[1])
                result[key] = evaluated
                v080._EVAL_CACHE[cache_key] = evaluated
                v080._EVAL_CACHE_MISSES += 1
                completed += 1
                if visible:
                    emit_progress("eval_step", label=label, current=completed, total=total)
        else:
            pool = v080._get_pool()
            futures = {
                key: (
                    pool.submit(
                        v080.evaluate_hud_state_on_segment,
                        state,
                        int(model.hidden_dim),
                        named.segment,
                        bool(same_hand),
                        float(control_dt_s),
                    ),
                    cache_key,
                )
                for key, (state, named, cache_key) in pending.items()
            }
            for key, (future, cache_key) in futures.items():
                raw = future.result()
                evaluated = v054.StudentEvalResult(raw[0], raw[1])
                result[key] = evaluated
                v080._EVAL_CACHE[cache_key] = evaluated
                v080._EVAL_CACHE_MISSES += 1
                completed += 1
                if visible:
                    emit_progress("eval_step", label=label, current=completed, total=total)

    if visible:
        emit_progress("eval_done", label=label, total=total, cached=cached_count)
    return result


def _progress_line_search(*args, **kwargs):
    """Wrap the installed fast/timed line search and expose each guard stage."""

    global _GUARD_DEPTH
    assert _ORIGINAL_LINE_SEARCH is not None
    assert _ORIGINAL_GROUPED_EVAL is not None

    validation_total = len(kwargs.get("validation_segments", ()))
    anchor_total = len(kwargs.get("anchor_segments", ()))
    emit_progress(
        "guard_start",
        alphas=len(v080.v058.DEFAULT_TRUST_ALPHAS),
        validation_total=validation_total,
        anchor_total=anchor_total,
    )

    validation_current = 0
    anchor_current = 0
    original_eval = v080_fast._evaluate_state_segment_groups

    def traced_eval(model, states, segments, *, same_hand: bool, control_dt_s: float, digests=None):
        nonlocal validation_current, anchor_current
        role = str(getattr(segments[0], "role", "eval")) if segments else "eval"
        if role == "round-train":
            phase = "train"
            emit_progress(
                "guard_phase_start",
                phase=phase,
                current=0,
                total=len(states),
                states=len(states),
            )
        elif role == "validation":
            phase = "validation"
            emit_progress(
                "guard_phase_start",
                phase=phase,
                current=validation_current,
                total=validation_total,
                states=len(states),
            )
        elif role.startswith("anchor-"):
            phase = "anchor"
            emit_progress(
                "guard_phase_start",
                phase=phase,
                current=anchor_current,
                total=anchor_total,
                states=len(states),
            )
        else:
            phase = role

        result = original_eval(
            model,
            states,
            segments,
            same_hand=same_hand,
            control_dt_s=control_dt_s,
            digests=digests,
        )

        if phase == "train":
            emit_progress(
                "guard_phase_step",
                phase=phase,
                current=len(states),
                total=len(states),
                states=len(states),
            )
        elif phase == "validation":
            validation_current = min(validation_total, validation_current + len(segments))
            emit_progress(
                "guard_phase_step",
                phase=phase,
                current=validation_current,
                total=validation_total,
                states=len(states),
            )
        elif phase == "anchor":
            anchor_current = min(anchor_total, anchor_current + len(segments))
            emit_progress(
                "guard_phase_step",
                phase=phase,
                current=anchor_current,
                total=anchor_total,
                states=len(states),
            )
        return result

    _GUARD_DEPTH += 1
    v080_fast._evaluate_state_segment_groups = traced_eval
    try:
        result = _ORIGINAL_LINE_SEARCH(*args, **kwargs)
    finally:
        v080_fast._evaluate_state_segment_groups = original_eval
        _GUARD_DEPTH -= 1

    chosen = result[0] if isinstance(result, tuple) and result else None
    emit_progress(
        "guard_done",
        accepted=chosen is not None,
        alpha=None if chosen is None else float(chosen.alpha),
    )
    return result


def install_progress_instrumentation() -> None:
    """Install observational callbacks before v0.7/v0.8 runtime patching."""

    global _INSTALLED, _ORIGINAL_LINE_SEARCH, _ORIGINAL_GROUPED_EVAL
    if _INSTALLED:
        return

    _ORIGINAL_LINE_SEARCH = v080_fast._fast_line_search
    _ORIGINAL_GROUPED_EVAL = v080_fast._evaluate_state_segment_groups

    v057._train_one_epoch = _progress_train_one_epoch
    v060._collect_expert_sequence = _progress_collect_expert_sequence
    v060._collect_mixture_rollout = _progress_collect_mixture_rollout
    v080._evaluate_states_on_segments = _progress_evaluate_states_on_segments
    v080_fast._fast_line_search = _progress_line_search

    # train_real_chart_v090_turbo captures the original expert collector at
    # import time.  If it is already imported, redirect that fallback too so
    # round-local expert collection emits progress while cached anchor teachers
    # remain untouched.
    turbo = sys.modules.get("train_real_chart_v090_turbo")
    if turbo is not None and hasattr(turbo, "_ORIGINAL_COLLECT_EXPERT"):
        turbo._ORIGINAL_COLLECT_EXPERT = _progress_collect_expert_sequence

    _INSTALLED = True
    print(f"progress-internals={PROGRESS_VERSION} bc=sequence/chunk guard=staged eval=task")

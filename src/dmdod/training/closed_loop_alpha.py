from __future__ import annotations

"""Opt-in fixed-Validation closed-loop weight-interpolation diagnostic.

Neither optimizer updates nor checkpoint selection depend on this module.
"""

from dataclasses import dataclass
import math
from typing import Callable

import torch

from dmdod.envs.n_key import NKeyOverloadTrace
from dmdod.training.real_chart import evaluate_role_continuous


ALPHAS = (0.0, 0.125, 0.25, 0.5, 1.0)
FOCUS_ANCHORS = (5, 7, 9, 13)
EndpointCounts = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class AlphaAnchorResult:
    index: int
    chart_name: str
    hits: int
    too_early: int
    keydowns: int
    overloaded: bool
    termination_s: float
    termination_reason: str
    overload_gauge: float

    @property
    def safe(self) -> bool:
        return not self.overloaded


@dataclass(frozen=True, slots=True)
class AlphaSweepRow:
    alpha: float
    summary: dict
    anchors: tuple[AlphaAnchorResult, ...]


def _state_parameter_keys(model) -> set[str]:
    return {name for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def _state_tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Exact comparison for dense and sparse state, without densifying matrices."""
    if a.shape != b.shape or a.dtype != b.dtype or a.layout != b.layout:
        return False
    a, b = a.detach(), b.detach()
    if a.device != b.device:
        a, b = a.cpu(), b.cpu()
    if a.layout == torch.strided:
        return bool(torch.equal(a, b))
    if a.layout == torch.sparse_coo:
        # COOs can be uncoalesced; canonicalize before comparing structure.
        a, b = a.coalesce(), b.coalesce()
        return bool(
            torch.equal(a.indices(), b.indices())
            and torch.equal(a.values(), b.values())
        )
    compressed = {
        torch.sparse_csr: ("crow_indices", "col_indices"),
        torch.sparse_csc: ("ccol_indices", "row_indices"),
        torch.sparse_bsr: ("crow_indices", "col_indices"),
        torch.sparse_bsc: ("ccol_indices", "row_indices"),
    }
    if a.layout not in compressed:
        raise TypeError(f"unsupported model state tensor layout: {a.layout}")
    row_attr, col_attr = compressed[a.layout]
    return bool(
        torch.equal(getattr(a, row_attr)(), getattr(b, row_attr)())
        and torch.equal(getattr(a, col_attr)(), getattr(b, col_attr)())
        and torch.equal(a.values(), b.values())
    )


def interpolate_state(
    model,
    best: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Interpolate trainable floating parameters; require all other state unchanged."""
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"unsupported alpha: {alpha}")
    current = model.state_dict()
    if set(best) != set(candidate) or set(best) != set(current):
        raise ValueError("alpha sweep model state keys mismatch")
    trainable = _state_parameter_keys(model)
    if not trainable.issubset(best):
        raise ValueError("trainable parameters absent from model state")

    for key in best:
        a, b, now = best[key], candidate[key], current[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.shape != now.shape or a.dtype != now.dtype:
            raise ValueError(f"alpha sweep state shape/dtype mismatch: {key}")
        if key in trainable and a.is_floating_point() and a.layout != torch.strided:
            raise ValueError(f"alpha sweep trainable sparse parameter unsupported: {key}")
        if (key not in trainable or not a.is_floating_point()) and not _state_tensor_equal(a, b):
            raise ValueError(f"alpha sweep frozen/buffer/nonfloating state changed: {key}")

    if alpha == 0.0:
        return {key: value.detach().clone() for key, value in best.items()}
    if alpha == 1.0:
        return {key: value.detach().clone() for key, value in candidate.items()}
    output = {}
    for key, a in best.items():
        b = candidate[key]
        if key in trainable and a.is_floating_point():
            # No arithmetic on frozen state or integer counters.
            output[key] = a + float(alpha) * (b - a)
        else:
            output[key] = a.detach().clone()
    return output


def _load_checked_state(model, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=True)
    model.prepare_recurrent_runtime()
    actual = model.state_dict()
    if set(actual) != set(state) or any(
        not _state_tensor_equal(actual[key], expected)
        for key, expected in state.items()
    ):
        raise RuntimeError("alpha sweep loaded model state does not match requested state")


def endpoint_counts(results) -> EndpointCounts:
    return (
        sum(not bool(stats.overloaded) for stats, _ in results),
        sum(int(stats.hits) for stats, _ in results),
        sum(int(stats.too_early_presses) for stats, _ in results),
        sum(int(downs) for _, downs in results),
    )


def check_endpoint(
    alpha: float,
    results,
    traces: tuple[NKeyOverloadTrace, ...],
    named_segments,
    expected: EndpointCounts,
    *,
    reference_results,
    reference_traces: tuple[NKeyOverloadTrace, ...],
) -> None:
    """Fail closed on aggregate, per-anchor or termination reproducibility."""
    if len(results) != len(named_segments) or len(traces) != len(named_segments):
        raise RuntimeError(f"alpha={alpha:g} endpoint count mismatch")
    if endpoint_counts(results) != expected:
        raise RuntimeError(
            f"alpha={alpha:g} endpoint FAIL: actual={endpoint_counts(results)} "
            f"expected={expected} [SAFE, Hits, TooEarly, Keydowns]"
        )
    if len(reference_results) != len(results) or len(reference_traces) != len(traces):
        raise RuntimeError(f"alpha={alpha:g} endpoint reference count mismatch")
    for index, ((a, down_a), (b, down_b), ta, tb) in enumerate(
        zip(results, reference_results, traces, reference_traces), 1
    ):
        # All scoring fields must match, not just aggregate SAFE/Hits.
        if a != b or int(down_a) != int(down_b) or ta != tb:
            raise RuntimeError(f"alpha={alpha:g} endpoint anchor #{index:02d} FAIL")


def _anchor_records(results, traces, segments) -> tuple[AlphaAnchorResult, ...]:
    if len(results) != len(traces) or len(results) != len(segments):
        raise RuntimeError("alpha sweep Validation result/trace length mismatch")
    records = []
    for index, ((stats, keydowns), trace, named) in enumerate(
        zip(results, traces, segments), 1
    ):
        terminal = trace.termination
        if (int(terminal.hits) != int(stats.hits)
            or int(terminal.too_early) != int(stats.too_early_presses)
            or int(terminal.keydowns) != int(keydowns)):
            raise RuntimeError(f"alpha sweep trace/stats mismatch at anchor #{index:02d}")
        records.append(AlphaAnchorResult(
            index=index,
            chart_name=str(named.chart_name),
            hits=int(stats.hits),
            too_early=int(stats.too_early_presses),
            keydowns=int(keydowns),
            overloaded=bool(stats.overloaded),
            termination_s=float(terminal.time_s),
            termination_reason=str(terminal.reason),
            overload_gauge=float(terminal.overload_value),
        ))
    return tuple(records)



def _validate_alphas(alphas: tuple[float, ...]) -> None:
    if (len(alphas) < 2 or alphas[0] != 0.0 or alphas[-1] != 1.0
        or any(not math.isfinite(x) or not 0.0 <= x <= 1.0 for x in alphas)
        or any(b <= a for a, b in zip(alphas, alphas[1:]))):
        raise ValueError("alpha sweep grid must increase strictly from 0 to 1")


def first_ordinal_keydown_difference(
    left, right, *, tolerance_s: float = 0.010,
):
    """Return (zero-based ordinal, left event, right event), or None.

    This is an ordinal comparison, not causal event matching: an extra
    KeyDown can shift the remainder of the stream.
    """
    if not math.isfinite(tolerance_s) or tolerance_s < 0.0:
        raise ValueError("invalid KeyDown timing tolerance")
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if (a is None or b is None or a.key != b.key
            or abs(a.time_s - b.time_s) > tolerance_s):
            return index, a, b
    return None


def _keydown_label(event) -> str:
    if event is None:
        return "none"
    return f"t={event.time_s:.3f}s {event.key} target={event.target_index}"


def format_keydown_transitions(
    physical_by_alpha: dict[float, tuple],
    *, anchor_index: int = 9,
) -> tuple[str, ...]:
    """#09 first physical-input differences; observational only."""
    pairs = ((0.0, 0.125), (0.125, 0.25), (0.0, 0.25))
    if not all(alpha in physical_by_alpha for pair in pairs for alpha in pair):
        raise ValueError("KeyDown comparison needs alphas 0, 0.125, 0.25")
    lines = [
        f"=== anchor #{anchor_index:02d} first physical KeyDown differences ===",
        "Ordinal events, same key within 10ms is treated as aligned.",
    ]
    for left_alpha, right_alpha in pairs:
        left, right = physical_by_alpha[left_alpha], physical_by_alpha[right_alpha]
        diff = first_ordinal_keydown_difference(left, right)
        prefix = f"alpha={left_alpha:.6f} vs alpha={right_alpha:.6f}"
        if diff is None:
            lines.append(f"{prefix}: no ordinal divergence ({len(left)} KeyDowns)")
            continue
        index, a, b = diff
        lines.append(
            f"{prefix}: first ordinal divergence KeyDown #{index + 1}: "
            f"{_keydown_label(a)} | {_keydown_label(b)}"
        )
        t = min(x.time_s for x in (a, b) if x is not None)
        for name, stream in ((f"{left_alpha:.6f}", left), (f"{right_alpha:.6f}", right)):
            near = tuple(e for e in stream if abs(e.time_s - t) <= 0.100)
            lines.append(
                f"  alpha={name} events within +/-100ms of {t:.3f}s "
                f"({len(near)}): "
                + (", ".join(_keydown_label(e) for e in near[:12]) or "none")
            )
    lines.append(
        "NOTE: insertion, omission and timing shifts can change event ordinals; "
        "this is not proof of the first causal divergence."
    )
    return tuple(lines)


def format_sweep_results(rows: tuple[AlphaSweepRow, ...]) -> tuple[str, ...]:
    lines = ["=== closed-loop alpha sweep ==="]
    for row in rows:
        s = row.summary
        mae = s["mae_ms"]
        mae_str = "inf" if mae == float("inf") else f"{mae:.2f}"
        lines.append(
            f"alpha={row.alpha:.6f} SAFE={s['safe']}/{s['anchors']} "
            f"H={s['hits']}/{s['targets']} X={s['x']:.2f}% "
            f"PP={s['pp']:.2f}% MAE={mae_str}ms "
            f"Early={s['early']} Keydowns={s['keydowns']} "
            f"Overloads={sum(a.overloaded for a in row.anchors)}"
        )
    lines.append("=== anchor transitions ===")
    for index in FOCUS_ANCHORS:
        if not rows or len(rows[0].anchors) < index:
            raise RuntimeError(f"alpha sweep missing requested anchor #{index:02d}")
        name = rows[0].anchors[index - 1].chart_name
        lines.append(f"#{index:02d} {name}:")
        for row in rows:
            a = row.anchors[index - 1]
            lines.append(
                f"  alpha={row.alpha:.6f} SAFE={int(a.safe)} "
                f"H={a.hits} Early={a.too_early} Keydowns={a.keydowns} "
                f"Overload={int(a.overloaded)} end={a.termination_s:.3f}s "
                f"reason={a.termination_reason} gauge={a.overload_gauge:.3f}"
            )
        lines.append(
            "  SAFE: " + " -> ".join(
                f"{row.alpha:.6f}:{'SAFE' if row.anchors[index - 1].safe else 'OVERLOAD'}"
                for row in rows
            )
        )
    lines.append("=== endpoint reproducibility ===")
    lines.append("alpha=0: PASS (existing best rollout; exact state loaded)")
    lines.append("alpha=1: PASS (existing candidate rollout; exact state loaded)")
    lines.append("NOTE: endpoint rollouts are reused, not independently repeated.")
    return tuple(lines)


def run_closed_loop_alpha_sweep(
    model,
    prepared,
    *,
    best_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
    best_results,
    candidate_results,
    best_traces: tuple[NKeyOverloadTrace, ...],
    candidate_traces: tuple[NKeyOverloadTrace, ...],
    expected_best: EndpointCounts,
    expected_candidate: EndpointCounts,
    validation_summary: Callable,
    evaluate: Callable = evaluate_role_continuous,
    alphas: tuple[float, ...] | None = None,
    compare_transitions: bool = False,
) -> tuple[AlphaSweepRow, ...]:
    """Evaluate specified intermediate alpha points; reuse existing endpoints."""
    alphas = ALPHAS if alphas is None else tuple(alphas)
    _validate_alphas(alphas)
    if compare_transitions and not {0.0, 0.125, 0.25}.issubset(alphas):
        raise ValueError("KeyDown comparisons need alpha=0, 0.125 and 0.25")
    if len(prepared.validation) != 20:
        raise RuntimeError("closed-loop alpha sweep requires fixed 20 Validation anchors")
    original_candidate = {k: v.detach().clone() for k, v in candidate_state.items()}
    # Preflight state compatibility before any model loading.
    interpolate_state(model, best_state, candidate_state, 0.5)
    records: list[AlphaSweepRow] = []
    physical_by_alpha: dict[float, tuple] = {}
    try:
        for alpha in alphas:
            state = interpolate_state(model, best_state, candidate_state, alpha)
            _load_checked_state(model, state)
            if alpha == 0.0:
                results = best_results
                traces = tuple(best_traces)
                reference_results = best_results
                reference_traces = tuple(best_traces)
                expected = expected_best
            elif alpha == 1.0:
                results = candidate_results
                traces = tuple(candidate_traces)
                reference_results = candidate_results
                reference_traces = tuple(candidate_traces)
                expected = expected_candidate
            else:
                captured: list[NKeyOverloadTrace] = []
                results = evaluate(
                    model,
                    prepared.validation,
                    label=f"alpha-{alpha:.3f}",
                    control_dt_s=prepared.control_dt_s,
                    physics_dt_s=prepared.physics_dt_s,
                    device=prepared.device,
                    verbose=False,
                    trace_collector=captured,
                )
                traces = tuple(captured)
            if alpha in (0.0, 1.0):
                check_endpoint(
                    alpha, results, traces, prepared.validation, expected,
                    reference_results=reference_results,
                    reference_traces=reference_traces,
                )
            if compare_transitions:
                physical_by_alpha[alpha] = tuple(traces[8].physical_keydowns)
            summary = validation_summary(results)
            records.append(AlphaSweepRow(
                alpha=alpha,
                summary=dict(summary),
                anchors=_anchor_records(results, traces, prepared.validation),
            ))
    except Exception as exc:
        print("=== endpoint reproducibility ===")
        print(f"alpha sweep: FAIL ({exc})")
        raise
    finally:
        _load_checked_state(model, original_candidate)
    output = tuple(records)
    for line in format_sweep_results(output):
        print(line)
    if compare_transitions:
        for line in format_keydown_transitions(physical_by_alpha):
            print(line)
    return output

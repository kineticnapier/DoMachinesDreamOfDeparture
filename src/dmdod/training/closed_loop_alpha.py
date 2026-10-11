from __future__ import annotations

"""Opt-in fixed-Validation closed-loop weight-interpolation diagnostic.

Neither optimizer updates nor checkpoint selection depend on this module.
"""

from dataclasses import dataclass
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


def interpolate_state(
    model,
    best: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Interpolate trainable floating parameters; require all other state unchanged."""
    if float(alpha) not in ALPHAS:
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
        if (key not in trainable or not a.is_floating_point()) and not torch.equal(a, b):
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
        not torch.equal(actual[key].detach().cpu(), expected.detach().cpu())
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


def format_sweep_results(rows: tuple[AlphaSweepRow, ...]) -> tuple[str, ...]:
    lines = ["=== closed-loop alpha sweep ==="]
    for row in rows:
        s = row.summary
        mae = s["mae_ms"]
        mae_str = "inf" if mae == float("inf") else f"{mae:.2f}"
        lines.append(
            f"alpha={row.alpha:.3f} SAFE={s['safe']}/{s['anchors']} "
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
                f"  alpha={row.alpha:.3f} SAFE={int(a.safe)} "
                f"H={a.hits} Early={a.too_early} Keydowns={a.keydowns} "
                f"Overload={int(a.overloaded)} end={a.termination_s:.3f}s "
                f"reason={a.termination_reason} gauge={a.overload_gauge:.3f}"
            )
        lines.append(
            "  SAFE: " + " -> ".join(
                f"{row.alpha:.3f}:{'SAFE' if row.anchors[index - 1].safe else 'OVERLOAD'}"
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
) -> tuple[AlphaSweepRow, ...]:
    """Only three intermediate rollouts; endpoints reuse existing Validation."""
    if len(prepared.validation) != 20:
        raise RuntimeError("closed-loop alpha sweep requires fixed 20 Validation anchors")
    original_candidate = {k: v.detach().clone() for k, v in candidate_state.items()}
    # Preflight state compatibility before any model loading.
    interpolate_state(model, best_state, candidate_state, 0.5)
    records: list[AlphaSweepRow] = []
    try:
        for alpha in ALPHAS:
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
    return output

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dmdod.envs.n_key import EpisodeTerminationTrace, NKeyOverloadTrace
from dmdod.training.closed_loop_alpha import (
    ALPHAS,
    _anchor_records,
    check_endpoint,
    endpoint_counts,
    format_sweep_results,
    interpolate_state,
    run_closed_loop_alpha_sweep,
)
from dmdod.training.config import load_training_config
from dmdod.training.human_visible_curriculum import validation_rank


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 3.0]))
        self.frozen = torch.nn.Parameter(
            torch.tensor([9.0]), requires_grad=False
        )
        self.register_buffer("int_buffer", torch.tensor([4], dtype=torch.int64))
        self.register_buffer("float_buffer", torch.tensor([2.0]))
        self.loads = 0
        self.states_initialized = 0

    def prepare_recurrent_runtime(self):
        self.loads += 1

    def initial_state(self, device):
        self.states_initialized += 1
        return torch.zeros(1)

    def snapshot(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}


def _states():
    model = _TinyModel()
    best = model.snapshot()
    candidate = {k: v.clone() for k, v in best.items()}
    candidate["weight"] = torch.tensor([3.0, -1.0])
    return model, best, candidate


def _trace(hits, early, downs, overloaded, *, end):
    return NKeyOverloadTrace(
        termination=EpisodeTerminationTrace(
            time_s=end,
            reason="Overload" if overloaded else "Time limit",
            hits=hits,
            too_early=early,
            keydowns=downs,
            next_target_index=4,
            overload_value=1.5 if overloaded else 0.0,
        ),
        keydown_times_s=tuple(float(i) / 10.0 for i in range(downs)),
        too_early_events=(),
    )


def _validation_results(safe_count, hits, early, downs, *, end=4.0):
    segments = [SimpleNamespace(chart_name=f"chart{i:02d}") for i in range(1, 21)]
    results = []
    traces = []
    for index in range(20):
        safe = index < safe_count
        stat = SimpleNamespace(
            hits=hits + index,
            targets=300,
            too_early_presses=early + index,
            overloaded=not safe,
        )
        physical_downs = downs + index
        results.append((stat, physical_downs))
        traces.append(_trace(
            stat.hits, stat.too_early_presses, physical_downs,
            stat.overloaded, end=end + index * .01,
        ))
    return segments, results, tuple(traces)


def _summary(results):
    return {
        "safe": sum(not x.overloaded for x, _ in results),
        "anchors": len(results),
        "hits": sum(x.hits for x, _ in results),
        "targets": sum(x.targets for x, _ in results),
        "x": 50.0,
        "pp": 25.0,
        "mae_ms": 30.0,
        "early": sum(x.too_early_presses for x, _ in results),
        "keydowns": sum(downs for _, downs in results),
    }


def test_alpha_endpoints_exact_without_arithmetic():
    model, best, candidate = _states()
    for alpha, expected in ((0.0, best), (1.0, candidate)):
        state = interpolate_state(model, best, candidate, alpha)
        assert all(torch.equal(state[k], expected[k]) for k in expected)
        assert all(state[k] is not expected[k] for k in expected)


def test_alpha_half_is_correct_and_preserves_frozen_parameters_and_buffers():
    model, best, candidate = _states()
    middle = interpolate_state(model, best, candidate, 0.5)
    assert torch.equal(middle["weight"], torch.tensor([2.0, 1.0]))
    for key in ("frozen", "int_buffer", "float_buffer"):
        assert torch.equal(middle[key], best[key])
    assert middle["int_buffer"].dtype == torch.int64


@pytest.mark.parametrize("name", ["frozen", "int_buffer", "float_buffer"])
def test_changed_frozen_state_is_rejected(name):
    model, best, candidate = _states()
    candidate[name] = candidate[name] + 1
    with pytest.raises(ValueError, match="frozen/buffer/nonfloating"):
        interpolate_state(model, best, candidate, 0.5)


def test_mismatched_shapes_and_unsupported_alpha_are_rejected():
    model, best, candidate = _states()
    with pytest.raises(ValueError, match="unsupported alpha"):
        interpolate_state(model, best, candidate, 0.3)
    candidate["weight"] = torch.zeros(4)
    with pytest.raises(ValueError, match="shape/dtype"):
        interpolate_state(model, best, candidate, 0.25)


def _run(*, incorrect_endpoint=False, failing_intermediate=False):
    model, best, candidate = _states()
    segments, best_results, best_traces = _validation_results(12, 80, 5, 100)
    _, candidate_results, candidate_traces = _validation_results(9, 70, 4, 90)
    expected_best = endpoint_counts(best_results)
    expected_candidate = endpoint_counts(candidate_results)
    if incorrect_endpoint:
        expected_candidate = replace_counts(expected_candidate, hit_delta=1)
    evaluated = []

    def evaluator(model, segments, *, trace_collector, **kwargs):
        alpha = (float(model.weight[0]) - 1.0) / 2.0
        evaluated.append(alpha)
        # This mirrors the existing per-anchor _evaluate_continuous() contract:
        # a new policy.initial_state() for every chart evaluation.
        for index in range(len(segments)):
            model.initial_state(kwargs["device"])
        safe_count = 12 if alpha < .25 else 9
        _, results, traces = _validation_results(
            safe_count, 78, 4, 99, end=4.0 - alpha
        )
        trace_collector.extend(traces)
        return results

    if failing_intermediate:
        def failing(model, segments, *, trace_collector, **kwargs):
            raise RuntimeError("simulated evaluator failure")
        evaluator = failing

    prepared = SimpleNamespace(
        validation=segments, control_dt_s=.01, physics_dt_s=.001,
        device=torch.device("cpu"),
    )
    model.load_state_dict(candidate)
    try:
        rows = run_closed_loop_alpha_sweep(
            model, prepared, best_state=best, candidate_state=candidate,
            best_results=best_results, candidate_results=candidate_results,
            best_traces=best_traces, candidate_traces=candidate_traces,
            expected_best=expected_best,
            expected_candidate=expected_candidate,
            validation_summary=_summary, evaluate=evaluator,
        )
        return model, candidate, rows, evaluated
    finally:
        assert all(torch.equal(model.state_dict()[k], candidate[k]) for k in candidate)


def replace_counts(counts, *, hit_delta):
    return (counts[0], counts[1] + hit_delta, counts[2], counts[3])


def test_sweep_evaluates_only_three_intermediate_alphas_and_restores_model(capsys):
    model, candidate, rows, evaluated = _run()
    assert tuple(row.alpha for row in rows) == ALPHAS
    assert evaluated == pytest.approx([0.125, 0.25, 0.5])
    assert model.states_initialized == 60
    assert model.loads >= 6  # 5 alpha states plus candidate restoration
    assert all(torch.equal(model.state_dict()[k], candidate[k]) for k in candidate)
    output = capsys.readouterr().out
    assert "=== closed-loop alpha sweep ===" in output
    assert "=== anchor transitions ===" in output
    assert "#09 chart09" in output
    assert "=== endpoint reproducibility ===" in output
    assert "alpha=0: PASS" in output and "alpha=1: PASS" in output
    assert "alpha=0.125" in output


def test_endpoint_failure_is_fatal_and_model_restored(capsys):
    with pytest.raises(RuntimeError, match="endpoint FAIL"):
        _run(incorrect_endpoint=True)
    assert "alpha sweep: FAIL" in capsys.readouterr().out


def test_evaluation_exception_also_restores_candidate():
    with pytest.raises(RuntimeError, match="simulated evaluator"):
        _run(failing_intermediate=True)


def test_anchor_overload_end_time_is_preserved_in_results():
    segments, results, traces = _validation_results(9, 49, 11, 60, end=3.888)
    records = _anchor_records(results, traces, segments)
    assert not records[8].safe
    assert records[8].overloaded
    assert records[8].termination_s == pytest.approx(3.968)
    assert records[8].termination_reason == "Overload"
    assert records[8].overload_gauge == pytest.approx(1.5)


def test_anchor_endpoint_mismatch_is_detected():
    segments, results, traces = _validation_results(12, 80, 5, 100)
    changed = list(results)
    changed[8] = (SimpleNamespace(
        hits=20, too_early_presses=9, overloaded=True, targets=300
    ), 200)
    with pytest.raises(RuntimeError, match="anchor #09"):
        check_endpoint(
            0.0, changed, traces, segments, endpoint_counts(changed),
            reference_results=results, reference_traces=traces,
        )


def test_optional_alpha_diagnostic_default_off_does_not_change_ranking():
    cfg = load_training_config("configs/training/human_visible_open_loop_smoke.toml")
    assert not cfg.human_visible.alpha_sweep_enabled
    best = {
        "safe": 12, "hits": 1786, "x": 50.0,
        "pp": 25.0, "mae_ms": 31.0, "early": 278,
    }
    candidate = dict(best, safe=9, hits=1546)
    assert validation_rank(candidate) < validation_rank(best)


def test_dedicated_alpha_smoke_is_opt_in_and_has_expected_endpoints():
    cfg = load_training_config("configs/training/human_visible_alpha_sweep_smoke.toml")
    h = cfg.human_visible
    assert h.alpha_sweep_enabled
    assert h.alpha_sweep_expected_best == (12, 1786, 278, 2064)
    assert h.alpha_sweep_expected_candidate == (9, 1546, 250, 1796)
    assert cfg.data.anchor_limit == cfg.data.validation_limit == 20
    assert h.updates_per_epoch == 2
    assert h.max_epochs == 1
    assert cfg.run.output.endswith("real_chart_humanvisible_alpha_sweep_smoke.pt")


def test_existing_closed_loop_evaluator_reinitializes_rnn_per_anchor():
    from dmdod.adofai_chart import parse_adofai_text
    from dmdod.adofai_timing import compile_adofai
    from dmdod.adofai_playable import build_playable_segment
    from dmdod.training.real_chart import NamedSegment, evaluate_role_continuous

    class IdlePolicy:
        key_count = 4

        def __init__(self):
            self.reset_count = 0

        def initial_state(self, device):
            self.reset_count += 1
            return torch.zeros(1, device=device)

        def eval(self):
            return self

        def forward_step(self, input_tensor, state):
            return torch.zeros(4, device=input_tensor.device), None, None, state

    chart = parse_adofai_text(
        '{"angleData":[0,90,0,0],"settings":{"bpm":120,"pitch":100,'
        '"countdownTicks":0,"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    segment = build_playable_segment(
        compiled, start_s=0.0, end_s=compiled.duration_s
    )
    named = NamedSegment(
        "validation", "tiny", "sha", 0.0, segment.duration_s, segment
    )
    policy = IdlePolicy()
    traces = []
    result = evaluate_role_continuous(
        policy, [named, named],
        label="rnn-reset-test", control_dt_s=0.01,
        physics_dt_s=0.001, device=torch.device("cpu"),
        verbose=False, trace_collector=traces,
    )
    assert policy.reset_count == 2
    assert len(result) == len(traces) == 2
    assert result[0] == result[1]

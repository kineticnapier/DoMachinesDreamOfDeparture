from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "stock-runtime-probe" / "Main.cs"


def test_probe_uses_exact_audited_overloads_and_read_only_harmony_calls() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    expected = (
        'typeof(scrPlayer), "Simulated_PlayerControl_Update", new[] { NullableUlong }',
        'typeof(scrPlayer), "CheckPostHoldFail", new[] { NullableUlong }',
        'typeof(scrController), "ProcessKeyInputs", new[] { typeof(ulong) }',
        'typeof(scrPlanet), "SwitchChosen", Type.EmptyTypes',
        'typeof(scrMarginTracker), "AddHit", new[] { typeof(HitMargin) }',
    )
    assert all(fragment in text for fragment in expected)
    assert "harmony.Patch(original, prefix, postfix)" in text
    assert "__result =" not in text
    assert "__runOriginal" not in text


def test_probe_keeps_internal_floor_and_counts_evaluator_only() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert '"floor_evaluator_only"' in text
    assert '"counter_snapshot_internal_evaluator_only"' in text
    assert '["visible_counter_state"] = null' in text
    assert '["student_visible_fields"] = new object[0]' in text


def test_logger_exceptions_are_isolated_and_writer_is_background_thread() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    assert "catch (Exception exception)" in text
    assert "EmitHookError" in text
    assert "IsBackground = true" in text
    assert "maximumQueuedLines" in text

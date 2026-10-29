from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_dmdogo_h12_blind_review import (
    FORMAT_H11, FORMAT_PUBLIC, FORMAT_KEY, compile_blind,
    render_html, main,
)


def fixture_source():
    # Full H1.1-shaped synthetic source with the exact predeclared
    # chart and target census; not a fabricated human demonstration.
    cohort_sizes = (1540, 3637, 5054)
    target_counts = (10, 10, 2)
    charts = []
    k = 0
    for ci, (n, targets) in enumerate(zip(cohort_sizes, target_counts)):
        rows = [
            {"event_index": i,
             "due_s": 1.0 + i*.12,
             "chord_size": 1,
             "original_axes": ["left_1" if i%2 == 0 else "right_1"]}
            for i in range(n)]
        candidates = []
        for j in range(targets):
            swap_at = 2+j
            original = rows[swap_at]["original_axes"]
            alternate = ["right_2"] if original == ["left_1"] else ["left_2"]
            all_axes = [x["original_axes"][:] for x in rows]
            all_axes[swap_at] = alternate
            score = -0.5 if k < 7 else 0.2
            candidates.append({
                "candidate_id": f"original-chart{ci}-join{j}",
                "flag_pair": [swap_at-1, swap_at],
                "join_window_index": 1,
                "changed_chunk_indices": [1],
                "changed_event_count": 1,
                "changed_events": [{
                    "event_index": swap_at,
                    "due_s": rows[swap_at]["due_s"],
                    "original_axes": original,
                    "alternative_axes": alternate,
                }],
                "alternative_axes_by_event_order": all_axes,
                "independent_intervention_only": True,
                "alternate_motor_result": "not_executed",
                "human_preference": "unobserved",
                "cost": {"final_delta": score},
            })
            k += 1
        charts.append({
            "chart": f"Sample chart {ci}",
            "chart_sha256": f"{ci+1:064x}",
            "event_count": n,
            "original_events_by_index": rows,
            "reconstructions": candidates,
        })
    return {
        "format": FORMAT_H11,
        "chart_count": 3,
        "target_count": 22,
        "strict_lower_heuristic_cost_count": 7,
        "complete_event_axis_assignments": True,
        "alternate_motor_executed": False,
        "human_preference_observed": False,
        "training_performed": False,
        "validation20_or_final_accessed": False,
        "frozen_planner_motor_timing_changed": False,
        "charts": charts,
    }


@pytest.fixture(scope="module")
def prepared():
    source = fixture_source()
    public, key = compile_blind(
        source, source_sha256="a"*64, seed=b"K"*32)
    return source, public, key


def test_balanced_22_paired_cases(prepared):
    _, public, key = prepared
    assert public["format"] == FORMAT_PUBLIC
    assert key["format"] == FORMAT_KEY
    assert public["packet_id"] == key["packet_id"]
    assert public["case_count"] == 22
    assert len(public["cases"]) == 22
    assert key["original_on_A_count"] == 11
    assert key["original_on_B_count"] == 11
    assert len({r["case_code"] for r in public["cases"]}) == 22
    for alias, n in (("Chart-1", 10), ("Chart-2", 10), ("Chart-3", 2)):
        cases = [x for x in public["cases"] if x["chart_alias"] == alias]
        matching = {
            x["case_code"] for x in cases}
        originals = [
            k["original_side"] for k in key["case_mapping"]
            if k["case_code"] in matching]
        assert len(cases) == n
        assert originals.count("A") == n//2
        assert originals.count("B") == n//2


def test_public_excludes_key_seed_objective_and_candidate_identity(prepared):
    _, public, key = prepared
    raw = json.dumps(public)
    assert "seed_hex" not in raw
    assert "source_h11_sha256" not in raw
    assert "candidate_id" not in raw
    assert "original_side" not in raw
    assert "alternative_side" not in raw
    assert "final_delta" not in raw
    assert "lower_heuristic" not in raw
    assert "original-chart" not in raw
    assert key["seed_hex"] not in raw
    assert key["source_h11_sha256"] not in raw


def test_blinded_ab_axes_correctly_match_private_side(prepared):
    source, public, key = prepared
    all_candidates = {
        item["candidate_id"]: (
            chart["original_events_by_index"],
            item["alternative_axes_by_event_order"])
        for chart in source["charts"]
        for item in chart["reconstructions"]
    }
    public_by_code = {p["case_code"]: p for p in public["cases"]}
    for saved in key["case_mapping"]:
        original, other = all_candidates[saved["candidate_id"]]
        published = public_by_code[saved["case_code"]]["events"]
        assert len(original) == len(published)
        for i, row in enumerate(published):
            assert row["i"] == original[i]["event_index"]
            a_is_original = saved["original_side"] == "A"
            assert row["a"] == (
                original[i]["original_axes"] if a_is_original else other[i])
            assert row["b"] == (
                other[i] if a_is_original else original[i]["original_axes"])


def test_seed_reproducibility_but_different_seed_reorders(prepared):
    source, public, key = prepared
    public2, key2 = compile_blind(
        source, source_sha256="a"*64, seed=b"K"*32)
    assert public2 == public and key2 == key
    changed, another_key = compile_blind(
        source, source_sha256="a"*64, seed=b"M"*32)
    assert changed["packet_id"] != public["packet_id"]
    assert another_key["seed_hex"] != key["seed_hex"]


def test_html_is_self_contained_without_unblinding_key(prepared):
    _, public, key = prepared
    rendered = render_html(public)
    assert rendered.startswith("<!DOCTYPE html>")
    assert "<script id=\"review-data\"" in rendered
    assert '"case_count":22' in rendered
    assert "https://" not in rendered
    assert key["seed_hex"] not in rendered
    assert "original-chart" not in rendered
    assert "source_h11_sha256" not in rendered
    assert "URL.createObjectURL" in rendered
    assert "preference" in rendered
    assert "changed" in rendered


@pytest.mark.parametrize("change,error", [
    ("absent_candidate", "per-chart target census"),
    ("invalid_axis", "bad alternative axis"),
    ("bad_event_count", "full Train event census"),
    ("forged_motor", "not an unmodified"),
    ("fake_human", "not an unmodified"),
    ("tampered_changed_event", "change-set identity differs"),
])
def test_malformed_or_fabricated_evidence_is_rejected(change,error):
    source=fixture_source()
    target=source["charts"][0]["reconstructions"][0]
    if change=="absent_candidate":
        source["charts"][0]["reconstructions"].pop()
    elif change=="invalid_axis":
        target["alternative_axes_by_event_order"][2]=["left_5"]
    elif change=="bad_event_count":
        source["charts"][0]["event_count"] -= 1
    elif change=="forged_motor":
        source["alternate_motor_executed"] = True
    elif change=="fake_human":
        source["human_preference_observed"] = True
    elif change=="tampered_changed_event":
        target["changed_events"][0]["event_index"] += 1
    with pytest.raises(ValueError, match=error):
        compile_blind(source, source_sha256="a"*64, seed=b"X"*32)


def test_short_secret_or_wrong_digest_is_rejected():
    source=fixture_source()
    with pytest.raises(ValueError, match="128 bits"):
        compile_blind(source, source_sha256="a"*64, seed=b"x")
    with pytest.raises(ValueError, match="source digest"):
        compile_blind(source, source_sha256="short", seed=b"X"*32)


def test_main_requires_separate_directories_and_create_once(
        tmp_path,monkeypatch):
    source=tmp_path/"h11.json"
    source.write_text(json.dumps(fixture_source()),encoding="utf-8")
    output_dir=tmp_path/"share"
    output_dir.mkdir()
    viewer=output_dir/"review.html"
    key_dir=tmp_path/"private"
    key=key_dir/"answer-key.json"
    monkeypatch.setattr(sys, "argv", [
        "h12", "--h11-report", str(source),
        "--viewer-output", str(viewer),
        "--private-key-output", str(output_dir/"wrong.json")])
    with pytest.raises(SystemExit):
        main()
    monkeypatch.setattr(sys, "argv", [
        "h12", "--h11-report", str(source),
        "--viewer-output", str(viewer),
        "--private-key-output", str(key)])
    main()
    assert viewer.is_file()
    assert key.is_file()
    assert len(viewer.read_text(encoding="utf-8")) > 10000
    before=viewer.read_bytes()
    with pytest.raises(SystemExit):
        main()
    assert before==viewer.read_bytes()

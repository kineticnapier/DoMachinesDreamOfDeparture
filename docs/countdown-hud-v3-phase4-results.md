# HUD v3 Phase 4 — Train bool second-event identifiability

## Outcome

**Observed counterfactual result: INDISTINGUISHABLE_BEFORE_44MS_DEADLINE.**

The Windows user ran `scripts/audit_bool_second_event_counterfactual_v01.py`
on the fixed Train bool chart and supplied the complete r1 JSON output.
The r1 JSON is currently on the user's Windows working tree at
`reports/countdown-bool-second-event-counterfactual-20261019-r1.json`.
This document records those observed results; it does not claim an
independent replay of that chart in another environment.

- Chart: `04042 - bool - vo[ICE].adofai`
- SHA256: `04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1`
- Observation: `mod-visible-tile-real-bpm-v2` HUD263 plus 15D stock countdown cues
- Fixed intervention: active floor-1 SetSpeed `bpmMultiplier` at action source index 2, 8.0 -> 1.0 (factor 0.125)
- Visible event marker and first three floors' observable topology: unchanged
- First target due in BOTH worlds: `3.999333624280351 s`
- Second due original: `4.020163490257674 s`
- Second due counterfactual: `4.16597255209893 s`
- Second due difference: **145.809062 ms**
- EARLIER-world 44ms motor lead deadline: `3.9761634902576737 s`
- Last sampled pre-deadline frame: **397 at 3.970 s**
- Compared observations: **398 HUD278 frames**
- Maximum HUD278 absolute difference: **0.0** (tolerance 5e-5)
- Countdown cue validity, kind, digit and render timestamp: identical
- First distinguishable frame before deadline: **none**

## Causal conclusion

The two possible worlds have indistinguishable current/past HUD278 observations
at every sampled 10ms policy frame through the earliest second-event
44ms preparation deadline, despite having different next due times.

Therefore no deterministic Timing policy receiving *only this HUD278 contract*
can distinguish the exact second target time between these worlds before
the preparation deadline. The second Train bool Miss in Phase 3 is consistent
with a causal input-information limitation, not merely an underperforming
Fingering Network or Motor. The experiment does **not** establish that all
possible hedged physical actions would miss, nor that no additional sensory
modality could distinguish these worlds.

The first target is identical in both worlds; this rules out confusing the
observed 2nd-event ambiguity with the previously repaired Countdown ambiguity.
HUD v3's Phase 3 8/9 result and first-target Perfect Hit remain intact.

This result is limited to the modeled **Mod numeric Tile/Real BPM + stock
countdown** HUD contract, the particular one-eighth SetSpeed intervention,
the fixed 10ms clock, and this chart. Real-world pixel capture, OCR, music
audio, future numerical annotations, other presentation channels and stock-only
observation were not tested. The provided r1 output does not by itself confirm
that every pytest test passed.

## Next research boundary

Do not tune Timing v1.4 lookahead, Motor lead, physical constants or the
Fingering checkpoint to chase a 9/9 result in this counterexample.
Investigate independent, *actually observable* evidence of the future
SetSpeed numeric value; otherwise leave this event abstained as expected.

Keep policy, renderer, evaluator separated: a new information channel must
be shown in the rendered observation and not silently populated from the
chart's future action data.

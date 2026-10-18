# HUD v3 Phase 4 — Train bool second-event counterfactual

## Status

The fixed counterfactual audit was added at
scripts/audit_bool_second_event_counterfactual_v01.py.
Do NOT claim the Train bool hypothesis is proved until Windows has run the
audit on the actual chart. That chart is not available in this environment.

Phase 3 fixed observations on Train bool:
- Chart SHA256: 04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1
- First target due: 3.999333624 s; HUD v3 Hit at 4.001 s.
- Second target due: 4.020163487 s; v1.4 first forecast at 4.000 s.
- Second target 44ms deadline: 3.976163487 s.
- Phase 3: v1.4 7/9, v3 8/9.
- Prior Oracle 9/9 is evidence of physical feasibility under privileged
  timing, not evidence that HUD alone knows the target's future time.

## Predeclared causal experiment

**Hypothesis:** The numerical value of floor-1 SetSpeed is hidden from
the Mod-HUD278 observation before the second event's preparation deadline.

1. Load the exact Train bool chart, rejecting an unexpected SHA.
2. Locate an existing active SetSpeed action at floor 1. Change ONLY
   its numeric BPM or multiplier to ONE EIGHTH of its original value.
   Preserve every marker, countdown setting and tile shape. This
   intervention factor is fixed and must not be tuned.
3. Compile both worlds. Require first due to be unchanged and second
   due gap to be at least 44ms. If no floor-1 SetSpeed exists,
   fail closed: THIS SPECIFIC HYPOTHESIS HAS NOT BEEN SHOWN.
4. Set the evaluator-only prefix cutoff to the EARLIER second due
   minus 44ms, sampling at 10ms intervals.
5. Compare EVERY prefix HUD278 sample, including the original 263D
   Mod-visible HUD (tolerance 5e-5), countdown suffix and cue
   timestamps/validity. Require identical first three floors'
   public geometry and markers.
6. Report INDISTINGUISHABLE_BEFORE_44MS_DEADLINE only if all gates
   pass; otherwise NOT_PROVEN. Do not change thresholds.

This demonstrates nonidentifiability of exact second-event due in two
hypothetical worlds, not impossibility of any hedged physical strategy.
True target times are evaluator-only; the renderer uses chart state
solely to render CURRENT screen-observable information.

**Scope:** Synthetic Mod-visible numeric Tile/Real BPM plus stock
countdown text. Not a claim about the actual stock-only screen,
audio, pixel capture latency, OCR or additional HUD channels.

## Windows commands, from repository root

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_audit_bool_second_event_counterfactual_v01.py --basetemp artifacts/pytest-phase4-counterfactual

.\.venv\Scripts\python.exe scripts/audit_bool_second_event_counterfactual_v01.py --output reports/countdown-bool-second-event-counterfactual-20261019-r1.json
~~~

The script refuses to overwrite an output. Neither learning nor
Validation20/FINAL nor any frozen timing, fingering or motor changes
are part of this audit.

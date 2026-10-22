# Phase 20C — Frame-driven Stock reference state machine

## Scope and evidence boundary

This phase keeps the production scorer unchanged.  The research reference now
separates a virtual frame into the order observed in the audited DLL:

1. time/orbit update;
2. `scrPlayer.CheckPostHoldFail`;
3. automatic NoFail transition or normal fail;
4. queued `KeyDown` processing;
5. hit/invalid-input floor transition and trace emission.

The control flow is supported by the audited decompilation under
`F:/adofai-code` (DLL SHA256
`0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD`).
The replay uses an explicitly chosen 10 ms virtual frame grid with phase 0,
matching the repository control interval.  It is **not** evidence that a live
Unity build uses that frame rate or phase.

## Fixed First town replay

`reports/phase20c-frame-stock-state-20261022-r1.json` replays the saved Phase
16 physical KeyDown timestamps.  With NoFail enabled, target and category
parity with the repository trace remains 54/54.  The backlog remains 50
KeyDowns: ID 38 is judged against event 37 and ID 89 against event 88; the
automatic transition of event 89 occurs on the 5.320 s virtual update before
ID 90 is processed, allowing ID 90 to resynchronise.

Strict angle crossing and first virtual update are kept separate:

| case | strict crossing | first 10 ms update |
|---|---:|---:|
| event 36, NoFail | 3.249473567 s | 3.250 s |
| event 89, NoFail | 5.315650044 s | 5.320 s |
| event 36, normal fail | 3.313823567 s | 3.320 s |

Normal-fail replay is anchored at the independently known event-35 landing.
ID 38 is TooLate against event 36 at 3.261 s.  `CheckPostHoldFail` then stops
the game at the 3.320 s virtual update, before the queued 3.311 s input is
processed.  This replaces the Phase 20B event-driven 3.355 s observation; the
old time was delayed until the next `key_down()` call rather than an actual
frame transition.

## Result and limitations

Verdict: `FRAME_ACCURATE_REFERENCE_VALIDATED` for the declared virtual-frame
contract.  Phase 20B's 54/54 NoFail target/category comparison is preserved,
while frame scheduling changes the normal-fail stop time.

Unsupported special branches remain explicit in the underlying reference
model.  Live Stock parity remains unverified until the installed build and its
Unity update/capture phase are measured independently.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_stock_judgement_reference.py tests/test_stock_floor_state_machine.py tests/test_stock_frame_state_machine.py tests/test_audit_phase20b_stock_floor_state_v01.py tests/test_audit_phase20c_frame_stock_v01.py
.\.venv\Scripts\python.exe scripts/audit_phase20c_frame_stock_v01.py --output reports/phase20c-frame-stock-state-20261022-r1.json
```

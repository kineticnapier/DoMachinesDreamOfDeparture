# Prestige I — JRP BPM HUD source study

Pinned upstream: Jongye0l/JipperResourcePack at
861a1ff6d12ef1c785ca16fb980a7e9292cfee4f.

- Normal-mode implementation:
  https://github.com/Jongye0l/JipperResourcePack/blob/861a1ff6d12ef1c785ca16fb980a7e9292cfee4f/JipperResourcePack/OverlayContents/Overlay.cs#L614-L638
- Jongyeol-mode pseudo-BPM:
  https://github.com/Jongye0l/JipperResourcePack/blob/861a1ff6d12ef1c785ca16fb980a7e9292cfee4f/JipperResourcePack/Jongyeol/JOverlay.cs#L282-L385
- Text formatting:
  https://github.com/Jongye0l/JipperResourcePack/blob/861a1ff6d12ef1c785ca16fb980a7e9292cfee4f/JipperResourcePack/OverlayContents/Overlay.cs#L550-L571

## Confirmed formulas for normal mode

TBPM = conductor.bpm * song.pitch * GetPlanetSpeed(controller)
CBPM = song.pitch * 60 / (nextFloor.entryTime - currentFloor.entryTime)
KPS = CBPM / 60

On the last floor, CBPM falls back to TBPM. Current-floor-to-next-floor
entryTime includes Pause: JRP normal CBPM therefore carries a Pause-inclusive
interval. The existing DMDOD "Real BPM" deliberately excludes Pause and
is NOT interchangeable with the JRP CBPM feature.

Update timing matters: OnHit and Show trigger UpdateBpm. The number is
*not* recomputed each 10ms physics tick. If a simulated floor advances
without the display-refresh hook, the visible text must remain stale.

The display defaults to two decimal places (configurable 0..4). JRP
WriteRounded uses ties-to-even rounding and strips trailing decimal zeroes.
The model must ingest display-quantized numbers, not raw chart-derived
floating-point CBPM.

Jongyeol mode (optional) has CheckPseudo enabled by default and may
aggregate multiple floors into its CBPM. KPS can also be multiplied by
the aggregated count. Its immediate next-floor interval is *not*
necessarily equal to 60/CBPM. This normal-mode v1 adapter does NOT
emulate the Jongyeol pseudo branch.

## Implementation

- src/dmdod/features/jrp_bpm_v1.py: independent renderer-side adapter and
  on_show / on_hit_hook display cache. No changes to frozen HUD codecs.
- tests/test_jrp_bpm_v1.py: Pause, pitch, SetSpeed, final floor, rounding,
  and display update hooks.
- scripts/audit_jrp_bpm_v1.py: posthoc formula/precision audit.

CPU: 38/38 PASS for new JRP test plus unchanged previous HUD regressions.

Normal-mode formula audit, **not a Student rollout**:

- Nerfed chart, first 65s: 890 next-floor intervals with usable displayed
  CBPM, all 890 inferred from rounded CBPM within 0.1ms of compiled chart gap.
- Second inspected chart, first 50s: 863 usable intervals out of 865;
  all 863 within 0.1ms; two zero-time floor transitions intentionally
  unsupported by finite BPM encoding.
- Pause examples in Nerfed: floor 37 TBPM=124 CBPM=31 KPS=0.52,
  floor 52 TBPM=124 CBPM=62 KPS=1.03.

This does NOT establish 890/890 HUD-only event tracking. The audit
generates the HUD rendering from known chart data on the simulator
side, then checks the displayed-number precision; it does not model
actual post-Hit JRP text refreshes, pseudo mode, error correction,
Motor control, or unseen-chart robustness. The earlier 887/890
HUD-tracker baseline remains frozen.

## Student boundary

Only text-quantized, currently rendered numbers and past observations
may enter a Student. The source chart, target timestamps, floor ordinal,
and exact duration stay renderer/evaluator-side. Integrate through a
versioned alternative HUD encoder only after the display lifecycle and
Jongyeol pseudo branch have their own tests; do not silently reuse old
263D checkpoint input semantics or lift Stage A FAIL.

CPU commands:

    uv run pytest -q tests/test_jrp_bpm_v1.py
    uv run python scripts/audit_jrp_bpm_v1.py PATH_TO_CHART.adofai --end-s 65 --output diagnostics/jrp-normal-bpm-v1.json

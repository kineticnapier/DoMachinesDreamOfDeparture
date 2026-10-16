# Prestige I — JRP overlay v2: lifecycle + optional Jongyeol pseudo BPM

Source-pinned to `Jongye0l/JipperResourcePack` commit `861a1ff6d12ef1c785ca16fb980a7e9292cfee4f`.

## Source semantics

- Normal `Overlay.UpdateBpm`: TBPM = conductor BPM × pitch × planet speed; CBPM = pitch × 60 / (next-floor entryTime − current-floor entryTime); KPS = CBPM / 60.
- `Bpm.OnHit` and `Overlay.Show` trigger text updates. A floor advancing alone does **not** refresh it. Text remains unchanged during a Miss until some update hook fires; it is NOT safe to recompute CBPM from the current tile at each 10ms Student frame.
- `JOverlay.UpdateBpm` is used only in the optional **Jongyeol mode**. With `Jbpm.Settings.CheckPseudo` enabled, some short-angle patterns merge several physical floors into one displayed CBPM; KPS = CBPM/60 × count. `_pseudoFloor` suppresses subsequent updates until the selected group has passed. Normal CBPM and pseudo CBPM are **not interchangeable**.
- Display numerics are rounded 0–4 decimals and trailing zeros trimmed. Only the rendered strings are made available for a future Student observation.

## Modules

- `src/dmdod/features/jrp_overlay_v2.py`: JRP event-driven text renderer with separate normal/Jongyeol modes and diagnostic-only source-floor information. Renderer **must** be isolated from the Student. The `from_segment` adapter reconstructs approximate source `angleLength` from DMDOD compiled timing, **not proven native `scrFloor.angleLength` equivalence**.
- `src/dmdod/envs/jrp_overlay_diagnostic.py`: A read-only `DiagnosticHudNKeyRealChartMotorEnv` subclass whose JRP overlay refreshes on **DMDOD's accepted Hit** events. This approximates native JRP's patched `scrPlayer.Hit`/`scrController.Hit` callbacks; their behavior on Early/Miss still requires native game parity tests. Scorer and Motor are unchanged.
- `tests/test_jrp_overlay_v2.py` and `tests/test_jrp_overlay_diagnostic.py`: pure JRP pseudo/lifecycle tests and real 1ms Motor accepted-hit integration. 52 related tests passed on Linux CPU including historical HUD regressions.
- `scripts/audit_jrp_overlay_v2.py`: render a hypothetical perfect-Hit sequence from compiled real-chart data. This is a **renderer test, not gameplay or HUD-only Timing**.

## Result on the inspected first 65s of 命日 (Nerfed)

Under 890 *hypothetical successful Hit hooks* (all source target events), normal display changed text 349 times and did not change on 541 hooks. In optional Jongyeol mode, it changed 343 times and kept the same text 547 times; pseudo overlay was active on 19 of these hook states. These are simulation numbers: native game hook parity, `scrFloor` native angleLength and mode-settings parity have not been proven.

## Integration policy

Keep frozen Timing v1–v1.4 and 263D HUD checkpoints unchanged. No true floor index, chart entryTime, pseudo group count, float-unrounded BPM, or `JrpDisplayedHud.source_floor` may be sent into Student. Model a **separate versioned JRP text-observation codec**, and validate causality against real hook traces before using its CBPM as a predictor. `JrpOverlayRenderer.visible` exposes only the exact rounded on-screen strings.

### CPU

```sh
PYTHONPATH=src python -m pytest -q tests/test_jrp_overlay_v2.py tests/test_jrp_overlay_diagnostic.py tests/test_jrp_bpm_v1.py tests/test_hud_bpm_unclipped.py
PYTHONPATH=src python scripts/audit_jrp_overlay_v2.py /mnt/data/nerfed.adofai --end-s 65 --output diagnostics/jrp-overlay-v2-nerfed65s.json
```

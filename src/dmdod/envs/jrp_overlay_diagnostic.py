from __future__ import annotations

"""Read-only overlay instrumentation for simulated *accepted Hit* events.

This is a diagnostic hook proxy, not a complete port of native JRP OnHit:
native JRP patches scrController.Hit or scrPlayer.Hit, which may differ from
the DMDOD scorer's accepted-hit transition on early/failed presses.
Nothing here modifies score, physics, frozen 263D HUD or Student actions.
"""

from dmdod.envs.n_key import DiagnosticHudNKeyRealChartMotorEnv
from dmdod.features.jrp_overlay_v2 import JrpOverlayRenderer


class JrpOverlayDiagnosticNKeyEnv(DiagnosticHudNKeyRealChartMotorEnv):
    """Diagnostic emulator; only displayed strings are in overlay_visible."""

    def __init__(self, segment, *, jrp_mode='normal', jrp_check_pseudo=True,
                 jrp_decimals=2, **kwargs):
        self._jrp_display = JrpOverlayRenderer.from_segment(
            segment, mode=jrp_mode, check_pseudo=jrp_check_pseudo,
            decimals=jrp_decimals)
        self._jrp_hook_calls = 0
        super().__init__(segment, **kwargs)

    @property
    def overlay_visible(self) -> tuple[str, str, str] | None:
        """Only valid Student export; no access to render source IDs."""
        return self._jrp_display.visible

    @property
    def overlay_diagnostic(self):
        """Evaluator-only: never feed this metadata into a policy."""
        return self._jrp_display.diagnostic

    @property
    def overlay_hit_hook_count(self) -> int:
        return self._jrp_hook_calls

    def reset(self):
        self._jrp_display.reset()
        self._jrp_hook_calls = 0
        result = super().reset()
        # Native JRP Overlay.Show triggers UpdateBpm once.
        initial_floor = self._floor_index_at_chart_time(self.segment.start_chart_s)
        self._jrp_display.on_show(initial_floor)
        return result

    def _score_event(self, event):
        # The scorer may expire one or more Misses before accepting this
        # keydown.  Therefore the pre-score ``_next_target_index`` is not
        # necessarily the floor whose Hit hook fired.  Diff the evaluator-only
        # used flags after scoring and render the target actually credited.
        used_before = tuple(self._used)
        previous_hits = self._hits
        reward = super()._score_event(event)
        if self._hits > previous_hits:
            credited = [i for i, (before, after) in enumerate(
                zip(used_before, self._used)) if not before and after]
            if len(credited) != 1:
                raise RuntimeError('accepted Hit did not identify exactly one credited floor')
            # Approximation of JRP OnHit: score ACCEPTED Hit. Native hook
            # behavior on judged-too-early events needs a separate game test.
            floor = self.segment.targets[credited[0]].floor_index
            self._jrp_display.on_hit_hook(floor)
            self._jrp_hook_calls += 1
        return reward

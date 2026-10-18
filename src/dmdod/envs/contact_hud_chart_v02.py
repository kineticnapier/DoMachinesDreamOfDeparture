from __future__ import annotations

"""Contact-chart HUD renderer; exposes only CURRENT displayed fields.

The environment-side renderer may use the simulator's chart clock to draw the
current HUD just like the existing eight-key HUD renderer. The *policy* sees
only the returned immutable ContactVisibleTimingHUD, never the segment,
clock, target schedule, or scorer. No input is synthesized from future notes.

The deterministic v1.4 Timing tracker does not read judgement feedback;
this first provider leaves feedback hidden instead of inventing a mapping.
"""

from dmdod.envs.contact_chart_16k import CoupledContactRealChartEnv
from dmdod.features.hud import hud_bpms_for_floor
from dmdod.planning.contact_timing_hud_v01 import (
    ContactVisibleTimingHUD,
    MOD_VISIBLE_BPM_CONTRACT,
)


class HudCoupledContactRealChartEnv(CoupledContactRealChartEnv):
    """Same 12-axis physics/scorer, with publicly displayed Tile/Real BPM."""

    def _observation(self, motor, episode_time_s: float) -> ContactVisibleTimingHUD:
        contact = super()._observation(motor, episode_time_s)
        # Renderer-only privileged clock, already used by existing HUD view.
        # Neither forecast adapter nor policy receives any of these fields.
        chart_time = self.segment.chart_time_from_episode(episode_time_s)
        index = self._floor_index_at_chart_time(chart_time)
        floor = self.segment.chart.floors[index]
        tile_bpm, real_bpm = hud_bpms_for_floor(self.segment, floor)
        return ContactVisibleTimingHUD(
            contact=contact, tile_bpm=tile_bpm, real_bpm=real_bpm,
            bpm_observation_contract=MOD_VISIBLE_BPM_CONTRACT,
            feedback_visible=False,
        )

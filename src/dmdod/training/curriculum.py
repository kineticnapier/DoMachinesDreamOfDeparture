from __future__ import annotations


def note_curriculum(final_notes: int) -> tuple[int, ...]:
    """Return the progressive toy-RL note counts up to ``final_notes``.

    The curriculum deliberately teaches one successful cue-driven press before
    asking the same policy to survive longer sequences under OVERLOAD rules.
    """

    if final_notes <= 0:
        raise ValueError("final_notes must be positive")

    stages = [stage for stage in (1, 2, 4, 8, 16) if stage < final_notes]
    stages.append(final_notes)
    return tuple(stages)


def curriculum_start_s(notes: int) -> float:
    """Use a shorter pre-roll for the smallest curriculum stages."""

    if notes <= 0:
        raise ValueError("notes must be positive")
    if notes == 1:
        return 0.450
    if notes == 2:
        return 0.500
    if notes <= 4:
        return 0.600
    return 0.750

# DMDOGO v0.6 — Pattern-first scheduler DP integrity

## Scope

The committed abstract scheduler (`src/dmdod/planning/pattern_scheduler_v06.py`) has
no Motor or Network access. Its output proves only periodic assignment with
declared logical-key reuse constraints, **not physical ADOFAI Hit**.

The recorded Train bool schedule used three abstract keys for 1,122 events,
minimum measured reuse 83.319 ms and zero abstract reuse violations. This
does not establish that a physical three-axis mapping can survive the 90 ms
reservation/eligibility window.

## Exactness repair

The original inter-segment DP retained only the lowest-cost state for each
last segment candidate. This was invalid: a different path ending in the
same candidate may have a *different last-use time* for a key not used by a
short intervening segment. It may therefore enable a cheaper continuation.

Regression with six synthetic events at
`[0, .03125, .09225, .26225, .2935, .3055]`, three logical keys and
100 ms minimum reuse:

- previous per-candidate pruning: objective width-cost 14, pattern changes 4;
- exact path enumeration: width-cost 14, pattern changes **3**;
- repaired DP retains nondominated objective/last-use histories and reaches
  the better result.

State A dominates state B only when its total objective is no worse *and* its
last-use time is no later for **every** key. The scheduler fails explicitly
with `dp_frontier_state_limit_exceeded` if the retained frontier exceeds
10,000 states; it does not silently discard necessary paths or claim physical
infeasibility. This is a bounded research implementation, not a hard real-time
complexity guarantee.

Additional corrections:

- The first gap of a chart now starts a new segment when it exceeds the
  declared restart threshold.
- NaN/infinite timing and reuse settings fail closed.
- Simultaneous independent events may not consume the same logical key even
  if the configured minimum reuse interval is zero.

## Motor handoff

The user's local `scripts/run_dmdogo_v06_pattern_first.py`,
`src/dmdod/planning/pattern_motor_adapter_v06.py` and
`tests/test_pattern_motor_v06.py` were untracked when inspected. These local
files are **not** included in these commits and have not been overwritten.
The GitHub-side scheduler repair preserves its public types and API.

Integration needs to distinguish three cases:

1. **Abstract valid:** periodic logical-key assignment meets reuse rules.
2. **Temporarily occupied:** mapped physical axis is reserved at 90 ms
   decision horizon; waiting for release may retain the scheduled choice.
3. **Infeasible by deadline:** at the final safe selection deadline the mapped
   allocation remains physically ineligible; an explicit local alternative
   or failure is required.

A temporary reservation at 90 ms is **not** proof of physical impossibility.
Conversely, no future Motor state or judgement may be assumed as observed.
Never bypass live eligibility or the 12-axis/16-contact topology to preserve
a periodic string.

## Verification

Focused tests are located at `tests/test_pattern_scheduler_v06.py`.
Run on the user's Windows checkout after `git pull --ff-only`:

```powershell
uv run pytest -q tests/test_pattern_scheduler_v06.py
```

No local Windows/CUDA replay or repository-wide pytest was executed as part
of this GitHub-only repair. Existing untracked adapter and artifact directories
must remain untouched.

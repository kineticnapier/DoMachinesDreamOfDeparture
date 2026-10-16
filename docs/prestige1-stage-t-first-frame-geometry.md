# Prestige I Stage T — Geometry at the first frame of a floor switch

## Question

The frozen stateless Hybrid improved Train-heldout near MAE from
36.926ms to 18.723ms, but all 148 observed true note-switch frames
fell back to the near-only GRU, with +0ms MAE 101.273ms.

Frozen standard Geometry immediately abstains if its current-floor
feature vector changes between successive 10ms observations. This
audit asks whether that single conservative guard is responsible for
missing an otherwise useful, visible angular calculation.

No training, no new Hybrid adoption, no Motor or Stage A changes.

## Candidate estimators (HUD-only)

At each frame, original Geometry is evaluated as-is on
successive 263D visible HUDs. If and ONLY if its abstention reason
is floor_changed, two counterfactuals are evaluated. Each passes a
temporary copied previous HUD into the UNCHANGED Geometry function,
with previous current-floor features replaced by the current
HUD current-floor features. This removes exactly the floor equality
guard without disabling the original current-floor/presence,
Midspin/multiplanet, special-event, full-turn, orbit-validity, Tile
BPM or angular-velocity consistency guards.

- current_crossing_step: previous orbit xy remains unchanged, so the
  angular direction is inferred from actual k-1 to k orbit movement,
  even though k crossed the floor boundary.
- carried_previous_step: requires k>=2; angular displacement from
  HUD[k-2] to HUD[k-1] is copied into an artificial previous orbit xy
  relative to the unchanged current orbit. Thus the angular direction
  is inferred from earlier motion, rather than crossing movement.

These are explicitly COUNTERFACTUAL guard tests: synthetic prior-HUD
fields do not represent an actual observation. Neither candidate
receives oracle event ID, future chart time, privileged floor index,
hidden event clock, near/mask or label. They run on all frames first,
BEFORE the evaluator uses event labels to select switch examples.

Both candidates intentionally abstain if other existing safeguards
reject them, or if there is insufficient history. They may be badly
wrong when floor progression changes orbit anchors or the simulator
requires information not encoded in this simple heuristic.

## Evaluation

Use the exact frozen Train20 expert-driven HUD capture, separated
into Train-fit and chart-disjoint Train-heldout. Restore and SHA-check
Stage A FAIL, Stage T checkpoint, near-only GRU, old distance audit,
old countdown audit, near-only countdown report, stateless Hybrid
report, and prior HUD switch diagnostic. Recompute the Hybrid and
ensure frame counts, near accuracy and switch count still agree.

After candidate predictions are frozen, identify Auto target-switch
pairs using oracle event IDs only as evaluator labels. Report:

- number of true note-switch frames and prior frozen Geometry reasons;
- angular step change through the boundary compared with immediately
  preceding angular step; direction flips and absolute-step ratios;
- candidate eligibility and rejection reason counts on true switches;
- candidate absolute signed-error MAE, P95, near-MAE and same-frame
  baseline GRU/Hybrid errors on exactly matched eligible frames;
- if the next frame remains the same oracle event and regular Geometry
  is eligible, the candidate-to-regular-Geometry countdown step
  residual (vs actual elapsed time), without oracle-triggered reset;
- counts/reasons for true new events that do not regain frozen Geometry
  before the next note, observation end or chart end;
- top eight largest counterfactual errors to avoid cherry-picking.

Candidate MAEs are conditional on visible eligibility, so compare
only with GRU on the same frames and inspect the coverage denominator.
Even if a counterfactual is near-perfect at labelled switches, this
does not establish standalone HUD-only switch detection, unknown
special-event correctness, or successful closed-loop Motor control.

No parameter tuning or threshold sweep is performed.

## Command (Windows)

First inspect the worktree; never discard or stash uncommitted work.

~~~powershell
git status --short
git pull --ff-only

uv run pytest -q tests/test_prestige1_autoplay_first_frame_geometry_audit.py

uv run python -u scripts/audit_prestige1_autoplay_first_frame_geometry.py --config configs/training/prestige1.toml
~~~

Output, created once:

- diagnostics/prestige1/stage_timing_autoplay_first_frame_geometry_train20.json

The command refuses to overwrite any existing output. It does not
modify prior reports, checkpoints, frozen Stage A FAIL, Validation,
FINAL, Fingering, Motor, or experiment configuration.

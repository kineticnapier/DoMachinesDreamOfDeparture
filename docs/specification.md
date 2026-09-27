# v0.1 Specification

## 1. Objective

Investigate whether a learning agent under human-like perceptual and physical constraints can acquire effective ADOFAI fingering strategies different from those used by humans.

The objective is not to build the strongest possible bot. Physical ability should be controlled as far as practical so that differences in learning, prediction, and fingering can be studied.

## 2. Architecture

```text
ADOFAI Environment
        |
   perception
        v
      Agent
        |
  motor command
        v
 Neural Delay
        |
        v
   Hand Model
        |
        v
 Keyboard Model
        |
        v
      Input
```

Agent, Perception, Hand Model, Keyboard Model, Environment, and Logger/Evaluator should remain independently replaceable.

## 3. Forbidden Information and Actions

Under evaluation conditions, the agent must not:

- read exact target input times from `.adofai`
- read internal beat or judgement timing
- use Otto or Autoplay information
- alter game speed or judgement windows
- directly emit a key press

The agent may output only motor commands to the simulated body.

## 4. Time

The body simulation uses a fixed time step, initially:

```text
Δt = 1 ms (1000 Hz)
```

The policy update frequency is independent of the body simulation and initially set to 100 Hz.

## 5. Hand Model v0.1

The first model has two fingers only:

```text
Left
Right
```

Each finger `i` has state:

```text
S_i = (x_i, v_i, a_i, f_i)
```

- `x_i`: finger position
- `v_i`: finger velocity
- `a_i`: muscle activation
- `f_i`: fatigue

The agent action is:

```text
u_i ∈ [-1, 1]
```

`+1` means maximum downward command, `0` no active command, and `-1` maximum upward command.

## 6. Muscle Response

Motor commands do not become force instantaneously.

```text
τ_a da_i/dt = u_i - a_i
F_i = F_max (1 - f_i) a_i
m_i d²x_i/dt² = F_i - c_i dx_i/dt - k_i(x_i - x_0,i)
```

These equations are replaceable approximations, not a definitive physiological model. Their parameters and structure should be validated against measurements.

## 7. Keyboard

A key has separate actuation and reset points. Initial example values:

```text
travel      = 4.0 mm
actuation   = 2.0 mm
reset       = 1.8 mm
```

`KEY_DOWN` occurs only when the key crosses the actuation point downward, and another press cannot occur until the key returns past the reset point.

## 8. Fatigue

Temporary v0.1 model:

```text
df_i/dt = α |a_i| - β f_i
0 <= f_i <= 1
```

Fatigue reduces available force. This model is explicitly provisional and should later be calibrated or replaced using measurements.

## 9. Finger Coupling

The two fingers are not perfectly independent. v0.1 uses a simple coupling term, for example:

```text
F'_i = F_i (1 - γ |a_j|)
```

A future ten-finger model should support finger-specific coupling, for example through a coupling matrix.

## 10. Perception

### Mode A: Symbolic

Used for development and body-model validation. The agent receives abstract state such as current angle, relative information about upcoming tiles, and rotation speed, but never exact target input times.

### Mode B: Visual

Used for final evaluation. Perception comes from rendered game frames and includes a finite frame rate and perceptual delay.

```text
observation at t = I(t - τ_visual)
```

Internal prediction of future events is allowed; humans also anticipate upcoming events and prepare movements in advance.

## 11. Reinforcement Learning

The initial implementation should use an algorithm supporting continuous actions. PPO is a candidate, but the learning algorithm is not fixed.

```text
a_t = (u_left, u_right)
R = R_hit - λ_t |timing_error| - λ_e E - R_miss
E = Σ a_i²
```

The reward must not directly prescribe a particular fingering pattern.

## 12. Curriculum

```text
Stage 0  single timed press
Stage 1  constant BPM
Stage 2  increasing BPM
Stage 3  speed where alternating fingers become advantageous
Stage 4  changing BPM
Stage 5  simple ADOFAI patterns
Stage 6  high difficulty
Stage 7  U-level frontier
```

Stage 3 is the main v0.1 experiment. Alternating fingers must not be explicitly taught; the experiment tests whether the agent discovers such a strategy independently.

## 13. Human Calibration

Body parameters should be calibrated against human measurements rather than chosen only because they appear plausible.

Candidate measurements:

- single-finger tapping
- two-finger alternation
- simultaneous presses
- gradually increasing rate
- sustained tapping
- irregular timing

With human measurements `H_j` and model measurements `M_j(θ)`, calibration can be expressed conceptually as:

```text
θ* = argmin_θ Σ_j w_j D(M_j(θ), H_j)
```

## 14. Held-out Validation

Matching humans only on calibration tasks is insufficient. Some patterns and timing conditions must be held out to test whether the model reproduces human-like performance on unseen tasks.

```text
Calibration: single finger, LR alternating, sustained tapping
Held out:    LLR, LRR, LLRR, irregular BPM, short bursts
```

## 15. Logging

At minimum, record:

```text
timestamp
observation
policy output
finger position
finger velocity
muscle activation
fatigue
key state
target timing
actual timing
timing error
judgement
reward
```

The evaluator may retain exact target timings for scoring and analysis, but they must never leak into the agent observation.

## 16. Out of Scope for v0.1

- ten fingers
- wrist and arm
- detailed tendon and skeletal simulation
- 3D human body
- individual anatomical muscles
- final visual perception
- audio perception
- direct control of the real game

The initial target is:

```text
2-finger physical model
+ simplified rhythm environment
+ reinforcement learning agent
```

## 17. v0.1 Success Criterion

Create a sequence whose rate is physically impossible for the simulated body to clear with one finger. Without giving fingering instructions or fingering-specific rewards, the agent must discover a two-finger periodic motion or another effective strategy and clear it.

```text
single finger: impossible
learned strategy: clear
```

This is the first milestone of v0.1.

[日本語](仕様書.md)

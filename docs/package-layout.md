# Package layout

`src/dmdod/` is intentionally a small namespace root. Domain code belongs in
subpackages; new top-level modules should be treated as an architecture change.

## Domains

- `adofai/` — chart parsing, geometry, timing, rules, playable segments.
- `motor/` — body, keyboard, motor environment, N-key capacity and actuation.
- `policies/` — recurrent, predictive, visual, toy, and N-key policies.
- `teachers/` — privileged and finger-routing teachers.
- `data/` — multi-chart and TUF dataset tooling.
- `envs/` — rhythm, geometry, real-chart, N-key, and simulator environments.
- `features/` — perception, HUD, real-chart, visual, and pattern features.
- `evaluation/` — evaluators, parallel rollout, HUD evaluation, benchmarks.
- `connectome/` — MaleCNS preprocessing and connectome policy backends.
- `integrations/` — external application transports such as ExtremeEditor.
- `cli/` — terminal/progress frontends.
- `training/` — current training algorithms, configuration, orchestration.
- `legacy/four_key/` — superseded four-key implementation retained for
  reproducibility.

Only `calibration.py` and `profiles.py` remain as small cross-cutting modules.

## Compatibility

Historical imports such as:

```python
from dmdod.n_key_motor import NKeyAction
from dmdod.fly_connectome_policy import NKeyFlyConnectomeActorCritic
```

continue to resolve through temporary aliases installed by `dmdod.__init__`.
New code must use structured imports:

```python
from dmdod.motor.n_key import NKeyAction
from dmdod.connectome.fly_policy import NKeyFlyConnectomeActorCritic
```

The compatibility aliases are migration support, not the preferred API.

## Architecture guard

`tests/test_training_architecture.py` enforces that the package root contains
only:

- `__init__.py`
- `calibration.py`
- `profiles.py`

It also checks representative legacy aliases. This prevents the flat-module
layout from gradually returning.

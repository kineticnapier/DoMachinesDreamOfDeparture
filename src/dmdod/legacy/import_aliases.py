from __future__ import annotations

"""Lazy compatibility imports for the retired flat dmdod module layout."""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys
from types import ModuleType


LEGACY_SUBMODULE_ALIASES: dict[str, str] = {
    "adofai_chart": "dmdod.adofai.chart",
    "adofai_geometry": "dmdod.adofai.geometry",
    "adofai_playable": "dmdod.adofai.playable",
    "adofai_rules": "dmdod.adofai.rules",
    "adofai_timing": "dmdod.adofai.timing",

    "body": "dmdod.motor.body",
    "fast_motor": "dmdod.motor.fast",
    "keyboard": "dmdod.motor.keyboard",
    "motor_env": "dmdod.motor.env",
    "n_key_capacity": "dmdod.motor.capacity",
    "n_key_motor": "dmdod.motor.n_key",

    "recurrent_policy": "dmdod.policies.recurrent",
    "predictive_recurrent_policy": "dmdod.policies.predictive_recurrent",
    "toy_policy": "dmdod.policies.toy",
    "visual_policy": "dmdod.policies.visual",
    "n_key_policy": "dmdod.policies.n_key",

    "finger_agnostic_teacher": "dmdod.teachers.finger_agnostic",
    "privileged_teacher": "dmdod.teachers.privileged",

    "multichart_dataset": "dmdod.data.multichart",
    "tuf_dataset": "dmdod.data.tuf",
    "tuf_balanced_dataset": "dmdod.data.tuf_balanced",
    "tuf_dataset_parallel": "dmdod.data.tuf_parallel",
    "tuf_dataset_preflight": "dmdod.data.preflight",

    "evaluator": "dmdod.evaluation.evaluator",
    "flat_hud_eval": "dmdod.evaluation.flat_hud",
    "parallel_hud_eval": "dmdod.evaluation.parallel_hud",
    "parallel_rollout": "dmdod.evaluation.rollout",
    "benchmark": "dmdod.evaluation.benchmark",

    "perception": "dmdod.features.perception",
    "planet_perception": "dmdod.features.planet",
    "real_chart_features": "dmdod.features.real_chart",
    "real_chart_hud": "dmdod.features.hud",
    "real_chart_hud_features": "dmdod.features.hud_features",
    "visual_observation": "dmdod.features.visual",
    "pattern_memory": "dmdod.features.pattern_memory",

    "rhythm_env": "dmdod.envs.rhythm",
    "geometry_rhythm_env": "dmdod.envs.geometry_rhythm",
    "motion_geometry_env": "dmdod.envs.motion_geometry",
    "pattern_geometry_env": "dmdod.envs.pattern_geometry",
    "real_chart_env": "dmdod.envs.real_chart",
    "n_key_real_chart": "dmdod.envs.n_key",
    "simulator": "dmdod.envs.simulator",

    "n_key_training": "dmdod.training.n_key",
    "n_key_dagger_continuation": "dmdod.training.dagger_continuation",
    "training_progress": "dmdod.training.progress",
    "curriculum": "dmdod.training.curriculum",
    "ablation": "dmdod.training.ablation",
    "fitting": "dmdod.training.fitting",

    "four_key_calibration": "dmdod.legacy.four_key.calibration",
    "four_key_motor": "dmdod.legacy.four_key.motor",
    "four_key_policy": "dmdod.legacy.four_key.policy",
    "four_key_real_chart": "dmdod.legacy.four_key.real_chart",
    "four_key_training": "dmdod.legacy.four_key.training",

    "fly_connectome_policy": "dmdod.connectome.fly_policy",
    "random_connectome_policy": "dmdod.connectome.random_policy",
    "malecns_connectome": "dmdod.connectome.malecns",
    "modern_cli": "dmdod.cli.modern",
    "modern_cli_bootstrap": "dmdod.cli.bootstrap",
    "modern_cli_dagger": "dmdod.cli.dagger",
    "modern_cli_live": "dmdod.cli.live",
    "extremeeditor_ipc": "dmdod.integrations.extremeeditor",
}


class _LegacyAliasLoader(Loader):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

    def create_module(self, spec):
        return None

    def exec_module(self, module: ModuleType) -> None:
        target = import_module(self.target_name)
        preserved = {
            "__name__": module.__name__,
            "__package__": module.__package__,
            "__loader__": module.__loader__,
            "__spec__": module.__spec__,
        }
        for name, value in target.__dict__.items():
            if name in preserved:
                continue
            module.__dict__[name] = value
        module.__dict__.update(preserved)
        module.__dict__["__legacy_target__"] = self.target_name


class _LegacyAliasFinder(MetaPathFinder):
    marker = "dmdod-legacy-alias-finder"

    def find_spec(self, fullname: str, path=None, target=None):
        prefix = "dmdod."
        if not fullname.startswith(prefix):
            return None
        short_name = fullname[len(prefix):]
        if "." in short_name:
            return None
        target_name = LEGACY_SUBMODULE_ALIASES.get(short_name)
        if target_name is None:
            return None
        return spec_from_loader(fullname, _LegacyAliasLoader(target_name))


def install_legacy_import_aliases() -> None:
    if any(
        getattr(finder, "marker", None) == _LegacyAliasFinder.marker
        for finder in sys.meta_path
    ):
        return
    # Put the alias finder after the normal built-in/frozen finders but before
    # PathFinder reaches the filesystem. The flat files are intentionally gone.
    insert_at = max(0, len(sys.meta_path) - 1)
    sys.meta_path.insert(insert_at, _LegacyAliasFinder())

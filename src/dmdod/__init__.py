"""Do Machines Dream of Departure? simulator package."""

from .adofai.chart import (
    AdoFaiAction,
    AdoFaiChart,
    load_adofai,
    parse_adofai_bytes,
    parse_adofai_text,
)
from .adofai.geometry import AdoFaiFloorGeometry, build_floor_geometry
from .adofai.playable import (
    PlayableChartSegment,
    PlayableChartTarget,
    build_playable_segment,
)
from .adofai.rules import (
    ABSOLUTE_MIN_S,
    COUNTED_BASE_DEG,
    EARLY_LATE_PERFECT_BASE_DEG,
    EARLY_LATE_PERFECT_MIN_S,
    LENIENT_COUNTED_MIN_S,
    LENIENT_OPTION_MINIMUM_BPM_CUSTOM,
    MOBILE_COUNTED_MIN_S,
    MOBILE_EARLY_LATE_PERFECT_MIN_S,
    MOBILE_PERFECT_MIN_S,
    MULTIPRESS_DAMAGE,
    MULTIPRESS_RECOVERY_PER_BEAT,
    MULTIPRESS_RESET_AFTER_BEATS,
    NORMAL_COUNTED_MIN_S,
    NORMAL_THRESHOLD_BPM,
    OVERLOAD_DAMAGE,
    OVERLOAD_LIMIT,
    OVERLOAD_RECOVERY_PER_BEAT,
    PERFECT_BASE_DEG,
    PURE_PERFECT_MIN_S,
    STRICT_COUNTED_MIN_S,
    STRICT_OPTION_MINIMUM_BPM_CUSTOM,
    OverloadCounter,
    TimingDifficulty,
    TimingJudgement,
    TimingWindows,
    classify_normal_timing,
    classify_timing,
    normal_accuracy_percent,
    normal_timing_windows,
    timing_windows,
    x_accuracy_components,
    x_accuracy_percent,
    x_accuracy_weight,
)
from .adofai.timing import (
    AdoFaiFloorTiming,
    CompiledAdoFaiChart,
    CompiledChartFloor,
    VisibleChartFloor,
    build_stock_timing,
    compile_adofai,
    load_compiled_adofai,
)
from .motor.body import (
    BilateralConfig,
    BilateralState,
    BodyConfig,
    FingerConfig,
    FingerState,
    HandConfig,
    HandState,
    TwoFingerBody,
)
from .calibration import INITIAL_RATE_TARGETS, RateTarget
from .evaluation.evaluator import EvaluationSummary, Judgement, TargetHit, TimingEvaluator
from .motor.keyboard import KeyConfig, KeyEvent, KeyState, TwoKeyKeyboard
from .motor.env import (
    LeftThresholdReflexPolicy,
    MotorAction,
    MotorDiagnostics,
    MotorEnv,
    MotorObservation,
    MotorPolicy,
    MotorTransition,
    TimedKeyEvent,
)
from .envs.pattern_geometry import (
    PatternGeometryObservation,
    PatternGeometryStep,
    PatternMemoryGeometryEnv,
)
from .features.pattern_memory import PatternMemory, PatternMemoryEntry, PatternMemoryFeatures
from .features.perception import VisualCueConfig, VisualCueEncoder, VisualCueObservation
from .features.planet import (
    PlanetGeometryObservation,
    PlanetVisionConfig,
    StraightPlanetGeometryEncoder,
)
from .profiles import PERSONAL_BLUE_SWITCH_V0_1_NAME, personal_blue_switch_v0_1
from .envs.real_chart import (
    RealChartMotorEnv,
    RealChartObservation,
    RealChartStep,
    RelativeVisibleFloor,
)
from .features.real_chart import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_EVENT_MARKERS,
    REAL_CHART_FLOOR_FEATURE_DIM,
    REAL_CHART_INPUT_DIM,
    RealChartFeatureConfig,
    encode_real_chart_observation,
)
from .envs.rhythm import (
    EpisodeStats,
    RewardConfig,
    RhythmMotorEnv,
    RhythmObservation,
    RhythmStep,
    make_regular_targets,
)
from .envs.simulator import Simulation, SimulationConfig, StepResult

__all__ = [
    "AdoFaiAction",
    "AdoFaiChart",
    "AdoFaiFloorGeometry",
    "AdoFaiFloorTiming",
    "CompiledAdoFaiChart",
    "CompiledChartFloor",
    "VisibleChartFloor",
    "PlayableChartSegment",
    "PlayableChartTarget",
    "build_playable_segment",
    "load_adofai",
    "parse_adofai_bytes",
    "parse_adofai_text",
    "build_floor_geometry",
    "build_stock_timing",
    "compile_adofai",
    "load_compiled_adofai",
    "ABSOLUTE_MIN_S",
    "COUNTED_BASE_DEG",
    "EARLY_LATE_PERFECT_BASE_DEG",
    "EARLY_LATE_PERFECT_MIN_S",
    "LENIENT_COUNTED_MIN_S",
    "LENIENT_OPTION_MINIMUM_BPM_CUSTOM",
    "MOBILE_COUNTED_MIN_S",
    "MOBILE_EARLY_LATE_PERFECT_MIN_S",
    "MOBILE_PERFECT_MIN_S",
    "MULTIPRESS_DAMAGE",
    "MULTIPRESS_RECOVERY_PER_BEAT",
    "MULTIPRESS_RESET_AFTER_BEATS",
    "NORMAL_COUNTED_MIN_S",
    "NORMAL_THRESHOLD_BPM",
    "OVERLOAD_DAMAGE",
    "OVERLOAD_LIMIT",
    "OVERLOAD_RECOVERY_PER_BEAT",
    "PERFECT_BASE_DEG",
    "PURE_PERFECT_MIN_S",
    "STRICT_COUNTED_MIN_S",
    "STRICT_OPTION_MINIMUM_BPM_CUSTOM",
    "OverloadCounter",
    "TimingDifficulty",
    "TimingJudgement",
    "TimingWindows",
    "classify_normal_timing",
    "classify_timing",
    "normal_accuracy_percent",
    "normal_timing_windows",
    "timing_windows",
    "x_accuracy_components",
    "x_accuracy_percent",
    "x_accuracy_weight",
    "BilateralConfig",
    "BilateralState",
    "BodyConfig",
    "FingerConfig",
    "FingerState",
    "HandConfig",
    "HandState",
    "TwoFingerBody",
    "RateTarget",
    "INITIAL_RATE_TARGETS",
    "EvaluationSummary",
    "Judgement",
    "TargetHit",
    "TimingEvaluator",
    "PERSONAL_BLUE_SWITCH_V0_1_NAME",
    "personal_blue_switch_v0_1",
    "KeyConfig",
    "KeyEvent",
    "KeyState",
    "TwoKeyKeyboard",
    "MotorAction",
    "MotorObservation",
    "TimedKeyEvent",
    "MotorDiagnostics",
    "MotorTransition",
    "MotorPolicy",
    "MotorEnv",
    "LeftThresholdReflexPolicy",
    "PatternMemory",
    "PatternMemoryEntry",
    "PatternMemoryFeatures",
    "PatternMemoryGeometryEnv",
    "PatternGeometryObservation",
    "PatternGeometryStep",
    "VisualCueConfig",
    "VisualCueEncoder",
    "VisualCueObservation",
    "PlanetGeometryObservation",
    "PlanetVisionConfig",
    "StraightPlanetGeometryEncoder",
    "RelativeVisibleFloor",
    "RealChartObservation",
    "RealChartStep",
    "RealChartMotorEnv",
    "RealChartFeatureConfig",
    "DEFAULT_REAL_CHART_FEATURE_CONFIG",
    "REAL_CHART_EVENT_MARKERS",
    "REAL_CHART_FLOOR_FEATURE_DIM",
    "REAL_CHART_INPUT_DIM",
    "encode_real_chart_observation",
    "EpisodeStats",
    "RewardConfig",
    "RhythmMotorEnv",
    "RhythmObservation",
    "RhythmStep",
    "make_regular_targets",
    "Simulation",
    "SimulationConfig",
    "StepResult",
]


# Temporary import compatibility while the flat package layout is retired.
# Internal code should import the structured package paths directly. Keeping
# these aliases here lets historical scripts/tests continue importing old
# submodule names without leaving duplicate files in src/dmdod/.
import importlib as _importlib
import sys as _sys

_COMPAT_SUBMODULES = {
    # Core package split
    "adofai_chart": ".adofai.chart",
    "adofai_geometry": ".adofai.geometry",
    "adofai_playable": ".adofai.playable",
    "adofai_rules": ".adofai.rules",
    "adofai_timing": ".adofai.timing",

    "body": ".motor.body",
    "fast_motor": ".motor.fast",
    "keyboard": ".motor.keyboard",
    "motor_env": ".motor.env",
    "n_key_capacity": ".motor.capacity",
    "n_key_motor": ".motor.n_key",

    "recurrent_policy": ".policies.recurrent",
    "predictive_recurrent_policy": ".policies.predictive_recurrent",
    "toy_policy": ".policies.toy",
    "visual_policy": ".policies.visual",
    "n_key_policy": ".policies.n_key",

    "finger_agnostic_teacher": ".teachers.finger_agnostic",
    "privileged_teacher": ".teachers.privileged",

    "multichart_dataset": ".data.multichart",
    "tuf_dataset": ".data.tuf",
    "tuf_balanced_dataset": ".data.tuf_balanced",
    "tuf_dataset_parallel": ".data.tuf_parallel",
    "tuf_dataset_preflight": ".data.preflight",

    "evaluator": ".evaluation.evaluator",
    "flat_hud_eval": ".evaluation.flat_hud",
    "parallel_hud_eval": ".evaluation.parallel_hud",
    "parallel_rollout": ".evaluation.rollout",
    "benchmark": ".evaluation.benchmark",

    "perception": ".features.perception",
    "planet_perception": ".features.planet",
    "real_chart_features": ".features.real_chart",
    "real_chart_hud": ".features.hud",
    "real_chart_hud_features": ".features.hud_features",
    "visual_observation": ".features.visual",
    "pattern_memory": ".features.pattern_memory",

    "rhythm_env": ".envs.rhythm",
    "geometry_rhythm_env": ".envs.geometry_rhythm",
    "motion_geometry_env": ".envs.motion_geometry",
    "pattern_geometry_env": ".envs.pattern_geometry",
    "real_chart_env": ".envs.real_chart",
    "n_key_real_chart": ".envs.n_key",
    "simulator": ".envs.simulator",

    # Already-migrated packages
    "fly_connectome_policy": ".connectome.fly_policy",
    "random_connectome_policy": ".connectome.random_policy",
    "malecns_connectome": ".connectome.malecns",
    "modern_cli": ".cli.modern",
    "modern_cli_bootstrap": ".cli.bootstrap",
    "modern_cli_dagger": ".cli.dagger",
    "modern_cli_live": ".cli.live",
    "extremeeditor_ipc": ".integrations.extremeeditor",

for _legacy_name, _target_name in _COMPAT_SUBMODULES.items():
    _sys.modules.setdefault(
        f"{__name__}.{_legacy_name}",
        _importlib.import_module(_target_name, __name__),
    )

del _legacy_name, _target_name, _importlib, _sys

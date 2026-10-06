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
from .body import (
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
from .evaluator import EvaluationSummary, Judgement, TargetHit, TimingEvaluator
from .keyboard import KeyConfig, KeyEvent, KeyState, TwoKeyKeyboard
from .motor_env import (
    LeftThresholdReflexPolicy,
    MotorAction,
    MotorDiagnostics,
    MotorEnv,
    MotorObservation,
    MotorPolicy,
    MotorTransition,
    TimedKeyEvent,
)
from .pattern_geometry_env import (
    PatternGeometryObservation,
    PatternGeometryStep,
    PatternMemoryGeometryEnv,
)
from .pattern_memory import PatternMemory, PatternMemoryEntry, PatternMemoryFeatures
from .perception import VisualCueConfig, VisualCueEncoder, VisualCueObservation
from .planet_perception import (
    PlanetGeometryObservation,
    PlanetVisionConfig,
    StraightPlanetGeometryEncoder,
)
from .profiles import PERSONAL_BLUE_SWITCH_V0_1_NAME, personal_blue_switch_v0_1
from .real_chart_env import (
    RealChartMotorEnv,
    RealChartObservation,
    RealChartStep,
    RelativeVisibleFloor,
)
from .real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_EVENT_MARKERS,
    REAL_CHART_FLOOR_FEATURE_DIM,
    REAL_CHART_INPUT_DIM,
    RealChartFeatureConfig,
    encode_real_chart_observation,
)
from .rhythm_env import (
    EpisodeStats,
    RewardConfig,
    RhythmMotorEnv,
    RhythmObservation,
    RhythmStep,
    make_regular_targets,
)
from .simulator import Simulation, SimulationConfig, StepResult

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
    "adofai_chart": ".adofai.chart",
    "adofai_geometry": ".adofai.geometry",
    "adofai_playable": ".adofai.playable",
    "adofai_rules": ".adofai.rules",
    "adofai_timing": ".adofai.timing",
    "fly_connectome_policy": ".connectome.fly_policy",
    "random_connectome_policy": ".connectome.random_policy",
    "malecns_connectome": ".connectome.malecns",
    "modern_cli": ".cli.modern",
    "modern_cli_bootstrap": ".cli.bootstrap",
    "modern_cli_dagger": ".cli.dagger",
    "modern_cli_live": ".cli.live",
    "extremeeditor_ipc": ".integrations.extremeeditor",
}

for _legacy_name, _target_name in _COMPAT_SUBMODULES.items():
    _sys.modules.setdefault(
        f"{__name__}.{_legacy_name}",
        _importlib.import_module(_target_name, __name__),
    )

del _legacy_name, _target_name, _importlib, _sys

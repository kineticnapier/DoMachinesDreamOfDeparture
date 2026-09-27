"""Do Machines Dream of Departure? simulator package."""

from .adofai_rules import (
    NORMAL_THRESHOLD_BPM,
    OVERLOAD_LIMIT,
    OverloadCounter,
    TimingJudgement,
    TimingWindows,
    classify_normal_timing,
    normal_timing_windows,
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
from .perception import VisualCueConfig, VisualCueEncoder, VisualCueObservation
from .profiles import PERSONAL_BLUE_SWITCH_V0_1_NAME, personal_blue_switch_v0_1
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
    "NORMAL_THRESHOLD_BPM",
    "OVERLOAD_LIMIT",
    "OverloadCounter",
    "TimingJudgement",
    "TimingWindows",
    "classify_normal_timing",
    "normal_timing_windows",
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
    "VisualCueConfig",
    "VisualCueEncoder",
    "VisualCueObservation",
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

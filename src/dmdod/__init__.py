"""Do Machines Dream of Departure? simulator package."""

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
from .keyboard import KeyConfig, KeyEvent, KeyState, TwoKeyKeyboard
from .profiles import PERSONAL_BLUE_SWITCH_V0_1_NAME, personal_blue_switch_v0_1
from .simulator import Simulation, SimulationConfig, StepResult

__all__ = [
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
    "PERSONAL_BLUE_SWITCH_V0_1_NAME",
    "personal_blue_switch_v0_1",
    "KeyConfig",
    "KeyEvent",
    "KeyState",
    "TwoKeyKeyboard",
    "Simulation",
    "SimulationConfig",
    "StepResult",
]

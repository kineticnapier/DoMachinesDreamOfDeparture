"""Do Machines Dream of Departure? simulator package."""

from .body import BodyConfig, FingerConfig, FingerState, TwoFingerBody
from .calibration import INITIAL_GLOBAL_RATE_CEILING_HZ, INITIAL_RATE_TARGETS, RateTarget
from .keyboard import KeyConfig, KeyEvent, KeyState, TwoKeyKeyboard
from .simulator import Simulation, SimulationConfig, StepResult

__all__ = [
    "BodyConfig",
    "FingerConfig",
    "FingerState",
    "TwoFingerBody",
    "RateTarget",
    "INITIAL_RATE_TARGETS",
    "INITIAL_GLOBAL_RATE_CEILING_HZ",
    "KeyConfig",
    "KeyEvent",
    "KeyState",
    "TwoKeyKeyboard",
    "Simulation",
    "SimulationConfig",
    "StepResult",
]

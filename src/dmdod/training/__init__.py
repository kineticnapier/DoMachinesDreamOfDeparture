"""Stable training infrastructure for current DMDOD experiments.

Version-numbered scripts are historical experiment entry points. New training
code should live in this package and be selected through configuration instead
of creating another train_real_chart_vXYZ.py file.
"""

from .config import TrainingConfig, load_training_config

__all__ = ["TrainingConfig", "load_training_config"]

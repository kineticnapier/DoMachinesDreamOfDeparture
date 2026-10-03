from __future__ import annotations

"""v1.6.3: trust-region continuous-action DAgger for N-key policies.

A full BC epoch can move a continuous controller across a sharp closed-loop
boundary even when its teacher-forced loss improves.  This trainer therefore
uses each BC epoch only to propose a direction in parameter space.  The
proposal is line-searched from the current accepted Train-safe state with fixed
alphas.  Only a Train-safe interpolation that improves the Train selection key
is accepted, and the next epoch starts from that accepted state rather than a
rejected proposal.  Validation is evaluated once after Train-only selection;
Final is never touched.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v160_n_key_bootstrap as v160
import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
from dmdod.multichart_dataset import discover_multichart_dataset
from dmdod.n_key_names import *  # type: ignore[import-not-found]

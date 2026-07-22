"""Transition-centric incomplete-information world-action models.

This package deliberately does not replace the earlier ``incomplete_info``
modules.  It reuses their promoted observation/action tokenizers as frozen
front ends while learning a new predictive state for planning.
"""

from .config import FactorizedDynamicsConfig, PredictiveBeliefConfig
from .dynamics import FactorizedWorldActionDynamics
from .encoder import PredictiveBeliefPretrainer, WorldActionBeliefEncoder
from .losses import factorized_world_action_dynamics_loss, predictive_belief_loss
from .module import WorldActionDynamicsModule

__all__ = [
    "FactorizedDynamicsConfig",
    "FactorizedWorldActionDynamics",
    "PredictiveBeliefConfig",
    "PredictiveBeliefPretrainer",
    "WorldActionBeliefEncoder",
    "WorldActionDynamicsModule",
    "factorized_world_action_dynamics_loss",
    "predictive_belief_loss",
]
